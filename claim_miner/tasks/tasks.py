import datetime
from typing import (
    Dict,
    Optional,
    List,
    ForwardRef,
    Literal,
    Union,
    Type,
    Tuple,
    Any,
    TYPE_CHECKING,
    TypeVar,
    Generic,
)
from logging import getLogger
import re

from sqlalchemy import select, and_, String, delete, or_, Select, BigInteger
from sqlalchemy.sql import case, literal_column
from sqlalchemy.orm import joinedload, subqueryload, aliased as sa_aliased
from sqlalchemy.sql.functions import count, cast, coalesce
from pydantic import Field, model_validator, field_validator

from .. import Session, pyd_models, hashfs, escape_fn
from ..task_registry import CMTask, CMTemplatedTask, hookimpl, TaskRegistry
from ..llm import (
    processing_models,
    parsers_by_name,
    parser_type,
    processing_model_from_name,
)
from ..utils import classproperty, safe_lang_detect

from ..pyd_models import (
    fragment_type,
    BASE_EMBED_MODEL,
    BASE_DOC_EMBED_MODEL,
    embedding_model,
    topic_type,
    link_type,
    TaskTriggerModel,
    AnalysisModel,
    StatementModel,
    TaskTemplateModel,
    ClusterDataModel,
    DocumentModel,
    FragmentModel,
    AnalysisWithTemplateModel,
    visible_statement_types,
    process_status,
    permission_values,
    AnyTopicModel,
    DocumentOrFragmentModel,
)

logger = getLogger(__name__)


async def theme_id_from_collection(collection, session=None):
    if session is None:
        async with Session() as session:
            return await theme_id_from_collection(collection, session)
    from ..models import Collection

    collection = await session.scalar(select(Collection).filter_by(name=collection))
    if collection:
        return collection.params.get("theme_id")


class ClusterAnalysisModel(AnalysisModel):
    analyzer_name: Literal["cluster"] = "cluster"
    algorithm: str = "dbscan"
    model: embedding_model = BASE_EMBED_MODEL
    eps: float = 0.1
    min_samples: int = 6
    scale: Optional[fragment_type] = None
    clusters: Optional[List[ClusterDataModel]] = None
    # TODO: collection_name or id must be there


class ClusterTask(CMTask[ClusterAnalysisModel]):
    name = "cluster"
    task_scale = (topic_type.collection,)
    materialize_analysis = True
    can_delete_results = True
    edit_template_name = "claim_cluster_analysis.html"

    def get_channel_key(self):
        return self.analysis.collection_name

    @classmethod
    def add_parse_params(cls, parser):
        parser.add_argument("--analysis-ids", type=int, action="append")
        parser.add_argument("--min-samples", type=int, default=6)
        parser.add_argument("--eps", type=float, default=0.1)
        parser.add_argument(
            "--scale", choices=[t.name for t in fragment_type], default=None
        )
        parser.add_argument("--algorithm", choices=["dbscan"], default="dbscan")

    async def run(self):
        from .cluster import do_cluster_claims

        await do_cluster_claims(self.analysis.id)
        await self.set_completed()

    async def delete_results(self, session):
        analysis = await self.get_db_analysis(session)
        assert analysis
        await session.refresh(analysis, "clusters")
        for cluster in analysis.clusters:
            await session.delete_all(cluster)
        analysis.status = process_status.pending
        analysis.results = {}

    async def enrich_edit_form_data(self, session, base_vars):
        from ..models import Analysis, ClusterData, InClusterData, User

        analysis = base_vars["analysis"]
        # TODO: Order by distance from distinguished here also?
        clusters = await session.execute(
            select(ClusterData)
            .join(Analysis)
            .filter(Analysis.id == analysis.id)
            .order_by(ClusterData.cluster_size.desc())
            .options(
                joinedload(ClusterData.distinguished_claim),
                subqueryload(ClusterData.has_cluster_rels).subqueryload(
                    InClusterData.fragment
                ),
            )
        )
        clusters = [cluster for (cluster,) in clusters]

        user_ids = {analysis.creator_id}
        for cluster in clusters:
            user_ids.update(icd.confirmed_by_id for icd in cluster.has_cluster_rels)
        user_ids.discard(None)
        usernames = await session.execute(
            select(User.id, User.handle).filter(User.id.in_(list(user_ids)))
        )
        usernames = dict(list(usernames))
        base_vars.update(
            dict(grouped_clusters=((analysis, clusters),), usernames=usernames)
        )

    @classmethod
    def task_form_before(cls):
        """Return a form for the creation or edition of the task object before execution, if needed."""
        # Note: Scale and algorithm are implicit for now
        return """
  <p>
  <label for="min_samples">Minimum number of statements per cluster:</label>
  <input name="min_samples" type="number" min="2" max="100" step="1" value="6"/>
  </p>
  <p>
  <label for="eps">Epsilon (cluster breadth):</label>
  <input name="eps" type="number" min="0" max="1" step="0.05" value="0.1"/>
  </p>"""


class AutoclassifyAnalysisModel(AnalysisModel):
    analyzer_name: Literal["autoclassify"] = "autoclassify"
    target_id: Optional[int] = None
    target: Optional[AnyTopicModel] = None
    model: embedding_model = BASE_EMBED_MODEL


class AutoclassifyTask(CMTask[AutoclassifyAnalysisModel]):
    name = "autoclassify"
    task_scale = (topic_type.standalone,)

    @classmethod
    async def compute_default_triggers(cls, session, task_registry: TaskRegistry):
        yield TaskTriggerModel(
            target_analyzer_id=task_registry.analyzer_by_name[cls.name].id,
            analyzer_trigger_id=task_registry.analyzer_by_name["embed_fragment"].id,
            automatic=True,
        )

    @classmethod
    def enrich_trigger_params(cls, session, params, task, collection):
        if task.name == "embed_fragment":
            template = task.analysis.get_task_template()
            params["model"] = template.model.name

    async def run(self):
        from .cluster import do_autoclassify, do_autoclassify_all

        analysis_model = self.analysis
        if analysis_model.target_id:
            return await do_autoclassify(analysis_model.target_id)
        else:
            return await do_autoclassify_all(analysis_model.collection_name)

    @classmethod
    async def check_trigger_conditions(
        cls,
        session,
        trigger,
        target_id: Optional[int],
        task: Optional[CMTask],
        collection_name: Optional[str],
    ) -> bool:
        from ..models import ClusterData, Analysis, Topic

        if not target_id:
            return False
        topic_type = await session.scalar(select(Topic.type).filter_by(id=target_id))
        if topic_type == topic_type.fragment:
            return False
        template = task.analysis.get_task_template() if task else None
        id = await session.scalar(
            select(ClusterData.id)
            .filter(ClusterData.auto_include_diameter > 0)
            .join(Analysis)
            .filter(cast(Analysis.params["model"], String) == template.model.name)
            .limit(1)
        )
        return bool(id)

    @classmethod
    def count_status_query(cls, collection_name=None, **kwargs):
        from ..models import Statement, Collection

        # TODO: Count as complete if in min_pos, max_pos
        q = select(
            1, literal_column("'inapplicable'"), None, count(Statement.id), 0
        ).filter(Statement.type == topic_type.standalone)
        if collection_name:
            q = q.join(Collection, Statement.collections).filter_by(
                name=collection_name
            )
        return q

    @classmethod
    def query_with_status(
        cls,
        status: process_status,
        collection_name: Optional[str] = None,
        task_template_name=None,
        analyzer_version: Optional[int] = None,
        **kwargs,
    ) -> Tuple[bool, Select[Tuple[BigInteger]]]:
        from ..models import Statement, Collection

        if status == process_status.inapplicable:
            q = select(Statement.id).filter(Statement.type == topic_type.standalone)
        else:
            return (False, select(Statement.id).filter(False))
        if collection_name:
            q = q.join(Collection, Statement.collections).filter_by(
                name=collection_name
            )
        return (False, q)


class DownloadAnalysisModel(AnalysisModel):
    analyzer_name: Literal["download"] = "download"
    target_id: Optional[int] = None
    target: Optional[DocumentModel] = None


class DownloadTask(CMTask[DownloadAnalysisModel]):
    name = "download"
    version = 2
    task_scale = (topic_type.document,)
    collection_specific_results = False
    can_delete_results = True
    can_apply_bulk = True

    @classmethod
    async def compute_default_triggers(cls, session, task_registry: TaskRegistry):
        yield TaskTriggerModel(
            target_analyzer_id=task_registry.analyzer_by_name[cls.name].id,
            creation_trigger_id=topic_type.document,
            automatic=True,
        )

    @classproperty
    def trigger_task_permission(cls) -> Optional[ForwardRef("permission")]:
        from ..pyd_models import permission

        return permission.add_document

    @classmethod
    async def status_for(cls, session, target_id, **params) -> process_status:
        from ..models import Document

        document = await session.get(Document, target_id)
        if document.file_identity:
            return process_status.complete
        if document.load_status == "error":
            return process_status.error
        return process_status.pending

    @classmethod
    def count_status_query(cls, collection_name=None, **kwargs):
        from ..models import Document, Collection

        q = select(
            1,
            case(
                (Document.file_identity.is_not(None), "complete"),
                (Document.return_code.is_(None), "not_requested"),
                else_="error",
            ),
            None,
            count(Document.id),
            0,
        ).group_by(Document.file_identity.is_not(None), Document.return_code.is_(None))
        if collection_name:
            q = q.join(Collection, Document.collections).filter_by(name=collection_name)
        return q

    @classmethod
    def query_with_status(
        cls,
        status: process_status,
        collection_name: Optional[str] = None,
        task_template_name=None,
        analyzer_version: Optional[int] = None,
        **kwargs,
    ) -> Tuple[bool, Select[Tuple[BigInteger]]]:
        from ..models import Document, Collection

        if status == process_status.complete:
            q = select(Document.id).filter(Document.file_identity.is_not(None))
        elif status == process_status.not_requested:
            q = select(Document.id).filter(Document.return_code.is_(None))
        elif status == process_status.error:
            q = select(Document.id).filter(Document.return_code >= 300)
        else:
            return (False, select(Document.id).filter(False))
        if collection_name:
            q = q.join(Collection, Document.collections).filter_by(name=collection_name)
        return (False, q)

    async def run(self):
        from .download import do_download

        await do_download(self.analysis.target_id)

    async def delete_results(self, session):
        from ..models import Document

        document = await session.get(Document, self.analysis.target_id)
        assert document
        if document.file_identity:
            await session.scalar(
                select(count(Document.id)).filter_by(
                    file_identity=document.file_identity
                )
            )
            if count == 1:
                f = hashfs(document.file_identity)
                f.delete()
        document.file_identity = None
        document.file_length = None

    @classmethod
    async def check_trigger_conditions(
        cls,
        session,
        trigger,
        target_id: Optional[int],
        task: Optional[CMTask],
        collection_name: Optional[str],
    ) -> bool:
        from ..models import Document

        if not target_id:
            return False
        topic = await session.get(Document, target_id)
        return topic and (topic.file_identity is None)


class EmbedTask:  # mixin
    collection_specific_results = False
    can_apply_bulk = True

    @classmethod
    def add_parse_params(cls, parser):
        super().add_parse_params(parser)
        parser.add_argument("--batch-size", type=int, default=0)


class EmbedTaskTemplateModel(TaskTemplateModel):
    analyzer_name: Literal["embed_fragment"] = "embed_fragment"
    model: embedding_model = BASE_EMBED_MODEL


class EmbedFragmentAnalysisModel(AnalysisWithTemplateModel):
    analyzer_name: Literal["embed_fragment"] = "embed_fragment"
    batch_size: int = 0
    target_id: Optional[int] = None  # Or union with List[int]?
    target: Optional[AnyTopicModel] = None
    task_template: Optional[EmbedTaskTemplateModel] = None


class EmbedFragmentTask(
    EmbedTask, CMTemplatedTask[EmbedFragmentAnalysisModel, EmbedTaskTemplateModel]
):
    name = "embed_fragment"
    task_scale = (topic_type.standalone, topic_type.fragment)

    @classmethod
    async def compute_default_triggers(cls, session, task_registry: TaskRegistry):
        from ..models import TaskTemplate, Collection, model_names

        # Ensure a few default task templates.
        new_templates = []
        for model, model_name in model_names.items():
            if model_name not in task_registry.task_template_by_nickname:
                new_templates.append(
                    TaskTemplate(
                        nickname=model_name,
                        analyzer_id=task_registry.analyzer_by_name[cls.name].id,
                        params=dict(model=model.name),
                    )
                )
        if new_templates:
            for template in new_templates:
                session.add(template)
            await session.commit()
            template_models = [t.as_model(session) for t in new_templates]
            for template in template_models:
                task_registry.update_template(template)
        # Now we can compute the triggers
        templates = {
            t.model.name: t for t in task_registry.task_templates_by_name[cls.name]
        }
        yield TaskTriggerModel(
            target_analyzer_id=task_registry.analyzer_by_name[cls.name].id,
            creation_trigger_id=topic_type.standalone,
            task_template_id=templates[BASE_EMBED_MODEL.name].id,
            automatic=True,
        )
        yield TaskTriggerModel(
            target_analyzer_id=task_registry.analyzer_by_name[cls.name].id,
            creation_trigger_id=topic_type.fragment,
            task_template_id=templates[BASE_EMBED_MODEL.name].id,
            automatic=True,
        )
        collections = [c for (c,) in await session.execute(select(Collection))]
        for collection in collections:
            for model in collection.params.get("embeddings", []):
                yield TaskTriggerModel(
                    target_analyzer_id=task_registry.analyzer_by_name[cls.name].id,
                    creation_trigger_id=topic_type.standalone,
                    task_template_id=templates[model].id,
                    collection_id=collection.id,
                )
                yield TaskTriggerModel(
                    target_analyzer_id=task_registry.analyzer_by_name[cls.name].id,
                    creation_trigger_id=topic_type.fragment,
                    task_template_id=templates[model].id,
                    collection_id=collection.id,
                )

    @classmethod
    async def status_for(
        cls, session, target_id: int, task_template: EmbedTaskTemplateModel, **params
    ) -> process_status:
        from ..models import embed_db_model_by_name

        Embedding = embed_db_model_by_name[task_template.model.name]
        exists = await session.scalar(
            select(Embedding.fragment_id.is_not(None)).filter_by(fragment_id=target_id)
        )
        return process_status.complete if exists else process_status.pending

    @classmethod
    def count_status_query(cls, collection_name=None, **kwargs):
        from ..models import (
            StatementOrFragment,
            Collection,
            TopicCollection,
            embed_db_model_by_name,
        )

        registry = TaskRegistry.get_registry()
        queries = []
        for task_template in registry.task_templates_by_name[cls.name]:
            Embedding = embed_db_model_by_name[task_template.model.name]
            # TODO: quote nickname, danger of injection
            q = (
                select(
                    1,
                    case(
                        (Embedding.fragment_id.is_not(None), "complete"),
                        else_="not_requested",
                    ),
                    literal_column(escape_fn(task_template.nickname)),
                    count(StatementOrFragment.id),
                    0,
                )
                .outerjoin(Embedding, Embedding.fragment_id == StatementOrFragment.id)
                .group_by(Embedding.fragment_id.is_not(None))
            )
            if collection_name:
                q = (
                    q.join(
                        TopicCollection,
                        coalesce(StatementOrFragment.doc_id, StatementOrFragment.id)
                        == TopicCollection.topic_id,
                    )
                    .join(Collection, TopicCollection.collection)
                    .filter_by(name=collection_name)
                )
            queries.append(q)
        q0 = queries.pop()
        return q0.union_all(*queries)

    @classmethod
    def query_with_status(
        cls,
        status: process_status,
        collection_name: Optional[str] = None,
        task_template_name=None,
        analyzer_version: Optional[int] = None,
        **kwargs,
    ) -> Tuple[bool, Select[Tuple[BigInteger]]]:
        from ..models import (
            StatementOrFragment,
            Collection,
            TopicCollection,
            embed_db_model_by_name,
        )

        registry = TaskRegistry.get_registry()
        task_template = registry.task_template_by_nickname.get(task_template_name)
        if not task_template:
            return (False, select(StatementOrFragment.id).filter(False))
        Embedding = embed_db_model_by_name[task_template.model.name]
        q = select(StatementOrFragment.id)
        if status == process_status.complete:
            q = q.join(Embedding, Embedding.fragment_id == StatementOrFragment.id)
        elif status == process_status.not_requested:
            q = q.outerjoin(
                Embedding, Embedding.fragment_id == StatementOrFragment.id
            ).filter(Embedding.fragment_id.is_(None))
        else:
            return (False, select(StatementOrFragment.id).filter(False))
        if collection_name:
            q = (
                q.join(
                    TopicCollection,
                    coalesce(StatementOrFragment.doc_id, StatementOrFragment.id)
                    == TopicCollection.topic_id,
                )
                .join(Collection, TopicCollection.collection)
                .filter_by(name=collection_name)
            )
        return (False, q)

    async def run(self):
        from .embed import do_embed_fragment

        analysis_model = self.analysis
        task_template = analysis_model.get_task_template()
        return await do_embed_fragment(
            analysis_model.target_id, embedding_model[task_template.model.name]
        )

    @classmethod
    def batch_size(cls, model=None, **kwargs):
        from ..embed import embedder_registry

        model = model or BASE_EMBED_MODEL
        model = embedding_model[model] if isinstance(model, str) else model
        embedder = embedder_registry[model.name]
        return kwargs.get("batch_size", 0) or embedder.batch_size

    @classmethod
    def add_parse_params(cls, parser):
        from ..models import embed_db_model_by_name

        super().add_parse_params(parser)
        parser.add_argument(
            "--model",
            choices=embed_db_model_by_name.keys(),
            default=BASE_EMBED_MODEL.name,
        )


class EmbedDocAnalysisModel(AnalysisModel):
    analyzer_name: Literal["embed_doc"] = "embed_doc"
    target_id: int = None
    target: Optional[AnyTopicModel] = None
    model: embedding_model = BASE_DOC_EMBED_MODEL


class EmbedDocTask(EmbedTask, CMTask[EmbedDocAnalysisModel]):
    name = "embed_doc"
    task_scale = (topic_type.document,)
    can_delete_results = True

    @classmethod
    async def compute_default_triggers(cls, session, task_registry: TaskRegistry):
        yield TaskTriggerModel(
            target_analyzer_id=task_registry.analyzer_by_name[cls.name].id,
            analyzer_trigger_id=task_registry.analyzer_by_name["process_pdf"].id,
            params=dict(model=BASE_DOC_EMBED_MODEL.name),
            automatic=True,
        )
        yield TaskTriggerModel(
            target_analyzer_id=task_registry.analyzer_by_name[cls.name].id,
            analyzer_trigger_id=task_registry.analyzer_by_name["process_html"].id,
            params=dict(model=BASE_DOC_EMBED_MODEL.name),
            automatic=True,
        )
        yield TaskTriggerModel(
            target_analyzer_id=task_registry.analyzer_by_name[cls.name].id,
            analyzer_trigger_id=task_registry.analyzer_by_name["process_text"].id,
            params=dict(model=BASE_DOC_EMBED_MODEL.name),
            automatic=True,
        )

    @classmethod
    async def status_for(
        cls, session, target_id, model=BASE_DOC_EMBED_MODEL, **params
    ) -> process_status:
        from ..models import Document, embed_db_model_by_name

        document = await session.get(Document, target_id)
        if not document.text_identity:
            return process_status.not_ready
        Embedding = embed_db_model_by_name[model.name]
        exists = await session.scalar(
            select(Embedding.doc_id.is_not(None)).filter_by(
                doc_id=target_id, fragment_id=None
            )
        )
        return process_status.complete if exists else process_status.pending

    @classmethod
    def count_status_query(cls, collection_name=None, **kwargs):
        from ..models import Document, Collection, embed_db_model_by_name

        model = kwargs.get("model", BASE_DOC_EMBED_MODEL)
        Embedding = embed_db_model_by_name[model.name]
        q = (
            select(
                1,
                case(
                    (Document.text_identity.is_(None), "not_ready"),
                    (Embedding.doc_id.is_not(None), "complete"),
                    else_="not_requested",
                ),
                None,
                count(Document.id),
                0,
            )
            .outerjoin(
                Embedding,
                and_(Embedding.doc_id == Document.id, Embedding.fragment_id.is_(None)),
            )
            .group_by(Document.text_identity.is_(None), Embedding.doc_id.is_not(None))
        )
        if collection_name:
            q = q.join(Collection, Document.collections).filter_by(name=collection_name)
        return q

    @classmethod
    def query_with_status(
        cls,
        status: process_status,
        collection_name: Optional[str] = None,
        task_template_name=None,
        analyzer_version: Optional[int] = None,
        **kwargs,
    ) -> Tuple[bool, Select[Tuple[BigInteger]]]:
        from ..models import Document, Collection, embed_db_model_by_name

        model = kwargs.get("model", BASE_DOC_EMBED_MODEL)
        Embedding = embed_db_model_by_name[model.name]
        q = select(Document.id)
        if status == process_status.not_ready:
            q = select(Document.id).filter(Document.text_identity.is_(None))
        elif status == process_status.complete:
            q = q.join(
                Embedding,
                and_(Embedding.doc_id == Document.id, Embedding.fragment_id.is_(None)),
            )
        elif status == process_status.not_requested:
            q = (
                q.filter(Document.text_identity.is_not(None))
                .outerjoin(
                    Embedding,
                    and_(
                        Embedding.doc_id == Document.id, Embedding.fragment_id.is_(None)
                    ),
                )
                .filter(Embedding.doc_id.is_(None))
            )
        else:
            return (False, select(Document.id).filter(False))
        if collection_name:
            q = q.join(Collection, Document.collections).filter_by(name=collection_name)
        return (False, q)

    async def run(self):
        from .embed import do_embed_doc

        analysis_model = self.analysis
        return await do_embed_doc([analysis_model.target_id], analysis_model.model)

    async def delete_results(self, session):
        from ..models import embed_db_model_by_name

        embedding_model = embed_db_model_by_name[self.analysis.model.name]
        await session.execute(
            delete(embedding_model).filter_by(
                doc_id=self.analysis.target_id, fragment_id=None
            )
        )


class GdeltAnalysisModel(AnalysisModel):
    analyzer_name: Literal["gdelt"] = "gdelt"
    source: Union[Literal["docs"], Literal["news"]] = "docs"
    limit: int = 10
    date: Optional[datetime.date] = None
    target_id: Optional[int] = None
    target: Optional[StatementModel] = None


class GdeltTask(CMTask[GdeltAnalysisModel]):
    name = "gdelt"
    core = False
    task_scale = (topic_type.standalone,)
    collection_specific_results = False
    task_creates: List[topic_type] = [topic_type.document]
    materialize_analysis = True
    can_reapply_complete: bool = True

    @classmethod
    def setup_models(cls):
        permission_values.add("bigdata_query")

    @classproperty
    def trigger_task_permission(cls) -> Optional[ForwardRef("permission")]:
        from ..pyd_models import permission

        return permission.bigdata_query

    @classmethod
    def query_with_status(
        cls,
        status: process_status,
        collection_name: Optional[str] = None,
        task_template_name=None,
        analyzer_version: Optional[int] = None,
        **kwargs,
    ) -> Tuple[bool, Select[Tuple[BigInteger]]]:
        from ..pyd_models import embedding_model
        from ..models import (
            Analysis,
            Analyzer,
            Collection,
            Statement,
            embed_db_model_by_name,
        )

        Embedding_Use4 = embed_db_model_by_name[
            embedding_model.universal_sentence_encoder_4.name
        ]
        registry = TaskRegistry.get_registry()
        if status == process_status.inapplicable:
            return (False, select(Statement.id).filter(False))
        elif status == process_status.not_requested:
            q = (
                select(Statement.id.distinct())
                .join(Embedding_Use4, Statement.id == Embedding_Use4.fragment_id)
                .outerjoin(Analysis, Analysis.target_id == Statement.id)
            )
            if analyzer_version:
                q = q.outerjoin(
                    Analyzer,
                    and_(
                        Analysis.analyzer_id == Analyzer.id,
                        Analyzer.name == cls.name,
                        Analyzer.version == analyzer_version,
                    ),
                )
            else:
                q = q.outerjoin(
                    Analyzer,
                    and_(
                        Analysis.analyzer_id == Analyzer.id, Analyzer.name == cls.name
                    ),
                )
            q = q.filter(Analysis.id.is_(None), Analyzer.id.is_(None))
        elif status == process_status.not_ready:
            q = (
                select(Statement.id)
                .outerjoin(Embedding_Use4, Statement.id == Embedding_Use4.fragment_id)
                .filter(Embedding_Use4.fragment_id.is_(None))
            )
        else:
            return super(GdeltTask, cls).query_with_status(
                status, collection_name, task_template_name, analyzer_version, **kwargs
            )
        if collection_name:
            q = q.join(Collection, Statement.collections).filter_by(
                name=collection_name
            )
        return (False, q)

    @classmethod
    def add_parse_params(cls, parser):
        parser.add_argument("--analysis-ids", type=int, action="append")
        parser.add_argument("--source", choices=["docs", "news"])
        parser.add_argument("--limit", type=int, default=10)
        parser.add_argument("--date", type=datetime.date)

    async def run(self):
        from .gdelt import do_gdelt

        query_time, document_ids = await do_gdelt(self.analysis.id)
        await self.set_completed(query_time)

    @classmethod
    def task_form_before(cls):
        """Return a form for the creation or edition of the task object before execution, if needed."""
        return """
  <p>
  <label for="limit">Number of documents:</label>
  <input name="limit" type="number" min="1" max="1000" default="100"/>
  </p>
  <p>
  <label for="date">Added after:</label>
  <input name="date" type="date"/>
  </p>
        """


class PromptTaskTemplateModel(TaskTemplateModel):
    model: processing_models = processing_models.gpt_3_5_turbo_1106
    parser: parser_type = parser_type.single_phrase
    link_type: pyd_models.link_type = pyd_models.link_type.freeform
    node_type: fragment_type = fragment_type.standalone
    backwards_link: bool = False
    prompt: Optional[str] = None

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, value: str | processing_models, _) -> processing_models:
        if isinstance(value, str):
            return processing_model_from_name(value)
        return value


U = TypeVar("U", bound=PromptTaskTemplateModel)


class PromptAnalysisModel(AnalysisWithTemplateModel, Generic[U]):
    smodel: Optional[embedding_model] = (
        None  # TODO Do I use this? Should I allow model override?
    )
    task_template: U
    autosave: bool = False


W = TypeVar("W", bound=PromptAnalysisModel)


class ProcessPromptTask(CMTemplatedTask[W, U], Generic[W, U]):
    collection_specific_results = False
    materialize_analysis = True
    can_delete_results = True
    task_creates: List[topic_type] = [topic_type.standalone]
    edit_template_name = "apply_prompt.html"
    task_template_form = "edit_prompt_template.html"

    @classmethod
    def setup_models(cls):
        permission_values.add("openai_query")
        permission_values.add("edit_prompts")

    @classproperty
    def trigger_task_permission(cls) -> Optional[ForwardRef("permission")]:
        from ..pyd_models import permission

        return permission.openai_query

    @classproperty
    def admin_task_permission(cls) -> Optional[ForwardRef("permission")]:
        from ..pyd_models import permission

        return permission.edit_prompts

    @classmethod
    def add_parse_params(cls, parser):
        parser.add_argument(
            "--task-template-nickname", type=str
        )  # Cannot access the registry yet, unfortunately
        parser.add_argument("--autosave", action="store_true")

    async def run(self):
        from .prompts import do_prompt_analysis

        analysis_model = self.analysis
        registry = TaskRegistry.get_registry()
        if not registry.task_template_by_nickname[
            analysis_model.task_template_nickname
        ]:
            # TODO: This is all wrong. Make sure parameters go to analysis.task_template.
            async with Session() as session:
                await registry.ensure_task_template(
                    session,
                    analysis_model.name,
                    analysis_model.task_template_nickname,
                    dict(
                        model=analysis_model.model,
                        parser=analysis_model.parser,
                        link_type=analysis_model.link_type,
                        node_type=analysis_model.node_type,
                        prompt=analysis_model.prompt,
                    ),
                    False,
                )
        await do_prompt_analysis(analysis_model.id)
        await self.set_completed()

    async def delete_results(self, session):
        analysis = await self.get_db_analysis(session)
        assert analysis
        await session.refresh(analysis, ["generated_topics"])
        for topic in analysis.generated_topics:
            # TODO: Check if multiple sources
            await session.delete(topic)
        analysis.status = process_status.pending
        analysis.results = {}

    @classmethod
    async def form_template_vars(cls):
        return dict(models={e.name: e.value for e in processing_models})


class SimplePromptTaskTemplateModel(PromptTaskTemplateModel):
    analyzer_name: Literal["simple_prompt_analyzer"] = "simple_prompt_analyzer"


class SimplePromptAnalysisModel(PromptAnalysisModel):
    analyzer_name: Literal["simple_prompt_analyzer"] = "simple_prompt_analyzer"
    target_id: Optional[int] = None
    target: Optional[AnyTopicModel] = None


class ProcessSimplePromptTask(
    ProcessPromptTask[SimplePromptAnalysisModel, SimplePromptTaskTemplateModel]
):
    name = "simple_prompt_analyzer"
    task_scale = (topic_type.standalone,)

    @classmethod
    async def status_for(
        cls, session, target_id, nickname=None, **params
    ) -> process_status:
        return await cls.status_from_analyses(
            session, target_id, params, nickname=nickname
        )


class FragmentPromptTaskTemplateModel(PromptTaskTemplateModel):
    analyzer_name: Literal["fragment_prompt_analyzer"] = "fragment_prompt_analyzer"


class FragmentPromptAnalysisModel(PromptAnalysisModel):
    analyzer_name: Literal["fragment_prompt_analyzer"] = "fragment_prompt_analyzer"
    source_ids: List[int]
    theme_id: int
    theme: Optional[StatementModel] = None
    context: Optional[List[FragmentModel]] = None

    @model_validator(mode="before")
    @classmethod
    def check_source(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if data.get("source_ids") is None:
                data["source_ids"] = []
                data["status"] = "not_ready"
        else:
            if getattr(data, "source_ids", None) is None:
                data.source_ids
                data.status = process_status.not_ready
        return data


class ProcessFragmentPromptTask(
    ProcessPromptTask[FragmentPromptAnalysisModel, FragmentPromptTaskTemplateModel]
):
    name = "fragment_prompt_analyzer"
    task_scale = (topic_type.fragment,)

    @classmethod
    def add_parse_params(cls, parser):
        super(ProcessFragmentPromptTask, cls).add_parse_params(parser)
        parser.add_argument("--source-ids", type=int, action="append")


class InferQuotesByProximityTaskTemplateModel(TaskTemplateModel):
    analyzer_name: Literal["infer_quotes_proximity"] = "infer_quotes_proximity"
    min_rank: Optional[float] = 0.85
    limit: Optional[float] = 4
    model: embedding_model = BASE_EMBED_MODEL


class InferQuotesByProximityAnalysisModel(AnalysisWithTemplateModel):
    analyzer_name: Literal["infer_quotes_proximity"] = "infer_quotes_proximity"
    target_id: Optional[int] = None
    target: Optional[StatementModel] = None
    recursive: Optional[bool] = False
    task_template: Optional[InferQuotesByProximityTaskTemplateModel] = None


class InferQuotesByProximityTask(
    CMTemplatedTask[
        InferQuotesByProximityAnalysisModel, InferQuotesByProximityTaskTemplateModel
    ]
):
    name = "infer_quotes_proximity"
    collection_specific_results = False
    materialize_analysis = True
    can_apply_bulk = True
    task_scale = (topic_type.standalone,)
    can_delete_results = True
    task_creates: List[topic_type] = [topic_type.link]
    task_template_form = "edit_infer_quotes_by_proximity_template.html"

    @classproperty
    def trigger_task_permission(cls) -> Optional[ForwardRef("permission")]:
        from ..pyd_models import permission

        return permission.add_claim

    async def run(self):
        from ..models import (
            ClaimLink,
            Collection,
            Analysis,
            Topic,
            Statement,
            graph_subquery,
        )
        from ..search import search

        async with Session() as session:
            analysis = self.analysis
            analysis_db = await session.get(Analysis, analysis.id)
            collection = None
            if analysis.collection_id:
                collection = await session.get(Collection, analysis.collection_id)
            collections = [collection] if collection else []
            existing_links = await session.scalars(
                select(ClaimLink).filter_by(
                    source=analysis.target_id, link_type=link_type.quote
                )
            )
            existing_targets = {cl.target: cl for cl in existing_links}
            task_template = await TaskRegistry.get_registry().get_task_template_by_id(
                analysis.task_template_id, session
            )
            targets = [analysis.target_id]

            if self.analysis.recursive:
                rec_q = graph_subquery(analysis.target_id)
                targets = list(
                    await session.scalars(
                        select(Statement.id)
                        .filter_by(scale=fragment_type.standalone_claim)
                        .join(rec_q, Statement.id == rec_q.c.target)
                    )
                )

            for target_id in targets:
                try:
                    r = await search(
                        session,
                        target_id,
                        None,
                        collection,
                        model=task_template.model,
                        include_claims=False,
                        scales=[
                            fragment_type.sentence,
                            fragment_type.paragraph,
                            fragment_type.quote,
                        ],
                        limit=task_template.limit,
                        min_distance=task_template.min_rank,
                        one_per_doc=True,
                    )
                except Exception as e:
                    if e.__class__.__name__ == "NotFound":
                        pass  # Missing embedding
                    else:
                        raise e

                lowest_rank = None
                for search_result in r:
                    quote = search_result["target"]
                    lowest_rank = min(lowest_rank or 10000, search_result["rank"])
                    if quote.id in existing_targets:
                        continue
                    cl = ClaimLink(
                        source=target_id,
                        link_type=link_type.quote,
                        target=quote.id,
                        from_analyses=[analysis_db],
                        collections=collections,
                        score=search_result["rank"],
                    )
                    session.add(cl)
                # TODO: Should we destroy automated quotes of lower rank? We need to find a way to confirm scores.
                if lowest_rank:
                    for cl in existing_targets.values():
                        if cl.score and cl.score < lowest_rank:
                            session.delete(cl)
            await session.commit()

        await self.set_completed()

    @classmethod
    def task_form_before(cls):
        """Return a form for the creation or edition of the task object before execution, if needed."""
        # Note: Scale and algorithm are implicit for now
        return """
  <p>
    <input type="checkbox" name="recursive">Apply to whole graph</input>
  </p>"""


class ExtractTextAnalysisModel(AnalysisModel):
    target_id: Optional[int] = None
    target: Optional[DocumentModel] = None
    post_process_text: bool = True
    include_hallucinations: bool = True


W2 = TypeVar("W2", bound=ExtractTextAnalysisModel)


class ExtractTextTask(CMTask[W2], Generic[W2]):
    task_scale = (topic_type.document,)
    collection_specific_results = False
    can_delete_results = True
    can_apply_bulk = True
    mimetypes: List[str]
    task_creates: List[topic_type] = [topic_type.fragment]
    version = 2

    @classmethod
    async def status_for(cls, session, target_id, **params) -> process_status:
        from ..models import Document

        document = await session.get(Document, target_id)
        if not document.file_identity:
            return process_status.not_ready
        if document.base_type not in cls.mimetypes:
            return process_status.inapplicable
        if document.text_identity:
            return process_status.complete
        if document.load_status == "error":
            return process_status.error
        return process_status.not_requested

    @classmethod
    def count_status_query(cls, collection_name=None, **kwargs):
        from ..models import Document, Collection, Analyzer

        current_version = cls.version
        status_exp = case(
            (Document.base_type.not_in(cls.mimetypes), "inapplicable"),
            (Document.text_identity.is_not(None), "complete"),
            (Document.file_identity.is_(None), "not_ready"),
            else_="not_requested",
        )
        q0 = select(
            Document.id,
            case(
                (Document.base_type.not_in(cls.mimetypes), "inapplicable"),
                (Document.text_identity.is_not(None), "complete"),
                (Document.file_identity.is_(None), "not_ready"),
                else_="not_requested",
            ).label("status"),
            coalesce(Analyzer.version, current_version).label("version"),
        ).outerjoin(Analyzer, Analyzer.id == Document.text_analyzer_id)
        if collection_name:
            q0 = q0.join(Collection, Document.collections).filter_by(
                name=collection_name
            )
        q0 = q0.cte()
        q = (
            select(q0.c.version, q0.c.status, None, count(q0.c.id), 0)
            .group_by(q0.c.status, q0.c.version)
            .order_by(q0.c.version)
        )
        return q

    @classmethod
    def query_with_status(
        cls,
        status: process_status,
        collection_name: Optional[str] = None,
        task_template_name=None,
        analyzer_version: Optional[int] = None,
        **kwargs,
    ) -> Tuple[bool, Select[Tuple[BigInteger]]]:
        from ..models import Document, Collection, Analyzer

        if status == process_status.complete:
            q = select(Document.id).filter(
                Document.text_identity.is_not(None),
                Document.base_type.in_(cls.mimetypes),
            )
        elif status == process_status.not_ready:
            q = select(Document.id).filter(Document.file_identity.is_(None))
        elif status == process_status.inapplicable:
            q = select(Document.id).filter(Document.base_type.not_in(cls.mimetypes))
        elif status == process_status.not_requested:
            q = select(Document.id).filter(
                Document.file_identity.is_not(None),
                Document.base_type.in_(cls.mimetypes),
                Document.text_identity.is_(None),
            )
        else:
            return (False, select(Document.id).filter(False))
        if analyzer_version:
            if analyzer_version == cls.version:
                q = q.outerjoin(
                    Analyzer, Analyzer.id == Document.text_analyzer_id
                ).filter(
                    or_(
                        Analyzer.version == analyzer_version, Analyzer.version.is_(None)
                    )
                )
            else:
                q = q.join(
                    Analyzer, Analyzer.id == Document.text_analyzer_id
                ).filter_by(version=analyzer_version)
        if collection_name:
            q = q.join(Collection, Document.collections).filter_by(name=collection_name)
        return (False, q)

    @classmethod
    async def compute_default_triggers(cls, session, task_registry: TaskRegistry):
        yield TaskTriggerModel(
            target_analyzer_id=task_registry.analyzer_by_name[cls.name].id,
            analyzer_trigger_id=task_registry.analyzer_by_name["download"].id,
            automatic=True,
        )
        yield TaskTriggerModel(
            target_analyzer_id=task_registry.analyzer_by_name[cls.name].id,
            creation_trigger_id=topic_type.document,
            automatic=True,
        )

    @classmethod
    async def check_trigger_conditions(
        cls,
        session,
        trigger,
        target_id: Optional[int],
        task: Optional[CMTask],
        collection_name: Optional[str],
    ) -> bool:
        from ..models import Document

        if not target_id:
            return False
        topic = await session.get(Document, target_id)
        return topic and topic.file_identity and topic.base_type in cls.mimetypes

    async def delete_results(self, session):
        from ..models import Document, Fragment, Topic

        document = await session.get(Document, self.analysis.target_id)
        assert document
        if document.text_identity:
            await session.scalar(
                select(count(Document.id)).filter_by(
                    text_identity=document.text_identity
                )
            )
            if count == 1:
                f = hashfs(document.text_identity)
                f.delete()
        document.text_identity = None
        document.text_length = None
        # Question : Should we delete quotes? I would say no.
        topic = Topic.__table__
        await session.execute(
            delete(topic).filter(
                topic.c.id.in_(
                    select(Fragment.id).filter(
                        Fragment.doc_id == document.id,
                        Fragment.scale.in_(
                            (fragment_type.paragraph, fragment_type.sentence)
                        ),
                    )
                )
            )
        )


class ExtractHtmlTextAnalysisModel(ExtractTextAnalysisModel):
    analyzer_name: Literal["process_html"] = "process_html"


class ProcessHtmlTask(ExtractTextTask[ExtractHtmlTextAnalysisModel]):
    mimetypes = ["text/html"]
    name = "process_html"

    async def run(self):
        from .process_html import do_process_html

        return await do_process_html(
            self.analysis.target_id,
            self.analysis.post_process_text,
            self.analysis.include_hallucinations,
        )


class ExtractPdfTextAnalysisModel(ExtractTextAnalysisModel):
    analyzer_name: Literal["process_pdf"] = "process_pdf"


class ProcessPdfTask(ExtractTextTask[ExtractPdfTextAnalysisModel]):
    name = "process_pdf"
    mimetypes = ["application/pdf"]

    async def run(self):
        from .process_pdf import do_process_pdf

        return await do_process_pdf(
            self.analysis.target_id,
            self.analysis.post_process_text,
            self.analysis.include_hallucinations,
        )


class ExtractPlainTextAnalysisModel(ExtractTextAnalysisModel):
    analyzer_name: Literal["process_text"] = "process_text"


class ProcessTextTask(ExtractTextTask[ExtractPlainTextAnalysisModel]):
    name = "process_text"
    mimetypes = ["text/plain", "text/markdown", "application/json"]

    async def run(self):
        from .process_text import do_process_text

        return await do_process_text(
            self.analysis.target_id,
            self.analysis.post_process_text,
            self.analysis.include_hallucinations,
        )


class FeedlyTaskTemplateModel(TaskTemplateModel):
    analyzer_name: Literal["feedly_feed"] = "feedly_feed"
    stream_id: str
    post_process_text: bool = True
    include_hallucinations: bool = True


class FeedlyAnalysisModel(AnalysisWithTemplateModel):
    analyzer_name: Literal["feedly_feed"] = "feedly_feed"
    task_template: Optional[FeedlyTaskTemplateModel] = None


class FeedlyTask(CMTemplatedTask[FeedlyAnalysisModel, FeedlyTaskTemplateModel]):
    name = "feedly_feed"
    materialize_analysis = True
    core = False
    task_scale = (topic_type.collection,)
    collection_specific_results = False
    task_template_form = "edit_feedly_template.html"
    task_creates: List[topic_type] = [topic_type.document]
    can_reapply_complete: bool = True
    can_apply_bulk = True

    async def run(self):
        from .feedly import do_feedly_feed

        tmpl = self.analysis.get_task_template()
        last_query_time = await do_feedly_feed(self.analysis.id)
        await self.set_completed(last_query_time)

    @classmethod
    def setup_models(cls):
        permission_values.add("admin_feedly")

    @classproperty
    def trigger_task_permission(cls) -> Optional[ForwardRef("permission")]:
        from ..pyd_models import permission

        return permission.add_document

    @classproperty
    def admin_task_permission(cls) -> Optional[ForwardRef("permission")]:
        from ..pyd_models import permission

        return permission.admin_feedly

    @classmethod
    def add_parse_params(cls, parser):
        parser.add_argument("--analysis-ids", type=int, action="append")
        parser.add_argument("--task-template-nickname", type=str)


class BaseTaskMap:
    @hookimpl
    def register_tasks(self, task_map: Dict[str, Type[CMTask[AnalysisModel]]]):
        task_map |= {
            t.name: t
            for t in (
                DownloadTask,
                EmbedDocTask,
                EmbedFragmentTask,
                GdeltTask,
                FeedlyTask,
                ProcessHtmlTask,
                ProcessTextTask,
                ProcessPdfTask,
                ClusterTask,
                AutoclassifyTask,
                ProcessSimplePromptTask,
                ProcessFragmentPromptTask,
                InferQuotesByProximityTask,
            )
        }

        return task_map
