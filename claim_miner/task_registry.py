from __future__ import annotations

from logging import getLogger
import asyncio
from abc import ABC, abstractmethod
from itertools import groupby, chain
from typing import (
    Mapping,
    Type,
    Optional,
    Tuple,
    List,
    ForwardRef,
    Set,
    Dict,
    AsyncGenerator,
    TypeVar,
    Generic,
    get_args,
)
from collections import defaultdict
from datetime import datetime

from sqlalchemy import select, Select, BigInteger, lambda_stmt
from sqlalchemy.sql import or_, and_
from sqlalchemy.sql.functions import count, coalesce
from sqlalchemy.orm import joinedload, subqueryload, with_polymorphic
from sqlalchemy.orm.util import AliasedClass
import pluggy

from .utils import classproperty, filter_dict
from .pyd_models import (
    topic_type,
    AnalyzerModel,
    AnalysisModel,
    TaskTemplateModel,
    process_status,
    fragment_type,
)

logger = getLogger(__name__)


def batch(iterable, batch_size):
    batch = []
    for val in iterable:
        batch.append(val)
        if len(val) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# TODO: Make the analysis/template models into a paramtric types

T = TypeVar("T", bound=AnalysisModel)
U = TypeVar("U", bound=TaskTemplateModel)


class CMTask(ABC, Generic[T]):
    name: str
    params: Mapping
    version: int = 1
    core: bool = True
    materialize_analysis: bool = False
    can_delete_results: bool = False
    can_apply_bulk: bool = False
    can_reapply_complete: bool = False  # Can we re-trigger this task in complete state
    task_scale: Tuple[topic_type, ...]
    collection_specific_results: bool = True
    task_creates: List[topic_type] = []
    # The analysis parameters that make it count as a distinct repeatable task
    edit_template_name = "analysis_info.html"
    task_template_form = None

    @classproperty
    def analysis_model(cls) -> Type[T]:
        for base in cls.__orig_bases__:
            args = get_args(base)
            if args:
                assert issubclass(args[0], AnalysisModel)
                return args[0]
        assert False, "No analysis model"

    @classproperty
    def task_template_model(cls) -> Optional[Type[TaskTemplateModel]]:
        return None

    @classmethod
    def setup_models(cls):
        pass

    @classproperty
    def create_tasks_with_status(cls):
        from .pyd_models import process_status

        return process_status.pending

    @classmethod
    async def compute_default_triggers(cls, session, task_registry: TaskRegistry):
        if False:
            yield None

    @classproperty
    def trigger_task_permission(cls) -> Optional[ForwardRef("permission")]:
        # The permission needed to trigger this task, if any
        return None

    @classproperty
    def admin_task_permission(cls) -> Optional[ForwardRef("permission")]:
        # The permission needed to set up this task
        from .pyd_models import permission

        return permission.admin

    def __init__(self, analysis: Optional[T] = None, **kwargs) -> None:
        """Task initializer should consume the task's arguments; kwargs should be empty.
        All arguments must be optional in the initializer.
        """
        self.analysis: T = analysis or self.analysis_model(**kwargs)

    def as_dict(self):
        return self.analysis.model_dump()

    def get_channel_key(self):
        return (
            str(
                getattr(self.analysis, "target_id", None)
                or getattr(self.analysis, "theme_id", "")
            )
            or None
        )

    async def complete_params(self, session=None):
        """Sometimes parameters can be inferred from one another."""
        pass

    @classmethod
    async def get_task_object(cls, session, **kwargs):
        return None

    @classmethod
    async def get_analyzer_model(cls):
        registry = await TaskRegistry.get_full_registry()
        return registry.analyzer_by_name[cls.name]

    @classmethod
    async def get_analyzer_id(cls):
        analyzer = await cls.get_analyzer_model()
        return analyzer.id

    async def delete_results(self, session):
        from .pyd_models import process_status

        if not self.can_delete_results:
            raise NotImplementedError()
        # Default implementation
        analysis = await self.get_db_analysis(session)
        assert analysis
        await session.refresh(analysis, ["generated_topics"])
        for topic in analysis.generated_topics:
            await session.delete(topic)
        analysis.status = process_status.pending
        analysis.results = {}

    async def before_run(self):
        from .pyd_models import process_status
        from .models import Session, Analysis

        if not self.analysis.id:
            return
        async with Session() as session:
            a = await session.get(Analysis, self.analysis.id)
            a.status = process_status.ongoing
            await session.commit()
            return a

    async def set_completed(self, timestamp: Optional[datetime] = None):
        if not self.materialize_analysis:
            return
        from .models import Session, Analysis

        async with Session() as session:
            timestamp = timestamp or datetime.utcnow()
            self.analysis.status = process_status.complete
            self.analysis.completed = timestamp
            if self.analysis.id:
                a = await session.get(Analysis, self.analysis.id)
                a.status = process_status.complete
                a.completed_at = timestamp
            else:
                a = Analysis.from_model(self.analysis)
                session.add(a)
                await session.flush()
                self.analysis.id = a.id
            await session.commit()

    @abstractmethod
    async def run(self):
        pass

    async def status(self, session=None) -> process_status:
        if session is None:
            from . import Session

            async with Session() as session:
                return await self.status(session)
        return await self.status_for(
            session, **self.analysis.model_dump()
        )  # Should I not send the analysis instead?

    @classmethod
    async def filter_not_done(cls, ids, **params):
        from . import Session
        from .pyd_models import process_status

        with Session() as session:
            for id_ in ids:
                if (
                    await cls.status_for(session, id_, **params)
                    == process_status.complete
                ):
                    continue
                yield id

    @classmethod
    async def status_for(cls, session, target_id, **params) -> process_status:
        from .pyd_models import process_status

        return process_status.not_requested

    @classmethod
    def count_status_query(cls, collection_name=None, **kwargs):
        from .pyd_models import topic_type
        from .models import (
            Topic,
            model_by_topic_type,
            Analyzer,
            Analysis,
            Collection,
            TaskTemplate,
            poly_type_clause,
        )

        registry = TaskRegistry.get_registry()
        analyzer = registry.analyzer_by_name[cls.name]
        # TODO: is there a way to detect when the template and collection conflict?
        if cls.materialize_analysis:
            ModelClasses = [model_by_topic_type[s] for s in cls.task_scale]
            if Collection in ModelClasses:
                assert len(ModelClasses) == 1
                ModelClass = Collection
            else:
                # elif len(ModelClasses) > 1:
                ModelClass = with_polymorphic(Topic, ModelClasses, aliased=True)
            # else:
            #     ModelClasse = ModelClasses[0]
            target_q = (
                select(
                    Analyzer.version,
                    Analysis.status,
                    TaskTemplate.nickname,
                    count(ModelClass.id.distinct()).label("num_topics"),
                    count(Analysis.id.distinct()).label("num_analysis"),
                )
                .filter_by(name=cls.name)
                .outerjoin(Analysis, Analysis.analyzer_id == Analyzer.id)
                .outerjoin(TaskTemplate, Analysis.task_template)
                .group_by(Analysis.status, Analyzer.version, TaskTemplate.nickname)
                .order_by(Analyzer.version, TaskTemplate.nickname, Analysis.status)
            )
            if cls.task_scale == (topic_type.collection,):
                target_q = target_q.join(ModelClass, Analysis.collection)
            else:
                target_q = target_q.join(
                    ModelClass,
                    ModelClass.id == coalesce(Analysis.target_id, Analysis.theme_id),
                ).filter(poly_type_clause(ModelClass))
                if collection_name:
                    target_q = target_q.join(Collection, Analysis.collection).where(
                        Collection.name == collection_name
                    )
            return target_q
        raise NotImplementedError()

    @classmethod
    def query_with_status(
        cls,
        status: process_status,
        collection_name: Optional[str] = None,
        task_template_name: Optional[str] = None,
        analyzer_version: Optional[int] = None,
        **kwargs,
    ) -> Tuple[bool, Select[Tuple[BigInteger]]]:
        "Return either the target_ids or analyzer_ids if materialized"
        from .models import (
            Topic,
            model_by_topic_type,
            Analysis,
            Collection,
            Analyzer,
            TaskTemplate,
            process_status,
            poly_type_clause,
            TopicCollection,
            Fragment,
        )

        query_on_analysis = False
        if cls.materialize_analysis:
            ModelClasses = [model_by_topic_type[s] for s in cls.task_scale]
            if Collection in ModelClasses:
                assert len(ModelClasses) == 1
                ModelClass: Type[Collection] | AliasedClass[Topic] = Collection
            else:
                ModelClass = with_polymorphic(Topic, ModelClasses, aliased=True)
            if status >= process_status.pending:
                query_on_analysis = True
                q: Select[Tuple[BigInteger]] = select(Analysis.id).filter_by(
                    status=status
                )
                q = q.join(Analyzer, Analysis.analyzer).filter_by(name=cls.name)
                if analyzer_version:
                    q = q.filter_by(version=analyzer_version)
                if task_template_name:
                    q = q.join(TaskTemplate, Analysis.task_template).filter_by(
                        nickname=task_template_name
                    )
                if ModelClass == Collection:
                    q = q.join(ModelClass, Analysis.collection)
                else:
                    # TODO: When is it the theme?
                    q = q.join(
                        ModelClass,
                        ModelClass.id
                        == coalesce(Analysis.target_id, Analysis.theme_id),
                    )
                q = q.filter(poly_type_clause(ModelClass))
            elif status in (process_status.not_ready, process_status.inapplicable):
                # Those are more likely to be overridden by subclasses
                return (query_on_analysis, select(ModelClass.id).filter(False))
            elif status == process_status.not_requested:
                q = (
                    select(ModelClass.id.distinct())
                    .filter(poly_type_clause(ModelClass))
                    .outerjoin(
                        Analysis,
                        ModelClass.id
                        == coalesce(
                            Analysis.target_id,
                            Analysis.theme_id,
                            Analysis.collection_id,
                        ),
                    )
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
                            Analysis.analyzer_id == Analyzer.id,
                            Analyzer.name == cls.name,
                        ),
                    )
                q = q.filter(Analysis.id.is_(None), Analyzer.id.is_(None))
                if task_template_name:
                    q = q.outerjoin(
                        TaskTemplate,
                        and_(
                            Analysis.task_template_id == TaskTemplate.id,
                            TaskTemplate.nickname == task_template_name,
                        ),
                    ).filter(TaskTemplate.id.is_(None))
            if collection_name and ModelClass != Collection:
                if Fragment in ModelClasses:
                    q = (
                        q.join(
                            TopicCollection,
                            TopicCollection.topic_id
                            == coalesce(ModelClass.Fragment.doc_id, ModelClass.id),
                        )
                        .join(Collection, TopicCollection.collection)
                        .filter_by(name=collection_name)
                    )
                else:
                    q = q.join(Collection, ModelClass.collections).filter_by(
                        name=collection_name
                    )
            return (query_on_analysis, q)
        raise NotImplementedError()

    @classmethod
    async def batch_tasks(cls, ids, check=True, **kwargs):
        # Obsolete code. Rewrite in bulk_tasks
        from . import Session
        from .pyd_models import process_status

        async with Session() as session:
            # TODO: Maybe instead build the batch according to data size?
            batch_size = cls.batch_size(**kwargs)
            batch = []
            for id_ in ids:
                if check:
                    status = await cls.status_for(session, id_, **kwargs)
                    if status == process_status.complete:
                        logger.info("Already done: {id_} in task {cls.name}")
                        continue
                    if status < process_status.not_requested:
                        logger.info("Not ready: {id_} in task {cls.name}")
                        continue
                if batch_size == 1:
                    task = cls(target_id=id_, **kwargs)
                    await task.complete_params()
                    yield task
                else:
                    batch.append(id_)
                    if len(batch) >= batch_size:
                        task = cls(target_id=batch, **kwargs)
                        await task.complete_params()
                        yield task
                        batch = []
            if batch:
                task = cls(target_id=batch, **kwargs)
                await task.complete_params()
                yield task

    async def get_db_analysis(self, session):
        if self.analysis and self.analysis.id:
            from .models import Analysis

            return await session.get(Analysis, self.analysis.id)

    async def schedule(self):
        from . import dispatcher, Session

        if self.materialize_analysis:
            if not self.analysis.id:
                from .models import Analysis

                async with Session() as session:
                    analysis = await Analysis.from_model(session, self.analysis)
                    session.add(analysis)
                    await session.commit()
                    self.analysis.id = analysis.id  # Or recreate the model?
            await dispatcher.trigger_task(self.name, analysis_id=self.analysis.id)
        else:
            await dispatcher.trigger_task(self.name, **self.as_dict())

    @classmethod
    def add_parse_params(cls, parser):
        pass

    @classmethod
    def batch_size(cls, **params):
        return 1

    @classmethod
    def get_template_params(cls, params, inverted=False):
        return params if inverted else {}

    @classmethod
    async def ensure_task_template_cls(
        cls, session, params: Mapping, nickname=None
    ) -> Optional[TaskTemplateModel]:
        return None

    async def ensure_task_template(self, session) -> Optional[TaskTemplateModel]:
        return await self.ensure_task_template_cls(session, self.as_dict())

    @classmethod
    async def analyses_for(
        cls,
        session,
        target_id: int,
        params: Optional[Dict] = None,
        nickname: Optional[str] = None,
        collection_name: Optional[str] = None,
        task_template: Optional[TaskTemplateModel] = None,
    ) -> List[AnalysisModel]:
        from .models import Analysis, Collection

        analyzer_model = await cls.get_analyzer_model()
        params = params or {}
        collection_name = collection_name or params.pop("collection_name", None)
        q = select(Analysis).filter_by(
            target_id=target_id, analyzer_id=analyzer_model.id, part_of_id=None
        )
        if collection_name and cls.collection_specific_results:
            q = q.join(Collection, Analysis.collection).filter_by(name=collection_name)
        analyses = list(await session.scalars(q))
        remaining_params = cls.get_template_params(params, True)
        if remaining_params:

            def filter_by_params(analysis):
                for k, v in analysis.params.items():
                    if remaining_params.get(k) != v:
                        return False
                return True

            analyses = [a for a in analyses if filter_by_params(a)]
        return [
            a.as_model(session, analyzer_name=analyzer_model.name) for a in analyses
        ]

    @classmethod
    async def results_for(
        cls,
        session,
        target_id: int,
        params: Optional[Dict] = None,
        nickname: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> List[AnalysisModel]:
        from .pyd_models import AnalysisModel, process_status, topic_type

        params = params or {}
        collection_name = collection_name or params.pop("collection_name", None)
        analyses = await cls.analyses_for(
            session, target_id, params, None, collection_name
        )
        if analyses:
            return analyses
        if target_id:
            params["target_id"] = target_id

        sub_params = cls.get_template_params(params, True)
        status: Optional[process_status] = await cls.status_for(
            session, collection=collection_name, nickname=None, **sub_params
        )
        analysis_model = cls.analysis_model
        if target_id and "target_id" not in analysis_model.model_fields:
            sub_params.pop("target_id")
            if "theme_id" in analysis_model.model_fields:
                sub_params["theme_id"] = target_id
        if status > process_status.not_ready:
            return [
                analysis_model(
                    status=status, collection_name=collection_name, **sub_params
                )
            ]
        return []

    @classmethod
    async def status_from_analyses(
        cls,
        session,
        target_id: int,
        params: Optional[Dict] = None,
        nickname: Optional[str] = None,
        collection_name: Optional[str] = None,
        task_template: Optional[TaskTemplateModel] = None,
    ) -> process_status:
        from .pyd_models import process_status

        analyses = await cls.analyses_for(
            session, target_id, params, collection_name=collection_name
        )
        if analyses:
            return max(analysis.status for analysis in analyses if analysis.status)
        return process_status.not_requested

    @classmethod
    def user_can_trigger(cls, user, collection=None):
        permission = cls.trigger_task_permission
        if permission is None:
            return True
        if collection:
            return collection.user_can(user, permission)
        else:
            return user.can(permission)

    @classmethod
    def user_can_admin(cls, user, collection=None):
        permission = cls.admin_task_permission
        if permission is None:
            return True
        if collection:
            return collection.user_can(user, permission)
        else:
            return user.can(permission)

    @classmethod
    async def check_trigger_conditions(
        cls,
        session,
        trigger,
        target_id: Optional[int],
        task: Optional[CMTask],
        collection_name: Optional[str],
    ) -> bool:
        return True

    @classmethod
    async def apply_topic_trigger(
        cls, session, trigger, topic_id, collection_name
    ) -> List[CMTask]:
        from .models import Topic

        topic = None
        triggering_task_id = None
        if topic_id:
            topic = await session.get(Topic, topic_id)
            await session.refresh(
                topic, ["from_analyses"]
            )  # WHY did this not work with a subqueryload?
            if topic.from_analyses:
                triggering_task_id = topic.from_analyses[0].id
        if not await cls.check_trigger_conditions(
            session, trigger, topic_id, None, collection_name
        ):
            return []
        params = trigger.params.copy()
        if trigger.task_template_id:
            params["task_template_id"] = trigger.task_template_id
        new_task = cls(
            target_id=topic_id,
            collection=collection_name,
            triggered_by_analysis_id=triggering_task_id,
            **params,
        )
        return [new_task]

    @classmethod
    def enrich_trigger_params(cls, session, params, task, collection):
        pass

    @classmethod
    async def apply_task_trigger(cls, session, trigger, task: CMTask) -> List[CMTask]:
        from .models import (
            Collection,
            Analysis,
            Topic,
            with_polymorphic,
            process_status,
        )

        if task.status == process_status.error:
            return
        collection_id = (
            trigger.collection_id or task.analysis.collection_id
        )  # Use triggering task's collection for generic triggers
        collection = (
            (await session.get(Collection, collection_id)) if collection_id else None
        )
        collection_name = collection.name if collection else None
        target_id: Optional[int] = getattr(task.analysis, "target_id", None)
        if target_id:
            # Make sure it's loaded properly
            await session.execute(
                select(
                    with_polymorphic(Topic, "*", flat=True, aliased=False)
                ).filter_by(id=target_id)
            )
        if not await cls.check_trigger_conditions(
            session, trigger, target_id, task, collection_name
        ):
            return []
        params = trigger.params.copy()
        cls.enrich_trigger_params(session, params, task, collection)
        # TODO: Improve on this!
        dummy_task = cls(
            target_id=target_id,
            collection_name=collection_name,
            task_template_id=trigger.task_template_id,
            **params,
        )
        dummy_analysis = await Analysis.from_model(session, dummy_task.analysis)
        params = filter_dict(dummy_analysis.params)
        tasks: List[CMTask] = []
        if trigger.params.get("use_output", False):
            # Note this case is not used yet. May not be needed.
            assert task.analysis.id
            analysis = await session.get(
                Analysis,
                task.analysis.id,
                options=[joinedload(Analysis.generated_topics)],
            )
            for output in analysis.generated_topics:
                existing = await session.scalars(
                    select(Analysis).filter_by(
                        analyzer_id=trigger.target_analyzer_id,
                        target_id=output.id,
                        collection=collection,
                        task_template_id=trigger.task_template_id,
                        params=params,
                    )
                )
                first = existing.first()
                if first:
                    new_task = cls(analysis=first.as_model(session))
                else:
                    new_task = cls(
                        target_id=output.id,
                        collection_name=collection_name,
                        task_template_id=trigger.task_template_id,
                        triggered_by_analysis_id=analysis.id,
                        **params,
                    )
                    new_task.analysis.status = await new_task.status(session)
                tasks.append(new_task)
        else:
            existing = await session.scalars(
                select(Analysis).filter_by(
                    analyzer_id=trigger.target_analyzer_id,
                    target_id=target_id,
                    collection=collection,
                    task_template_id=trigger.task_template_id,
                    params=params,
                )
            )
            first = existing.first()
            if first:
                new_task = cls(analysis=first.as_model(session))
            else:
                new_task = cls(
                    target_id=target_id,
                    collection_name=collection_name,
                    task_template_id=trigger.task_template_id,
                    triggered_by_analysis_id=task.analysis.id,
                    **params,
                )
                new_task.analysis.status = await new_task.status(session)
            tasks.append(new_task)
        return tasks

    async def enrich_edit_form_data(self, session, base_vars):
        pass

    @classmethod
    def setup_routes(cls, app_router, api_router):
        pass

    @classmethod
    async def form_template_vars(cls):
        return {}

    @classmethod
    def task_form_before(cls):
        """Return a form for the creation or edition of the task object before execution, if needed."""
        return None

    @classmethod
    async def task_form_args(cls, session) -> Mapping:
        return {}

    def task_form_after(self):
        """Return a form for operations on a finished task object."""
        pass

    async def handle_task_form_after(self, session, form):
        pass


class CMTemplatedTask(CMTask[T], Generic[T, U]):
    @classproperty
    def task_template_model(cls) -> Type[U]:
        for sup in cls.mro():
            for base in getattr(sup, "__orig_bases__", ()):
                args = get_args(base)
                if len(args) > 1:
                    return args[1]
        assert False, "Cannot find task template model"

    def __init__(self, analysis: Optional[T] = None, **kwargs) -> None:
        """Task initializer should consume the task's arguments; kwargs should be empty.
        All arguments must be optional in the initializer.
        """
        if (
            not analysis
            and "task_template_id" not in kwargs
            and "task_template_nickname" in kwargs
        ):
            nickname = kwargs["task_template_nickname"]
            template = TaskRegistry.get_registry().task_template_by_nickname.get(
                nickname
            )
            if not template:
                tkwargs = self.get_template_params(kwargs)
                template = self.task_template_model(
                    nickname=kwargs["task_template_nickname"],
                    analyzer_name=self.name,
                    **tkwargs,
                )
            kwargs["task_template"] = template
            kwargs = self.get_template_params(kwargs, True)
        super(CMTemplatedTask, self).__init__(analysis=analysis, **kwargs)

    @classmethod
    def get_template_params(cls, params, inverted=False):
        fields = cls.task_template_model.model_fields
        common = set(
            (
                "analyzer_id",
                "analyzer_name",
                "analyzer",
                "collection_id",
                "collection_name",
            )
        )
        return {
            k: v
            for (k, v) in params.items()
            if (k in common) or ((k in fields) is not inverted)
        }

    @classmethod
    async def ensure_task_template_cls(
        cls, session, params: Mapping, nickname=None
    ) -> Optional[U]:
        params = cls.get_template_params(params)
        if params or nickname:
            return await TaskRegistry.get_registry().ensure_task_template(
                session, cls.name, nickname, params
            )
        return None

    async def ensure_task_template(self, session) -> Optional[U]:
        return await self.ensure_task_template_cls(session, self.as_dict())

    async def before_run(self):
        analysis = await super(CMTemplatedTask, self).before_run()
        if analysis is not None:
            # Ensure template in registry
            await TaskRegistry.get_registry().get_task_template_by_id(
                analysis.task_template_id
            )

    @classmethod
    async def analyses_for(
        cls,
        session,
        target_id: int,
        params: Optional[Dict] = None,
        nickname: Optional[str] = None,
        collection_name: Optional[str] = None,
        task_template: Optional[U] = None,
    ) -> List[AnalysisModel]:
        from .models import Analysis, Collection

        analyzer_model = await cls.get_analyzer_model()
        params = params or {}
        if not task_template:
            nickname = nickname or params.pop("nickname", None)
            task_template = await cls.ensure_task_template_cls(
                session, params, nickname
            )
            if not task_template:
                return []
        collection_name = collection_name or params.pop("collection_name", None)
        q = select(Analysis).filter_by(
            target_id=target_id,
            analyzer_id=analyzer_model.id,
            task_template_id=task_template.id,
            part_of_id=None,
        )
        if collection_name and cls.collection_specific_results:
            q = q.join(Collection, Analysis.collection).filter_by(name=collection_name)
        analyses = list(await session.scalars(q))
        remaining_params = cls.get_template_params(params, True)
        if remaining_params:

            def filter_by_params(analysis):
                for k, v in analysis.params.items():
                    if remaining_params.get(k) != v:
                        return False
                return True

            analyses = [a for a in analyses if filter_by_params(a)]
        return [
            a.as_model(session, analyzer_name=analyzer_model.name) for a in analyses
        ]

    @classmethod
    async def status_from_analyses(
        cls,
        session,
        target_id: int,
        params: Optional[Dict] = None,
        nickname: Optional[str] = None,
        collection_name: Optional[str] = None,
        task_template: Optional[U] = None,
    ) -> process_status:
        from .pyd_models import process_status

        analyses = await cls.analyses_for(
            session, target_id, params, nickname, collection_name, task_template
        )
        if analyses:
            return max(analysis.status for analysis in analyses if analysis.status)
        return process_status.not_requested

    @classmethod
    async def results_for(
        cls,
        session,
        target_id: int,
        params: Optional[Dict] = None,
        nickname: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> List[AnalysisModel]:
        from .pyd_models import AnalysisModel, process_status, topic_type

        params = params or {}
        nickname = nickname or params.pop("nickname", None)
        task_template = await cls.ensure_task_template_cls(session, params, nickname)
        if not task_template:
            return []
        collection_name = collection_name or params.pop("collection_name", None)
        analyses = await cls.analyses_for(
            session, target_id, params, nickname, collection_name, task_template
        )
        if analyses:
            return analyses
        if task_template:
            params["task_template"] = task_template
        if target_id:
            params["target_id"] = target_id

        sub_params = cls.get_template_params(params, True)
        status: Optional[process_status] = await cls.status_for(
            session, collection=collection_name, nickname=nickname, **sub_params
        )
        if task_template:
            sub_params["task_template_nickname"] = task_template.nickname
        analysis_model = cls.analysis_model
        if target_id and "target_id" not in analysis_model.model_fields:
            sub_params.pop("target_id")
            if "theme_id" in analysis_model.model_fields:
                sub_params["theme_id"] = target_id
        if status > process_status.not_ready:
            return [
                analysis_model(
                    status=status, collection_name=collection_name, **sub_params
                )
            ]
        return []

    @classmethod
    def add_parse_params(cls, parser):
        registry = TaskRegistry.get_registry()
        templates = registry.task_templates_by_name[cls.name]
        parser.add_argument(
            "--task-template-nickname",
            type=str,
            choices=[t.nickname for t in templates],
            help="Task template nickname",
        )


class TaskRegistry:
    task_registry: TaskRegistry

    def __init__(self) -> None:
        assert (
            getattr(TaskRegistry, "task_registry", None) is None
        ), "Do not double-initialize task registry"
        TaskRegistry.task_registry = self
        self.task_by_name: Mapping[str, Type[CMTask]] = {}
        self.analyzer_by_name: Mapping[str, AnalyzerModel] = {}
        self.analyzer_by_id: Mapping[int, AnalyzerModel] = {}
        self.task_template_by_id: Dict[int, TaskTemplateModel] = {}
        self.task_template_by_nickname: Dict[str, TaskTemplateModel] = {}
        self.task_templates_by_name: Dict[str, List[TaskTemplateModel]] = defaultdict(
            list
        )
        self.background_tasks: Set[asyncio.Task] = set()
        self.analysis_model_by_name: Dict[str, Type[AnalysisModel]] = {}

    async def load_analyzers(self, session):
        from .models import Analyzer

        anas = await session.execute(select(Analyzer))
        self.analyzer_by_id = {a.id: a.as_model(session) for (a,) in anas}
        # Get latest version of analyzer for each task name
        analyzers = list(self.analyzer_by_id.values())
        analyzers.sort(key=lambda a: (a.name, -a.version))
        self.analyzer_by_name = {
            a.name: a
            for a in (next(i) for (_, i) in groupby(analyzers, lambda a: a.name))
        }
        # Add analyzer objects for missing tasks
        missing = [
            task
            for (name, task) in self.load_tasks().items()
            if name not in self.analyzer_by_name
            or self.analyzer_by_name[name].version != task.version
        ]
        if missing:
            extras = [
                Analyzer(name=task.name, version=task.version)
                for task in missing
                if task.name != "dummy"
            ]
            session.add_all(extras)
            await session.commit()
            extra_models = [a.as_model(session) for a in extras]
            self.analyzer_by_id.update({a.id: a for a in extra_models})
            self.analyzer_by_name.update({a.name: a for a in extra_models})

    def get_task_cls_by_name(self, name) -> Type[CMTask]:
        return self.task_by_name.get(name, CMTask)

    async def load_templates(self, session):
        from .models import TaskTemplate

        tmpl = await session.execute(select(TaskTemplate))
        self.task_template_by_id = {t.id: t.as_model(session) for (t,) in tmpl}
        self.task_template_by_nickname = {
            t.nickname: t for t in self.task_template_by_id.values() if t.nickname
        }
        for t in self.task_template_by_id.values():
            self.task_templates_by_name[self.analyzer_by_id[t.analyzer_id].name].append(
                t
            )

    async def compute_default_triggers(self, session):
        for task in self.task_by_name.values():
            async for trigger in task.compute_default_triggers(session, self):
                yield trigger

    def analysis_from_params(self, analyzer_name: str, **kwargs) -> AnalysisModel:
        analysis_model = self.analysis_model_by_name[analyzer_name]
        assert analysis_model
        return analysis_model(**kwargs)

    async def get_task_template(
        self, nickname: str, session=None
    ) -> Optional[TaskTemplateModel]:
        if template := self.task_template_by_nickname.get(nickname):
            return template
        from .models import TaskTemplate, Session

        if session is None:
            async with Session() as session:
                return await self.get_task_template(nickname, session)
        template_db = await session.scalar(
            select(TaskTemplate).filter_by(nickname=nickname)
        )
        if not template_db:
            return None
        template = template_db.as_model(session)
        self.update_template(template)

    async def get_task_template_by_id(
        self, id: int, session=None
    ) -> Optional[TaskTemplateModel]:
        if template := self.task_template_by_id.get(id):
            return template
        from .models import TaskTemplate, Session

        if session is None:
            async with Session() as session:
                return await self.get_task_template_by_id(id, session)
        template_db = await session.get(TaskTemplate, id)
        if not template_db:
            return None
        template = template_db.as_model(session)
        self.update_template(template)

    async def ensure_default_triggers(
        self, session, collection_id: Optional[int] = None
    ):
        from .models import TaskTrigger, Collection

        q = select(TaskTrigger)
        if collection_id:
            q = q.filter_by(collection_id=collection_id)
        existing = [t for (t,) in await session.execute(q)]
        existing_by_sig = {t.signature(): t for t in existing}
        found = set()
        async for trigger_model in self.compute_default_triggers(session):
            sig = trigger_model.signature()
            if collection_id and trigger_model.collection_id != collection_id:
                continue
            if sig not in existing_by_sig:
                session.add(await TaskTrigger.from_model(session, trigger_model))
            found.add(sig)
        # delete any trigger that cannot be found in defaults now, unless it was manually created.
        extra = [
            t for s, t in existing_by_sig.items() if not (t.creator_id or s in found)
        ]
        for t in extra:
            await session.delete(t)
        await session.commit()

    async def handle_created_objects(self):
        from .models import (
            TaskTrigger,
            TopicCollection,
            Collection,
            created_objects,
            Session,
            batch_lambda_query,
        )

        tasks = []
        # Note that created objects can even come from another query.
        if not created_objects:
            return
        created_objects_ = set(created_objects)
        created_objects.clear()
        async with Session() as session:
            # logger.debug("handle_created_objects %s", created_objects_)
            topic_types = {topic[0] for topic in created_objects_}
            topic_ids = {topic[1] for topic in created_objects_}
            collections_by_id = defaultdict(list)

            def q1(ids):
                return lambda_stmt(
                    lambda: select(
                        TopicCollection.topic_id, TopicCollection.collection_id
                    )
                    .filter(TopicCollection.topic_id.in_(ids))
                    .order_by(TopicCollection.topic_id)
                )

            r = await batch_lambda_query(session, q1, topic_ids)
            # Note: topic_ids order is not strict thanks to batching.
            for topic_id, g in groupby(r, lambda c: c[0]):
                collections_by_id[topic_id].extend([c[1] for c in g])
            all_collections = set(chain(*collections_by_id.values()))
            collection_names = {
                id: name
                for (id, name) in await session.execute(
                    select(Collection.id, Collection.name)
                )
            }
            triggers = await session.execute(
                select(TaskTrigger)
                .filter_by(automatic=True)
                .filter(
                    and_(
                        TaskTrigger.creation_trigger_id.in_(list(topic_types)),
                        TaskTrigger.analyzer_trigger_id.is_(None),
                        or_(
                            TaskTrigger.collection_id.in_(all_collections),
                            TaskTrigger.collection_id.is_(None),
                        ),
                    )
                )
                .order_by(TaskTrigger.creation_trigger_id, TaskTrigger.collection_id)
            )
            triggers_by_type = {
                tt: list(g)
                for (tt, g) in groupby(
                    (t for (t,) in triggers), lambda t: t.creation_trigger_id
                )
            }
            if not triggers_by_type:
                return
            for tlist in triggers_by_type.values():
                tlist.sort(key=lambda t: t.collection_id or 0)
            for topic_type, topic_id in created_objects_:
                triggers = triggers_by_type.get(topic_type, [])
                for trigger in triggers:
                    if (
                        trigger.collection_id is None
                        or trigger.collection_id in collections_by_id.get(topic_id, [])
                    ):
                        task_class = self.get_task_cls_by_name(
                            self.analyzer_by_id[trigger.target_analyzer_id].name
                        )
                        tasks.extend(
                            await task_class.apply_topic_trigger(
                                session,
                                trigger,
                                topic_id,
                                collection_names.get(trigger.collection_id),
                            )
                        )
        logger.debug("%s", tasks)
        # TODO: deduplicate
        for task in tasks:
            await task.schedule()

    async def trigger_task_on_task_end(self, task: CMTask):
        from .pyd_models import process_status
        from .models import Collection, Session, TaskTrigger, Analysis, Topic

        analysis_model: AnalysisModel = task.analysis
        async with Session() as session:
            if task.materialize_analysis:
                assert (
                    task.analysis.status == process_status.complete
                ), "task did not call set_completed"
            q = (
                select(TaskTrigger)
                .filter_by(
                    automatic=True,
                    analyzer_trigger_id=self.analyzer_by_name[task.name].id,
                )
                .options(joinedload(TaskTrigger.collection))
            )
            if task.collection_specific_results and analysis_model.collection_name:
                q = q.join(Collection, TaskTrigger.collection).filter_by(
                    name=analysis_model.collection_name
                )
            elif target_id := getattr(analysis_model, "target_id", None):
                target = await session.get(
                    Topic, target_id, options=[subqueryload(Topic.collections)]
                )
                if not target:
                    logger.warn(f"Lost target {target_id}")
                    return
                if target.collections:
                    q = q.filter(
                        or_(
                            TaskTrigger.collection_id.is_(None),
                            TaskTrigger.collection_id.in_(
                                [c.id for c in target.collections]
                            ),
                        )
                    )
                else:
                    q = q.filter(TaskTrigger.collection_id.is_(None))
            triggers = await session.scalars(q.order_by(TaskTrigger.collection_id))
            tasks: List[CMTask] = []
            for trigger in triggers:
                task_class = self.get_task_cls_by_name(
                    self.analyzer_by_id[trigger.target_analyzer_id].name
                )
                if task_class != CMTask:
                    tasks.extend(
                        await task_class.apply_task_trigger(session, trigger, task)
                    )
        # TODO: deduplicate
        for task in tasks:
            await task.schedule()

    async def compute_task_cascade_for_delete(
        self, task: CMTask
    ) -> AsyncGenerator[CMTask, None]:
        # async yield for
        async for task in self.compute_task_cascade(task, for_delete=True):
            yield task

    async def compute_task_cascade(
        self, task: CMTask, for_delete: bool = False
    ) -> AsyncGenerator[CMTask, None]:
        from .models import (
            TaskTrigger,
            Session,
            Analysis,
            Collection,
            Topic,
            process_status,
        )

        if for_delete and not task.can_delete_results:
            return
        async with Session() as session:
            q = (
                select(TaskTrigger)
                .filter_by(analyzer_trigger_id=self.analyzer_by_name[task.name].id)
                .options(joinedload(TaskTrigger.collection))
            )
            if task.collection_specific_results and task.analysis.collection_id:
                collection_name = task.analysis.collection_name or await session.scalar(
                    select(Collection.name).filter_by(id=task.analysis.collection_id)
                )
                q = q.join(Collection, TaskTrigger.collection).filter_by(
                    name=collection_name
                )
            triggers = await session.scalars(q)
            for trigger in triggers:
                task_class = self.get_task_cls_by_name(
                    self.analyzer_by_id[trigger.target_analyzer_id].name
                )
                if task_class == CMTask:
                    continue
                tasks = await task_class.apply_task_trigger(session, trigger, task)
                for dep_task in tasks:
                    if dep_task.materialize_analysis:
                        if not dep_task.analysis.id:
                            continue
                    else:
                        dep_task.analysis.status = await dep_task.status(session)
                    if for_delete and dep_task.status != process_status.complete:
                        continue
                    async for t in self.compute_task_cascade(dep_task, for_delete):
                        yield t

            if task.analysis.id and not for_delete and False:
                # WIP
                analysis = await session.get(
                    Analysis,
                    task.analysis.id,
                    options=[
                        subqueryload(Analysis.generated_topics).subqueryload(
                            Topic.collections
                        )
                    ],
                )
                await session.refresh(analysis, ["generated_topics"])
                types = {t.type for t in analysis.generated_topics}
                collection_id = task.analysis.collection_id
                collection = (
                    (await session.get(Collection, collection_id))
                    if collection_id
                    else None
                )
                collection_cond = TaskTrigger.collection_id.is_(None)
                if collection_id:
                    collection_cond = or_(
                        collection_cond, TaskTrigger.collection_id == collection_id
                    )
                collection_ids = list(set((None, collection_id)))
                triggers = await session.scalars(
                    select(TaskTrigger)
                    .filter(
                        and_(
                            TaskTrigger.creation_trigger_id.in_(types),
                            TaskTrigger.analyzer_trigger_id.is_(None),
                            collection_cond,
                        )
                    )
                    .options(joinedload(TaskTrigger.collection))
                    .order_by(TaskTrigger.creation_trigger_id)
                )
                triggers_by_type = {
                    tt: list(g)
                    for (tt, g) in groupby(triggers, lambda t: t.creation_trigger_id)
                }
                for output in analysis.generated_topics:
                    triggers = triggers_by_type.get(output.type, [])
                    for trigger in triggers:
                        task_class = self.get_task_cls_by_name(
                            self.analyzer_by_id[trigger.target_analyzer_id].name
                        )
                        if task_class == CMTask:
                            continue
                        for t in await task_class.apply_topic_trigger(
                            session, trigger, output.id, collection
                        ):
                            t.analysis.status = await t.status(session)
                            yield t

            yield task

    async def delete_task_data(self, session, task: CMTask, delete_tasks=False):
        from .models import Analysis

        async for subtask in self.compute_task_cascade_for_delete(task):
            await subtask.delete_results(session)
            if delete_tasks and subtask.materialize_analysis:
                analysis = await session.get(Analysis, subtask.analysis.id)
                await session.delete(analysis)
        # commit outside

    async def trigger_task_on_task_error(self, task: CMTask):
        if task.analysis.id:
            from .pyd_models import process_status
            from .models import Session, Analysis

            async with Session() as session:
                analysis = await session.get(Analysis, task.analysis.id)
                analysis.status = process_status.error
                await session.commit()

    def load_tasks(self):
        if not self.task_by_name:
            from .pyd_models import finalize_permissions

            pm = get_plugin_manager()
            pm.hook.register_tasks(task_map=self.task_by_name)
            for task in self.task_by_name.values():
                task.setup_models()
                analysis_model = task.analysis_model
                f = analysis_model.model_fields["analyzer_name"]
                name = f.annotation.__args__[0]
                self.analysis_model_by_name[name] = analysis_model
            finalize_permissions()
        return self.task_by_name

    @classmethod
    def get_registry(cls):
        if not getattr(cls, "task_registry", None):
            TaskRegistry()
            cls.task_registry.load_tasks()
        return cls.task_registry

    @classmethod
    async def get_full_registry(cls):
        from . import Session

        registry = cls.get_registry()
        if not registry.analyzer_by_id:
            async with Session() as session:
                await registry.load_analyzers(session)
                await registry.load_templates(session)
                await registry.ensure_default_triggers(session)
            from .models import finalize_db_models

            await finalize_db_models()
        return registry

    def update_template(self, task_template: TaskTemplateModel, original_nickname=None):
        self.task_template_by_id[task_template.id] = task_template
        if original_nickname and original_nickname != task_template.nickname:
            self.task_template_by_nickname.pop(original_nickname, None)
        if task_template.nickname:
            self.task_template_by_nickname[task_template.nickname] = task_template
        analyzer = self.analyzer_by_id[task_template.analyzer_id]
        templates = self.task_templates_by_name[analyzer.name]
        self.task_templates_by_name[analyzer.name] = [
            t for t in templates if t.id != task_template.id
        ] + [task_template]

    def lookup_task_template(
        self, analyzer_name: str, params: Optional[Dict] = None
    ) -> Optional[TaskTemplateModel]:
        # TODO: Send a signal to task handlers to reload. Note: Kafka not perfect for this, because consumed by single handler.
        params = params or {}
        for t in self.task_templates_by_name[analyzer_name]:
            # TODO transform task into params. Sigh.
            t_params = t.model_dump()
            for k in ("id", "nickname", "analyzer_id", "collection_id", "draft"):
                t_params.pop(k, None)
            if t_params == filter_dict(params):
                return t
        return None

    async def ensure_task_template(
        self,
        session,
        analyzer_name: str,
        nickname: Optional[str] = None,
        params: Optional[Dict] = None,
        draft=True,
        collection_id=None,
    ) -> TaskTemplateModel:
        from .models import TaskTemplate

        params = params or {}
        template = None
        if nickname:
            template = await self.get_task_template(nickname)
        if template is None:
            template = self.lookup_task_template(analyzer_name, params)
        if template is not None:
            return template
        # TODO: What if param collision with and without nickname? Just accept there's an error for now.
        assert params
        analyzer = self.analyzer_by_name[analyzer_name]
        template = TaskTemplate(
            params=params,
            nickname=nickname,
            analyzer_id=analyzer.id,
            draft=draft,
            collection_id=collection_id,
        )
        session.add(template)
        await session.commit()
        template_model = template.as_model(session)
        self.task_template_by_id[template.id] = template_model
        self.task_templates_by_name[analyzer_name].append(template_model)
        if nickname:
            self.task_template_by_nickname[nickname] = template_model
        return template_model

    def tasks_by_target_type(self, type_: topic_type):
        return (t for t in self.task_by_name.values() if type_ in t.task_scale)

    async def active_tasks(
        self, session, scale_filter=None, collection: Optional[str] = None
    ) -> AsyncGenerator[Tuple[Type[CMTask], Optional[TaskTemplateModel]], None]:
        # TODO: Take into account collection-specific triggers
        from .pyd_models import AnalysisWithTemplateModel
        from .models import Collection

        for task in self.task_by_name.values():
            if scale_filter is not None and scale_filter not in task.task_scale:
                continue
            templates = self.task_templates_by_name.get(task.name)
            if templates:
                collection_id = None
                if collection and task.collection_specific_results:
                    collection_id = await session.scalar(
                        select(Collection.id).filter_by(name=collection)
                    )
                for tmpl in templates:
                    if (
                        tmpl.collection_id == collection_id
                        or not task.collection_specific_results
                    ):
                        yield (task, tmpl)
            elif not issubclass(task.analysis_model, AnalysisWithTemplateModel):
                yield (task, None)

    async def all_task_status(
        self,
        session,
        target,
        collection: Optional[str] = None,
        scale: Optional[fragment_type] = None,
        params: Optional[Dict] = None,
    ) -> AsyncGenerator[
        Tuple[Type[CMTask], Optional[TaskTemplateModel], AnalysisModel], None
    ]:
        from .pyd_models import topic_type
        from .models import Analysis

        q = select(Analysis).filter(
            coalesce(Analysis.target_id, Analysis.theme_id) == target.id
        )
        if params is not None:
            for k, v in params.items():
                q = q.filter(Analysis.params[k] == v)
        analyses = await session.execute(
            q.order_by(Analysis.analyzer_id, Analysis.task_template_id).options(
                joinedload(Analysis.collection)
            )
        )
        analyses = {
            k: list(g)
            for (k, g) in groupby(
                (a for (a,) in analyses),
                lambda a: (a.analyzer_name, a.task_template_id),
            )
        }
        async for task, template in self.active_tasks(
            session, scale_filter=scale or target.type, collection=collection
        ):
            results: List[AnalysisModel] = []
            local_analyses = analyses.get(
                (task.name, template.id if template else None)
            )
            if local_analyses and task.collection_specific_results:
                local_analyses = list(
                    filter(
                        lambda analysis: collection
                        == (analysis.collection.name if analysis.collection else None),
                        local_analyses,
                    )
                )
            if local_analyses:
                for analysis in local_analyses:
                    yield (task, template, analysis.as_model(session))
                continue
            elif template:
                results = await task.results_for(
                    session,
                    target.id,
                    params,
                    nickname=template.nickname,
                    collection_name=collection,
                )
            else:
                results = await task.results_for(
                    session, target.id, params, collection_name=collection
                )
            if results:
                for result in results:
                    yield (task, template, result)

    async def task_from_analysis_id(self, analysis_id: int) -> Optional[CMTask]:
        from .models import Session, Analysis

        async with Session() as session:
            analysis = await session.get(
                Analysis,
                analysis_id,
                options=[
                    joinedload(Analysis.collection),
                    joinedload(Analysis.task_template),
                ],
            )
            if analysis:
                if analysis.task_template_id:
                    await self.get_task_template_by_id(
                        analysis.task_template_id, session
                    )
                return self.task_from_analysis(analysis.as_model(session))
        return None

    def task_from_analysis(self, analysis_model: AnalysisModel) -> CMTask:
        from .pyd_models import AnalysisModel, process_status

        task_class = self.get_task_cls_by_name(analysis_model.analyzer_name)
        if task_class != CMTask:
            return task_class(analysis=analysis_model)

    async def topic_created(self, topic):
        pass  # TODO

    async def task_complete(self, task: CMTask, analysis: AnalysisModel):
        pass  # TODO


hookspec = pluggy.HookspecMarker("claim_miner")
hookimpl = pluggy.HookimplMarker("claim_miner")
plugin_manager = None


class TaskMapSpec:
    @hookspec
    def register_tasks(
        self, task_map: Mapping[str, Type[CMTask]]
    ) -> Mapping[str, Type[CMTask]]:
        return task_map


def get_plugin_manager():
    global plugin_manager
    if plugin_manager is not None:
        return plugin_manager
    pm = pluggy.PluginManager("claim_miner")
    pm.add_hookspecs(TaskMapSpec)
    from .tasks.tasks import BaseTaskMap

    pm.register(BaseTaskMap())
    pm.load_setuptools_entrypoints("claim_miner")
    plugin_manager = pm
    return plugin_manager


def get_task_map() -> Mapping[str, Type[CMTask]]:
    return TaskRegistry.get_registry().task_by_name
