"""
Copyright Society Library and Conversence 2022-2024
"""

from __future__ import annotations

import asyncio
from typing import Optional, Union, Annotated, List

from sqlalchemy.orm import (
    joinedload,
    subqueryload,
    contains_eager,
    aliased as sa_aliased,
)
from sqlalchemy.sql.functions import count
from fastapi import Form
from fastapi.responses import ORJSONResponse, Response
from pydantic import Discriminator

from .. import Session, select
from ..pyd_models import AnalysisModel, process_status, fragment_type, BASE_EMBED_MODEL
from ..models import (
    Analysis,
    Topic,
    with_polymorphic,
    Collection,
    Document,
    ClusterData,
    TaskTemplate,
    Statement,
    Fragment,
    PolyTopic,
    embed_db_model_by_name,
)
from ..app import NotFound, Unauthorized, BadRequest
from ..task_registry import TaskRegistry, CMTask
from ..auth import user_with_coll_permission_t_dep, active_user_t_dep
from . import api_router, get_collection

AllAnalysisModels = Annotated[
    Union[tuple(TaskRegistry.get_registry().analysis_model_by_name.values())],
    Discriminator("analyzer_name"),
]


# TODO: make the subtypes explicit?
@api_router.get(
    "/analysis/{subtype}/list",
)
@api_router.get("/c/{collection}/analysis/{subtype}/list")
async def get_analyses_of_type(
    subtype: str,
    current_user: user_with_coll_permission_t_dep("access"),
    collection: Optional[str] = None,
    target_id: Optional[int] = None,
    nickname: Optional[str] = None,
    offset: int = 0,
    limit: int = 0,
) -> List[AllAnalysisModels]:
    async with Session() as session:
        registry = TaskRegistry.get_registry()
        analyzer = registry.analyzer_by_name.get(subtype, None)
        if not analyzer:
            raise NotFound()
        theme_alias = with_polymorphic(
            Topic, [Statement, Fragment], flat=True, aliased=True
        )
        target_alias = with_polymorphic(Topic, "*", flat=True, aliased=True)
        output_alias = with_polymorphic(Topic, "*", flat=True, aliased=True)
        collection_ob = await get_collection(
            collection, session, current_user.id if current_user else None
        )

        options = [
            joinedload(Analysis.analyzer),
            # joinedload(Analysis.part_of),
            # subqueryload(Analysis.has_parts),
            joinedload(Analysis.task_template),
            subqueryload(Analysis.context),
            subqueryload(Analysis.clusters).subqueryload(ClusterData.fragments),
            contains_eager(Analysis.theme.of_type(theme_alias)),
        ]
        q = (
            select(Analysis)
            .filter_by(analyzer_id=analyzer.id)
            .order_by(Analysis.id)
            .outerjoin(Analysis.theme.of_type(theme_alias))
            .outerjoin(Analysis.target.of_type(target_alias))
        )
        if collection:
            q = q.join(Collection, Analysis.collection).filter_by(name=collection)
        else:
            options.append(joinedload(Analysis.collection))
        if target_id:
            q = q.filter(Analysis.target_id == target_id)
        else:
            options.append(contains_eager(Analysis.target.of_type(target_alias)))
        if nickname:
            q = q.join(TaskTemplate, Analysis.task_template).filter_by(
                nickname=nickname
            )
        if offset:
            q = q.offset(offset)
        if limit:
            q = q.limit(limit)
        q = q.options(*options)
        analyses = list(await session.scalars(q))
        await session.execute(
            select(output_alias)
            .join(Analysis, output_alias.from_analyses)
            .filter(Analysis.id.in_([a.id for a in analyses]))
        )

        # Base permissions on first of target, collection, theme or output? Actually should depend on analyzer!
        models = [analysis.as_model(session) for analysis in analyses]
        for model in models:
            task_class = TaskRegistry.get_registry().get_task_cls_by_name(
                model.analyzer_name
            )
            if not task_class.user_can_admin(current_user, collection_ob):
                model.task_template = None
        return models


@api_router.get("/analysis/{id}")
@api_router.get("/c/{collection}/analysis/{id}")
async def get_analysis(
    id: int,
    current_user: user_with_coll_permission_t_dep("access"),
    collection: Optional[str] = None,
) -> AllAnalysisModels:
    async with Session() as session:
        analysis = await session.get(
            Analysis,
            id,
            options=(
                joinedload(Analysis.analyzer),
                joinedload(Analysis.task_template),
                joinedload(Analysis.part_of),
                joinedload(Analysis.collection),
                joinedload(Analysis.target.of_type(PolyTopic)),
                subqueryload(Analysis.has_parts),
                subqueryload(Analysis.context),
                subqueryload(Analysis.context),
                subqueryload(Analysis.clusters).subqueryload(ClusterData.fragments),
            ),
        )
        if not analysis:
            raise NotFound()
        topic = with_polymorphic(Topic, "*")
        await session.execute(
            select(Analysis.id, topic).join(topic, Analysis.generated_topics)
        )
        if analysis.target_id:
            await session.get(Topic, analysis.target_id)
        if analysis.theme_id:
            await session.get(Topic, analysis.theme_id)
        # Base permissions on first of target, collection, theme or output? Actually should depend on analyzer!
        model: AnalysisModel = analysis.as_model(session)
        collection_ob = await get_collection(
            collection, session, current_user.id if current_user else None
        )
        task_class = TaskRegistry.get_registry().get_task_cls_by_name(
            model.analyzer_name
        )
        if not task_class.user_can_admin(current_user, collection_ob):
            model.task_template = None
        return model


@api_router.get("/statement/{doc_id}/analysis")
@api_router.get("/c/{collection}/document/{doc_id}/analysis")
async def get_document_analyses(
    doc_id: int,
    current_user: user_with_coll_permission_t_dep("access"),
    collection: Optional[str] = None,
) -> List[AllAnalysisModels]:
    async with Session() as session:
        document = await session.get(Document, doc_id)
        if not document:
            raise NotFound()
        theme_alias = with_polymorphic(
            Topic, [Statement, Fragment], flat=True, aliased=True
        )
        output_alias = with_polymorphic(Topic, "*", flat=True, aliased=True)

        q = (
            select(Analysis)
            .filter_by(target_id=doc_id)
            .outerjoin(Analysis.theme.of_type(theme_alias))
            .options(
                joinedload(Analysis.analyzer),
                joinedload(Analysis.task_template),
                joinedload(Analysis.part_of),
                joinedload(Analysis.collection),
                subqueryload(Analysis.has_parts),
                subqueryload(Analysis.context),
                subqueryload(Analysis.context),
                contains_eager(Analysis.theme.of_type(theme_alias)),
            )
        )
        if collection:
            q = q.join(Collection, Analysis.collection).filter_by(name=collection)
        analyses = list(await session.scalars(q))
        await session.execute(
            select(output_alias)
            .join(Analysis, output_alias.from_analyses)
            .filter(Analysis.id.in_([a.id for a in analyses]))
        )

        # Base permissions on first of target, collection, theme or output? Actually should depend on analyzer!
        return [analysis.as_model(session) for analysis in analyses]


@api_router.get("/statement/{stmt_id}/analysis")
@api_router.get("/c/{collection}/statement/{stmt_id}/analysis")
async def get_statement_analyses(
    stmt_id: int,
    current_user: user_with_coll_permission_t_dep("access"),
    collection: Optional[str] = None,
) -> List[AllAnalysisModels]:
    async with Session() as session:
        statement = await session.get(Document, stmt_id)
        if not statement:
            raise NotFound()
        theme_alias = with_polymorphic(
            Topic, [Statement, Fragment], flat=True, aliased=True
        )
        output_alias = with_polymorphic(Topic, "*", flat=True, aliased=True)

        q = (
            select(Analysis)
            .filter_by(target_id=stmt_id)
            .outerjoin(Analysis.theme.of_type(theme_alias))
            .options(
                joinedload(Analysis.analyzer),
                joinedload(Analysis.task_template),
                joinedload(Analysis.part_of),
                joinedload(Analysis.collection),
                subqueryload(Analysis.has_parts),
                subqueryload(Analysis.context),
                subqueryload(Analysis.context),
                contains_eager(Analysis.theme.of_type(theme_alias)),
            )
        )
        if collection:
            q = q.join(Collection, Analysis.collection).filter_by(name=collection)
        analyses = list(await session.scalars(q))
        await session.execute(
            select(output_alias)
            .join(Analysis, output_alias.from_analyses)
            .filter(Analysis.id.in_([a.id for a in analyses]))
        )

        # Base permissions on first of target, collection, theme or output? Actually should depend on analyzer!
        return [analysis.as_model(session) for analysis in analyses]


# Focus on target and output. context and theme can wait.
# Start from objects, but I'll want generic query routes.
# Maybe /analysis/subtype/ as a query
# In general, /object/id/analysis/subtype/id should be allowed


@api_router.post("/analysis")
@api_router.post("/c/{collection}/analysis")
async def post_analysis(
    current_user: active_user_t_dep,
    analysis: AllAnalysisModels,
    collection: Optional[str] = None,
):
    from ..models import Collection

    registry = await TaskRegistry.get_full_registry()
    task_class = registry.get_task_cls_by_name(analysis.analyzer_name)
    if task_class == CMTask:
        raise NotFound()
    async with Session() as session:
        collection_ob: Collection = (
            (
                await session.scalar(
                    select(Collection).where(Collection.name == collection)
                )
            )
            if collection
            else None
        )
        if not task_class.user_can_trigger(current_user, collection_ob):
            raise Unauthorized()
        if analysis.status is None:
            analysis.status = process_status.pending
        if collection:
            analysis.collection_id = collection_ob.id
        analysis.creator_id = current_user.id
        analysis_ob = await Analysis.from_model(session, analysis)
        if task_class.materialize_analysis:
            session.add(analysis_ob)
            await session.commit()
            analysis.id = analysis_ob.id
            task = task_class(analysis=analysis)
            await task.schedule()
            location = f"{collection_ob.path}/analysis/{analysis.id}"
            return ORJSONResponse(
                analysis.model_dump(mode="json"),
                status_code=201,
                headers=dict(location=location),
            )
        else:
            task = registry.task_from_analysis(analysis)
            await task.schedule()
            return Response(status_code=202)  # Accepted


@api_router.get("/analysis/{analysis_id}/wait")
@api_router.get("/c/{collection}/analysis/{analysis_id}/wait")
async def wait_for_analysis(
    current_user: active_user_t_dep, analysis_id: int, collection: Optional[str] = None
) -> AllAnalysisModels:
    async with Session() as session:
        analysis = await session.get(Analysis, analysis_id)
        if not analysis:
            raise NotFound()
        while analysis.status < process_status.complete:
            await asyncio.sleep(0.75)
            await session.refresh(analysis, ["status"])
        await session.refresh(
            analysis,
            [
                "analyzer",
                "theme",
                "target",
                "generated_topics",
                "collection",
                "task_template",
            ],
        )
        topic = with_polymorphic(Topic, "*")
        await session.execute(
            select(Analysis.id, topic).join(topic, Analysis.generated_topics)
        )
        if analysis.target_id:
            await session.get(Topic, analysis.target_id)

        result = analysis.as_model(session)

        # AD HOC, demo-driven
        if result.analyzer_name == "graph_auto_search":
            doc_q = (
                select(
                    count(Document.id),
                    Document.load_status,
                    Document.text_identity.is_(None),
                )
                .join(Analysis, Document.from_analyses)
                .filter_by(id=analysis_id)
                .group_by(Document.load_status, Document.text_identity.is_(None))
            )
            results = list(await session.execute(doc_q))
            num_docs = sum(x[0] for x in results)
            num_error = sum(x[0] for x in results if x[1] == "error")
            num_read = sum(x[0] for x in results if x[1] == "loaded")
            num_extracted = sum(x[0] for x in results if x[1] == "loaded" and not x[2])
            num_pending = num_docs - num_error - num_extracted
            Embedding = embed_db_model_by_name[BASE_EMBED_MODEL.name]
            doc = sa_aliased(Document, flat=True)
            r = await session.execute(
                select(count(Fragment.id), count(Embedding.fragment_id))
                .filter_by(scale=fragment_type.paragraph)
                .outerjoin(Embedding, Embedding.fragment_id == Fragment.id)
                .join(doc, Fragment.document)
                .join(Analysis, doc.from_analyses)
                .filter_by(id=analysis_id)
            )
            num_paras, num_embeddings = r.one()
            doc_embedded = await session.scalar(
                select(count(doc.id.distinct()))
                .join(Analysis, doc.from_analyses)
                .filter_by(id=analysis_id)
                .join(Fragment, Document.paragraphs)
                .join(Embedding, Embedding.fragment_id == Fragment.id)
            )
            # read is 1/3, extraction is 1/3, embedding is 1/3. To be adjusted.
            estimate = (
                num_error
                + num_read / 3
                + (1 + num_embeddings / num_paras) * (num_extracted / 3)
            ) / num_docs
            result.completion_estimate = estimate
        return result


@api_router.post("/analysis/{analysis_id}/trigger")
@api_router.post("/c/{collection}/analysis/{analysis_id}/trigger")
async def analysis_trigger(
    current_user: active_user_t_dep, analysis_id: int, collection: Optional[str] = None
) -> AllAnalysisModels:
    registry = await TaskRegistry.get_full_registry()
    async with Session() as session:
        analysis = await session.get(Analysis, analysis_id)
        if not analysis:
            raise NotFound()
        task_class = registry.task_by_name.get(analysis.analyzer_name)
        collection_ob: Collection = (
            (
                await session.scalar(
                    select(Collection).where(Collection.name == collection)
                )
            )
            if collection
            else None
        )
        if not task_class.user_can_trigger(current_user, collection_ob):
            raise Unauthorized()
        if analysis.status <= process_status.not_ready:
            raise BadRequest("Task cannot be triggered")
        if (
            analysis.status == process_status.complete
            and not task_class.can_reapply_complete
        ):
            raise BadRequest("Cannot re-trigger a complete task")
        analysis.status = process_status.pending
        await session.commit()
        analysis_model = analysis.as_model(session)
        task = registry.task_from_analysis(analysis_model)
        await task.schedule()
        return analysis_model


@api_router.delete("/analysis/{analysis_id}")
@api_router.delete("/c/{collection}/analysis/{analysis_id}")
async def delete_analysis_data(
    current_user: active_user_t_dep, analysis_id: int, collection: Optional[str] = None
) -> AllAnalysisModels:
    # Note this deletes the analysis data, not the analysis.
    # TODO: Maybe add a parameter for that?
    registry = await TaskRegistry.get_full_registry()
    async with Session() as session:
        analysis = await session.get(Analysis, analysis_id)
        if not analysis:
            raise NotFound()
        task_class = registry.task_by_name.get(analysis.analyzer_name)
        collection_ob: Collection = (
            (
                await session.scalar(
                    select(Collection).where(Collection.name == collection)
                )
            )
            if collection
            else None
        )

        if not task_class.can_delete_results:
            raise BadRequest("Analysis data cannot be deleted")

        if not collection_ob.user_can(current_user, "admin"):
            raise Unauthorized()
        if analysis.status != process_status.complete:
            raise BadRequest("Analysis not complete")

        analysis_model = analysis.as_model(session)
        task = registry.task_from_analysis(analysis_model)

        await registry.delete_task_data(session, task, True)
        analysis.status = process_status.not_requested
        await session.commit()
        analysis_model.status = process_status.not_requested

        return analysis_model
