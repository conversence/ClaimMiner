from typing import Optional, List, Annotated

from sqlalchemy import select
from sqlalchemy.orm import joinedload, subqueryload, with_polymorphic, contains_eager
from starlette import status
from fastapi import Request, Form
from starlette.responses import RedirectResponse

from .. import Session
from ..auth import user_with_permission_c_dep, active_user_c_dep
from ..pyd_models import topic_type, process_status
from ..models import (
    Collection,
    Analysis,
    Topic,
    Statement,
    Fragment,
    TaskTemplate,
    ClusterData,
    ClaimLink,
    analysis_output_table,
    claim_neighbourhood,
    AnyClaimOrHyperedge,
    poly_type_clause,
    model_by_topic_type,
)
from ..task_registry import TaskRegistry, CMTask
from ..app import NotFound, Unauthorized, BadRequest, app_router
from . import (
    update_fragment_selection,
    get_base_template_vars,
    templates,
    get_collection,
)


@app_router.get("/analysis/{analysis_id:int}")
@app_router.get("/c/{collection}/analysis/{analysis_id:int}")
async def route_to_analysis(
    request: Request,
    current_user: user_with_permission_c_dep("access"),
    analysis_id: int,
    collection: Optional[str] = None,
):
    async with Session() as session:
        analysis = await session.get(
            Analysis,
            analysis_id,
            options=[joinedload(Analysis.analyzer), joinedload(Analysis.target)],
        )
        if analysis is None:
            raise NotFound()
        task_class = TaskRegistry.get_registry().get_task_cls_by_name(
            analysis.analyzer.name
        )
        collection_path = f"/c/{collection}" if collection else ""
        if task_class.materialize_analysis:
            return RedirectResponse(
                f"/f{collection_path}/analysis/{analysis.analyzer.name}/{analysis_id}"
            )
        if analysis.target:
            collection_ob = await get_collection(
                collection, session, current_user.id if current_user else None
            )
            return RedirectResponse(f"/f{analysis.target.web_path(collection_ob)}")


@app_router.get("/{topictype:topic_type}/{topic_id:int}/analysis/{task_name}")
@app_router.get(
    "/c/{collection}/{topictype:topic_type}/{topic_id:int}/analysis/{task_name}"
)
async def show_topic_dematerialized_analysis_fallback(
    request: Request,
    current_user: user_with_permission_c_dep("access"),
    topic_id: int,
    task_name: str,
    topictype: topic_type,
    collection: Optional[str] = None,
):
    registry = await TaskRegistry.get_full_registry()
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        model_class = model_by_topic_type[topictype]
        topic = await session.get(model_class, topic_id)
        if not topic:
            raise NotFound()
        task_class = registry.get_task_cls_by_name(task_name)
        if task_class == CMTask:
            raise NotFound()
        analyzer = registry.analyzer_by_name[task_name]
        if task_class.materialize_analysis:
            q = select(Analysis).filter_by(analyzer_id=analyzer.id)
            if topic.type == topic_type.collection:
                q = q.filter_by(collection_id=topic_id)
            else:
                q = q.filter_by(target_id=topic_id)
                if topic.type == collection_ob:
                    q = q.filter_by(collection_id=collection_ob.id)
            analyses = list(await session.scalars(q))
            if len(analyses) == 1:
                return RedirectResponse(
                    f"/f{collection_ob.path}/analysis/{task_name}/{analyses[0].id}"
                )
            return templates.TemplateResponse(
                request,
                "list_analyses.html",
                dict(analyzer=analyzer, analyses=analyses, target=topic, **base_vars),
            )
        # TODO: what if the task needs a template?
        # https://sentry.conversence.com/organizations/conversence/issues/1151
        task: CMTask = task_class(target_id=topic_id)
        task.analysis.status = await task.status(session)
        template = task.edit_template_name
        dependent_tasks = []
        if task.analysis.status == process_status.complete:
            async for d in registry.compute_task_cascade_for_delete(task):
                dependent_tasks.append(d)
            dependent_tasks.pop()  # remove self
        can_trigger = collection_ob.user_can(current_user, task.trigger_task_permission)
        base_vars |= dict(
            analyzer=analyzer,
            analysis=task.analysis,
            theme=None,
            outgoing_links=[],
            target=topic,
            dependent_tasks=dependent_tasks,
            incoming_links=[],
            result_nodes=[],
            fragments=[],
            target_nghd=None,
            can_trigger=can_trigger,
            task=task,
        )
        await task.enrich_edit_form_data(session, base_vars)
    return templates.TemplateResponse(request, template, base_vars)


@app_router.post("/{topictype:topic_type}/{topic_id:int}/analysis/{task_name}")
@app_router.post(
    "/c/{collection}/{topictype:topic_type}/{topic_id:int}/analysis/{task_name}"
)
async def alter_topic_dematerialized_analysis_fallback(
    request: Request,
    current_user: user_with_permission_c_dep("access"),
    topic_id: int,
    task_name: str,
    topictype: topic_type,
    collection: Optional[str] = None,
):
    registry = await TaskRegistry.get_full_registry()
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        model_class = model_by_topic_type[topictype]
        topic = await session.get(model_class, topic_id)
        if not topic:
            raise NotFound()
        task_class = registry.get_task_cls_by_name(task_name)
        if task_class == CMTask:
            raise NotFound()
        analyzer = registry.analyzer_by_name[task_name]
        if task_class.materialize_analysis:
            raise BadRequest("Only use this for non-materialized analysis")
        task: CMTask = task_class(target_id=topic_id)
        task.analysis.status = await task.status(session)
        dependent_tasks = []
        template = task.edit_template_name
        can_trigger = collection_ob.user_can(current_user, task.trigger_task_permission)
        form = await request.form()
        if task.analysis.status == process_status.complete:
            if "delete_results" in form:
                if not collection_ob.user_can(current_user, "admin"):
                    raise Unauthorized()
                await registry.delete_task_data(session, task, True)
                task.analysis.status == process_status.not_requested
                await session.commit()
            else:
                if task.analysis.status == process_status.complete:
                    async for d in registry.compute_task_cascade_for_delete(task):
                        dependent_tasks.append(d)
                    dependent_tasks.pop()  # remove self
        if "launch" in form:
            if not can_trigger:
                raise Unauthorized()
            if task.analysis.status <= process_status.not_ready or (
                task.analysis.status == process_status.complete
                and not task.can_reapply_complete
            ):
                raise BadRequest()
            await task.schedule()
            task.analysis.status = process_status.pending
        base_vars |= dict(
            analyzer=analyzer,
            analysis=task.analysis,
            theme=None,
            outgoing_links=[],
            target=topic,
            dependent_tasks=dependent_tasks,
            can_trigger=can_trigger,
            incoming_links=[],
            result_nodes=[],
            fragments=[],
            target_nghd=None,
        )
        await task.enrich_edit_form_data(session, base_vars)
    return templates.TemplateResponse(request, template, base_vars)


@app_router.get("/analysis/{task_name}/{analysis_id:int}")
@app_router.get("/c/{collection}/analysis/{task_name}/{analysis_id:int}")
async def show_analysis_fallback(
    request: Request,
    current_user: user_with_permission_c_dep("access"),
    analysis_id: int,
    task_name: str,
    collection: Optional[str] = None,
):
    registry = await TaskRegistry.get_full_registry()
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        theme_alias = with_polymorphic(Topic, "*", flat=True, aliased=True)
        target_alias = with_polymorphic(Topic, "*", flat=True, aliased=True)

        analysis = await session.scalar(
            select(Analysis)
            .filter_by(id=analysis_id)
            .outerjoin(Analysis.theme.of_type(theme_alias))
            .outerjoin(Analysis.target.of_type(target_alias))
            .options(
                joinedload(Analysis.analyzer),
                joinedload(Analysis.task_template),
                joinedload(Analysis.part_of),
                joinedload(Analysis.collection),
                subqueryload(Analysis.has_parts),
                subqueryload(Analysis.context),
                contains_eager(Analysis.target.of_type(target_alias)),
                contains_eager(Analysis.theme.of_type(theme_alias)),
            )
        )
        if not analysis:
            raise NotFound()
        analyzer = analysis.analyzer
        if analyzer.name != task_name:
            return RedirectResponse(
                f"/f{collection_ob.path}/analysis/{analyzer.name}/{analysis_id}"
            )
        task: CMTask = registry.task_from_analysis(analysis.as_model(session))

        # TODO: Is there a permission to see the analysis?

        # if analyzer.name == 'cluster':
        #     if not collection_ob:
        #         assert analysis.collection_id
        #         collection_ob = await session.get(Collection, analysis.collection_id)
        #     return RedirectResponse(f"/f{collection_ob.path}/analysis/cluster/{analysis_id}")

        template = task.edit_template_name
        theme = analysis.theme
        target = analysis.target
        fragments = {f.id: f for f in analysis.context}
        # TODO: Filter fragments in collection
        PolyTopic = with_polymorphic(Topic, "*")
        r = await session.execute(
            select(PolyTopic)
            .join(analysis_output_table)
            .filter_by(analysis_id=analysis.id)
            .options(
                subqueryload(PolyTopic.ClaimLink.source_topic.of_type(PolyTopic)),
                subqueryload(PolyTopic.ClaimLink.target_topic.of_type(PolyTopic)),
            )
        )
        result_nodes = {f.id: f for (f,) in r}
        # TODO: Filter topics in collection
        outgoing_links = []
        incoming_links = []
        target_nghd = {}
        nghd_target = (
            target if (target and target.type == topic_type.standalone) else theme
        )
        if nghd_target and nghd_target.type == topic_type.standalone:
            target_nghd = await claim_neighbourhood(nghd_target.id, session)
            r = await session.execute(
                select(AnyClaimOrHyperedge, ClaimLink)
                .filter(poly_type_clause(AnyClaimOrHyperedge))
                .join(
                    ClaimLink,
                    (
                        (AnyClaimOrHyperedge.id == ClaimLink.source)
                        & (ClaimLink.target == nghd_target.id)
                    )
                    | (
                        (AnyClaimOrHyperedge.id == ClaimLink.target)
                        & (ClaimLink.source == nghd_target.id)
                    ),
                )
            )
            related_link = r.fetchall()
            outgoing_links = [
                (fragment, link)
                for (fragment, link) in related_link
                if link.source == nghd_target.id
            ]
            incoming_links = [
                (fragment, link)
                for (fragment, link) in related_link
                if link.target == nghd_target.id
            ]
        dependent_tasks = []
        can_delete = task.can_delete_results
        if can_delete and analysis.status == process_status.complete:
            async for d in registry.compute_task_cascade_for_delete(task):
                dependent_tasks.append(d)
            dependent_tasks.pop()  # remove self
        can_trigger = collection_ob.user_can(current_user, task.trigger_task_permission)
        base_vars |= dict(
            analyzer=analyzer,
            analysis=analysis,
            theme=theme,
            outgoing_links=outgoing_links,
            target=target,
            dependent_tasks=dependent_tasks,
            can_trigger=can_trigger,
            incoming_links=incoming_links,
            result_nodes=result_nodes,
            fragments=fragments,
            target_nghd=target_nghd,
            task=task,
        )
        await task.enrich_edit_form_data(session, base_vars)

    return templates.TemplateResponse(request, template, base_vars)


@app_router.post("/analysis/{task_name}/{analysis_id:int}")
@app_router.post("/c/{collection}/analysis/{task_name}/{analysis_id:int}")
async def alter_analysis_fallback(
    request: Request,
    current_user: user_with_permission_c_dep("access"),
    analysis_id: int,
    task_name: str,
    collection: Optional[str] = None,
):
    registry = await TaskRegistry.get_full_registry()
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        theme_alias = with_polymorphic(Topic, "*", flat=True, aliased=True)
        target_alias = with_polymorphic(Topic, "*", flat=True, aliased=True)

        analysis = await session.scalar(
            select(Analysis)
            .filter_by(id=analysis_id)
            .outerjoin(Analysis.theme.of_type(theme_alias))
            .outerjoin(Analysis.target.of_type(target_alias))
            .options(
                joinedload(Analysis.analyzer),
                joinedload(Analysis.task_template),
                joinedload(Analysis.part_of),
                joinedload(Analysis.collection),
                subqueryload(Analysis.has_parts),
                subqueryload(Analysis.context),
                contains_eager(Analysis.target.of_type(target_alias)),
                contains_eager(Analysis.theme.of_type(theme_alias)),
            )
        )
        if not analysis:
            raise NotFound()
        analyzer = analysis.analyzer
        if analyzer.name != task_name:
            return RedirectResponse(
                f"/f{collection_ob.path}/analysis/{analyzer.name}/{analysis_id}"
            )
        task: CMTask = registry.task_from_analysis(analysis.as_model(session))

        # TODO: Is there a permission to see the analysis?

        # if analyzer.name == 'cluster':
        #     if not collection_ob:
        #         assert analysis.collection_id
        #         collection_ob = await session.get(Collection, analysis.collection_id)
        #     return RedirectResponse(f"/f{collection_ob.path}/analysis/cluster/{analysis_id}")

        template = task.edit_template_name
        theme = analysis.theme
        target = analysis.target
        fragments = {f.id: f for f in analysis.context}
        # TODO: Filter fragments in collection
        PolyTopic = with_polymorphic(Topic, "*")
        r = await session.execute(
            select(PolyTopic)
            .join(analysis_output_table)
            .filter_by(analysis_id=analysis.id)
        )
        result_nodes = {f.id: f for (f,) in r}
        # TODO: Filter topics in collection
        outgoing_links = []
        incoming_links = []
        target_nghd = {}
        nghd_target = (
            target if (target and target.type == topic_type.standalone) else theme
        )
        if nghd_target and nghd_target.type == topic_type.standalone:
            target_nghd = await claim_neighbourhood(nghd_target.id, session)
            r = await session.execute(
                select(AnyClaimOrHyperedge, ClaimLink)
                .filter(poly_type_clause(AnyClaimOrHyperedge))
                .join(
                    ClaimLink,
                    (
                        (AnyClaimOrHyperedge.id == ClaimLink.source)
                        & (ClaimLink.target == nghd_target.id)
                    )
                    | (
                        (AnyClaimOrHyperedge.id == ClaimLink.target)
                        & (ClaimLink.source == nghd_target.id)
                    ),
                )
            )
            related_link = r.fetchall()
            outgoing_links = [
                (fragment, link)
                for (fragment, link) in related_link
                if link.source == nghd_target.id
            ]
            incoming_links = [
                (fragment, link)
                for (fragment, link) in related_link
                if link.target == nghd_target.id
            ]
        dependent_tasks = []
        can_trigger = collection_ob.user_can(current_user, task.trigger_task_permission)
        form = await request.form()
        if task.analysis.status == process_status.complete:
            if "delete_results" in form:
                if not collection_ob.user_can(current_user, "admin"):
                    raise Unauthorized()
                await registry.delete_task_data(session, task, True)
                task.analysis.status = process_status.not_requested
                await session.commit()
            else:
                async for d in registry.compute_task_cascade_for_delete(task):
                    dependent_tasks.append(d)
                if dependent_tasks:
                    dependent_tasks.pop()  # remove self
        if "launch" in form:
            if not can_trigger:
                raise Unauthorized()
            if task.analysis.status <= process_status.not_ready or (
                task.analysis.status == process_status.complete
                and not task.can_reapply_complete
            ):
                raise BadRequest()
            await task.schedule()
            task.analysis.status = process_status.pending

        can_delete = task.can_delete_results
        if can_delete and analysis.status == process_status.complete:
            async for d in registry.compute_task_cascade_for_delete(task):
                dependent_tasks.append(d)
            dependent_tasks.pop()  # remove self
        base_vars |= dict(
            analyzer=analyzer,
            analysis=analysis,
            theme=theme,
            outgoing_links=outgoing_links,
            target=target,
            dependent_tasks=dependent_tasks,
            can_trigger=can_trigger,
            incoming_links=incoming_links,
            result_nodes=result_nodes,
            fragments=fragments,
            target_nghd=target_nghd,
            task=task,
        )
        await task.enrich_edit_form_data(session, base_vars)

    return templates.TemplateResponse(request, template, base_vars)


@app_router.post("/analysis/{task_name}/{analysis_id:int}")
@app_router.post("/c/{collection}/analysis/{task_name}/{analysis_id:int}")
async def edit_analysis_fallback(
    task_name: str,
    analysis_id: int,
    request: Request,
    current_user: user_with_permission_c_dep("access"),
    collection: Optional[str] = None,
):
    registry = await TaskRegistry.get_full_registry()
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        theme_alias = with_polymorphic(Topic, "*", flat=True, aliased=True)
        target_alias = with_polymorphic(Topic, "*", flat=True, aliased=True)

        analysis = await session.scalar(
            select(Analysis)
            .filter_by(id=analysis_id)
            .outerjoin(Analysis.theme.of_type(theme_alias))
            .outerjoin(Analysis.target.of_type(target_alias))
            .options(
                joinedload(Analysis.analyzer),
                joinedload(Analysis.task_template),
                joinedload(Analysis.part_of),
                joinedload(Analysis.collection),
                subqueryload(Analysis.has_parts),
                subqueryload(Analysis.context),
                contains_eager(Analysis.target.of_type(target_alias)),
                contains_eager(Analysis.theme.of_type(theme_alias)),
            )
        )
        if not analysis:
            raise NotFound()
        # TODO: Is there a permission to see the analysis?

        analyzer = analysis.analyzer
        if analyzer.name != task_name:
            return RedirectResponse(
                f"/f{collection_ob.path}/analysis/{analyzer.name}/{analysis_id}"
            )
        # Update the analysis based on form paramters. Iffy.
        form = await request.form()
        task_class = registry.get_task_cls_by_name(analyzer.name)
        baseline_task = task_class()
        params = dict(form)
        for k in ("selection_changes", "selection_reset"):
            params.pop(k, None)
        updated_task = task_class(**params)
        await updated_task.complete_params(session)
        changed = False
        for k, v in updated_task.__dict__.items():
            if k.startswith("_") or k in ("params",):
                continue
            if k == "nickname":
                tt = await registry.get_task_template(v, session)
                if tt:
                    k = "task_template_id"
                    v = tt.id
                else:
                    continue
            if (
                v
                and v != getattr(baseline_task, k, None)
                and v != getattr(analysis, k, None)
            ):
                changed = True
                setattr(analysis, k, v)
        for k, v in updated_task.params.items():
            if v and v != baseline_task.params.get(k, None):
                changed = True
                analysis.params[k] = v
        if changed:
            await session.commit()

        analysis_model = analysis.as_model(session)
        task = registry.task_from_analysis(analysis_model)

        # TODO: If the analysis is pending, it might mean that I'm asking to re-run it. But maybe not.

        template = task.edit_template_name
        theme = analysis.theme
        target = analysis.target
        fragments = {f.id: f for f in analysis.context}
        # TODO: Filter fragments in collection
        PolyTopic = with_polymorphic(Topic, "*")
        r = await session.execute(
            select(PolyTopic)
            .join(analysis_output_table)
            .filter_by(analysis_id=analysis.id)
        )
        result_nodes = {f.id: f for (f,) in r}
        # TODO: Filter topics in collection
        outgoing_links = []
        incoming_links = []
        target_nghd = {}
        nghd_target = (
            target if (target and target.type == topic_type.standalone) else theme
        )
        if nghd_target and nghd_target.type == topic_type.standalone:
            target_nghd = await claim_neighbourhood(nghd_target.id, session)
            r = await session.execute(
                select(AnyClaimOrHyperedge, ClaimLink)
                .filter(poly_type_clause(AnyClaimOrHyperedge))
                .join(
                    ClaimLink,
                    (
                        (AnyClaimOrHyperedge.id == ClaimLink.source)
                        & (ClaimLink.target == nghd_target.id)
                    )
                    | (
                        (AnyClaimOrHyperedge.id == ClaimLink.target)
                        & (ClaimLink.source == nghd_target.id)
                    ),
                )
            )
            related_link = r.fetchall()
            outgoing_links = [
                (fragment, link)
                for (fragment, link) in related_link
                if link.source == nghd_target.id
            ]
            incoming_links = [
                (fragment, link)
                for (fragment, link) in related_link
                if link.target == nghd_target.id
            ]
        can_trigger = collection_ob.user_can(current_user, task.trigger_task_permission)
        base_vars |= dict(
            analyzer=analyzer,
            analysis=analysis,
            theme=theme,
            outgoing_links=outgoing_links,
            target=target,
            can_trigger=can_trigger,
            incoming_links=incoming_links,
            result_nodes=result_nodes,
            fragments=fragments,
            target_nghd=target_nghd,
        )
        await task.enrich_edit_form_data(session, base_vars)

    return templates.TemplateResponse(request, template, base_vars)


@app_router.post("/analysis/{task_name}")
@app_router.post("/c/{collection_name}/analysis/{task_name}")
async def post_new_analysis_fallback(
    request: Request,
    current_user: active_user_c_dep,
    task_name: str,
    collection_name: Optional[str] = None,
):
    # TODO: Is there a way to get the parameters come from the task (other than redefining the function...)
    registry = await TaskRegistry.get_full_registry()
    task_class = registry.get_task_cls_by_name(task_name)
    if task_class == CMTask:
        raise NotFound()
    analysis_model_cls = task_class.analysis_model
    async with Session() as session:
        collection_ob: Collection = await get_collection(
            collection_name, session, current_user.id if current_user else None
        )
        if not task_class.user_can_trigger(current_user, collection_ob):
            raise Unauthorized()
        form = await request.form()
        params = dict(form)
        sources: List[int] = []
        if "source_ids" in analysis_model_cls.model_fields:
            sources = list(
                update_fragment_selection(
                    request,
                    params.pop("selection_changes", {}),
                    params.pop("reset_fragments", False),
                )
            )
            if not sources:
                raise BadRequest("No source fragments selected")
            params["source_ids"] = sources
        if "status" not in params:
            params["status"] = "pending"
        task = task_class(collection_name=collection_name, **params)
        await task.complete_params(session)
        if task.materialize_analysis:
            analysis_model = task.analysis
            analysis_model.creator_id = current_user.id
            analysis = await Analysis.from_model(session, analysis_model)
            session.add(analysis)
            await session.commit()
            if (
                task.task_template_model
                and not analysis_model.task_template
                and analysis.task_template_id
            ):
                analysis_model.task_template = await registry.get_task_template_by_id(
                    analysis.task_template_id, session
                )
            if analysis.status == process_status.pending:
                task.analysis.id = analysis.id
                await task.schedule()
            return RedirectResponse(
                f"/f{collection_ob.path}/analysis/{task_name}/{analysis.id}",
                status_code=status.HTTP_303_SEE_OTHER,
            )
        else:
            await task.schedule()
            target_id = form.get("target_id") or form.get("theme_id")
            if not target_id:
                raise NotFound()
            target = await session.get(Topic, int(target_id))
            if target.type == topic_type.standalone:
                return RedirectResponse(
                    f"/f{collection_ob.path}/claim/{target.id}",
                    status_code=status.HTTP_303_SEE_OTHER,
                )
            elif target.type == topic_type.document:
                return RedirectResponse(
                    f"/f{collection_ob.path}/document/{target.id}",
                    status_code=status.HTTP_303_SEE_OTHER,
                )
            elif target.type == topic_type.cluster:
                return RedirectResponse(
                    f"/f{collection_ob.path}/document/{target.id}",
                    status_code=status.HTTP_303_SEE_OTHER,
                )
            else:
                # Not sure what to do here
                return RedirectResponse(
                    f"/f{collection_ob.path}", status_code=status.HTTP_303_SEE_OTHER
                )


# TODO: make the subtypes explicit?
@app_router.get(
    "/analysis/{task_name}/list",
)
@app_router.get("/c/{collection}/analysis/{task_name}/list")
async def get_analyses_of_type(
    request: Request,
    task_name: str,
    current_user: user_with_permission_c_dep("admin"),
    collection: Optional[str] = None,
    target_id: Optional[int] = None,
    nickname: Optional[str] = None,
    offset: int = 0,
    limit: int = 0,
):
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        registry = TaskRegistry.get_registry()
        analyzer = registry.analyzer_by_name.get(task_name, None)
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
            .order_by(Analysis.id.desc())
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
        # models = [analysis.as_model(session) for analysis in analyses]
        # for model in models:
        #     task_class = TaskRegistry.get_registry().get_task_cls_by_name(model.analyzer_name)
        #     if not task_class.user_can_admin(current_user, collection_ob):
        #         model.task_template = None
    return templates.TemplateResponse(
        request,
        "list_analyses.html",
        dict(error="", success="", analyzer=analyzer, analyses=analyses, **base_vars),
    )
