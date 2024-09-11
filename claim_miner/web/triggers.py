"""
Copyright Society Library and Conversence 2022-2024
"""

from typing import Optional, Literal, Union, Annotated
from itertools import groupby
from orjson import loads

# clustering
from sqlalchemy.future import select
from sqlalchemy.sql import or_
from sqlalchemy.orm import joinedload
from fastapi import Request, Form, status
from fastapi.responses import RedirectResponse
from pygraphviz import AGraph

from ..app import BadRequest
from ..auth import user_with_coll_permission_c_dep
from ..models import TaskTrigger, Analyzer, TaskTemplate
from ..pyd_models import topic_type
from .. import Session
from ..task_registry import TaskRegistry
from . import get_base_template_vars, templates, app_router


@app_router.get("/task_trigger")
@app_router.get("/c/{collection}/task_trigger")
async def list_task_triggers(
    request: Request,
    current_user: user_with_coll_permission_c_dep("admin"),
    collection: Optional[str] = None,
):
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob = base_vars.get("collection")
        r = await session.execute(
            select(TaskTrigger)
            .filter_by(collection_id=collection_ob.id if collection_ob else None)
            .options(
                joinedload(TaskTrigger.target_analyzer),
                joinedload(TaskTrigger.task_template),
                joinedload(TaskTrigger.analyzer_trigger),
            )
            .order_by(
                TaskTrigger.creation_trigger_id,
                TaskTrigger.analyzer_trigger_id,
                TaskTrigger.id,
            )
        )
        triggers = [t for (t,) in r.fetchall()]
    return templates.TemplateResponse(
        request, "list_task_triggers.html", dict(triggers=triggers, **base_vars)
    )


@app_router.get("/task_trigger/map")
@app_router.get("/c/{collection}/task_trigger/map")
async def map_task_triggers(
    request: Request,
    current_user: user_with_coll_permission_c_dep("admin"),
    collection: Optional[str] = None,
):
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob = base_vars.get("collection")
        cond = TaskTrigger.collection_id.is_(None)
        if collection_ob:
            cond = or_(cond, TaskTrigger.collection_id == collection_ob.id)
        r = await session.execute(
            select(TaskTrigger)
            .filter(cond)
            .order_by(TaskTrigger.collection_id.nulls_first(), TaskTrigger.id)
            .options(
                joinedload(TaskTrigger.target_analyzer),
                joinedload(TaskTrigger.analyzer_trigger),
                joinedload(TaskTrigger.collection),
            )
        )
        triggers = [t for (t,) in r.fetchall()]
        ttypes = {t.creation_trigger_id.name for t in triggers if t.creation_trigger_id}
        task_names = {
            t.analyzer_trigger.name for t in triggers if t.analyzer_trigger_id
        } | {t.target_analyzer.name for t in triggers}
        g = AGraph(directed=True)
        g.node_attr["shape"] = "rect"
        g.add_nodes_from(task_names | ttypes)
        for t in task_names:
            g.get_node(t).attr["color"] = "blue"
        for t in ttypes:
            g.get_node(t).attr["color"] = "green"
        for t in triggers:
            source = (
                t.analyzer_trigger.name
                if t.analyzer_trigger_id
                else t.creation_trigger_id.name
            )
            target = t.target_analyzer.name
            g.add_edge(source, target)
            e = g.get_edge(source, target)
            if l := e.attr.get("label"):
                e.attr["label"] = f"{l}, {t.id}"
            else:
                e.attr["label"] = t.id
            e.attr["labelURL"] = (
                f"/c/{t.collection.name}/task_trigger/{t.id}"
                if t.collection_id
                else f"/task_trigger/{t.id}"
            )
            if collection_ob and t.collection_id:
                if t.automatic:
                    line_color = "darkblue"
                else:
                    line_color = "darkslategray3"
            else:
                if t.automatic:
                    line_color = "black"
                else:
                    line_color = "grey80"
            e.attr["color"] = line_color
        registry = TaskRegistry.get_registry()
        target_tasks = {
            registry.get_task_cls_by_name(t.target_analyzer.name) for t in triggers
        }
        for t in target_tasks:
            for ttype in t.task_creates:
                if ttype.name in ttypes:
                    g.add_edge(t.name, ttype.name)
                    e = g.get_edge(t.name, ttype.name)
                    e.attr["style"] = "dotted"
        g.layout(prog="dot")
        return g.draw(format="svg")


async def show_edit_trigger(
    request: Request, session, base_vars, id: Optional[int] = None
):
    registry = TaskRegistry.get_registry()
    if id:
        trigger = await session.get(TaskTrigger, id)
    else:
        trigger = TaskTrigger()
    analyzers = {a.id: a.name for a in registry.analyzer_by_id.values()}
    task_templates = list(registry.task_template_by_id.values())
    task_templates.sort(key=lambda t: t.analyzer_id)
    task_templates_by_analyzer_id = {
        id: {tt.id: tt.nickname for tt in g}
        for (id, g) in groupby(task_templates, lambda t: t.analyzer_id)
    }
    return templates.TemplateResponse(
        request,
        "edit_task_trigger.html",
        dict(
            trigger=trigger,
            analyzers=analyzers,
            topic_type=topic_type,
            task_templates_by_analyzer=task_templates_by_analyzer_id,
            **base_vars,
        ),
    )


@app_router.get("/task_trigger/new")
@app_router.get("/c/{collection}/task_trigger/new")
async def pre_create_triggers(
    request: Request,
    current_user: user_with_coll_permission_c_dep("admin"),
    collection: Optional[str] = None,
):
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        return await show_edit_trigger(request, session, base_vars)


@app_router.get("/task_trigger/{id}")
@app_router.get("/c/{collection}/task_trigger/{id}")
async def get_triggers(
    request: Request,
    current_user: user_with_coll_permission_c_dep("admin"),
    id: int,
    collection: Optional[str] = None,
):
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        return await show_edit_trigger(request, session, base_vars, id)


@app_router.post("/task_trigger/new")
@app_router.post("/c/{collection}/task_trigger/new")
async def post_trigger(
    request: Request,
    current_user: user_with_coll_permission_c_dep("admin"),
    trigger_type: Annotated[Union[Literal["task_based", "topic_based"], str], Form()],
    conditions: Annotated[str, Form()],
    target_analyzer_id: Annotated[int, Form()],
    params: Annotated[str, Form()],
    automatic: Annotated[Optional[bool], Form()] = False,
    analyzer_trigger_id: Annotated[Optional[int], Form()] = None,
    creation_trigger_id: Annotated[Optional[topic_type], Form()] = None,
    task_template_id: Annotated[Optional[int], Form()] = None,
    collection: Optional[str] = None,
):
    registry = TaskRegistry.get_registry()
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob = base_vars.get("collection")
        trigger = TaskTrigger(
            collection=collection_ob,
            analyzer_trigger_id=analyzer_trigger_id,
            creation_trigger_id=creation_trigger_id,
            automatic=automatic,
            conditions=loads(conditions or "{}") or {},
            params=loads(params or "{}") or {},
            target_analyzer_id=target_analyzer_id,
            task_template_id=task_template_id,
            creator_id=current_user.id,
        )
        session.add(trigger)
        await session.commit()
        return RedirectResponse(
            f"/f{collection_ob.path}/task_trigger/{trigger.id}",
            status_code=status.HTTP_303_SEE_OTHER,
        )


@app_router.post("/task_trigger/{id}")
@app_router.post("/c/{collection}/task_trigger/{id}")
async def update_trigger(
    request: Request,
    id: int,
    current_user: user_with_coll_permission_c_dep("admin"),
    trigger_type: Annotated[Union[Literal["task_based", "topic_based"], str], Form()],
    conditions: Annotated[str, Form()],
    target_analyzer_id: Annotated[int, Form()],
    params: Annotated[str, Form()],
    automatic: Annotated[Optional[bool], Form()] = False,
    delete_trigger: Annotated[Optional[bool], Form()] = False,
    analyzer_trigger_id: Annotated[Optional[int], Form()] = None,
    creation_trigger_id: Annotated[Optional[topic_type], Form()] = None,
    task_template_id: Annotated[Optional[int], Form()] = None,
    collection: Optional[str] = None,
):
    registry = TaskRegistry.get_registry()
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        trigger = await session.get(TaskTrigger, id)
        if delete_trigger:
            await session.delete(trigger)
            await session.commit()
            return RedirectResponse(
                ".", status_code=status.HTTP_303_SEE_OTHER
            )  # TODO maybe HX-Refresh
        if trigger_type == "task_based":
            assert analyzer_trigger_id
            trigger.analyzer_trigger_id = analyzer_trigger_id
            trigger.creation_trigger_id = None
        else:
            assert creation_trigger_id
            trigger.creation_trigger_id = creation_trigger_id
            trigger.analyzer_trigger_id = None
        trigger.automatic = automatic
        trigger.conditions = loads(conditions or "{}")
        trigger.params = loads(params or "{}")
        trigger.target_analyzer_id = target_analyzer_id
        trigger.task_template_id = task_template_id
        await session.commit()
        return await show_edit_trigger(request, session, base_vars, id)
