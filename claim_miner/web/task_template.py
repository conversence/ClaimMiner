"""
Copyright Society Library and Conversence 2022-2024
"""

from typing import Annotated, Optional
from itertools import groupby

from fastapi import Request, Form, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import aliased
from sqlalchemy.sql.functions import count

from .. import Session, select
from . import get_base_template_vars, templates, app_router
from ..models import Topic, Analyzer, TaskTemplate, Analysis, Collection
from ..app import BadRequest, NotFound, Unauthorized
from ..auth import user_with_permission_c_dep, user_c_dep
from ..task_registry import TaskRegistry
from ..utils import as_bool

prompt_analyzer_names = ("fragment_prompt_analyzer", "simple_prompt_analyzer")


@app_router.get("/c/{collection}/template")
@app_router.get("/template")
async def list_all_templates(
    request: Request,
    current_user: user_with_permission_c_dep("access"),
    collection: Optional[str] = None,
):
    registry = task_class = TaskRegistry.get_registry()
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob = base_vars["collection"]
        task_templates = list(
            await session.scalars(
                select(TaskTemplate)
                .filter_by(collection_id=collection_ob.id)
                .order_by(TaskTemplate.analyzer_id, TaskTemplate.nickname)
            )
        )
        templates_by_analyzer_id = {
            k: list(g) for k, g in groupby(task_templates, lambda x: x.analyzer_id)
        }
        for task in registry.task_by_name.values():
            analyzer = registry.analyzer_by_name[task.name]
            if analyzer.id not in templates_by_analyzer_id and task.task_template_model:
                templates_by_analyzer_id[analyzer.id] = []
        authorized = []
        analyzers = {}
        can_edit = {}
        for analyzer_id, task_templates in templates_by_analyzer_id.items():
            analyzer = registry.analyzer_by_id[analyzer_id]
            analyzers[analyzer_id] = analyzer
            task_class = registry.get_task_cls_by_name(analyzer.name)
            if collection_ob.user_can(current_user, task_class.trigger_task_permission):
                authorized.append(analyzer_id)
                can_edit[analyzer_id] = collection_ob.user_can(
                    current_user, task_class.admin_task_permission
                )
        templates_by_analyzer_id = {
            k: l for k, l in templates_by_analyzer_id.items() if k in authorized
        }
        if not templates_by_analyzer_id:
            raise Unauthorized()
    return templates.TemplateResponse(
        request,
        "list_templates.html",
        dict(
            all_analyzers=True,
            analyzers=analyzers,
            templates=templates_by_analyzer_id,
            can_edit=can_edit,
            **base_vars,
        ),
    )


@app_router.get("/c/{collection}/analyzer/{analyzer_name}/template")
@app_router.get("/analyzer/{analyzer_name}/template")
async def list_templates(
    request: Request,
    current_user: user_with_permission_c_dep("access"),
    analyzer_name: str,
    collection: Optional[str] = None,
):
    async with Session() as session:
        analyzer = TaskRegistry.get_registry().analyzer_by_name.get(analyzer_name)
        if not analyzer:
            raise NotFound()
        task_class = TaskRegistry.get_registry().get_task_cls_by_name(analyzer_name)
        if not task_class.task_template_model:
            raise BadRequest("Task {analyzer_name} does not use templates")
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob = base_vars["collection"]
        if not collection_ob.user_can(current_user, task_class.trigger_task_permission):
            raise Unauthorized()
        can_edit = collection_ob.user_can(
            current_user, task_class.admin_task_permission
        )
        task_templates = list(
            await session.scalars(
                select(TaskTemplate)
                .filter_by(analyzer_id=analyzer.id)
                .join(Collection, isouter=collection is None)
                .filter_by(name=collection)
                .order_by(TaskTemplate.nickname)
            )
        )

    return templates.TemplateResponse(
        request,
        "list_templates.html",
        dict(
            all_analyzers=False,
            analyzers={analyzer.id: analyzer},
            templates={analyzer.id: task_templates},
            can_edit={analyzer.id: can_edit},
            **base_vars,
        ),
    )


@app_router.get("/c/{collection}/analyzer/{analyzer_name}/template/{nickname}")
@app_router.get("/analyzer/{analyzer_name}/template/{nickname}")
async def show_edit_template(
    request: Request,
    current_user: user_with_permission_c_dep("access"),
    analyzer_name: str,
    nickname: str,
    force_edit: bool = False,
    collection: Optional[str] = None,
):
    async with Session() as session:
        analyzer = TaskRegistry.get_registry().analyzer_by_name.get(analyzer_name)
        if not analyzer:
            raise NotFound()
        task_class = TaskRegistry.get_registry().get_task_cls_by_name(analyzer_name)
        if not task_class.task_template_model:
            raise BadRequest("Task {analyzer_name} does not use templates")
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob = base_vars["collection"]
        if not collection_ob.user_can(current_user, task_class.trigger_task_permission):
            raise Unauthorized()
        q = (
            select(
                TaskTemplate, count(Analysis.id.distinct()), count(Topic.id.distinct())
            )
            .filter_by(nickname=nickname, analyzer_id=analyzer.id)
            .join(Collection, TaskTemplate.collection, isouter=collection is None)
            .outerjoin(Analysis, TaskTemplate.analyses)
            .outerjoin(Topic, Analysis.generated_topics)
            .group_by(TaskTemplate.id)
        )
        r = await session.execute(q)
        (template, analysis_count, fragment_count) = r.first()
        await session.refresh(template, ["analyzer"])
        can_edit = collection_ob.user_can(
            current_user, task_class.admin_task_permission
        ) and (
            template.draft
            or (force_edit and collection_ob.user_can(current_user, "admin"))
        )
        template_model = template.as_model(session)
        form_template = task_class.task_template_form or "edit_template.html"
        form_template_vars = await task_class.form_template_vars()
    return templates.TemplateResponse(
        request,
        form_template,
        dict(
            error="",
            template=template_model,
            analysis_count=analysis_count,
            fragment_count=fragment_count,
            analyzer=analyzer,
            can_edit=can_edit,
            **form_template_vars,
            **base_vars,
        ),
    )


@app_router.post(
    "/c/{collection}/analyzer/{analyzer_name}/template/{original_nickname}"
)
@app_router.post("/analyzer/{analyzer_name}/template/{original_nickname}")
async def do_edit_template(
    request: Request,
    current_user: user_with_permission_c_dep("access"),
    analyzer_name: str,
    original_nickname: str,
    force_edit: Annotated[bool, Form()] = False,
    collection: Optional[str] = None,
):
    errors = []
    async with Session() as session:
        analyzer = TaskRegistry.get_registry().analyzer_by_name.get(analyzer_name)
        if not analyzer:
            raise NotFound()
        task_class = TaskRegistry.get_registry().get_task_cls_by_name(analyzer_name)
        if not task_class.task_template_model:
            raise BadRequest("Task {analyzer_name} does not use templates")
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob = base_vars["collection"]
        if not collection_ob.user_can(current_user, task_class.admin_task_permission):
            raise Unauthorized()
        analyzer_alias = aliased(Analyzer)
        q = (
            select(
                TaskTemplate, count(Analysis.id.distinct()), count(Topic.id.distinct())
            )
            .filter_by(nickname=original_nickname, analyzer_id=analyzer.id)
            .join(Collection, TaskTemplate.collection, isouter=collection is None)
            .outerjoin(Analysis, TaskTemplate.analyses)
            .outerjoin(Topic, Analysis.generated_topics)
            .group_by(TaskTemplate.id)
        )
        r = await session.execute(q)
        (template, analysis_count, fragment_count) = r.first()
        can_edit = template.draft or (
            force_edit and collection_ob.user_can(current_user, "admin")
        )
        if not can_edit:
            raise Unauthorized()
        await session.refresh(template, ["analyzer"])
        template_model = template.as_model(session)
        form_data = dict(await request.form())
        for k in ("force_edit", "collection_name", "analyzer"):
            form_data.pop(k, None)
        form_data["draft"] = bool(form_data.get("draft", False))
        for k, f in template_model.__class__.model_fields.items():
            if f._attributes_set["annotation"] in (bool, Optional[bool]):
                form_data[k] = as_bool(form_data.get(k))
        modified_model = template_model.model_copy(update=form_data)
        if template.draft and not modified_model.draft:
            try:
                # TODO: fail on empty strings?
                full_model = task_class.task_template_model(
                    **modified_model.model_dump()
                )
                modified_model = full_model
            except Exception as e:
                modified_model.draft = True
                errors.append(str(e))
        modified_template = await TaskTemplate.from_model(session, modified_model)
        template = await session.merge(modified_template)
        form_template = task_class.task_template_form or "edit_template.html"
        form_template_vars = await task_class.form_template_vars()
        await session.commit()
        TaskRegistry.get_registry().update_template(modified_model, original_nickname)
        if original_nickname != template.nickname:
            return RedirectResponse(
                f"/f{collection.path}/analyzer/{analyzer_name}/template/{template.nickname}",
                status_code=status.HTTP_303_SEE_OTHER,
            )

    return templates.TemplateResponse(
        request,
        form_template,
        dict(
            error="".join(errors),
            template=modified_model,
            analysis_count=analysis_count,
            fragment_count=fragment_count,
            analyzer=analyzer,
            force_edit=force_edit,
            can_edit=can_edit,
            **form_template_vars,
            **base_vars,
        ),
    )


@app_router.post("/c/{collection}/analyzer/{analyzer_name}/template")
@app_router.post("/analyzer/{analyzer_name}/template")
async def add_template(
    request: Request,
    current_user: user_with_permission_c_dep("access"),
    nickname: Annotated[str, Form()],
    analyzer_name: str,
    collection: Optional[str] = None,
):
    errors = []
    async with Session() as session:
        analyzer = TaskRegistry.get_registry().analyzer_by_name.get(analyzer_name)
        if not analyzer:
            raise NotFound()
        task_class = TaskRegistry.get_registry().get_task_cls_by_name(analyzer_name)
        if not task_class.task_template_model:
            raise BadRequest("Task {analyzer_name} does not use templates")
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob = base_vars["collection"]
        if not collection_ob.user_can(current_user, task_class.admin_task_permission):
            raise Unauthorized()
        template = TaskTemplate(
            nickname=nickname,
            analyzer_id=analyzer.id,
            draft=True,
            collection_id=collection_ob.id,
        )
        session.add(template)
        await session.commit()
        return RedirectResponse(
            f"/f{collection_ob.path}/analyzer/{analyzer_name}/template/{nickname}",
            status_code=status.HTTP_303_SEE_OTHER,
        )
