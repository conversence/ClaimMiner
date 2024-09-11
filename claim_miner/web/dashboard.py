"""
Copyright Society Library and Conversence 2022-2024
"""

from typing import Optional, Type, Union, Literal, Annotated
from collections import defaultdict
from logging import getLogger

from sqlalchemy.sql import or_, and_
from sqlalchemy.sql.functions import count, coalesce
from fastapi import Request, Form

from .. import Session, select
from ..pyd_models import process_status, topic_type
from ..models import (
    Analysis,
    Analyzer,
    TaskTemplate,
    Topic,
    Fragment,
    TopicCollection,
    aliased,
    embed_db_model_by_name,
    with_polymorphic,
    poly_type_clause,
    model_by_topic_type,
)
from ..auth import user_with_permission_c_dep
from ..app import BadRequest
from . import get_base_template_vars, templates, app_router
from ..task_registry import TaskRegistry, CMTask
from ..bulk_tasks import (
    bulk_tasks,
    deletable_tasks,
    bulk_status,
    deletable_status,
    no_action_status,
    bulk_apply_tasks as do_bulk_apply_tasks,
)

logger = getLogger(__name__)


def emb_by_model(row):
    return {model: getattr(row, f"emb_{model}") for model in embed_db_model_by_name}


@app_router.get("/c/{collection}/dashboard")
@app_router.get("/dashboard")
async def dashboard(
    request: Request,
    current_user: user_with_permission_c_dep("admin"),
    collection: Optional[str] = None,
):
    registry = TaskRegistry.get_registry()
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob = base_vars["collection"]

        Target = Topic.__table__
        content_q = select(Target.c.type, count(Target.c.id)).group_by(Target.c.type)
        if collection_ob:
            content_q = content_q.outerjoin(Fragment, Fragment.id == Target.c.id).join(
                TopicCollection,
                and_(
                    TopicCollection.collection_id == collection_ob.id,
                    TopicCollection.topic_id == coalesce(Fragment.doc_id, Target.c.id),
                ),
            )
        content = dict((await session.execute(content_q)).fetchall())
        content[topic_type.collection] = len(base_vars["collection_names"])

        analyzer = aliased(Analyzer, flat=True)
        generated_q = (
            select(
                Analysis.status,
                analyzer.name,
                analyzer.version,
                TaskTemplate.nickname,
                Topic.type,
                count(Topic.id.distinct()).label("num_topics"),
                count(Analysis.id.distinct()).label("num_analysis"),
            )
            .join(analyzer, Analysis.analyzer)
            .outerjoin(TaskTemplate, Analysis.task_template)
            .outerjoin(Topic, Analysis.generated_topics)
            .group_by(
                Analysis.status,
                analyzer.name,
                analyzer.version,
                TaskTemplate.nickname,
                Topic.type,
            )
            .order_by(
                analyzer.name,
                analyzer.version,
                TaskTemplate.nickname,
                Analysis.status,
                Topic.type,
            )
        )
        if collection_ob:
            generated_q = generated_q.where(Analysis.collection_id == collection_ob.id)
        logger.debug("%s", generated_q)
        generated_topics = list(await session.execute(generated_q))
        logger.debug("%s", generated_topics)

        target_topics = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: (0, 0)))
        )
        versions = defaultdict(set)
        for task in registry.task_by_name.values():
            found = False
            for (
                version,
                status,
                template_name,
                num_topics,
                num_analysis,
            ) in await session.execute(task.count_status_query(collection)):
                status = process_status[status] if isinstance(status, str) else status
                target_topics[task.task_scale[0]][(task.name, version, template_name)][
                    status
                ] = (num_topics, num_analysis)
                versions[task.name].add(version)
                found = True
            if not found:
                if templates_ := registry.task_templates_by_name[task.name]:
                    for template in templates_:
                        target_topics[task.task_scale[0]][
                            (task.name, task.version, template.nickname)
                        ][process_status.complete] = (0, 0)
                else:
                    target_topics[task.task_scale[0]][(task.name, task.version, None)][
                        process_status.complete
                    ] = (0, 0)

        target_topics_control = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: (0, 0)))
        )
        if False:
            for task in registry.task_by_name.values():
                for template in registry.task_templates_by_name[task.name] or [None]:
                    template_name = template.nickname if template else None
                    for status in process_status:
                        (q_on_analysis, q) = task.query_with_status(
                            status, collection, template_name
                        )
                        target_topics_control[task.task_scale[0]][
                            (task.name, 1, template_name)
                        ][status] = (
                            1 if q_on_analysis else 0,
                            len(list(await session.execute(q))),
                        )

        show_versions = {taskname: len(v) > 1 for taskname, v in versions.items()}
        for tt, analyzers in target_topics.items():
            for (task_name, _, _), status_counts in analyzers.items():
                if process_status.not_requested not in status_counts:
                    topic_count = content.get(tt, 0)
                    task_cls = registry.get_task_cls_by_name(task_name)
                    if len(task_cls.task_scale) > 1:
                        target = with_polymorphic(
                            Topic, [model_by_topic_type[s] for s in task_cls.task_scale]
                        )
                        topic_count = await session.scalar(
                            select(count(target.id)).filter(poly_type_clause(target))
                        )
                    not_requested = topic_count - sum(
                        [t for (t, _) in status_counts.values()]
                    )
                    if not_requested:
                        status_counts[process_status.not_requested] = (not_requested, 0)
        return templates.TemplateResponse(
            request,
            "dashboard.html",
            dict(
                deletable_tasks=list(deletable_tasks),
                bulk_tasks=list(bulk_tasks),
                bulk_status=[s.name for s in bulk_status],
                deletable_status=[s.name for s in deletable_status],
                no_action_status=[s.name for s in no_action_status],
                versions=show_versions,
                content=content,
                generated_topics=generated_topics,
                target_topics=target_topics,
                target_topics_control=target_topics_control,
                **base_vars,
            ),
        )


@app_router.post("/c/{collection}/dashboard")
@app_router.post("/dashboard")
async def bulk_apply_tasks(
    request: Request,
    current_user: user_with_permission_c_dep("admin"),
    action: Annotated[Union[Literal["trigger"], Literal["delete"]], Form()] = "trigger",
    collection: Optional[str] = None,
):
    registry = TaskRegistry.get_registry()
    form = await request.form()
    logger.debug("%s", form)
    async with Session() as session:
        statuses = [s for s in process_status if f"status_{s.name}" in form]
        if not statuses:
            raise BadRequest("Select at least one status")
        tasks = []
        for k in form:
            if k.startswith("task__"):
                (_, task_name, version, template_name) = k.split("__", 3)
            else:
                continue
            task_cls: Type[CMTask] = registry.get_task_cls_by_name(task_name)
            if task_cls == CMTask:
                raise BadRequest("Invalid task name")
            if template_name:
                template = await registry.get_task_template(template_name, session)
                if not template:
                    raise BadRequest("Invalid task template name")
            template_name = template_name or None
            tasks.append((task_cls, version, template_name))
        if not tasks:
            raise BadRequest("Select at least one task")
        for task_cls, version, template_name in tasks:
            for status in statuses:
                await do_bulk_apply_tasks(
                    task_cls.name,
                    status,
                    collection,
                    template_name,
                    delete=action == "delete",
                    session=session,
                )
        if action == "delete":
            await session.commit()

        return await dashboard(request, current_user, collection)
