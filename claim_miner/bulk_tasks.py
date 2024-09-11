"""
Copyright Society Library and Conversence 2022-2024
"""

from typing import Optional, Type, Union, Literal, Annotated, List
from collections import defaultdict

from sqlalchemy.sql import or_, and_
from sqlalchemy.sql.functions import count
from fastapi import Request, Form

from . import Session, select
from .pyd_models import process_status, topic_type
from .models import (
    Analysis,
    Analyzer,
    TaskTemplate,
    Topic,
    Fragment,
    TopicCollection,
    embed_db_model_by_name,
    with_polymorphic,
    poly_type_clause,
    model_by_topic_type,
)
from .auth import user_with_permission_c_dep
from .app import BadRequest
from .task_registry import TaskRegistry, CMTask


def emb_by_model(row):
    return {model: getattr(row, f"emb_{model}") for model in embed_db_model_by_name}


registry = TaskRegistry.get_registry()
bulk_tasks = {
    name for (name, task) in registry.task_by_name.items() if task.can_apply_bulk
}
deletable_tasks = {
    name for (name, task) in registry.task_by_name.items() if task.can_delete_results
}
no_action_status = set(
    (process_status.inapplicable, process_status.ongoing, process_status.not_ready)
)
bulk_status = set(
    (
        process_status.error,
        process_status.not_requested,
        process_status.pending,
        process_status.complete,
    )
)
deletable_status = {
    s
    for s in process_status
    if s >= process_status.pending and s not in no_action_status
}


async def bulk_apply_tasks(
    task_name: str,
    status: process_status,
    collection_name: str = None,
    template_name: Optional[str] = None,
    target_ids: Optional[List[int]] = None,
    analysis_ids: Optional[List[int]] = None,
    delete=False,
    session=None,
    **kwargs,
):
    if session is None:
        async with Session() as session:
            await bulk_apply_tasks(
                task_name,
                status,
                collection_name,
                template_name,
                target_ids,
                delete,
                session=session,
                **kwargs,
            )
            if delete:
                await session.commit()
            return

    task_cls = registry.get_task_cls_by_name(task_name)

    if delete:
        assert (
            task_name in deletable_tasks
        ), f"Cannot delete results of task {task_name}"
        if not target_ids:
            assert status in deletable_status, f"Can't delete status {status}"
    else:
        assert (
            target_ids or analysis_ids or task_name in bulk_tasks
        ), f"Cannot trigger task {task_name} in bulk"
        if not target_ids:
            assert status not in no_action_status and not (
                status == process_status.complete and not task_cls.can_reapply_complete
            ), f"Can't trigger on status {status}"

    if target_ids:
        query_on_analysis, q = False, None
    elif analysis_ids:
        query_on_analysis, q = True, None
    else:
        query_on_analysis, q = task_cls.query_with_status(
            status, collection_name, template_name
        )

    if query_on_analysis:
        analyses = await session.scalars(
            select(Analysis).filter(
                Analysis.id.in_(analysis_ids if analysis_ids else q.subquery())
            )
        )
        for analysis in analyses:
            task = task_cls(analysis=analysis.as_model(session))
            # print(task.analysis)
            if delete:
                await task.delete_results(session)
                await session.commit()  # commit each delete
            else:
                await task.schedule()
    else:
        if target_ids is None:
            target_ids = await session.scalars(q)
        for target_id in target_ids:
            task = task_cls(
                target_id=target_id,
                task_template_nickname=template_name,
                collection_name=collection_name,
                **kwargs,
            )
            if q is None:
                task_status = await task_cls.status_for(
                    session,
                    target_id,
                    collection_name=collection_name,
                    template_name=template_name,
                )
                if delete:
                    if task_status not in deletable_status:
                        continue
                else:
                    if status in no_action_status or (
                        status == process_status.complete
                        and not task_cls.can_reapply_complete
                    ):
                        continue
            if delete:
                await task.delete_results(session)
                await session.commit()  # commit each delete
            else:
                await task.schedule()
