"""
Copyright Society Library and Conversence 2022-2024
"""

from typing import Annotated, Optional
from collections import defaultdict
import re
from logging import getLogger

from fastapi import Form, Request, status
from fastapi.responses import RedirectResponse, HTMLResponse
from sqlalchemy.sql.functions import count

from .. import Session, select, as_bool
from ..pyd_models import permission, embedding_model, topic_type, BASE_EMBED_MODEL
from ..models import (
    TopicCollection,
    Statement,
    Collection,
    CollectionPermissions,
    Topic,
)
from ..app import BadRequest
from ..auth import active_user_c_dep, user_with_coll_permission_c_dep
from . import get_base_template_vars, templates, app_router
from ..task_registry import TaskRegistry

logger = getLogger(__name__)


@app_router.get("/")
async def list_collections(request: Request, current_user: active_user_c_dep):
    async with Session() as session:
        base_vars = await get_base_template_vars(request, current_user, None, session)
        filter = None
        q1 = select(Collection, CollectionPermissions).join(
            CollectionPermissions,
            (CollectionPermissions.collection_id == Collection.id)
            & (CollectionPermissions.user_id == current_user.id),
            isouter=current_user.can("access"),
        )
        if not current_user.can("access"):
            filter = CollectionPermissions.permissions.any(permission.access)
            q1 = q1.filter(filter)
        r1 = list(await session.execute(q1))
        q2 = (
            select(
                TopicCollection.collection_id,
                Topic.type,
                count(TopicCollection.topic_id.distinct()),
            )
            .join(Topic, TopicCollection.topic)
            .group_by(TopicCollection.collection_id, Topic.type)
        )
        if filter:
            q2 = q2.filter(
                TopicCollection.collection_id.in_(
                    select(CollectionPermissions.collection_id).filter(
                        filter & CollectionPermissions.user_id == current_user.id
                    )
                )
            )
        counts = defaultdict(dict)
        r2 = await session.execute(q2)
        for col_id, typ, cnt in r2:
            counts[col_id][typ] = cnt
        data = [
            (
                coll,
                counts[coll.id].get(topic_type.document, 0),
                counts[coll.id].get(topic_type.standalone, 0),
            )
            for coll, perms in r1
        ]
        for coll, perm in r1:
            coll.user_permissions = perm
        logger.debug("%s", data)
    return templates.TemplateResponse(
        request, "list_collections.html", dict(data=data, **base_vars)
    )


@app_router.get("/c/{collection}")
@app_router.post("/c/{collection}")  # Why post?
async def show_collection(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    collection: str,
):
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        num_docs = await session.scalar(
            select(count(TopicCollection.topic_id))
            .join(Topic, TopicCollection.topic)
            .filter(
                TopicCollection.collection_id == collection_ob.id,
                Topic.type == topic_type.document,
            )
        )
        num_frags = await session.scalar(
            select(count(TopicCollection.topic_id))
            .join(Topic, TopicCollection.topic)
            .filter(
                TopicCollection.collection_id == collection_ob.id,
                Topic.type == topic_type.standalone,
            )
        )
        theme = None
        if theme_id := collection_ob.params.get("theme_id"):
            theme = await session.get(Statement, int(theme_id))
        tasks = []
        async for task_data in TaskRegistry.task_registry.all_task_status(
            session, collection_ob, collection
        ):
            tasks.append(task_data)
            base_vars |= await task_data[0].task_form_args(session)

    return templates.TemplateResponse(
        request,
        "view_collection.html",
        dict(
            num_docs=num_docs,
            num_frags=num_frags,
            tasks=tasks,
            theme=theme,
            **base_vars,
        ),
    )


@app_router.get("/c/{collection}/edit")
async def edit_collection_get(
    request: Request,
    current_user: user_with_coll_permission_c_dep("admin"),
    collection: str,
):
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        num_docs = await session.scalar(
            select(count(TopicCollection.topic_id))
            .join(Topic, TopicCollection.topic)
            .filter(
                TopicCollection.collection_id == collection_ob.id,
                Topic.type == topic_type.document,
            )
        )
        num_frags = await session.scalar(
            select(count(TopicCollection.topic_id))
            .join(Topic, TopicCollection.topic)
            .filter(
                TopicCollection.collection_id == collection_ob.id,
                Topic.type == topic_type.standalone,
            )
        )
        theme = None
        if theme_id := collection_ob.params.get("theme_id"):
            theme = await session.get(Statement, int(theme_id))

    return templates.TemplateResponse(
        request,
        "edit_collection.html",
        dict(num_docs=num_docs, num_frags=num_frags, theme=theme, **base_vars),
    )


@app_router.post("/c/{collection}/edit")
async def edit_collection(
    request: Request,
    collection: str,
    current_user: user_with_coll_permission_c_dep("admin"),
    theme_id: Annotated[Optional[int], Form()] = None,
):
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        theme = None
        recalc_triggers = False
        collection_ob: Collection = base_vars["collection"]
        num_docs = await session.scalar(
            select(count(TopicCollection.topic_id)).filter(
                TopicCollection.collection_id == collection_ob.id
            )
        )
        num_frags = await session.scalar(
            select(count(TopicCollection.topic_id)).filter(
                TopicCollection.collection_id == collection_ob.id
            )
        )
        original_theme_id = collection_ob.params.get("theme_id")
        if theme_id:
            collection_ob.params["theme_id"] = theme_id
            theme = await session.get(Statement, int(theme_id))
        elif "theme_id" in collection_ob.params:
            del collection_ob.params["theme_id"]
        if (theme_id is not None) != (original_theme_id is not None):
            recalc_triggers = True

        params = dict(collection_ob.params)
        embeddings = []
        for model in embedding_model:
            if model == BASE_EMBED_MODEL:
                continue
            async with request.form() as form:
                if as_bool(form.get(model.name)):
                    embeddings.append(model.name)
        old_embeddings = set(params.get("embeddings", ()))
        if old_embeddings != set(embeddings):
            params["embeddings"] = embeddings
            recalc_triggers = True
        # TODO: Set ordering
        collection_ob.params = params
        # TODO: re-generate triggers on new collection
        # TODO: Import roots
        await session.commit()
        if recalc_triggers:
            await TaskRegistry.get_registry().ensure_default_triggers(
                session, collection_ob.id
            )
    return templates.TemplateResponse(
        request,
        "edit_collection.html",
        dict(theme=theme, num_docs=num_docs, num_frags=num_frags, **base_vars),
    )


@app_router.post("/")
async def add_collection(
    request: Request,
    name: Annotated[str, Form()],
    current_user: user_with_coll_permission_c_dep("admin"),
):
    if not re.match(r"^\w+$", name):
        raise BadRequest("Spaces not allowed in collection names")
    async with Session() as session:
        base_vars = await get_base_template_vars(request, current_user, None, session)
        collection_ob = Collection(name=name)
        session.add(collection_ob)
        await session.commit()
        await TaskRegistry.get_registry().ensure_default_triggers(
            session, collection_ob.id
        )
    return RedirectResponse(
        f"/f{collection_ob.path}/edit", status_code=status.HTTP_303_SEE_OTHER
    )
