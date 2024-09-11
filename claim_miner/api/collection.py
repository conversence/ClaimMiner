from __future__ import annotations

from typing import List, Union, Optional

from sqlalchemy import select
from sqlalchemy.sql.functions import count
from sqlalchemy.orm import subqueryload
from fastapi.responses import ORJSONResponse

from .. import Session
from ..app import BadRequest
from ..auth import user_with_coll_permission_t_dep, active_user_t_dep
from ..models import Collection, CollectionPermissions, TopicCollection
from ..pyd_models import (
    CollectionModel,
    CollectionPermissionsModel,
    permission,
    CollectionExtendedModel,
    CollectionExtendedAdminModel,
    PartialCollectionModel,
)
from ..task_registry import TaskRegistry
from . import api_router, get_collection as get_collection_fn


@api_router.get("/c")
async def get_collections(
    current_user: active_user_t_dep,
) -> List[Union[CollectionExtendedModel, CollectionExtendedAdminModel]]:
    async with Session() as session:
        qfilter = None
        model = CollectionExtendedModel
        q1 = select(Collection, CollectionPermissions).join(
            CollectionPermissions,
            (CollectionPermissions.collection_id == Collection.id)
            & (CollectionPermissions.user_id == current_user.id),
            isouter=current_user.can("access"),
        )
        if not current_user.can("access"):
            qfilter = CollectionPermissions.permissions.any(permission.access)
            q1 = q1.filter(qfilter)
        if current_user.can("admin"):
            q1 = q1.options(subqueryload(Collection.permissions))
            model = CollectionExtendedAdminModel
        r1 = list(await session.execute(q1))
        collections_by_id = {c.id: c.as_model(session, model) for c, _ in r1}
        for c, p in r1:
            if p:
                collections_by_id[c.id].user_permissions = p.as_model(
                    session, CollectionPermissionsModel
                )
        q2 = select(
            TopicCollection.collection_id, count(TopicCollection.topic_id.distinct())
        ).group_by(TopicCollection.collection_id)
        if qfilter:
            q2 = q2.filter(
                TopicCollection.collection_id.in_(
                    select(CollectionPermissions.collection_id).filter(
                        qfilter & CollectionPermissions.user_id == current_user.id
                    )
                )
            )
        for id, num_docs in await session.execute(q2):
            collections_by_id[id].num_documents = num_docs
        q3 = select(
            TopicCollection.collection_id, count(TopicCollection.topic_id.distinct())
        ).group_by(TopicCollection.collection_id)
        if qfilter:
            q3 = q3.filter(
                TopicCollection.collection_id.in_(
                    select(CollectionPermissions.collection_id).filter(
                        qfilter & CollectionPermissions.user_id == current_user.id
                    )
                )
            )
        for id, num_statements in await session.execute(q3):
            collections_by_id[id].num_statements = num_statements
        return list(collections_by_id.values())


@api_router.get("/c/{collection}")
async def get_collection(
    current_user: user_with_coll_permission_t_dep("access"), collection: str
) -> Union[CollectionExtendedModel, CollectionExtendedAdminModel]:
    async with Session() as session:
        collection_db = await get_collection_fn(collection, session, current_user.id)
        model = CollectionExtendedModel
        if current_user.can("admin"):
            await session.refresh(collection_db, ["permissions"])
            model = CollectionExtendedAdminModel
        collection_m = collection_db.as_model(session, model)
        collection_m.num_documents = await session.scalar(
            select(count(TopicCollection.topic_id)).filter_by(
                collection_id=collection_db.id
            )
        )
        collection_m.num_statements = await session.scalar(
            select(count(TopicCollection.topic_id)).filter_by(
                collection_id=collection_db.id
            )
        )
        return collection_m


@api_router.post("/c", status_code=201)
async def add_collection(
    current_user: user_with_coll_permission_t_dep("admin"), collection: CollectionModel
) -> CollectionModel:
    async with Session() as session:
        collection_db = await Collection.from_model(session, collection)
        session.add(collection_db)
        await session.commit()
        await TaskRegistry.get_registry().ensure_default_triggers(
            session, collection_db.id
        )
        collection = collection_db.as_model(session)
        location = f"/api/c/{collection.name}"
        return ORJSONResponse(
            collection.model_dump(mode="json"),
            status_code=201,
            headers=dict(location=location),
        )


@api_router.patch("/c/{collection}")
async def modify_collection(
    edited_collection: PartialCollectionModel,
    current_user: user_with_coll_permission_t_dep("admin"),
    collection: str,
) -> CollectionModel:
    async with Session() as session:
        collection_db = await get_collection_fn(collection, session, current_user.id)
        data = edited_collection.model_dump()
        if id_ := data.pop("id", None):
            if id_ != collection_db.id:
                raise BadRequest("Id mismatch")
        for k, v in data.items():
            if v is not None:
                setattr(collection_db, k, v)
        await session.commit()
        await TaskRegistry.get_registry().ensure_default_triggers(
            session, collection_db.id
        )
        return collection_db.as_model(session)
