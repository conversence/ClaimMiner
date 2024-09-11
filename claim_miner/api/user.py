from typing import Optional, List

from sqlalchemy import select
from fastapi.responses import FileResponse, ORJSONResponse

from .. import Session, hashfs
from ..app import NotFound, Forbidden, BadRequest
from ..auth import (
    user_with_coll_permission_t_dep,
    user_t_dep,
    maybe_user_t_dep,
    get_password_hash,
)
from ..models import User
from ..uri import normalize
from ..pyd_models import UserModel, UserModelWithPw, UserModelOptional
from . import api_router


@api_router.get("/user/me")
async def get_my_user(current_user: user_t_dep) -> UserModel:
    async with Session() as session:
        user = await session.get(User, current_user.id)
        return user.as_model(session)


@api_router.get("/user/{user_id}")
async def get_user(
    current_user: user_with_coll_permission_t_dep("admin"), user_id: int
) -> UserModel:
    async with Session() as session:
        user = await session.get(User, user_id)
        if not user:
            raise NotFound()
        return user.as_model(session)


@api_router.post("/user", status_code=201)
async def add_user(
    new_user: UserModelWithPw, current_user: maybe_user_t_dep
) -> UserModel:
    async with Session() as session:
        if not (current_user and current_user.can("admin")):
            new_user.confirmed = False
            new_user.permissions = []
        new_user.passwd = get_password_hash(new_user.passwd)
        new_user_db = await User.from_model(session, new_user)
        session.add(new_user_db)
        await session.commit()
        return new_user_db.as_model(session)


@api_router.patch("/user/{user_id}")
@api_router.patch("/user/me")
async def modify_user(
    edited_user: UserModelOptional,
    current_user: user_t_dep,
    user_id: Optional[int] = None,
) -> UserModel:
    user_id = user_id or current_user.id
    async with Session() as session:
        if not (current_user.id == user_id or current_user.can("admin")):
            raise Forbidden()
        user = await session.get(User, user_id)
        if not user:
            raise NotFound()
        data = edited_user.model_dump()
        data.pop("id", None)
        if current_user.id == user_id:
            if "passwd" in data:
                data["passwd"] = get_password_hash(data["passwd"])
        else:
            data.pop("passwd", None)
        for k, v in data.items():
            if v is not None:
                setattr(user, k, v)
        await session.commit()
        return user.as_model(session)
