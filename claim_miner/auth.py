"""
Copyright Society Library and Conversence 2022-2024
"""

from datetime import datetime, timedelta
from typing import Optional, Annotated, Union
from logging import getLogger

from pydantic import BaseModel
from fastapi import HTTPException, Depends, Cookie
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from jose import jwt, JWTError
from starlette import status
from starlette.middleware.authentication import AuthenticationMiddleware
from fastapi_oauth2.exceptions import OAuth2Error
from fastapi_oauth2.middleware import (
    Auth,
    User as OAuth2User,
    OAuth2Config,
    OAuth2Middleware,
    OAuth2Backend,
    Claims,
)
from fastapi_oauth2.router import router as oauth2_router
from fastapi_oauth2.client import OAuth2Client
from social_core.backends.google import GoogleOAuth2
from social_core.backends.github import GithubOAuth2
from pyisemail import is_email

from . import Session, select, db_config_get, config
from .models import (
    User,
    Collection,
    CollectionPermissions,
    Document,
    TopicCollection,
    Statement,
)
from .pyd_models import UserModel, UserModelWithPw, permission
from .app import app, Forbidden, Unauthorized, sentry_sdk

logger = getLogger(__name__)

ACCESS_TOKEN_EXPIRE_MINUTES = db_config_get("token_minutes", 180)

SECRET_KEY = db_config_get("auth_secret")
ALGORITHM = "HS256"

oauth2_specs = {
    "google-oauth2": dict(
        backend=GoogleOAuth2,
        scope=["openid", "profile", "email"],
        claims=Claims(
            identity=lambda user: f"{user.provider}:{user.sub}",
        ),
    ),
    "github": dict(
        backend=GithubOAuth2,
        scope=["user:email"],
        claims=Claims(
            # Map the `picture` claim to the `avatar_url` key in the user data.
            picture="avatar_url",
            # Calculate the `identity` claim based on the user data.
            identity=lambda user: f"{user.provider}:{user.id}",
        ),
    ),
}


async def on_auth_success(auth: Auth, user: OAuth2User):
    """This could be async function as well."""
    if not user.email_verified:
        return False
    async with Session() as session:
        existing = await session.scalar(
            select(User).filter_by(external_id=user.identity)
        )
        if not existing:
            existing = await session.scalar(select(User).filter_by(email=user.email))
        if existing:
            if not existing.confirmed:
                existing.confirmed = True
            if existing.external_id is None:
                existing.external_id = user.identity
            elif existing.external_id != user.identity:
                # Borderline case: Two identities yield same email. Use latest for now
                existing.external_id = user.identity
            if not existing.picture_url:
                existing.picture_url = user.picture
            if not existing.name:
                existing.name = user.name
        else:
            base_handle = user.email.split("@")[0]
            for i in range(100):
                handle = f"{base_handle}{i}" if i else base_handle
                existing = await session.scalar(select(User).filter_by(handle=handle))
                if not existing:
                    break
            else:
                raise Exception(f"Could not find a unique handle for {user.email}")

            user = User(
                external_id=user.identity,
                passwd="",
                email=user.email,
                picture_url=user.picture,
                confirmed=True,
                handle=handle,
                name=user.name,
            )
            session.add(user)
        await session.commit()
        user.id = existing.id


class OAuth2Backend2(OAuth2Backend):
    async def authenticate(self, request):
        # Spurious empty basic authenticaation
        if request.headers.get("authorization", "").startswith("Basic "):
            return Auth(), OAuth2User()
        return await super(OAuth2Backend2, self).authenticate(request)


class OAuth2Middleware2(OAuth2Middleware):
    def __init__(self, app, config, callback, **kwargs) -> None:
        # This is just a way to change the backend.
        if isinstance(config, dict):
            config = OAuth2Config(**config)
        elif not isinstance(config, OAuth2Config):
            raise TypeError("config is not a valid type")
        self.default_application_middleware = app
        self.auth_middleware = AuthenticationMiddleware(
            app, backend=OAuth2Backend2(config, callback), **kwargs
        )


def make_oauth2_config():
    clients_names = [
        c.strip() for c in db_config_get("oauth2_clients", "").split(",") if c
    ]
    if not clients_names:
        return None
    clients = []
    for client_name in clients_names:
        clients.append(
            OAuth2Client(
                client_id=config.get(f"oauth2_{client_name}", "client_id"),
                client_secret=config.get(f"oauth2_{client_name}", "client_secret"),
                # redirect_uri=f'http://localhost:8000/oauth2/{client_name}/callback',
                **oauth2_specs[client_name],
            )
        )
    config_ = OAuth2Config(
        allow_http=True,
        jwt_secret=SECRET_KEY,
        jwt_expires=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        jwt_algorithm=ALGORITHM,
        clients=clients,
    )
    app.include_router(oauth2_router, include_in_schema=False)
    app.add_middleware(
        OAuth2Middleware2,
        config=config_,
        callback=on_auth_success,
    )


make_oauth2_config()


async def doc_collection_constraints(
    current_user: UserModel,
    query,
    collection=None,
    perm: permission = permission.access,
    include_in_collection=True,
    include_outside_collection=None,
):
    if collection:
        include_in_collection = True
        collection_name = collection if isinstance(collection, str) else collection.name
    if include_outside_collection is None:
        include_outside_collection = not collection
    assert include_in_collection or include_outside_collection
    if not collection:
        if current_user.can(perm):
            return query
        if include_in_collection:
            if include_outside_collection:
                return (
                    query.outerjoin(
                        TopicCollection, Document.id == TopicCollection.topic_id
                    )
                    .outerjoin(
                        CollectionPermissions,
                        (
                            CollectionPermissions.collection_id
                            == TopicCollection.collection_id
                        )
                        & (CollectionPermissions.user_id == current_user.id),
                    )
                    .filter(
                        (TopicCollection.collection_id.is_(None))
                        | (
                            (CollectionPermissions.permissions.is_not(None))
                            & CollectionPermissions.permissions.any(perm)
                        )
                    )
                )
            else:
                return (
                    query.join(TopicCollection, Document.id == TopicCollection.topic_id)
                    .join(
                        CollectionPermissions,
                        (
                            CollectionPermissions.collection_id
                            == TopicCollection.collection_id
                        )
                        & (CollectionPermissions.user_id == current_user.id),
                    )
                    .filter(
                        (CollectionPermissions.permissions.is_not(None))
                        & CollectionPermissions.permissions.any(perm)
                    )
                )
        else:
            return query.outerjoin(
                TopicCollection, Document.id == TopicCollection.topic_id
            ).filter(TopicCollection.collection_id.is_(None))
    else:
        # todo: no access needed for main connection.
        generic_permissions = current_user.can(perm)
        coll_perm_cond = (
            CollectionPermissions.permissions.is_not(None)
        ) & CollectionPermissions.permissions.any(perm)
        if include_outside_collection:
            query = (
                query.outerjoin(
                    TopicCollection, Document.id == TopicCollection.topic_id
                )
                .outerjoin(Collection, Collection.id == TopicCollection.collection_id)
                .filter(
                    (Collection.name == collection_name) | (Collection.name.is_(None))
                )
            )
            if not generic_permissions:
                query = query.outerjoin(
                    CollectionPermissions,
                    (
                        CollectionPermissions.collection_id
                        == TopicCollection.collection_id
                    )
                    & (CollectionPermissions.user_id == current_user.id),
                ).filter((Collection.name.is_(None)) | coll_perm_cond)
        else:
            query = (
                query.join(TopicCollection, Document.id == TopicCollection.topic_id)
                .join(Collection, Collection.id == TopicCollection.collection_id)
                .filter(Collection.name == collection_name)
            )
            if not generic_permissions:
                query = query.join(
                    CollectionPermissions,
                    (
                        CollectionPermissions.collection_id
                        == TopicCollection.collection_id
                    )
                    & (CollectionPermissions.user_id == current_user.id),
                ).filter(coll_perm_cond)
    return query


async def check_doc_access(
    current_user: UserModel,
    doc_id,
    collection=None,
    perm: permission = permission.access,
):
    if current_user.can(perm):
        return True
    q = await doc_collection_constraints(
        current_user, select(Document).filter(Document.id == doc_id), collection, perm
    )
    async with Session() as session:
        r = await session.execute(q)
        r = r.first()
    if r is None:
        raise Forbidden()
    return True


async def fragment_collection_constraints(
    current_user: UserModel,
    query,
    collection=None,
    Target=Statement,
    perm: permission = permission.access,
    include_in_collection=True,
    include_outside_collection=None,
):
    if collection:
        include_in_collection = True
        collection_name = collection if isinstance(collection, str) else collection.name
    if include_outside_collection is None:
        include_outside_collection = not collection
    assert include_in_collection or include_outside_collection
    if not collection:
        if current_user.can(perm):
            return query
        if include_in_collection:
            if include_outside_collection:
                return (
                    query.outerjoin(
                        TopicCollection, Target.id == TopicCollection.topic_id
                    )
                    .outerjoin(
                        CollectionPermissions,
                        (
                            CollectionPermissions.collection_id
                            == TopicCollection.collection_id
                        )
                        & (CollectionPermissions.user_id == current_user.id),
                    )
                    .filter(
                        (TopicCollection.collection_id.is_(None))
                        | (
                            (CollectionPermissions.permissions.is_not(None))
                            & CollectionPermissions.permissions.any(perm)
                        )
                    )
                )
            else:
                return (
                    query.join(TopicCollection, Target.id == TopicCollection.topic_id)
                    .join(
                        CollectionPermissions,
                        (
                            CollectionPermissions.collection_id
                            == TopicCollection.collection_id
                        )
                        & (CollectionPermissions.user_id == current_user.id),
                    )
                    .filter(
                        (CollectionPermissions.permissions.is_not(None))
                        & CollectionPermissions.permissions.any(perm)
                    )
                )
        else:
            return query.outerjoin(
                TopicCollection, Target.id == TopicCollection.topic_id
            ).filter(TopicCollection.collection_id.is_(None))
    else:
        # todo: no access needed for main connection.
        generic_permissions = current_user.can(perm)
        coll_perm_cond = (
            CollectionPermissions.permissions.is_not(None)
        ) & CollectionPermissions.permissions.any(perm)
        if include_outside_collection:
            query = (
                query.outerjoin(TopicCollection, Target.id == TopicCollection.topic_id)
                .outerjoin(Collection, Collection.id == TopicCollection.collection_id)
                .filter(
                    (Collection.name == collection_name) | (Collection.name.is_(None))
                )
            )
            if not generic_permissions:
                query = query.outerjoin(
                    CollectionPermissions,
                    (
                        CollectionPermissions.collection_id
                        == TopicCollection.collection_id
                    )
                    & (CollectionPermissions.user_id == current_user.id),
                ).filter((Collection.name.is_(None)) | coll_perm_cond)
        else:
            query = (
                query.join(TopicCollection, Target.id == TopicCollection.topic_id)
                .join(Collection, Collection.id == TopicCollection.collection_id)
                .filter(Collection.name == collection_name)
            )
            if not generic_permissions:
                query = query.join(
                    CollectionPermissions,
                    (
                        CollectionPermissions.collection_id
                        == TopicCollection.collection_id
                    )
                    & (CollectionPermissions.user_id == current_user.id),
                ).filter(coll_perm_cond)
    return query


async def check_fragment_access(
    current_user: UserModel,
    fragment_id,
    collection=None,
    perm: permission = permission.access,
):
    if current_user.can(perm):
        return True
    q = await fragment_collection_constraints(
        current_user, select(Statement), collection, Statement, perm
    )
    q = q.filter(Statement.id == fragment_id)
    async with Session() as session:
        r = await session.execute(q)
        r = r.first()
    if r is None:
        raise Forbidden()
    return True


class Token(BaseModel):
    access_token: str
    token_type: str


pwd_context = CryptContext(schemes=["scram"], deprecated="auto")

# 2a allows compatibility with pg_crypto if I want to check in DB. Not used yet.
alt_pwd_context = CryptContext(
    schemes=["bcrypt"], deprecated="auto", bcrypt__ident="2a"
)


def verify_password(plain_password, hashed_password):
    if hashed_password.startswith("$scram$"):
        return pwd_context.verify(plain_password, hashed_password)
    elif hashed_password.startswith("$2a$"):
        return alt_pwd_context.verify(plain_password, hashed_password)
    return False


def get_password_hash(password):
    return pwd_context.hash(password)


def set_sentry_agent(agent: User):
    if sentry_sdk:
        from sentry_sdk.hub import Hub

        hub = Hub.current
        with hub.configure_scope() as sentry_scope:
            sentry_scope.user = dict(
                id=agent.id, username=agent.handle, email=agent.email
            )


async def get_agent_by_username(session, handle: str) -> Optional[UserModelWithPw]:
    agent: Optional[User] = await session.scalar(select(User).filter_by(handle=handle))
    if agent:
        set_sentry_agent(agent)
        return agent.as_model(session, UserModelWithPw)


async def get_agent_by_email(session, email: str) -> Optional[UserModelWithPw]:
    agent: Optional[User] = await session.scalar(select(User).filter_by(email=email))
    if agent:
        set_sentry_agent(agent)
        return agent.as_model(session, UserModelWithPw)


async def send_recovery_email(agent: UserModelWithPw, root: str):
    assert is_email(agent.email)
    token = create_access_token(agent.id)
    print(f"token link: {root}/change_password/{token}")

    # await send_email(
    #     subject='Claim Miner Account Recovery',
    #     to=agent.email,
    #     from_email='claimminer@conversence.com',
    #     template_name='recovery',
    #     context=dict(agent=agent, token= token)
    # )
    return token


async def get_agent_by_id(session, id: int) -> Optional[UserModelWithPw]:
    agent: Optional[User] = await session.get(User, id)
    if agent:
        set_sentry_agent(agent)
        return agent.as_model(session, UserModelWithPw)


async def get_agent_by_provider_id(session, identity: str) -> Optional[UserModelWithPw]:
    agent: Optional[User] = await session.scalar(
        select(User).filter_by(external_id=identity)
    )
    if agent:
        set_sentry_agent(agent)
        return agent.as_model(session, UserModelWithPw)


async def authenticate_agent(
    session, handle: str, password: str
) -> Optional[UserModel]:
    agent = await get_agent_by_username(session, handle)
    if agent and verify_password(password, agent.passwd):
        return agent


def create_access_token(
    id: int, expiration_minutes=ACCESS_TOKEN_EXPIRE_MINUTES, subtype="claimminer"
):
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = dict(sub=f"{subtype}:{id}", exp=expire)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_agent(authorization: str) -> Optional[UserModel]:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        if not authorization:
            return None
        parts = authorization.split()
        assert len(parts) < 3
        if len(parts) == 2:
            scheme, token = authorization.split()
            assert scheme.lower() == "bearer"
        else:
            token = parts[0]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if provider := payload.get("provider"):
            as_model = OAuth2User(**payload)
            as_model.use_claims(oauth2_specs[payload["provider"]]["claims"])
            identity = as_model.identity
        else:
            provider = "claimminer"
            subtype, identity = payload.get("sub").split(":", 1)
            identity = int(identity)
            assert subtype == provider
    except JWTError as e:
        logger.exception("", exc_info=e)
        raise credentials_exception from e
    async with Session() as session:
        if provider == "claimminer":
            agent = await get_agent_by_id(session, identity)
        else:
            agent = await get_agent_by_provider_id(session, identity)
    if agent is None:
        raise credentials_exception
    return agent


async def get_current_agent_cookie(
    Authorization: Annotated[str, Cookie()],
) -> Optional[UserModel]:
    return await get_current_agent(Authorization)


async def get_current_active_agent_cookie(
    current_agent: Annotated[UserModelWithPw, Depends(get_current_agent_cookie)],
) -> Optional[UserModel]:
    if not current_agent:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Please login"
        )
    if not current_agent.confirmed:
        raise Unauthorized("User not confirmed")
    return current_agent


user_c_dep = Annotated[UserModel, Depends(get_current_agent_cookie)]
active_user_c_dep = Annotated[UserModel, Depends(get_current_active_agent_cookie)]
optional_active_user_c_dep = Annotated[
    Optional[UserModel], Depends(get_current_active_agent_cookie)
]


def user_with_permission_c(
    perm: permission,
    current_user: Annotated[UserModel, Depends(get_current_active_agent_cookie)],
) -> Optional[UserModel]:
    if not current_user.can(perm):
        raise Forbidden()
    return current_user


def user_with_permission_c_curry(perm: permission):
    def user_with_permission_inner(
        current_user: Annotated[UserModel, Depends(get_current_active_agent_cookie)],
    ):
        return user_with_permission_c(perm, current_user)

    return user_with_permission_inner


async def user_with_collection_permission_c(
    perm: permission,
    current_user: Annotated[UserModel, Depends(get_current_active_agent_cookie)],
    collection: Optional[Union[str, Collection]] = None,
) -> Optional[UserModel]:
    if current_user.can(perm):
        return current_user
    if not collection:
        raise Forbidden()
    async with Session() as session:
        collection_name = collection if isinstance(collection, str) else collection.name
        q = (
            select(CollectionPermissions.permissions)
            .join(Collection)
            .filter(
                CollectionPermissions.user_id == current_user.id,
                Collection.name == collection_name,
            )
        )
        r = await session.execute(q)
        r = r.first()
        if r is None or perm not in r[0]:
            raise Forbidden()
    return current_user


def user_with_collection_permission_c_curry(perm: permission):
    async def user_with_permission_inner(
        current_user: Annotated[UserModel, Depends(get_current_active_agent_cookie)],
        collection: Optional[str] = None,
    ):
        return await user_with_collection_permission_c(perm, current_user, collection)

    return user_with_permission_inner


def user_with_permission_c_dep(perm: str) -> UserModel:
    return Annotated[UserModel, Depends(user_with_permission_c_curry(permission[perm]))]


def user_with_coll_permission_c_dep(perm: str) -> UserModel:
    return Annotated[
        UserModel, Depends(user_with_collection_permission_c_curry(permission[perm]))
    ]


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token", auto_error=False)


async def get_current_agent_token(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> Optional[UserModel]:
    return await get_current_agent(token)


async def get_current_active_agent_token(
    current_agent: Annotated[UserModelWithPw, Depends(get_current_agent_token)],
) -> UserModel:
    if not current_agent:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please login",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not current_agent.confirmed:
        raise Unauthorized("User not confirmed")
    return current_agent


maybe_user_t_dep = Annotated[Optional[UserModel], Depends(get_current_agent_token)]
user_t_dep = Annotated[UserModel, Depends(get_current_agent_token)]
active_user_t_dep = Annotated[UserModel, Depends(get_current_active_agent_token)]


def user_with_permission_t(
    perm: permission,
    current_user: Annotated[UserModel, Depends(get_current_active_agent_token)],
) -> Optional[UserModel]:
    if not current_user.can(perm):
        raise Forbidden()
    return current_user


def user_with_permission_t_curry(perm: permission):
    def user_with_permission_inner(
        current_user: Annotated[UserModel, Depends(get_current_active_agent_token)],
    ):
        return user_with_permission_t(perm, current_user)

    return user_with_permission_inner


async def user_with_collection_permission_t(
    perm: permission,
    current_user: Annotated[UserModel, Depends(get_current_active_agent_token)],
    collection: Optional[Union[str, Collection]] = None,
) -> Optional[UserModel]:
    if current_user.can(perm):
        return current_user
    if not collection:
        raise Forbidden()
    async with Session() as session:
        collection_name = collection if isinstance(collection, str) else collection.name
        q = (
            select(CollectionPermissions.permissions)
            .join(Collection)
            .filter(
                CollectionPermissions.user_id == current_user.id,
                Collection.name == collection_name,
            )
        )
        r = await session.execute(q)
        r = r.first()
        if r is None or perm not in r[0]:
            raise Forbidden()
    return current_user


def user_with_collection_permission_t_curry(perm: permission):
    async def user_with_permission_inner(
        current_user: Annotated[UserModel, Depends(get_current_active_agent_token)],
        collection: Optional[str] = None,
    ):
        return await user_with_collection_permission_t(perm, current_user, collection)

    return user_with_permission_inner


def user_with_permission_t_dep(perm: str):
    return Annotated[UserModel, Depends(user_with_permission_t_curry(permission[perm]))]


def user_with_coll_permission_t_dep(perm: str):
    return Annotated[
        UserModel, Depends(user_with_collection_permission_t_curry(permission[perm]))
    ]
