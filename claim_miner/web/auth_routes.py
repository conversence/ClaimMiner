"""
Copyright Society Library and Conversence 2022-2024
"""

from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional
from logging import getLogger

from fastapi import Form, Request, status, HTTPException
from fastapi.responses import Response, RedirectResponse
from pyisemail import is_email

from .. import Session, select, as_bool, production
from . import get_base_template_vars
from ..models import User, CollectionPermissions, Collection
from ..pyd_models import permission
from ..app import sentry_sdk
from ..auth import (
    verify_password,
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    user_with_coll_permission_c_dep,
    get_password_hash,
    user_c_dep,
    get_agent_by_email,
    send_recovery_email,
    get_current_agent,
    active_user_c_dep,
)
from . import templates, app_router, spa_router

logger = getLogger(__name__)


@app_router.get("/login")
async def login_get(request: Request):
    base_vars = await get_base_template_vars(
        request,
    )
    return templates.TemplateResponse(
        request, "login.html", dict(error="", **base_vars)
    )


@app_router.post("/login")
async def login_post(
    request: Request, username: Annotated[str, Form()], password: Annotated[str, Form()]
):
    async with Session() as session:
        base_vars = await get_base_template_vars(request, None, None, session)
        try:
            user = await session.scalar(select(User).filter_by(handle=username))
            if not user:
                error = "Invalid username or password"
            elif not user.confirmed:
                error = "Account not confirmed. Please ask an administrator."
            elif not verify_password(password, user.passwd):
                error = "Invalid username or password"
            else:
                # See other will mutate back to GET
                path = request.session.pop("after_login", "/")
                response = RedirectResponse(
                    f"/f{path}", status_code=status.HTTP_303_SEE_OTHER
                )
                expiry = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
                expires = datetime.now(timezone.utc) + expiry
                token = create_access_token(user.id, expiry)
                response.set_cookie(
                    key="Authorization",
                    value=f"Bearer {token}",
                    path="/",
                    samesite="strict",
                    expires=expires,
                    secure=production,
                )
                response.headers["HX-Location"] = str(
                    dict(path=f"/f{path}", target="#content")
                )
                return response
        except TypeError as e:
            error = str(e)
    return templates.TemplateResponse(
        request, "login.html", dict(error=error, **base_vars)
    )


@app_router.get("/register")
async def register_get(request: Request):
    base_vars = await get_base_template_vars(
        request,
    )
    return templates.TemplateResponse(
        request, "register.html", dict(error="", **base_vars)
    )


@app_router.post("/register")
async def register_post(
    request: Request,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    name: Annotated[str, Form()],
    email: Annotated[str, Form()],
):
    try:
        error = None
        async with Session() as session:
            base_vars = await get_base_template_vars(request, session=session)
            existing = await session.execute(
                select(User.id, User.passwd, User.confirmed).filter_by(handle=username)
            )
            existing = existing.first()
            if existing:
                error = f"Username {username} already exists"
            else:
                if not is_email(email):
                    error = "Invalid email address"
                else:
                    user = User(
                        handle=username,
                        passwd=get_password_hash(password),
                        email=email,
                        name=name,
                    )
                    session.add(user)
                    await session.commit()
                    return f"User {username} created. Please wait for approval by administrator."
    except Exception as e:
        error = f"Error creating user {username}: {e}"
    return templates.TemplateResponse(
        request, "register.html", dict(error=error, **base_vars)
    )


@app_router.get("/logout", response_class=RedirectResponse)
async def logout():
    response = RedirectResponse("/f/login")
    response.delete_cookie("Authorization")
    return response


@app_router.get("/admin")
@app_router.get("/c/{collection}/admin")
async def admin_get(
    request: Request,
    current_user: user_with_coll_permission_c_dep("admin"),
    collection: Optional[str] = None,
):
    error = ""
    users = []
    try:
        async with Session() as session:
            base_vars = await get_base_template_vars(request, None, collection, session)
            collection_ob: Collection = base_vars["collection"]
            users = await session.execute(select(User).order_by(User.email))
            users = [user for (user,) in users]
            if collection_ob:
                await session.refresh(collection_ob, ["permissions"])
                permissions_per_user = {
                    cp.user_id: [p.name for p in cp.permissions]
                    for cp in collection_ob.permissions
                }
            else:
                permissions_per_user = {}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("", exc_info=e)
        error = str(e)
    return templates.TemplateResponse(
        request,
        "admin.html",
        dict(
            users=users,
            error=error,
            permissions_per_user=permissions_per_user,
            permissions=permission._member_names_,
            **base_vars,
        ),
    )


@app_router.post("/admin")
@app_router.post("/c/{collection}/admin")
async def admin_post(
    request: Request,
    current_user: user_with_coll_permission_c_dep("admin"),
    collection: Optional[str] = None,
):
    error = ""
    users = []
    try:
        async with request.form() as form, Session() as session:
            base_vars = await get_base_template_vars(
                request, current_user, collection, session
            )
            collection_ob: Collection = base_vars["collection"]
            users = await session.execute(select(User).order_by(User.email))
            users = [user for (user,) in users]
            if collection_ob:
                await session.refresh(collection_ob, ["permissions"])
                collection_permissions_per_user = {
                    cp.user_id: cp for cp in collection_ob.permissions
                }
                permissions_per_user = {
                    uid: [p.name for p in cp.permissions]
                    for (uid, cp) in collection_permissions_per_user.items()
                }
            else:
                permissions_per_user = {}
            for user in users:
                confirmed = as_bool(form.get(f"{user.id}_confirmed"))
                if confirmed != user.confirmed:
                    user.confirmed = confirmed
                new_permissions = set(
                    filter(
                        lambda p: as_bool(form.get(f"{user.id}_{p.name}")), permission
                    )
                )
                if collection_ob:
                    new_permissions = {p for p in new_permissions if not user.can(p)}
                    r = collection_permissions_per_user.get(user.id)
                    if new_permissions or r:
                        r = r or CollectionPermissions(
                            user_id=user.id, collection_id=collection_ob.id
                        )
                        if set(r.permissions or ()) != new_permissions:
                            r.permissions = list(new_permissions)
                        session.add(r)
                else:
                    if set(user.permissions or ()) != new_permissions:
                        user.permissions = list(new_permissions)
                permissions_per_user[user.id] = [p.name for p in new_permissions]
            await session.commit()
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("", exc_info=e)
        error = str(e)
    return templates.TemplateResponse(
        request,
        "admin.html",
        dict(
            users=users,
            error=error,
            permissions_per_user=permissions_per_user,
            permissions=permission._member_names_,
            **base_vars,
        ),
    )


@app_router.get("/reset_password")
async def reset_password_get(request: Request):
    base_vars = await get_base_template_vars(request)
    return templates.TemplateResponse(
        request, "reset_password_request.html", dict(error="", **base_vars)
    )


@app_router.post("/reset_password")
async def reset_password_request(request: Request, email: Annotated[str, Form()]):
    base_vars = await get_base_template_vars(request)
    async with Session() as session:
        agent = await get_agent_by_email(session, email)
    if agent:
        protocol = "https" if production else "http"
        base_url = f"{protocol}://{request.headers['host']}"
        await send_recovery_email(agent, base_url)
    return templates.TemplateResponse(
        request,
        "reset_password_request.html",
        dict(
            error="IF you are a known user, a recovery link was sent to your email",
            **base_vars,
        ),
    )


@app_router.get("/change_password/{token}")
async def reset_password_step2_token(request: Request, token: str):
    base_vars = await get_base_template_vars(request)
    agent = await get_current_agent(token)
    if agent:
        response = templates.TemplateResponse(
            request, "change_password.html", dict(is_reset=True, **base_vars)
        )
        expiry = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        expires = datetime.now(timezone.utc) + expiry
        response.set_cookie(
            "Authorization",
            value=f"Bearer {token}",
            path="/",
            samesite="strict",
            expires=expires,
            secure=production,
        )
        return response
    else:
        return RedirectResponse(
            "/f/reset_password", status_code=status.HTTP_303_SEE_OTHER
        )


@app_router.get("/change_password")
async def reset_password_step2(request: Request, current_user: active_user_c_dep):
    base_vars = await get_base_template_vars(request)
    return templates.TemplateResponse(
        request, "change_password.html", dict(is_reset=False, **base_vars)
    )


@app_router.post("/change_password")
async def reset_password_step2_post(
    request: Request,
    current_user: active_user_c_dep,
    new_password: Annotated[str, Form()],
    confirm: Annotated[str, Form()],
    is_reset: Annotated[bool, Form()] = False,
    old_password: Annotated[Optional[str], Form()] = None,
):
    base_vars = await get_base_template_vars(request)
    error = None
    if new_password != confirm:
        error = "Passwords do not match"
    elif not is_reset:
        if not old_password:
            error = "Old password is required"
        elif not verify_password(old_password, current_user.passwd):
            error = "Old password is incorrect"

    if error:
        return templates.TemplateResponse(
            request,
            "change_password.html",
            dict(is_reset=is_reset, error=error, **base_vars),
        )
    async with Session() as session:
        user = await session.get(User, current_user.id)
        user.passwd = get_password_hash(new_password)
        await session.commit()
    return RedirectResponse("/f/", status_code=status.HTTP_303_SEE_OTHER)


@spa_router.get("/login")
@spa_router.get("/register")
@spa_router.post("/register")
@spa_router.get("/logout", response_class=RedirectResponse)
@spa_router.post("/change_password")
@spa_router.get("/reset_password")
@spa_router.post("/reset_password")
@spa_router.get("/change_password/{token}")
@spa_router.get("/change_password")
@spa_router.post("/login")
async def spa(request: Request):
    base_vars = await get_base_template_vars(request)
    base_vars["path"] = request.url.path
    return templates.TemplateResponse(request, "spa.html", base_vars)


if sentry_sdk:

    @app_router.get("/test_sentry")
    def test_sentry(current_user: user_c_dep):
        raise HTTPException(500, "Testing sentry")
