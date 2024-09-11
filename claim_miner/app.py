"""
Definition of the FastApi app.
"""

# Copyright Society Library and Conversence 2022-2024
from logging import getLogger
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, APIRouter, Request
from fastapi.responses import HTMLResponse, ORJSONResponse, Response, RedirectResponse
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from starlette import status
from starlette.middleware import Middleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.cors import CORSMiddleware

from . import dispatcher, config, production, is_testing, set_log_config
from .task_registry import TaskRegistry
from .utils import get_sentry_sdk

# Complete the models
TaskRegistry.get_registry()

set_log_config("web_logging")

logger = getLogger(__name__)

app_router = APIRouter(
    default_response_class=HTMLResponse,
    include_in_schema=False,
    prefix="/f",
)

api_router = APIRouter(
    default_response_class=ORJSONResponse,
    prefix="/api",
)

spa_router = APIRouter(default_response_class=HTMLResponse, include_in_schema=False)

# app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    await dispatcher.start()
    registry = await TaskRegistry.get_full_registry()
    for task_cls in registry.task_by_name.values():
        task_cls.setup_routes(app_router, api_router)
    import claim_miner.web.routes
    import claim_miner.api

    app.include_router(app_router)
    app.include_router(api_router)
    app.include_router(spa_router)
    yield

    # Clean up
    await dispatcher.stop()


# Permissive cors for now
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    ),
    Middleware(SessionMiddleware, secret_key=config.get("base", "secret_key")),
]

if production:
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    middleware.append(Middleware(ProxyHeadersMiddleware))
    # TODO: Consider TrustedHostMiddleware, GzipMiddleware, MessagePackMiddleware...

sentry_sdk = get_sentry_sdk("server")

app = FastAPI(
    name=config.get("base", "app_name"),
    debug=not production,
    lifespan=lifespan,
    middleware=middleware,
)

if not production and not is_testing:
    # Dev
    @app.middleware("http")
    async def debugger(request: Request, call_next):
        try:
            response = await call_next(request)
        except Exception as e:
            import pdb

            logger.exception("", exc_info=e)
            pdb.post_mortem()
            raise e
        return response


@app.middleware("http")
async def add_handle_created_objects(request: Request, call_next):
    response = await call_next(request)
    await TaskRegistry.get_registry().handle_created_objects()
    await dispatcher.flush()
    return response


class Forbidden(HTTPException):
    def __init__(
        self, detail: Any = None, headers: Dict[str, str] | None = None
    ) -> None:
        super().__init__(status.HTTP_403_FORBIDDEN, detail, headers)


class NotFound(HTTPException):
    def __init__(
        self, detail: Any = None, headers: Dict[str, str] | None = None
    ) -> None:
        super().__init__(status.HTTP_404_NOT_FOUND, detail, headers)


class BadRequest(HTTPException):
    def __init__(
        self, detail: Any = None, headers: Dict[str, str] | None = None
    ) -> None:
        super().__init__(status.HTTP_400_BAD_REQUEST, detail, headers)


class Unauthorized(HTTPException):
    def __init__(
        self, detail: Any = None, headers: Dict[str, str] | None = None
    ) -> None:
        super().__init__(status.HTTP_401_UNAUTHORIZED, detail, headers)


# These need to be defined early, before FastAPI has a chance to add them to the router


@app.exception_handler(Unauthorized)
async def redirect_to_login(request, exc):
    if request.url.path.startswith("/api/"):
        return Response("Unauthorized", status_code=401)
    if request.method == "GET" and not request.url.query:
        request.session["after_login"] = request.url.path
    return RedirectResponse("/login")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.exception("validation error %s", request, exc_info=exc)
    if any(er["loc"] == ("cookie", "Authorization") for er in exc._errors):
        return await redirect_to_login(request, exc)
    return await request_validation_exception_handler(request, exc)


# Import now to install the last middelware befere the app starts
import claim_miner.auth
