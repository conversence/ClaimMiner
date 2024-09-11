"""
In this package, the web application's routes are implemented.
"""
# Copyright Society Library and Conversence 2022-2024

from typing import Annotated

from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends

from ..app import BadRequest, api_router
from ..auth import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    verify_password,
    create_access_token,
    get_agent_by_username,
)
from ..models import CollectionScope
from .. import Session

get_collection = CollectionScope.get_collection


@api_router.post("/token")
async def get_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
):
    async with Session() as session:
        user = await get_agent_by_username(session, form_data.username)
    if not user:
        raise BadRequest("Invalid username or password")
    elif not user.confirmed:
        raise BadRequest("Account not confirmed. Please ask an administrator.")
    elif not verify_password(form_data.password, user.passwd):
        raise BadRequest("Invalid username or password")
    access_token = create_access_token(user.id, minutes)
    return {"access_token": access_token, "token_type": "bearer"}


# from . import auth_routes
from . import collection
from . import docs

# from . import scatterplot
# from . import claim_clusters
from . import search
from . import statement
from . import analysis
from . import user
# from . import task_template
# from . import dashboard
# from . import prompts
