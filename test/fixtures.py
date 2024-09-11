import os
from typing import Dict, Iterator
import logging

from httpx import AsyncClient
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

# from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, delete

from fastapi.security import OAuth2PasswordRequestForm
from fastapi.encoders import jsonable_encoder

import pytest


@pytest.fixture(scope="session")
def logger():
    return logging.getLogger("tests")


@pytest.fixture(scope="session")
async def sqla_engine(ini_file):
    os.environ["TARGET_DB"] = "test"
    from claim_miner import engine

    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture(scope="session")
def init_database(sqla_engine, ini_file):
    import sys

    sys.path.append("./scripts")
    from db_updater import (
        get_connection_data,
        db_state,
        read_structure,
        init_db,
        deploy,
        revert,
    )

    db = "test"
    conn_data = get_connection_data(ini_file, db)
    admin_conn_data = get_connection_data(ini_file, db, admin_password=True)
    init_db(conn_data)
    structures = read_structure()
    deploy(
        None,
        db_state(conn_data),
        structures,
        conn_data,
        admin_conn_data=admin_conn_data,
    )
    yield
    state = db_state(conn_data)
    revert(
        structures,
        db_state(conn_data),
        conn_data,
        admin_conn_data=admin_conn_data,
    )


@pytest.fixture(scope="session")
async def started_dispatcher():
    from claim_miner import dispatcher

    await dispatcher.start()
    return dispatcher


@pytest.fixture(scope="session")
async def registry(started_dispatcher, init_database):
    from claim_miner.task_registry import TaskRegistry

    registry = await TaskRegistry.get_full_registry()
    return registry


@pytest.fixture(scope="session")
async def models(registry, init_database):
    import claim_miner.models

    await claim_miner.models.finalize_db_models()
    return claim_miner.models


@pytest.fixture(scope="function")
def session_maker(models):
    from claim_miner import Session

    return Session


@pytest.fixture(scope="function")
async def session(session_maker):
    async with session_maker() as session:
        yield session


@pytest.fixture(scope="function")
async def dispatcher(started_dispatcher):
    started_dispatcher.clear()
    yield started_dispatcher
    started_dispatcher.clear()


@pytest.fixture(scope="function")
async def clean_tables(models, session_maker):
    async with session_maker() as session:
        await models.delete_data(session)
        await session.commit()
    yield True
    async with session_maker() as session:
        await models.delete_data(session)
        await session.commit()


@pytest.fixture(scope="session")
def app(models) -> FastAPI:
    from claim_miner.app import app

    # app.mount("/test_data", StaticFiles(directory="test/data"), name="test_data")
    return app


@pytest.fixture(scope="session")
async def client(app) -> Iterator[AsyncClient]:
    async with LifespanManager(app):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac


@pytest.fixture(scope="function")
async def admin_user(clean_tables, client):
    from claim_miner.pyd_models import UserModelWithPw
    from claim_miner.models import User, Session

    admin_user = UserModelWithPw(
        handle="admin",
        email="admin@example.com",
        passwd="admin",
        permissions=["admin"],
        confirmed=True,
    )
    response = await client.post(
        "/api/user", json=jsonable_encoder(admin_user.model_dump())
    )
    assert response.status_code == 201, response.json()
    yield admin_user
    async with Session() as session:
        await session.execute(delete(User).where(User.id == admin_user.id))
        await session.commit()


@pytest.fixture(scope="function")
async def admin_token(admin_user, client) -> str:
    form = OAuth2PasswordRequestForm(username="admin", password="admin")
    response = await client.post("/api/token", data=form.__dict__)
    assert response.status_code == 200, response.json()
    return response.json()["access_token"]


@pytest.fixture(scope="function")
def admin_headers(admin_token) -> Dict:
    return dict(Authorization=f"Bearer {admin_token}")


@pytest.fixture(scope="function")
async def admin_cookie_client(app, admin_headers) -> Iterator[AsyncClient]:
    async with LifespanManager(app):
        async with AsyncClient(
            app=app, base_url="http://test", cookies=admin_headers
        ) as ac:
            yield ac


@pytest.fixture(scope="function")
async def admin_headers_client(app, admin_headers) -> Iterator[AsyncClient]:
    async with LifespanManager(app):
        async with AsyncClient(
            app=app, base_url="http://test", headers=admin_headers
        ) as ac:
            yield ac


@pytest.fixture(scope="function")
async def collection(client, admin_headers, models, session_maker):
    from claim_miner.pyd_models import CollectionModel

    collection = CollectionModel(name="test")
    response = await client.post(
        "/api/c", json=jsonable_encoder(collection.model_dump()), headers=admin_headers
    )
    assert response.status_code == 201, response.json()
    collection = CollectionModel(**response.json())
    yield collection
    async with session_maker() as session:
        await session.execute(
            delete(models.Collection).where(models.Collection.id == collection.id)
        )
        await session.commit()


@pytest.fixture(scope="function")
async def base_question(client, admin_headers, models, session_maker, collection):
    from claim_miner.pyd_models import StatementModel, fragment_type

    statement = StatementModel(
        text="Why are concept maps useful?",
        scale=fragment_type.standalone_question,
        language="en",
    )
    response = await client.post(
        f"/api/c/{collection.name}/statement",
        json=jsonable_encoder(statement),
        headers=admin_headers,
    )
    assert response.status_code == 201, response.json()
    statement = StatementModel(**response.json())
    yield statement
    async with session_maker() as session:
        await session.execute(
            delete(models.Statement).where(models.Statement.id == statement.id)
        )
        await session.commit()


@pytest.fixture(scope="function")
async def simple_claim(client, admin_headers, models, session_maker, collection):
    from claim_miner.pyd_models import StatementModel, fragment_type

    statement = StatementModel(
        text="Maps allow to express more ideas",
        scale=fragment_type.standalone_claim,
        language="en",
    )
    response = await client.post(
        f"/api/c/{collection.name}/statement",
        json=jsonable_encoder(statement),
        headers=admin_headers,
    )
    assert response.status_code == 201, response.json()
    statement = StatementModel(**response.json())
    yield statement
    async with session_maker() as session:
        await session.execute(
            delete(models.Statement).where(models.Statement.id == statement.id)
        )
        await session.commit()


@pytest.fixture(scope="function")
async def collection_with_theme(
    client, admin_headers, models, session_maker, collection, base_question
):
    from claim_miner.pyd_models import PartialCollectionModel, CollectionModel

    collection_edit = PartialCollectionModel(
        params=collection.params | dict(theme_id=base_question.id)
    )
    response = await client.patch(
        f"/api/c/{collection.name}",
        json=collection_edit.model_dump(),
        headers=admin_headers,
    )
    assert response.status_code == 200, response.json()
    collection_ = CollectionModel(**response.json())
    # Make triggers automatic
    async with session_maker() as session:
        triggers = await session.scalars(
            select(models.TaskTrigger).filter_by(
                automatic=False, collection_id=collection.id
            )
        )
        for trigger in triggers:
            trigger.automatic = True
        await session.commit()
    return collection_


@pytest.fixture(scope="function")
async def simple_prompt_template(
    client, admin_headers, registry, models, session_maker
):
    from claim_miner.tasks.tasks import PromptTaskTemplateModel
    from claim_miner.pyd_models import link_type, fragment_type

    template_model = PromptTaskTemplateModel(
        analyzer_name="simple_prompt_analyzer",
        nickname="statement_from_question",
        prompt="The following is a claim. Write its negation.\r\n{theme}",
        parser="single_phrase",
        link_type=link_type.freeform,
        node_type=fragment_type.standalone_claim,
    )
    async with session_maker() as session:
        template = await models.TaskTemplate.from_model(session, template_model)
        session.add(template)
        await session.commit()
    template_model = template.as_model(session)
    registry.update_template(template_model)
    yield template_model
    async with session_maker() as session:
        await session.execute(
            delete(models.TaskTemplate).where(models.TaskTemplate.id == template.id)
        )
        await session.commit()
        # reset the registry's templates
        await registry.load_templates(session)


@pytest.fixture(scope="function")
async def many_claims(models, session_maker, registry, collection):
    from claim_miner.pyd_models import fragment_type

    with open("test/data/claims.txt") as f:
        claim_txt = f.readlines()
    async with session_maker() as session:
        collection_ob = await session.get(models.Collection, collection.id)
        claims = [
            models.Statement(
                text=s.strip(),
                scale=fragment_type.standalone_claim,
                language="en",
                collections=[collection_ob],
            )
            for s in claim_txt
        ]
        session.add_all(claims)
        await session.commit()
        await registry.handle_created_objects()
        claim_models = [s.as_model(session) for s in claims]
    yield claim_models
    async with session_maker() as session:
        await session.execute(
            delete(models.Statement).where(
                models.Statement.id.in_([s.id for s in claim_models])
            )
        )
        await session.commit()
