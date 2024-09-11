from io import BytesIO

import pytest
from sqlalchemy import select

pytestmark = pytest.mark.anyio


async def test_general_dashboard(
    admin_cookie_client,
    dispatcher,
    models,
    session,
    collection,
    simple_prompt_template,
    many_claims,
):
    response = await admin_cookie_client.get("/c/dashboard")


async def test_collection_dashboard(
    admin_cookie_client,
    dispatcher,
    models,
    session,
    collection,
    simple_prompt_template,
    many_claims,
):
    response = await admin_cookie_client.get("/c/{collection.name}/dashboard")


async def test_task_target_queries(
    admin_cookie_client,
    registry,
    dispatcher,
    models,
    session,
    collection,
    simple_prompt_template,
    many_claims,
):
    from claim_miner.pyd_models import process_status

    for collection_name in (None, collection.name):
        for t in registry.task_by_name.values():
            q = t.count_status_query(collection_name)
            assert q is not None
            r = await session.execute(q)
            templates = registry.task_templates_by_name.get(t.name, None)
            template_names = [tt.nickname for tt in templates] if templates else [None]
            for nickname in template_names:
                for status in process_status:
                    _, q = t.query_with_status(status, collection_name, nickname)
                    assert q is not None
                    r = await session.execute(q)
            else:
                _, q = t.query_with_status(status, collection_name)
                assert q is not None
                r = await session.execute(q)
