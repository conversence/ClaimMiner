from io import BytesIO

import pytest
from sqlalchemy import select
from fastapi.encoders import jsonable_encoder


pytestmark = [pytest.mark.anyio, pytest.mark.cost]


async def test_simple_prompt_form(
    admin_cookie_client, dispatcher, session, simple_claim, simple_prompt_template
):
    from claim_miner.pyd_models import fragment_type
    from claim_miner.models import Statement

    await dispatcher.apply_tasks()
    response = await admin_cookie_client.post(
        f"/f/analysis/{simple_prompt_template.analyzer_name}",
        data=dict(
            target_id=simple_claim.id,
            task_template_nickname=simple_prompt_template.nickname,
            autosave=True,
        ),
    )
    assert response.status_code in (200, 303)
    assert dispatcher.num_tasks() == 1
    await dispatcher.apply_tasks()
    claims = await session.scalars(
        select(Statement).filter_by(scale=fragment_type.standalone_claim)
    )
    claims = list(claims)
    assert len(claims) > 0
