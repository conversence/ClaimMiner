from datetime import date, timedelta

import pytest
from sqlalchemy import select
from fastapi.encoders import jsonable_encoder

pytestmark = [pytest.mark.anyio, pytest.mark.cost]


async def test_simple_prompt_form(
    admin_cookie_client, dispatcher, session, simple_claim
):
    from claim_miner.pyd_models import fragment_type, embedding_model
    from claim_miner.models import Document, model_names
    from claim_miner.tasks.tasks import EmbedFragmentTask, EmbedFragmentAnalysisModel

    embed_task = EmbedFragmentAnalysisModel(
        target_id=simple_claim.id,
        task_template_nickname=model_names[
            embedding_model.universal_sentence_encoder_4
        ],
    )
    task = EmbedFragmentTask(embed_task)
    await task.schedule()
    await dispatcher.apply_tasks()  # claim embedding
    date_limit = date.today() - timedelta(days=2)
    response = await admin_cookie_client.post(
        f"/f/analysis/gdelt",
        data=dict(date=date_limit.isoformat(), limit=2, target_id=simple_claim.id),
    )
    assert response.status_code in (200, 303)
    assert dispatcher.num_tasks() == 1
    await dispatcher.apply_tasks()
    docs = await session.scalars(select(Document))
    assert len(list(docs))
