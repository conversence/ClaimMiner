import os

import pytest
from sqlalchemy import select
import numpy as np

from claim_miner.embed import embedder_registry, tf_embed

pytestmark = [pytest.mark.anyio]


async def embedding_tester(embedding: str):
    e1 = np.array(await tf_embed("LLM assertions are uncertain", embedding))
    e2 = np.array(await tf_embed("LLMs can make unverified statements", embedding))
    e3 = np.array(
        await tf_embed("Global warming should not be controversial", embedding)
    )
    d1 = e1.dot(e2)
    d2 = e1.dot(e3)
    assert d1 * 0.85 > d2


@pytest.mark.parametrize(
    "embedding",
    [
        name
        for name, embedder in embedder_registry.items()
        if not getattr(embedder, "pricing", None)
    ],
)
async def test_free_embedder(embedding):
    await embedding_tester(embedding)


@pytest.mark.cost
@pytest.mark.parametrize(
    "embedding",
    [
        name
        for name, embedder in embedder_registry.items()
        if getattr(embedder, "pricing", None)
    ],
)
async def test_paid_embedder(embedding):
    await embedding_tester(embedding)
