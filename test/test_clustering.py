from io import BytesIO
from logging import getLogger

import pytest
from sqlalchemy import select
from sqlalchemy.orm import subqueryload

logger = getLogger(__name__)

pytestmark = pytest.mark.anyio


async def test_clustering(
    admin_cookie_client, dispatcher, session, collection, many_claims
):
    from claim_miner.models import ClusterData

    logger.debug("%d %s", 0, dispatcher.channels)
    await dispatcher.apply_tasks(until_empty=True)
    logger.debug("%d %s", 1, dispatcher.channels)
    response = await admin_cookie_client.post(
        f"/f/c/{collection.name}/analysis/cluster", data=dict(min_samples=3, eps=0.75)
    )
    assert response.status_code in (200, 303)
    assert dispatcher.num_tasks()
    logger.debug("%d %s", 2, dispatcher.channels)
    await dispatcher.apply_tasks()
    logger.debug("%d %s", 3, dispatcher.channels)
    clusters = list(await session.scalars(select(ClusterData)))
    # print([c.cluster_size for c in clusters])
    assert len(clusters) == 2
