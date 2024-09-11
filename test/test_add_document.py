from io import BytesIO

import pytest
from sqlalchemy import select
from sqlalchemy.orm import joinedload
from fastapi.encoders import jsonable_encoder

pytestmark = pytest.mark.anyio


async def test_add_html_url_form(admin_cookie_client, dispatcher, models, session):
    url = "https://www.conversence.com"
    response = await admin_cookie_client.post(
        "/f/document/upload",
        data=dict(upload_type="single", url=url),
        files=dict(file=""),
    )
    assert response.status_code == 200
    assert dispatcher.num_tasks() == 1
    assert len(dispatcher.channels["download"])
    docs = await session.scalars(select(models.Document))
    assert len(list(docs)) == 1


async def test_add_html_with_redirect(admin_cookie_client, dispatcher, models, session):
    url = "http://wiki.conversence.com"
    response = await admin_cookie_client.post(
        "/f/document/upload",
        data=dict(upload_type="single", url=url),
        files=dict(file=""),
    )
    assert response.status_code == 200
    assert dispatcher.num_tasks() == 1
    assert len(dispatcher.channels["download"])
    docs = await session.scalars(
        select(models.Document).options(joinedload(models.Document.uri))
    )
    docs = list(docs)
    assert len(docs) == 1
    doc = docs[0]
    assert doc.url == url
    await dispatcher.apply_tasks(until_empty=False)  # just the download
    await session.refresh(doc, ["uri"])
    assert doc.url != url


async def test_add_html_url_collection_form(
    admin_cookie_client, dispatcher, models, session, collection
):
    url = "https://www.conversence.com"
    response = await admin_cookie_client.post(
        f"/f/c/{collection.name}/document/upload",
        data=dict(upload_type="single", url=url),
        files=dict(file=""),
    )
    assert response.status_code == 200
    assert dispatcher.num_tasks() == 1
    assert len(dispatcher.channels["download"])
    doc = await session.scalar(select(models.Document))
    await session.refresh(doc, ["collections"])
    assert len(doc.collections) == 1


async def test_upload_html_form(admin_cookie_client, dispatcher, models, session):
    url = "https://example.com/test.html"
    content_txt = b"<html><body>This is a minimal html file.</body></html>"
    content = BytesIO(content_txt)
    content.seek(0)
    assert not len(dispatcher.channels["download"])
    response = await admin_cookie_client.post(
        "/f/document/upload",
        data=dict(upload_type="single", url=url),
        files=dict(file=("test.html", content)),
    )
    assert response.status_code == 200
    assert dispatcher.num_tasks() == 1
    assert len(dispatcher.channels["process_html"])
    docs = await session.scalars(select(models.Document))
    assert len(list(docs)) == 1


async def test_add_html_url_api(
    admin_headers_client, dispatcher, models, session, collection
):
    from claim_miner.pyd_models import DocumentModel

    document = DocumentModel(url="https://www.conversence.com")

    response = await admin_headers_client.post(
        f"/api/c/{collection.name}/document", json=jsonable_encoder(document)
    )
    assert response.status_code == 201
    assert dispatcher.num_tasks() == 1
    assert len(dispatcher.channels["download"])
    docs = await session.scalars(select(models.Document))
    assert len(list(docs)) == 1
