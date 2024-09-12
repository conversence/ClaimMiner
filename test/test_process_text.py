from io import BytesIO

import pytest
from sqlalchemy import select
from fastapi.encoders import jsonable_encoder


pytestmark = pytest.mark.anyio


@pytest.mark.parametrize("extension", ["txt", "html", "pdf"])
async def test_process_text_after_upload(
    admin_cookie_client, dispatcher, models, session, extension
):
    url = f"https://example.com/simple_test.{extension}"
    content = open(f"test/data/simple_test.{extension}", "rb")
    response = await admin_cookie_client.post(
        "/f/document/upload",
        data=dict(upload_type="single", url=url),
        files=dict(file=(f"simple_test.{extension}", content)),
    )
    assert response.status_code == 200
    assert dispatcher.num_tasks() == 1
    assert len(dispatcher.channels.keys()) == 1
    assert list(dispatcher.channels.keys())[0].startswith("process_")
    doc = await session.scalar(select(models.Document).limit(1))
    assert doc
    await dispatcher.apply_tasks()
    await session.refresh(doc, ["paragraphs"])
    assert len(doc.paragraphs) > 0
    assert len(dispatcher.channels["embed_doc"]) == 1
    assert len(dispatcher.channels["embed_fragment"]) > 0


async def test_process_text_after_add_url(
    admin_cookie_client, dispatcher, models, session
):
    url = "https://www.societylibrary.org/mission-vision"
    response = await admin_cookie_client.post(
        "/f/document/upload",
        data=dict(upload_type="single", url=url),
        files=dict(file=""),
    )
    assert response.status_code == 200
    assert dispatcher.num_tasks() == 1
    assert len(dispatcher.channels["download"])
    docs = await session.scalars(select(models.Document))
    docs = list(docs)
    assert len(docs) == 1
    await dispatcher.apply_tasks()
    assert dispatcher.num_tasks() == 1
    assert "process_html" in dispatcher.channels
    await dispatcher.apply_tasks()
    doc = docs[0]
    await session.refresh(doc, ["paragraphs"])
    assert len(doc.paragraphs) > 0
