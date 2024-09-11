import pytest

pytestmark = pytest.mark.anyio


async def test_home_page(admin_cookie_client):
    response = await admin_cookie_client.get("/")
    assert response.status_code == 200


async def test_add_collection_form(admin_cookie_client, dispatcher, models, session):
    name = "test_collection"
    response = await admin_cookie_client.post("/f/", data=dict(name=name))
    assert response.status_code == 303
    assert response.headers["location"] == f"/f/c/{name}/edit"
    response = await admin_cookie_client.get(response.headers["location"])
    assert response.is_success


async def test_add_collection_with_spaces_form(
    admin_cookie_client, dispatcher, models, session
):
    name = "test collection"
    response = await admin_cookie_client.post("/f/", data=dict(name=name))
    assert response.status_code == 400


async def test_display_claim(
    admin_cookie_client, dispatcher, collection_with_theme, base_question
):
    await dispatcher.apply_tasks(until_empty=True)  # embed_fragment on the theme
    response = await admin_cookie_client.get(
        f"/f/c/{collection_with_theme.name}/claim/{base_question.id}"
    )
    assert response.is_success


async def test_display_claim_search(
    admin_cookie_client, dispatcher, collection_with_theme, base_question, many_claims
):
    await dispatcher.apply_tasks(until_empty=True)  # embed_fragment on the theme
    response = await admin_cookie_client.get(
        f"/f/c/{collection_with_theme.name}/claim/{base_question.id}/search"
    )
    assert response.is_success
