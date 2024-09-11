"""
Copyright Society Library and Conversence 2022-2024
"""

from typing import Annotated, Optional
from logging import getLogger

from sqlalchemy.future import select
from fastapi import Form, Request

from .. import Session
from ..models import TaskTemplate, Analyzer, Collection
from ..pyd_models import embedding_model, search_mode
from ..auth import user_with_coll_permission_c_dep
from ..search import search
from . import update_fragment_selection, get_base_template_vars, templates, app_router

logger = getLogger(__name__)


@app_router.get("/search")
@app_router.get("/claim/propose")
@app_router.get("/c/{collection}/search")
@app_router.get("/c/{collection}/claim/propose")
async def search_form(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    collection: Optional[str] = None,
):
    update_fragment_selection(request, None, True)
    base_vars = await get_base_template_vars(request, current_user, collection)
    collection_ob: Collection = base_vars["collection"]
    is_proposal = request.url.path.split("/")[-1] == "propose"
    return templates.TemplateResponse(
        request,
        "search.html",
        dict(
            theme=None,
            include_paragraphs=True,
            is_proposal=is_proposal,
            tasks=[],
            mode=search_mode.semantic.name,
            lam=0.5,
            models=collection_ob.embed_models_names(),
            model=collection_ob.embed_model(),
            **base_vars,
        ),
    )


@app_router.post("/search")
@app_router.post("/claim/propose")
@app_router.post("/c/{collection}/search")
@app_router.post("/c/{collection}/claim/propose")
async def text_search(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    text: Annotated[str, Form()],
    model: Annotated[Optional[embedding_model], Form()] = None,
    selection_changes: Annotated[Optional[str], Form()] = None,
    min_distance: Annotated[Optional[float], Form()] = 0.0,
    max_distance: Annotated[Optional[float], Form()] = 100.0,
    reset_fragments: Annotated[bool, Form()] = False,
    mode: Annotated[search_mode, Form()] = search_mode.semantic,
    lam_percent: Annotated[int, Form()] = 50,
    offset: Annotated[int, Form()] = 0,
    limit: Annotated[int, Form()] = 10,
    claim: Annotated[bool, Form()] = False,
    paragraph: Annotated[bool, Form()] = False,
    sentence: Annotated[bool, Form()] = False,
    one_per_cluster: Annotated[bool, Form()] = False,
    one_per_doc: Annotated[bool, Form()] = False,
    show_quotes: Annotated[bool, Form()] = False,
    group_by_cluster: Annotated[bool, Form()] = False,
    only_with_quote: Annotated[bool, Form()] = False,
    collection: Optional[str] = None,
):
    lam = lam_percent / 100.0
    selection = update_fragment_selection(request, selection_changes, reset_fragments)
    is_proposal = request.url.path.split("/")[-1] == "propose"
    if is_proposal:
        claim = True
        paragraph = False
        model = None  # will be the collection's model
        mode = search_mode.semantic

    prev = max(offset - limit, 0) if offset > 0 else ""
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        model = model or collection_ob.embed_model()

        if not (paragraph or sentence or claim):
            base_vars = await get_base_template_vars(request, current_user, collection)
            return templates.TemplateResponse(
                request,
                "search.html",
                dict(
                    theme=None,
                    text=text,
                    mode=mode.name,
                    results=[],
                    offset=0,
                    error="Nothing to search for",
                    include_sentences=sentence,
                    limit=limit,
                    lam=lam,
                    prev="",
                    next="",
                    end=0,
                    selection=list(selection),
                    include_claims=claim,
                    tasks=[],
                    include_paragraphs=paragraph,
                    one_per_cluster=one_per_cluster,
                    one_per_doc=one_per_doc,
                    show_quotes=show_quotes,
                    group_by_cluster=group_by_cluster,
                    only_with_quote=only_with_quote,
                    model=model,
                    models=collection_ob.embed_models_names(),
                    **base_vars,
                ),
            )

        r = await search(
            session,
            None,
            text,
            collection_ob,
            mode,
            model,
            lam_percent / 100.0,
            offset,
            limit,
            claim,
            paragraph,
            min_distance,
            max_distance,
            one_per_doc,
            one_per_cluster,
            show_quotes,
            False,
            group_by_cluster,
            only_with_quote,
            include_sentences=sentence,
        )

    logger.debug("%s", r)
    next_ = (offset + limit) if len(r) == limit else ""
    end = offset + len(r)

    return templates.TemplateResponse(
        request,
        "search.html",
        dict(
            theme=None,
            text=text,
            mode=mode.name,
            results=r,
            offset=offset,
            is_proposal=is_proposal,
            limit=limit,
            lam=lam,
            prev=prev,
            next=next_,
            end=end,
            selection=list(selection),
            include_claims=claim,
            tasks=[],
            include_paragraphs=paragraph,
            model=model,
            one_per_cluster=one_per_cluster,
            one_per_doc=one_per_doc,
            show_quotes=show_quotes,
            group_by_cluster=group_by_cluster,
            only_with_quote=only_with_quote,
            models=collection_ob.embed_models_names(),
            **base_vars,
        ),
    )
