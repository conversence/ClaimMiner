from typing import Optional, Union, List, Any, Annotated

from pydantic import Field
from fastapi import Form

from .. import Session
from ..app import BadRequest
from ..auth import user_with_coll_permission_t_dep
from ..pyd_models import (
    search_mode,
    embedding_model,
    fragment_type,
    StatementModel,
    FragmentModel,
    BaseModel,
    BASE_EMBED_MODEL,
)
from ..search import search as base_search
from . import api_router, get_collection


class SearchResult(BaseModel):
    rank: Any
    result: Union[StatementModel, FragmentModel] = Field(discriminator="type")


@api_router.post("/search")
@api_router.post("/c/{collection}/search")
async def search(
    current_user: user_with_coll_permission_t_dep("access"),
    statement_id: Annotated[Optional[int], Form()] = None,
    search_text: Annotated[Optional[str], Form()] = None,
    offset: Annotated[int, Form()] = 0,
    limit: Annotated[int, Form()] = 20,
    mode: Annotated[search_mode, Form()] = search_mode.semantic,
    model: Annotated[embedding_model, Form()] = BASE_EMBED_MODEL,
    lam: Annotated[Optional[float], Form()] = 0.7,
    include_paragraphs: Annotated[bool, Form()] = False,
    include_statements: Annotated[bool, Form()] = True,
    one_per_doc: Annotated[bool, Form()] = False,
    one_per_cluster: Annotated[bool, Form()] = False,
    min_distance: Annotated[Optional[float], Form()] = None,
    max_distance: Annotated[Optional[float], Form()] = None,
    group_by_cluster: Annotated[bool, Form()] = False,
    only_with_quote: Annotated[bool, Form()] = False,
    scales: Annotated[
        List[fragment_type], Form()
    ] = [],  # This is strictly invalid but fastapi handles it well.
    include_sentences: Annotated[bool, Form()] = False,
    collection: Optional[str] = None,
) -> List[SearchResult]:
    if not (include_paragraphs or include_statements or include_sentences):
        raise BadRequest(
            "Specify include_paragraphs or include_statements or include_sentences"
        )
    if not (statement_id or search_text):
        raise BadRequest("Specify statement_id or search_text")
    async with Session() as session:
        collection_ob = await get_collection(collection, session, current_user.id)
        rows = await base_search(
            session,
            statement_id,
            search_text,
            collection_ob,
            mode,
            model,
            lam,
            offset,
            limit,
            include_statements,
            include_paragraphs,
            min_distance,
            max_distance,
            one_per_doc,
            one_per_cluster,
            group_by_cluster=group_by_cluster,
            only_with_quote=only_with_quote,
            scales=scales,
            include_from_analysis=True,
            include_sentences=include_sentences,
        )
        return [
            SearchResult.model_validate(
                dict(result=row["target"].as_model(session), rank=row["rank"])
            )
            for row in rows
        ]
