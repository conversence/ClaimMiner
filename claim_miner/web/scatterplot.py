"""
Copyright Society Library and Conversence 2022-2024
"""

# clustering
from collections import defaultdict
from html import escape
from typing import Optional, List, Union, Dict, Any

from fastapi import Request
import numpy as np
import orjson as json
from sqlalchemy.future import select
from sklearn.metrics.pairwise import cosine_similarity
import sklearn.manifold
import sklearn.decomposition
from ..utils import run_sync

from .. import Session
from ..models import (
    Statement,
    VisibleClaim,
    embed_db_model_by_name,
    Collection,
    TopicCollection,
    poly_type_clause,
)
from ..pyd_models import fragment_type, embedding_model
from ..app import BadRequest
from ..auth import user_with_coll_permission_c_dep, optional_active_user_c_dep
from ..embed import tf_embed
from ..utils import decode_uuid
from . import get_base_template_vars, templates, app_router, spa_router


@app_router.get("/claim/scatter")
@app_router.get("/c/{collection}/claim/scatter")
async def claim_scatter(
    request: Request,
    claim_id: Optional[Union[int, str]] = None,
    collection: Optional[str] = None,
    model: Optional[embedding_model] = None,
    keyword: Optional[str] = None,
    depth: int = 6,
    method: str = "TruncatedSVD",
):
    base_vars = await get_base_template_vars(request, None, collection)
    collection: Collection = base_vars["collection"]

    if claim_id:
        try:
            claim_id = int(claim_id)
        except ValueError as e:
            pass
        query = (
            select(Statement)
            .filter_by(scale=fragment_type.standalone_root)
            .filter_by(id=claim_id)
        )
        query = query.limit(1)
        async with Session() as session:
            claim = await session.scalar(query)
            await session.refresh(claim, ["collections"])
            if claim.collections:
                collection = claim.collections[0]

    coll_model_names = collection.embed_models_names()
    model = model or embedding_model[coll_model_names[0]]
    Embedding = embed_db_model_by_name[model.name]

    query = (
        select(Embedding.fragment_id, Embedding.embedding, VisibleClaim.text)
        .join(VisibleClaim)
        .filter(poly_type_clause(VisibleClaim))
    )
    if collection:
        query = (
            query.join(TopicCollection)
            .join(Collection)
            .filter(Collection.name == collection.name)
        )
    async with Session() as session:
        data = await session.execute(query)
    (fids, embeds, texts) = zip(*data)  # TODO: fails if empty
    embeds = list(embeds)
    num_claims = len(fids)
    keywords = [keyword] if keyword else []
    if len(keywords) == 1:
        keywords = [t.strip() for t in keywords[0].split(",") if t.strip()]
    if keywords:
        # TODO: cache keyword embed results to avoid recurring costs
        kwembeds = await tf_embed(keywords, model.name)
        all_embeds = embeds + list(kwembeds)
        similarities = await run_sync(
            lambda: cosine_similarity(embeds, kwembeds).tolist()
        )()
        num_total = len(all_embeds)
    else:
        all_embeds = embeds
        similarities = []
        num_total = num_claims
    kwargs: Dict[str, Any] = dict(n_components=2)
    if hasattr(sklearn.manifold, method):
        method_class = getattr(sklearn.manifold, method)
        kwargs["n_jobs"] = -1
        extra_args: dict = defaultdict(
            dict,
            dict(
                LocallyLinearEmbedding=dict(
                    method="hessian", n_neighbors=6, eigen_solver="dense"
                ),
                SpectralEmbedding=dict(eigen_solver="amg"),  # affinity="rbf"
            ),
        )[method]
        kwargs |= extra_args
    elif hasattr(sklearn.decomposition, method):
        method_class = getattr(sklearn.decomposition, method)
    else:
        raise BadRequest("Unknown method")
    method_ob = method_class(**kwargs)
    pos = await run_sync(lambda: method_ob.fit_transform(np.array(all_embeds)))()
    claims_data = [
        dict(id=id, x=float(x), y=float(y), t=escape(t))
        for (id, t, (x, y)) in zip(fids, texts, pos[:num_claims])
    ]
    keyword_data = [
        dict(id=n, x=float(x), y=float(y), t=escape(t))
        for ((n, t), (x, y)) in zip(enumerate(keywords), pos[num_claims:])
    ]
    template = "scatter.html"
    if not request.url.path.startswith("/f/"):
        template = "scatter_standalone.html"
    return templates.TemplateResponse(
        request,
        template,
        dict(
            data=json.dumps(claims_data).decode("utf-8"),
            method=method,
            keywords=", ".join(keywords),
            similarities=json.dumps(similarities).decode("utf-8"),
            model=model,
            models=coll_model_names,
            keyword_data=json.dumps(keyword_data).decode("utf-8"),
            **base_vars,
        ),
        200,
        {"HX-Retarget": "#body"},
    )
