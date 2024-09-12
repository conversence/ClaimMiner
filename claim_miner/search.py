from typing import Optional, List
from itertools import groupby, chain

from sqlalchemy import select, literal_column, func, desc
from sqlalchemy.sql import cast, functions, or_
from sqlalchemy.orm import subqueryload
from sqlalchemy.dialects.postgresql import websearch_to_tsquery
from pgvector.sqlalchemy import Vector

from . import dispatcher
from .app import NotFound
from .models import (
    Statement,
    Collection,
    embed_db_model_by_name,
    VisibleClaim,
    poly_type_clause,
    search_target_class,
    aliased,
    model_names,
    Document,
    UriEquiv,
    Fragment,
    ClaimLink,
    collection_filter,
    ClusterData,
    InClusterData,
    Analysis,
    analysis_output_table,
)
from .pyd_models import (
    UserModel,
    search_mode,
    embedding_model,
    link_type,
    visible_statement_types,
    fragment_type,
    uri_status,
)
from .embed import tf_embed
from .task_registry import TaskRegistry
from .uri_equivalence import get_existing_documents
from .utils import run_sync


async def search(
    session,
    statement_id: Optional[int] = None,
    search_text: Optional[str] = None,
    collection_ob: Optional[Collection] = None,
    mode: search_mode = search_mode.semantic,
    model: Optional[embedding_model] = None,
    lam: float = 0.5,
    offset: int = 0,
    limit: int = 10,
    include_claims: bool = True,
    include_paragraphs: bool = False,
    min_distance: Optional[float] = None,
    max_distance: Optional[float] = None,
    one_per_doc=False,
    one_per_cluster: bool = False,
    show_quotes: bool = False,
    show_keypoints: bool = False,
    group_by_cluster: bool = False,
    only_with_quote: bool = False,
    scales: Optional[List[fragment_type]] = None,
    include_from_analysis: bool = False,
    include_sentences: bool = False,
):
    assert statement_id or search_text and not (statement_id and search_text)
    model = model or collection_ob.embed_model()
    Embedding = embed_db_model_by_name[model.name]
    if statement_id:
        q = (
            select(VisibleClaim)
            .join(Embedding, VisibleClaim.id == Embedding.fragment_id)
            .filter(VisibleClaim.id == statement_id, poly_type_clause(VisibleClaim))
        )
        claim = await session.scalar(q)
        if not claim:
            await dispatcher.trigger_task(
                "embed_fragment",
                target_id=statement_id,
                task_template_nickname=model_names[model],
            )
            raise NotFound("Missing the embedding for this claim, try again shortly")

    scales = scales or []
    if include_claims:
        scales.extend(visible_statement_types)
    if include_paragraphs:
        scales.append(fragment_type.paragraph)
    else:
        include_paragraphs = fragment_type.paragraph in scales
    if include_sentences:
        scales.append(fragment_type.sentence)
    else:
        include_sentences = fragment_type.sentence in scales
    include_claims = bool(set(scales).intersection(set(visible_statement_types)))

    include_fragments = (
        include_paragraphs or include_sentences or fragment_type.quote in scales
    )
    Target = search_target_class(include_claims, include_fragments)
    neighbour_embedding = aliased(Embedding, name="neighbour_embedding")
    neighbour = aliased(Target, name="neighbour")
    neighbour_doc = aliased(Document, name="neighbour_doc")
    neighbour_uri = aliased(UriEquiv, name="neighbour_uri")
    target = aliased(Embedding, name="target")
    key_point = aliased(Fragment, name="key_point")
    key_point_doc = aliased(Document, name="key_point_doc")
    key_point_uri = aliased(UriEquiv, name="key_point_uri")
    quote = aliased(Fragment, name="quote")
    quote_doc = aliased(Document, name="quote_doc")
    quote_uri = aliased(UriEquiv, name="quote_uri")

    # keep in sync with search.py:
    # Statement.doc_id, Statement.id.label("fragment_id"), Document.url, Document.title, Statement.position, tsrank, Statement.text, Statement.scale
    columns = ["target_id", "doc_id", "rank"]
    query = select(neighbour.id, neighbour.doc_id).filter(poly_type_clause(neighbour))
    if mode != search_mode.text:
        query = query.join(
            neighbour_embedding, neighbour_embedding.fragment_id == neighbour.id
        )

    if include_fragments:
        query = query.join(
            neighbour_doc, neighbour_doc.id == neighbour.doc_id, isouter=include_claims
        )
    else:
        query = query.filter(neighbour.doc_id.is_(None))
    if collection_ob:
        query = collection_filter(
            query,
            collection_ob.name,
            include_claims,
            include_fragments,
            neighbour,
            neighbour_doc,
        )

    if len(scales) > 1:
        query = query.filter(neighbour.scale.in_(scales))
    else:
        query = query.filter(neighbour.scale == scales[0])

    # claim_analyzer = (
    #     TaskRegistry.get_registry().analyzer_by_name.get("extract_claims", None)
    # )
    if only_with_quote:
        query = query.join(
            analysis_output_table, analysis_output_table.c.topic_id == neighbour.id
        ).join(
            Analysis,
            (analysis_output_table.c.analysis_id == Analysis.id)
            # & (Analysis.analyzer_id == claim_analyzer.id)
            & (Analysis.target.is_not(None)),
        )

    if search_text:
        if mode == search_mode.text:
            tsquery = websearch_to_tsquery("english", search_text).label("tsquery")
            vtext = func.to_tsvector(neighbour.text)
            rank = func.ts_rank_cd(vtext, tsquery, 16).label("rank")
            query = query.add_columns(rank)
            query = query.filter(func.starts_with(neighbour.language, "en")).filter(
                neighbour.ptmatch("english")(tsquery)
            )
        else:
            text_embed = await tf_embed(search_text, model.name)
            if mode == search_mode.semantic:
                rank = (1 - neighbour_embedding.distance()(text_embed)).label("rank")
                query = query.add_columns(rank)
            elif mode == search_mode.mmr:
                mmr = func.mmr(
                    cast(text_embed, Vector),
                    None,
                    Embedding.__table__.name,
                    [t.name for t in scales],
                    limit + offset,
                    lam,
                    1000,
                ).table_valued("id", "score")
                rank = mmr.columns.score.label("rank")
                query = query.join(mmr, mmr.columns.id == neighbour.id).add_columns(
                    rank
                )
    elif statement_id:
        if mode == search_mode.semantic:
            query = query.filter(neighbour.id != statement_id)
            subq = (
                select(target.embedding)
                .filter_by(fragment_id=statement_id)
                .scalar_subquery()
            )
            rank = (1 - neighbour_embedding.distance()(subq)).label("rank")
            query = query.add_columns(rank)
        elif mode == search_mode.mmr:
            mmr = func.mmr(
                None,
                statement_id,
                Embedding.__table__.name,
                [t.name for t in scales],
                limit + offset,
                lam,
                1000,
            ).table_valued("id", "score")
            rank = mmr.columns.score.label("rank")
            query = query.join(mmr, mmr.columns.id == neighbour.id).add_columns(rank)

    if min_distance:
        query = query.filter(rank >= float(min_distance))
    if max_distance:
        query = query.filter(rank <= float(max_distance))
    if include_claims:
        if group_by_cluster:
            query = (
                query.outerjoin(InClusterData, neighbour.in_cluster_rels)
                .outerjoin(ClusterData, InClusterData.cluster)
                .add_columns(InClusterData.cluster_id, ClusterData.analysis_id)
            )
            columns.extend(["cluster_id", "cluster_analysis"])
            query = query.order_by(desc(rank))
        elif not include_fragments and one_per_cluster:
            # This is going to be very slow unless we have a max rank...
            cluster_col = functions.coalesce(
                InClusterData.cluster_id, neighbour.id
            ).label("cluster_id")
            query = query.outerjoin(
                InClusterData, neighbour.in_cluster_rels
            ).add_columns(cluster_col)
            query = query.order_by(cluster_col, desc(rank)).distinct(cluster_col)
            subq = query.cte()
            query = select(subq.c.id, subq.c.doc_id, subq.c.rank).order_by(
                desc(subq.c.rank)
            )
        else:
            query = query.order_by(desc(rank))
    elif include_fragments and one_per_doc:
        query = query.order_by(neighbour.doc_id, desc(rank)).distinct(neighbour.doc_id)
        subq = query.cte()
        query = select(subq.c.id, subq.c.doc_id, subq.c.rank).order_by(
            desc(subq.c.rank)
        )
    else:
        query = query.order_by(desc(rank))
    query = query.distinct()
    query = query.limit(limit).offset(offset)
    r = await session.execute(query)
    r = [dict(zip(columns, x)) for x in r]
    columns = ["target"]
    all_targets = [x["target_id"] for x in r]
    q2 = select(neighbour).filter(neighbour.id.in_(all_targets))
    if include_fragments:
        if include_claims:
            q2 = (
                q2.outerjoin(neighbour_doc, neighbour.Fragment.document)
                .outerjoin(neighbour_uri, neighbour_doc.uri)
                .add_columns(neighbour_doc.title, neighbour_uri.uri)
            )
        else:
            q2 = (
                q2.outerjoin(neighbour_doc, neighbour.document)
                .outerjoin(neighbour_uri, neighbour_doc.uri)
                .add_columns(neighbour_doc.title, neighbour_uri.uri)
            )
        columns.extend(["title", "uri"])
    if show_keypoints:
        q2 = (
            q2.outerjoin(
                ClaimLink,
                ClaimLink.source == neighbour.id
                and ClaimLink.link_type == link_type.key_point,
            )
            .outerjoin(key_point, key_point.id == ClaimLink.target)
            .outerjoin(key_point_doc, key_point.document)
            .outerjoin(key_point_uri, key_point_doc.uri)
            .add_columns(key_point, key_point_doc.title, key_point_uri.uri)
        )
        columns.extend(["key_point", "key_point_doc_title", "key_point_doc_url"])
    if show_quotes or only_with_quote:
        isouter = not only_with_quote
        q2 = (
            q2.join(
                analysis_output_table,
                analysis_output_table.c.topic_id == neighbour.id,
                isouter=isouter,
            )
            .join(
                Analysis,
                (analysis_output_table.c.analysis_id == Analysis.id)
                # & (Analysis.analyzer_id == claim_analyzer.id)
                & Analysis.target.is_not(None),
                isouter=isouter
            )
            .join(quote, Analysis.target, isouter=isouter)
        )
        if show_quotes:
            q2 = (
                q2.join(quote_doc, quote.document, isouter=isouter)
                .join(quote_uri, quote_doc.uri, isouter=isouter)
                .add_columns(quote, quote_doc.title, quote_uri.uri)
            )
            columns.extend(["quote", "quote_doc_title", "quote_doc_url"])
    if include_from_analysis and not include_fragments:
        q2 = q2.options(subqueryload(neighbour.from_analyses))
    q2r = await session.execute(q2)

    fragment_dict = {row[0].id: row for row in q2r}
    for row in r:
        extra = fragment_dict.get(row.pop("target_id"))
        if extra:
            row |= dict(zip(columns, extra))

    if group_by_cluster:
        r.sort(key=lambda x: (x["cluster_id"] or 0, -x["rank"]))
        groups = {k: list(g) for k, g in groupby(r, lambda x: x["cluster_id"])}
        non_clusters = groups.pop(None, [])
        for row in non_clusters:
            groups[row["target"].id] = [row]
        groups = list(groups.values())
        groups.sort(key=lambda g: -g[0]["rank"])
        r = list(chain(*groups))

    # if include_fragments:
    #     q3 = await session.execute(
    #         select(Document.id, Document.title, UriEquiv.uri
    #             ).join(UriEquiv, Document.uri
    #             ).filter(Document.id.in_(list(set(f.doc_id for f in fragment_dict.values() if f.doc_id)))))
    #     q3d = {doc_id: (title, uri) for (doc_id, title, uri) in q3}
    #     print(q3d)
    #     for x in r:
    #         target = x['target']
    #         if target.doc_id:
    #             x['title'] = q3d[target.doc_id][0]
    #             x['uri'] = q3d[target.doc_id][1]
    return r
