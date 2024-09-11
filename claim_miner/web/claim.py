"""
Copyright Society Library and Conversence 2022-2024
"""

from datetime import date
from io import TextIOWrapper, BytesIO
from csv import reader, writer
from collections import defaultdict
from typing import Annotated, Optional, Union
from logging import getLogger

import orjson as json
from fastapi import Form, Request, UploadFile, status
from fastapi.responses import RedirectResponse, Response
from werkzeug.http import parse_accept_header
from werkzeug.datastructures import Accept
from sqlalchemy import BigInteger, cast
from sqlalchemy.sql import desc, or_
from sqlalchemy.sql.functions import count
from sqlalchemy.sql.expression import func
from sqlalchemy.orm import joinedload, subqueryload
from sqlalchemy.dialects.postgresql import plainto_tsquery

from .. import Session, select, config, dispatcher
from ..models import (
    Document,
    Statement,
    Fragment,
    VisibleClaim,
    AnyClaimOrHyperedge,
    Analysis,
    InClusterData,
    Analyzer,
    ClaimLink,
    Collection,
    aliased,
    HyperEdge,
    TaskTemplate,
    claim_neighbourhood,
    poly_type_clause,
    graph_subquery,
)
from ..pyd_models import (
    fragment_type,
    link_type,
    embedding_model,
    search_mode,
    topic_type,
    process_status,
)
from ..app import NotFound, BadRequest
from ..auth import fragment_collection_constraints, user_with_coll_permission_c_dep
from . import (
    get_collection,
    get_base_template_vars,
    update_fragment_selection,
    templates,
    send_as_attachment,
    app_router,
)
from ..search import search
from ..utils import as_bool
from ..task_registry import TaskRegistry

logger = getLogger(__name__)

mimetypes = {
    "html": "text/html",
    "pdf": "application/pdf",
    "txt": "text/plain",
}


@app_router.post("/claim")
@app_router.post("/c/{collection}/claim")
async def add_claim(
    text: Annotated[str, Form()],
    node_type: Annotated[fragment_type, Form()],
    current_user: user_with_coll_permission_c_dep("add_claim"),
    collection: Optional[str] = None,
):
    # Obsolete
    if not text:
        return RedirectResponse("/f/claim", status_code=status.HTTP_303_SEE_OTHER)
    async with Session() as session:
        collection_ob = await get_collection(collection, session, current_user.id)
        claim = Statement(
            text=text, scale=node_type, language="en", created_by=current_user.id
        )
        if collection_ob:
            claim.collections = [collection_ob]
        session.add(claim)
        await session.commit()
    return RedirectResponse(
        f"/f{collection_ob.path}/claim/{claim.id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


@app_router.get("/claim")
@app_router.get("/c/{collection}/claim")
async def list_claims(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    collection: Optional[str] = None,
    start: int = 0,
    limit: int = 30,
    search_text: Optional[str] = None,
    type: Optional[str] = None,
):
    offset = start
    accept: Accept = parse_accept_header(request.headers.get("accept"))
    as_json = type == "json" or (
        accept.quality("application/json") > accept.quality("text/html")
    )
    as_csv = type == "csv" or (accept.quality("text/csv") > accept.quality("text/html"))
    as_data = as_json or as_csv
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        if search_text:
            tsquery = plainto_tsquery("english", search_text).label("tsquery")
            vtext = func.to_tsvector(VisibleClaim.text)
            tsrank = func.ts_rank_cd(vtext, tsquery, 16).label("rank")
        if as_data:
            query = (
                select(VisibleClaim)
                .filter(poly_type_clause(VisibleClaim))
                .offset(offset)
                .limit(limit)
            )
            if search_text:
                query = query.filter(VisibleClaim.ptmatch()(tsquery)).order_by(
                    desc(tsrank)
                )
            else:
                query = query.order_by(VisibleClaim.id)
            query = query.offset(offset).limit(limit)
        else:
            theme_analysis = aliased(Analysis)
            target_analysis = aliased(Analysis)
            output_analysis = aliased(Analysis)
            query = (
                select(
                    VisibleClaim,
                    count(theme_analysis.id).label("num_analysis"),
                    func.array_agg(output_analysis.id),
                    func.array_agg(target_analysis.id),
                )
                .outerjoin(theme_analysis, theme_analysis.theme_id == VisibleClaim.id)
                .outerjoin(
                    target_analysis, target_analysis.target_id == VisibleClaim.id
                )
                .outerjoin(output_analysis, VisibleClaim.from_analyses)
                .group_by(VisibleClaim.primary_key)
            )
            if search_text:
                query = (
                    query.filter(VisibleClaim.ptmatch()(tsquery))
                    .order_by(desc(tsrank))
                    .offset(offset)
                    .limit(limit)
                )
            else:
                id_query = (
                    select(Statement.id)
                    .filter(
                        Statement.scale >= fragment_type.standalone,
                        Statement.type == "standalone",
                    )
                    .order_by(func.substr(Statement.text, 0, 64), Statement.text)
                    .offset(offset)
                    .limit(limit)
                )
                if collection_ob or not current_user.can("access"):
                    id_query = await fragment_collection_constraints(
                        current_user, id_query, collection_ob, Fragment
                    )
                query = query.filter(VisibleClaim.id.in_(id_query.subquery())).order_by(
                    VisibleClaim.text
                )
        if collection_ob or not current_user.can("access") and (as_data or search_text):
            query = await fragment_collection_constraints(
                current_user, query, collection_ob, VisibleClaim
            )
        r = await session.execute(query)
        claims = [
            (claim, n, set(filter(None, target_a)), set(filter(None, from_a)))
            for (claim, n, target_a, from_a) in r
        ]
        claim_indices = []
        if offset == 0 and not as_data:
            query = (
                select(VisibleClaim)
                .filter_by(scale=fragment_type.standalone_root)
                .order_by(func.substr(VisibleClaim.text, 0, 64), VisibleClaim.text)
            )
            if collection_ob or not current_user.can("access"):
                query = await fragment_collection_constraints(
                    current_user, query, collection_ob, VisibleClaim
                )
            r = await session.execute(query)
            claim_indices = [c for (c,) in r.fetchall()]
    if as_json:
        title = f"claims_{collection_ob.name}.json" if collection_ob else "claims.json"
        return send_as_attachment(
            json.dumps(
                [
                    dict(id=claim.id, text=claim.text, type=claim.scale.name)
                    for (claim,) in claims
                ]
            ),
            "application/json",
            title,
        )
    elif as_csv:
        title = f"claims_{collection_ob.name}.csv" if collection_ob else "claims.csv"
        output = BytesIO()
        output_utf8 = TextIOWrapper(output, encoding="utf-8")
        csv = writer(output_utf8, dialect="excel", delimiter=";")
        csv.writerow(["id", "type", "text"])
        for (claim,) in claims:
            csv.writerow([claim.id, claim.scale.name, claim.text])
        output_utf8.detach()
        output.seek(0)
        return send_as_attachment(output, "text/csv", title)
    previous = max(offset - limit, 0) if offset > 0 else ""
    next_ = (offset + limit) if len(claims) == limit else ""
    end = offset + len(claims)
    return templates.TemplateResponse(
        request,
        "list_claims.html",
        dict(
            claims=claims,
            claim_indices=claim_indices,
            offset=offset,
            prev=previous,
            next=next_,
            limit=limit,
            end=end,
            search_text=search_text,
            **base_vars,
        ),
    )


@app_router.get("/claim/{id}/add_related")
@app_router.get("/c/{collection}/claim/{id}/add_related")
async def claim_add_related_get(
    request: Request,
    current_user: user_with_coll_permission_c_dep("add_claim"),
    id: int,
    text: Annotated[str, Form()] = "",
    node_type: Annotated[fragment_type, Form()] = fragment_type.standalone,
    link_type: Annotated[link_type, Form()] = link_type.freeform,
    reverse_link: Annotated[bool, Form()] = False,
    collection: Optional[str] = None,
):
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        collections = [collection_ob] if collection_ob else []
        nghd = await claim_neighbourhood(id, session)
        claim = nghd["node"]
        if not claim:
            raise NotFound()
        return templates.TemplateResponse(
            request,
            "propose_claim.html",
            dict(text=text, node_type=node_type, claim_nghd=nghd, **base_vars),
        )


@app_router.post("/claim/{id}/add_related")
@app_router.post("/c/{collection}/claim/{id}/add_related")
async def claim_add_related(
    request: Request,
    current_user: user_with_coll_permission_c_dep("add_claim"),
    id: int,
    text: Annotated[str, Form()],
    add: Annotated[bool, Form()] = False,
    node_type: Annotated[fragment_type, Form()] = fragment_type.standalone,
    link_type: Annotated[link_type, Form()] = link_type.freeform,
    reverse_link: Annotated[bool, Form()] = False,
    collection: Optional[str] = None,
):
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        collections = [collection_ob] if collection_ob else []
        if not add:
            # Navigating to another claim
            nghd = await claim_neighbourhood(id, session)
            claim = nghd["node"]
            if not claim:
                raise NotFound()
            return templates.TemplateResponse(
                request,
                "propose_claim.html",
                dict(text=text, node_type=node_type, claim_nghd=nghd, **base_vars),
            )
        # otherwise create the new claim
        new_claim = Statement(
            text=text, scale=node_type, language="en", collections=collections
        )
        if reverse_link:
            flink = ClaimLink(
                source_topic=new_claim,
                target=id,
                link_type=link_type,
                created_by=current_user.id,
            )
        else:
            flink = ClaimLink(
                source=id,
                target_topic=new_claim,
                link_type=link_type,
                created_by=current_user.id,
            )
        session.add(flink)
        await session.commit()
    return RedirectResponse(
        f"/f{collection_ob.path}/claim/{new_claim.id}",
        status_code=status.HTTP_303_SEE_OTHER,
    )


def send_results_csv(
    results, paragraph, show_quotes, target, group_by_cluster, collection_ob
):
    title = (
        f"claims_{collection_ob.name}_near_{target.id}.csv"
        if collection_ob
        else f"claims_near_{id}.csv"
    )
    output = BytesIO()
    output_utf8 = TextIOWrapper(output, encoding="utf-8")
    csv = writer(output_utf8, dialect="excel", delimiter=";")
    columns = ["id", "rank", "statement"]
    if group_by_cluster:
        columns.extend(["cluster_analysis", "cluster_id"])
    if paragraph:
        columns.append("url")
    if show_quotes:
        columns.extend(["quote", "quote_url"])
    csv.writerow(columns)
    data = [target.id, "target", target.text]
    if group_by_cluster:
        data.extend([None, None])
    if paragraph:
        data.append(None)
    if show_quotes:
        data.extend([None, None])
    csv.writerow(data)
    for row in results:
        logger.debug("%s", row)
        neighbour = row["target"]
        data = [neighbour.id, row["rank"], neighbour.text]
        if group_by_cluster:
            data.extend(
                [row.get("cluster_analysis", None), row.get("cluster_id", None)]
            )
        if paragraph:
            data.append(row.get("doc_url", None))
        if show_quotes:
            quote = row.get("quote")
            data.extend([quote.text if quote else None, row.get("quote_doc_url")])
        csv.writerow(data)
    output_utf8.detach()
    output.seek(0)
    return send_as_attachment(output, "text/csv", title)


@app_router.get("/claim/{id}/search")
@app_router.get("/c/{collection}/claim/{id}/search")
async def search_on_claim_get(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    id: int,
    model: Optional[embedding_model] = None,
    selection_changes: Optional[str] = None,
    min_distance: Optional[float] = None,
    max_distance: Optional[float] = None,
    reset_fragments: Optional[bool] = False,
    mode: search_mode = search_mode.semantic,
    lam_percent: int = 50,
    offset: int = 0,
    limit: int = 10,
    claim: bool = True,
    paragraph: bool = False,
    sentence: bool = True,
    one_per_cluster: bool = False,
    one_per_doc: bool = False,
    show_quotes: bool = False,
    group_by_cluster: bool = False,
    only_with_quote: bool = False,
    as_csv: bool = False,
    collection: Optional[str] = None,
):
    selection = update_fragment_selection(
        request, selection_changes, as_bool(reset_fragments)
    )
    lam = lam_percent / 100.0
    if not (paragraph or claim):
        paragraph = True
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        claim_ob = await session.get(Statement, id)
        r = await search(
            session,
            id,
            None,
            collection_ob,
            mode,
            model,
            lam,
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
        registry = TaskRegistry.get_registry()
        tasks = []
        if (paragraph or sentence) and selection:
            async for task_data in registry.all_task_status(
                session,
                claim_ob,
                collection,
                topic_type.fragment,
                dict(source_ids=list(selection)),
            ):
                tasks.append(task_data)
                base_vars |= await task_data[0].task_form_args(session)

    accept: Accept = parse_accept_header(request.headers.get("accept"))
    as_csv = as_csv or (
        type == "csv" or (accept.quality("text/csv") > accept.quality("text/html"))
    )
    if as_csv:
        return send_results_csv(
            r,
            paragraph or sentence,
            show_quotes,
            claim_ob,
            group_by_cluster,
            collection_ob,
        )

    prev = max(offset - limit, 0) if offset > 0 else ""
    next_ = (offset + limit) if len(r) == limit else ""
    return templates.TemplateResponse(
        request,
        "search.html",
        dict(
            theme=claim_ob,
            text=claim_ob.text,
            results=r,
            lam=lam,
            model=model,
            tasks=tasks,
            offset=offset,
            limit=limit,
            prev=prev,
            next=next_,
            selection=list(selection),
            include_paragraphs=paragraph,
            include_sentences=sentence,
            include_claims=claim,
            one_per_cluster=one_per_cluster,
            one_per_doc=one_per_doc,
            models=collection_ob.embed_models_names(),
            show_quotes=show_quotes,
            group_by_cluster=group_by_cluster,
            only_with_quote=only_with_quote,
            mode=mode.name,
            **base_vars,
        ),
    )


@app_router.post("/claim/{id}/search")
@app_router.post("/c/{collection}/claim/{id}/search")
async def search_on_claim_post(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    id: int,
    model: Annotated[Optional[embedding_model], Form()] = None,
    selection_changes: Annotated[Optional[str], Form()] = None,
    min_distance: Annotated[Optional[float], Form()] = None,
    max_distance: Annotated[Optional[float], Form()] = None,
    reset_fragments: Annotated[bool, Form()] = False,
    mode: Annotated[search_mode, Form()] = search_mode.semantic,
    lam_percent: Annotated[int, Form(gt=0, lt=100)] = 50,
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
    as_csv: Optional[bool] = False,
):
    selection = update_fragment_selection(
        request, selection_changes, as_bool(reset_fragments)
    )
    lam = lam_percent / 100.0
    if not (paragraph or claim or sentence):
        paragraph = True
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        claim_ob = await session.get(Statement, id)
        r = await search(
            session,
            id,
            None,
            collection_ob,
            mode,
            model,
            lam,
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

        registry = TaskRegistry.get_registry()
        tasks = []
        if (paragraph or sentence) and selection:
            async for task_data in registry.all_task_status(
                session,
                claim_ob,
                collection,
                topic_type.fragment,
                dict(source_ids=list(selection)),
            ):
                tasks.append(task_data)
                base_vars |= await task_data[0].task_form_args(session)

    accept: Accept = parse_accept_header(request.headers.get("accept"))
    as_csv = as_csv or (
        type == "csv" or (accept.quality("text/csv") > accept.quality("text/html"))
    )
    if as_csv:
        return send_results_csv(
            r, paragraph, show_quotes, claim_ob, group_by_cluster, collection_ob
        )

    prev = max(offset - limit, 0) if offset > 0 else ""
    next_ = (offset + limit) if len(r) == limit else ""
    return templates.TemplateResponse(
        request,
        "search.html",
        dict(
            theme=claim_ob,
            text=claim_ob.text,
            results=r,
            lam=lam,
            model=model,
            tasks=tasks,
            offset=offset,
            limit=limit,
            prev=prev,
            next=next_,
            selection=list(selection),
            include_paragraphs=paragraph,
            include_claims=claim,
            one_per_cluster=one_per_cluster,
            one_per_doc=one_per_doc,
            show_quotes=show_quotes,
            include_sentences=sentence,
            group_by_cluster=group_by_cluster,
            only_with_quote=only_with_quote,
            models=collection_ob.embed_models_names(),
            mode=mode.name,
            **base_vars,
        ),
    )


@app_router.get("/claim/{theme_id}/gdelt")
@app_router.get("/c/{collection}/claim/{theme_id}/gdelt")
async def gdelt_get(
    request: Request,
    current_user: user_with_coll_permission_c_dep("bigdata_query"),
    theme_id: int,
    collection: Optional[str] = None,
):
    base_vars = await get_base_template_vars(request, current_user, collection)
    return templates.TemplateResponse(
        request, "gdelt.html", dict(claim_id=theme_id, error="", **base_vars)
    )


@app_router.post("/claim/{theme_id}/gdelt")
@app_router.post("/c/{collection}/claim/{theme_id}/gdelt")
async def gdelt(
    request: Request,
    current_user: user_with_coll_permission_c_dep("bigdata_query"),
    theme_id: int,
    limit: Annotated[int, Form()],
    date: Annotated[Optional[date], Form()] = None,
    collection: Optional[str] = None,
):
    error = ""
    base_vars = await get_base_template_vars(request, current_user, collection)
    try:
        if date:
            # sanity
            since = date.strftime("%Y-%m-%d")
        await dispatcher.trigger_task(
            "gdelt", theme_id=theme_id, limit=limit, since=since
        )
        return RedirectResponse("/f/", status_code=status.HTTP_303_SEE_OTHER)
    except Exception as e:
        error = str(e)
    return templates.TemplateResponse(
        request, "gdelt.html", dict(claim_id=theme_id, error=error, **base_vars)
    )


@app_router.get("/claim/upload")
@app_router.get("/c/{collection}/claim/upload")
async def upload_claims_get(
    request: Request,
    current_user: user_with_coll_permission_c_dep("add_claim"),
    collection: Optional[str] = None,
):
    base_vars = await get_base_template_vars(request, current_user, collection)
    return templates.TemplateResponse(
        request,
        "upload_claims.html",
        dict(error="", success="", new_ids=[], **base_vars),
    )


@app_router.post("/claim/upload")
@app_router.post("/c/{collection}/claim/upload")
async def upload_claims_post(
    request: Request,
    current_user: user_with_coll_permission_c_dep("add_claim"),
    file: UploadFile,
    column: Annotated[int, Form()],
    node_type: Annotated[fragment_type, Form()],
    skip: Annotated[bool, Form()] = False,
    collection: Optional[str] = None,
):
    error = ""
    success = ""
    warning = ""
    repeats = []
    new_ids = []
    base_vars = await get_base_template_vars(request, current_user, collection)
    collection_ob: Collection = base_vars["collection"]
    assert file.size
    r = reader(TextIOWrapper(file.file, "utf-8"))
    if skip:
        next(r)
    claim_texts = [row[column - 1].strip() for row in r]
    claims = []
    async with Session() as session:
        for claim_text in claim_texts:
            # Check if it exists first
            existing = await session.execute(
                select(Statement)
                .filter(Statement.text == claim_text, poly_type_clause(Statement))
                .limit(1)
            )
            if existing := existing.first():
                # Ensure in collection, right type
                repeats.append(existing.id)
                continue
            claim = Statement(
                text=claim_text,
                scale=node_type,
                language="en",
                created_by=current_user.id,
            )
            if collection_ob:
                claim.collections = [collection_ob]
            session.add(claim)
            claims.append(claim)
        if claims:
            await session.commit()
            for claim in claims:
                # TODO Batch embedding requests
                new_ids.append(claim.id)
            success = f"Success, {len(new_ids)} created"
            if repeats:
                warning = f"Already existing: {repeats}"
        else:
            warning = "All those claim already exist"
    return templates.TemplateResponse(
        request,
        "upload_claims.html",
        dict(
            error=error, success=success, warning=warning, new_ids=new_ids, **base_vars
        ),
    )


@app_router.get("/claim/{id}")
@app_router.get("/c/{collection}/claim/{id}")
async def claim_info(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    id: int,
    collection: Optional[str] = None,
    start: int = 0,
    limit: int = 30,
):
    offset = start
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        claim = await session.scalar(
            select(AnyClaimOrHyperedge)
            .filter(AnyClaimOrHyperedge.id == id, poly_type_clause(AnyClaimOrHyperedge))
            .options(
                subqueryload(AnyClaimOrHyperedge.from_analyses)
                .joinedload(Analysis.task_template)
                .joinedload(TaskTemplate.analyzer),
                subqueryload(AnyClaimOrHyperedge.from_analyses).joinedload(
                    Analysis.analyzer
                ),
                subqueryload(AnyClaimOrHyperedge.outgoing_links).joinedload(
                    ClaimLink.target_topic
                ),
                subqueryload(AnyClaimOrHyperedge.incoming_links).joinedload(
                    ClaimLink.source_topic
                ),
                subqueryload(AnyClaimOrHyperedge.Statement.in_cluster_rels).joinedload(
                    InClusterData.cluster
                ),
            )
        )
        if not claim:
            raise NotFound()
        claim_nghd = await claim_neighbourhood(id, session)
        tasks = []
        async for task_data in TaskRegistry.get_registry().all_task_status(
            session, claim, collection
        ):
            tasks.append(task_data)
            base_vars |= await task_data[0].task_form_args(session)

        # Claim extraction analyses
        para_by_analysis = defaultdict(set)
        origin_paras = {
            o["source"]
            for o in (getattr(claim, "generation_data", {}) or {}).get("origin", ())
        }
        for analysis in claim.from_analyses:
            if isinstance(analysis.results, list):
                for r in analysis.results:
                    if r.get("fragment_id", None) == id:
                        origin_paras.update(r.get("sources", ()))
        if origin_paras:
            origin_paras_q = await session.execute(
                select(Fragment)
                .filter(Fragment.id.in_(list(origin_paras)))
                .options(subqueryload(Fragment.document))
            )
            origin_paras_d = {p.id: p for (p,) in origin_paras_q}
            for analysis in claim.from_analyses:
                if analysis.analyzer.name == "extract_claims":
                    await session.refresh(analysis, ["context"])
                    for para in origin_paras_d.values():
                        if para in analysis.context:
                            para_by_analysis[analysis.id].add(para)
                        if para.id == analysis.target_id:
                            para_by_analysis[analysis.id].add(para)
                else:
                    for r in analysis.results:
                        if r.get("fragment_id", None) == id:
                            for id_ in r.get("sources", ()):
                                para_by_analysis[analysis.id].add(origin_paras_d[id_])

        related_analysis = []
        if isinstance(claim, Fragment):
            source = aliased(Fragment, name="source")
            source_doc = aliased(Document, name="source_doc")
            # analyses using this claim as a theme
            # TODO: suspicious logic here! themes are statements, not fragments
            q1 = (
                select(Analysis, source, source_doc)
                .join(source, source.id == Analysis.theme_id)
                .join(source_doc, Fragment.document)
                .filter(cast(Analysis.params["theme"], BigInteger) == id)
                .options(joinedload(Analysis.analyzer))
            )

            r = await session.execute(q1)
            related_analysis = r.fetchall()

    next_ = (offset + limit) if len(related_analysis) == limit else ""
    prev = max(offset - limit, 0) if offset > 0 else ""
    outgoing_links = [(link.target_topic, link) for link in claim.outgoing_links]
    incoming_links = [(link.source_topic, link) for link in claim.incoming_links]
    return templates.TemplateResponse(
        request,
        "claim_info.html",
        dict(
            claim=claim,
            related_analysis=related_analysis,
            incoming_links=incoming_links,
            tasks=tasks,
            outgoing_links=outgoing_links,
            claim_nghd=claim_nghd,
            para_by_analysis=para_by_analysis,
            prev=prev,
            next=next_,
            offset=offset,
            limit=limit,
            **base_vars,
        ),
    )
