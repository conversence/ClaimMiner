"""
Copyright Society Library and Conversence 2022-2024
"""

from io import BytesIO, TextIOWrapper
from datetime import datetime
from csv import reader, writer
from collections import defaultdict
from pathlib import Path
import re
from typing import Annotated, Optional, List, Dict, Tuple
from itertools import chain
from io import StringIO
from logging import getLogger

import orjson as json
from fastapi import Form, Request, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import cast, Float, Boolean
from sqlalchemy.sql.functions import count, max as fmax, min as fmin, coalesce
from sqlalchemy.orm import joinedload, subqueryload
import isodate

from .. import Session, select, hashfs
from ..models import (
    Analysis,
    Document,
    Statement,
    Fragment,
    ClaimLink,
    Collection,
    UriEquiv,
    embed_db_model_by_name,
    aliased,
    analysis_output_table,
)
from ..pyd_models import fragment_type, visible_statement_types
from ..app import Forbidden, NotFound, BadRequest
from ..auth import (
    doc_collection_constraints,
    check_doc_access,
    user_with_coll_permission_c_dep,
)
from ..nlp import as_prompts
from ..uri import normalize, url_regexp
from .. import uri_equivalence
from . import (
    render_with_spans,
    get_collection,
    get_base_template_vars,
    send_as_attachment,
    templates,
    app_router,
)
from ..task_registry import TaskRegistry
from ..utils import deduplicate, safe_lang_detect

logger = getLogger(__name__)


mimetypes = {
    "html": "text/html",
    "pdf": "application/pdf",
    "txt": "text/plain",
}


@app_router.get("/document")
@app_router.get("/c/{collection}/document")
async def list_docs(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    collection: Optional[str] = None,
    start: int = 0,
    limit: int = 30,
):
    offset = start
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        sq = (
            select(Document.id).order_by(Document.id.desc()).offset(offset).limit(limit)
        )
        if collection_ob or not current_user.can("access"):
            sq = await doc_collection_constraints(current_user, sq, collection_ob)
        # subquery load instead of grouping by uri
        docs = await session.execute(
            select(Document)
            .filter(Document.id.in_(sq))
            .order_by(Document.id.desc())
            .options(joinedload(Document.uri))
        )
        docs = [d for (d,) in docs]
        q = (
            select(
                Document.id,
                count(Fragment.id.distinct()),
                count(analysis_output_table.c.topic_id.distinct()),
            )
            .outerjoin(Fragment, Document.paragraphs)
            .outerjoin(Analysis, Fragment.id == Analysis.target_id)
            .outerjoin(
                analysis_output_table,
                analysis_output_table.c.analysis_id == Analysis.id,
            )
            .group_by(Document.id)
            .filter(Document.id.in_(sq))
        )
        r = await session.execute(q)
        data = r.fetchall()
    num_paras = {x[0]: x[1] for x in data}
    num_generated = {x[0]: x[2] for x in data}
    previous = max(offset - limit, 0) if offset > 0 else ""
    next = (offset + limit) if len(docs) == limit else ""
    end = offset + len(docs)
    return templates.TemplateResponse(
        request,
        "list_docs.html",
        dict(
            docs=docs,
            offset=offset,
            end=end,
            prev=previous,
            next=next,
            limit=limit,
            num_paras=num_paras,
            num_generated=num_generated,
            **base_vars,
        ),
    )


@app_router.get("/document/{doc_id}/raw")
@app_router.get("/c/{collection}/document/{doc_id}/raw")
async def get_raw_doc(
    current_user: user_with_coll_permission_c_dep("access"),
    doc_id: int,
    collection: Optional[str] = None,
):
    async with Session() as session:
        collection_ob = await get_collection(collection, session, current_user.id)
        r = await session.execute(
            select(
                Document.file_identity, Document.mimetype, Document.public_contents
            ).filter_by(id=doc_id)
        )
        if r is None:
            raise NotFound()
        (file_identity, mimetype, public_contents) = r.first()
        if not (public_contents or current_user.can("admin")):
            raise Forbidden("Copyrighted content")
        await check_doc_access(current_user, doc_id, collection_ob)
        file_info = hashfs.get(file_identity)
        extension = mimetype.split("/")[1]
        return FileResponse(
            file_info.abspath, media_type=mimetype, filename=f"{doc_id}.{extension}"
        )


@app_router.get("/document/{doc_id}/text")
@app_router.get("/c/{collection}/document/{doc_id}/text")
async def get_text_doc(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    doc_id: int,
    collection: Optional[str] = None,
):
    async with Session() as session:
        collection_ob = await get_collection(collection, session, current_user.id)
        r = await session.execute(
            select(
                Document.text_identity, Document.mimetype, Document.public_contents
            ).filter_by(id=doc_id)
        )
        if r is None:
            raise NotFound()
        (text_identity, mimetype, public_contents) = r.first()
        if not (public_contents or current_user.can("admin")):
            raise Forbidden("Copyrighted content")
        await check_doc_access(current_user, doc_id, collection_ob)
        file_info = hashfs.get(text_identity)
        return FileResponse(
            file_info.abspath, media_type=mimetype, filename=f"{doc_id}.txt"
        )


@app_router.get("/document/{doc_id}/csv")
@app_router.get("/c/{collection}/document/{doc_id}/csv")
async def get_doc_csv(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    doc_id: int,
    collection: Optional[str] = None,
):
    async with Session() as session:
        collection_ob = await get_collection(collection, session, current_user.id)
        r = await session.execute(
            select(
                Document.text_identity, Document.mimetype, Document.public_contents
            ).filter_by(id=doc_id)
        )
        if r is None:
            raise NotFound()
        (text_identity, mimetype, public_contents) = r.first()
        if not (public_contents or current_user.can("admin")):
            raise Forbidden("Copyrighted content")
        await check_doc_access(current_user, doc_id, collection_ob)
        if not text_identity:
            raise BadRequest("Text file not ready")
        fragments = await session.scalars(
            select(Fragment)
            .filter_by(doc_id=doc_id)
            .order_by(Fragment.part_of.nulls_first(), Fragment.position)
        )
        fragments = list(fragments)
        paragraphs = [f for f in fragments if f.scale == fragment_type.paragraph]
        sentences = defaultdict(list)
        for f in fragments:
            if f.scale == fragment_type.sentence:
                sentences[f.part_of].append(f)
        generated_claims_by_para = await get_generated_claims_by_para(
            session, {p.id: p for p in paragraphs}
        )
        claims = {
            p_id: [c for (c, _a) in p_c]
            for (p_id, p_c) in generated_claims_by_para.items()
        }
        io = StringIO()
        csv = writer(io)
        if sentences:
            rows = ["paragraph #", "sentence #", "text"]
        else:
            rows = ["paragraph #", "text"]
        if claims:
            rows += ["claims"]
        csv.writerow(rows)
        if sentences:
            for para in paragraphs:
                first = True
                for sentence in sentences[para.id]:
                    row = [para.position, sentence.position, sentence.text]
                    # Attribute all claims to first sentence, 1st approximation
                    if claims and first:
                        first = False
                        row += ["\n".join(c.text for c in claims.get(para.id, ()))]
                    csv.writerow(row)
        else:
            for para in paragraphs:
                row = [para.position, para.text]
                if claims:
                    row += ["\n".join(c.text for c in claims.get(para.id, ()))]
                csv.writerow(row)
    io.seek(0)
    return send_as_attachment(io, "text/csv", f"document_{doc_id}.csv")


@app_router.get("/document/{doc_id}/links")
@app_router.get("/c/{collection}/document/{doc_id}/links")
async def get_text_links(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    doc_id: int,
    collection: Optional[str] = None,
):
    # Temporary code
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        doc = await session.get(Document, doc_id, options=[joinedload(Document.uri)])
        if doc is None:
            raise NotFound()
        if not (doc.public_contents or current_user.can("admin")):
            raise Forbidden("Copyrighted content")
        if not doc.file_identity:
            raise BadRequest("File not downloaded")
        await check_doc_access(current_user, doc_id, collection_ob)
        file_info = hashfs.get(doc.file_identity)
        mimetype = doc.base_type
        if mimetype == "text/html":
            from bs4 import BeautifulSoup

            with open(file_info.abspath, "r") as f:
                soup = BeautifulSoup(f.read(), "lxml")
            anchors = soup.find_all("a")
            links_set = {a.attrs.get("href") for a in anchors}
            links_set.discard(None)
        elif mimetype == "application/pdf":
            from pdfx import PDFx

            pdf = PDFx(file_info.abspath)
            links_d = pdf.get_references_as_dict()
            links_set = set(chain(*links_d.values()))
        elif mimetype in ("text/plain", "text/markdown"):
            with open(file_info.abspath, "r") as f:
                content = f.read()
            links_set = set(url_regexp.findall(content))
        links = list({normalize(u, doc.url) for u in links_set})
        links_set.discard(doc.url)
        links.sort()
        existing = await session.execute(
            select(UriEquiv, Document)
            .join(UriEquiv.document)
            .filter(UriEquiv.uri.in_(links))
        )
        existing = {ue.uri: (ue, doc) for (ue, doc) in existing}
    return templates.TemplateResponse(
        request,
        "doc_links.html",
        dict(doc=doc, links=links, existing=existing, **base_vars),
    )


def compose_url_jsonl(data, spec):
    if "|" in spec:
        specs = spec.split("|")
        for spec in specs:
            if url := compose_url_jsonl(data, spec):
                return url
    part_specs = spec.split(",")
    parts = []
    for spec in part_specs:
        spec = spec.strip()
        if spec.startswith("'"):
            parts.append(spec.strip("'"))
            continue
        slugify = spec.startswith("#")
        spec = spec.strip("#")
        prefix = None
        if "-" in spec:
            spec, prefix = spec.split("-")
            prefix = int(prefix)
        part = data.get(spec, None)
        if slugify:
            part = part.encode("ascii", "replace").decode("ascii")
            part = re.sub(r"\W+", "_", part, 0, re.ASCII)
            part = part.strip("_")
        if part == "n/a":
            part = None
        if not part:
            return None
        if prefix:
            part = part[prefix:]
        parts.append(part)
    return "/".join(parts)


def maybe_flatten(str_or_list):
    if isinstance(str_or_list, list):
        return "\n\n".join(str_or_list)
    return str_or_list


def get_text_jsonl(data, text_fields="text", text_process=None):
    text_fields = text_fields.split(",")
    text = "\n\n".join(
        maybe_flatten(data.get(field.strip(), "")) for field in text_fields
    )
    if text_process:
        for pat, repl in text_process:
            text = re.sub(pat, repl, text)
    return text


@app_router.get("/document/upload")
@app_router.get("/c/{collection}/document/upload")
async def upload_docs_get(
    request: Request,
    current_user: user_with_coll_permission_c_dep("add_document"),
    collection: Optional[str] = None,
):
    base_vars = await get_base_template_vars(request, current_user, collection)
    return templates.TemplateResponse(
        request,
        "upload_docs.html",
        dict(
            error="",
            success="",
            success_doc_id=None,
            use_title=True,
            new_ids=[],
            **base_vars,
        ),
    )


@app_router.get("/document/urls")
@app_router.get("/c/{collection}/document/urls")
async def get_doc_urls(
    current_user: user_with_coll_permission_c_dep("access"),
    collection: Optional[str] = None,
):
    async with Session() as session:
        q = select(UriEquiv.uri).join(Document, Document.uri_id == UriEquiv.id)
        if collection:
            q = q.join(Collection, Document.collections).filter_by(name=collection)
        uris = await session.scalars(q)
        return StreamingResponse(
            StringIO("\n".join(uris)), 200, media_type="plain/text"
        )


@app_router.post("/document/upload")
@app_router.post("/c/{collection}/document/upload")
async def upload_docs(
    request: Request,
    current_user: user_with_coll_permission_c_dep("add_document"),
    upload_type: Annotated[str, Form()],
    url: Annotated[Optional[str], Form()] = None,
    quote_text: Annotated[Optional[str], Form()] = None,
    file: Annotated[Optional[UploadFile], Form()] = None,
    url_spec: Annotated[Optional[str], Form()] = None,
    skip: Annotated[bool, Form()] = False,
    column: Annotated[int, Form()] = 1,
    title_column: Annotated[Optional[int], Form()] = None,
    quote_column: Annotated[Optional[int], Form()] = None,
    text_fields: Annotated[Optional[str], Form()] = "text",
    use_title: Annotated[bool, Form()] = False,
    use_published: Annotated[bool, Form()] = False,
    extra_newlines: Annotated[bool, Form()] = False,
    collection: Optional[str] = None,
):
    # TODO: Make this into three functions with different form parameters
    error = ""
    success = ""
    warning = ""
    new_ids = []
    collections = []
    success_doc_id = None
    try:
        async with Session() as session:
            base_vars = await get_base_template_vars(
                request, current_user, collection, session
            )
            if collection_ob := base_vars["collection"]:
                collections.append(collection_ob)
        if upload_type == "single":
            doc_given = False
            if not url:
                raise BadRequest("URL is required")
            url = url.strip()
            async with Session() as session:
                collections = [await session.merge(c) for c in collections]
                url = normalize(url)
                r = await session.scalar(
                    select(UriEquiv.id).filter_by(uri=url).limit(1)
                )
                # TODO: UX to load a new snapshot of an existing document
                if r is not None:
                    raise BadRequest("Document with this URL already exists")
                if file and file.size:
                    extension = (file.filename or ".").lower().split(".")[-1]
                    mimetype = mimetypes.get(extension)
                    if not mimetype:
                        warning = f"unknown file extension: {extension}"
                        logger.warning(warning)
                    file_identity = hashfs.put(file.file)
                    if Path(file_identity.abspath).stat().st_size > 1000:
                        # Avoid identifying small stubs
                        r = await session.execute(
                            select(UriEquiv)
                            .join(Document, Document.uri_id == UriEquiv.id)
                            .filter_by(file_identity=file_identity.id)
                            .limit(1)
                        )
                        if r := r.first():
                            (uri_equiv,) = r
                            await uri_equivalence.add_variant(session, url, uri_equiv)
                            await session.commit()
                            raise BadRequest(
                                f"Document with this file already exists at URL {uri_equiv.uri}"
                            )
                    uri = UriEquiv(uri=url)
                    doc = Document(
                        uri=uri,
                        file_identity=file_identity.id,
                        file_size=file.size,
                        mimetype=mimetype,
                        created_by=current_user.id,
                        return_code=200,
                        collections=collections,
                        retrieved=datetime.utcnow(),
                    )
                    doc_given = True
                else:
                    uri = UriEquiv(uri=url)  # tentatively canonical?
                    doc = Document(
                        uri=uri, created_by=current_user.id, collections=collections
                    )
                session.add(doc)
                if quote_text:
                    lang = safe_lang_detect(quote_text)
                    quote = Fragment(
                        document=doc,
                        text=quote_text,
                        scale=fragment_type.quote,
                        collections=collections,
                        created_by=current_user.id,
                        language=lang,
                    )
                    session.add(quote)
                await session.commit()
                success_doc_id = doc.id
            new_ids = [doc.id]
        elif upload_type == "csv":
            if not file:
                raise BadRequest("File is required")
            fs = file.file
            r = reader(TextIOWrapper(fs, "utf-8"))
            if skip:
                next(r)
            urls_and_quotes = [
                (
                    row[column - 1].strip(),
                    row[title_column - 1].strip() if title_column else None,
                    row[quote_column - 1].strip() if quote_column else None,
                )
                for row in r
            ]
            urls_and_quotes = [
                (normalize(url), title, quote)
                for (url, title, quote) in urls_and_quotes
                if url.startswith("http")
            ]  # basic sanity check
            urls = []
            quotes = defaultdict(list)
            titles = dict()
            for url, title, quote in urls_and_quotes:
                urls.append(url)
                if quote:
                    quotes[url].append(quote)
                if title:
                    titles[url] = title
            async with Session() as session:
                collections = [await session.merge(c) for c in collections]
                documents = await uri_equivalence.add_documents(
                    session, urls, collections
                )
                for uri, doc in documents.items():
                    if doc.id:
                        logger.warning(f"Already existing URL: {uri}")
                    else:
                        doc.created_by = current_user.id
                    if not doc.title:
                        doc.title = titles.get(uri)
                if quotes or titles:
                    quotes = list(
                        chain(
                            *[
                                [
                                    Fragment(
                                        text=quote,
                                        language="en",
                                        document=documents[uri],
                                        scale=fragment_type.quote,
                                    )
                                    for quote in deduplicate(dquotes)
                                ]
                                for (uri, dquotes) in quotes.items()
                            ]
                        )
                    )
                    # TODO: maybe attribute quotes to paragraphs for existing documents
                    session.add_all(quotes)
                await session.commit()
            success = f"{len(documents)} documents added"
        elif upload_type == "jsonl":
            if not file:
                raise BadRequest("File is required")
            fs = file.file
            r = TextIOWrapper(fs, "utf-8")
            published_field = "date_published"
            urls = set()
            docs = []
            async with Session() as session:
                collections = [await session.merge(c) for c in collections]
                for line in r:
                    data = json.loads(line)
                    url = normalize(compose_url_jsonl(data, url_spec))
                    if url in urls:
                        continue
                    urls.add(url)
                    text = get_text_jsonl(
                        data, text_fields, [[r" \n\n", " "]] if extra_newlines else None
                    )
                    title = data.get("title") if use_title else None
                    published = None
                    if use_published:
                        published = data[published_field]
                        if published == "n/a":
                            published = None
                        if published:
                            if " " in published:
                                published = "T".join(published.split())
                            if "T" in published:
                                published = isodate.parse_datetime(published)
                            else:
                                published = isodate.parse_date(published)
                    if isinstance(text, list):
                        text = "\n".join(text)
                    existing = await session.scalar(
                        select(count(UriEquiv.uri)).filter_by(uri=url)
                    )
                    if existing:
                        continue
                    json_as_file = hashfs.put(BytesIO(line.encode("utf-8")))
                    txt_as_file = hashfs.put(BytesIO(text.encode("utf-8")))
                    docs.append(
                        Document(
                            uri=UriEquiv(uri=url),
                            created_by=current_user.id,
                            title=title,
                            collections=collections,
                            file_identity=json_as_file.id,
                            file_size=Path(json_as_file.abspath).stat().st_size,
                            text_identity=txt_as_file.id,
                            text_size=Path(txt_as_file.abspath).stat().st_size,
                            mimetype="text/plain",
                            created=published,
                            return_code=200,
                        )
                    )
                for doc in docs:
                    session.add(doc)
                await session.commit()
            success = f"{len(docs)} documents added"
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.exception("", exc_info=e)
        error = str(e)

    return templates.TemplateResponse(
        request,
        "upload_docs.html",
        dict(
            error=error,
            success=success,
            success_doc_id=success_doc_id,
            new_ids=new_ids,
            **base_vars,
        ),
    )


@app_router.get("/document/{doc_id}/completion_prompts")
@app_router.get("/c/{collection}/document/{doc_id}/completion_prompts")
async def as_completions(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    doc_id: int,
    collection: Optional[str] = None,
):
    io = BytesIO()
    async with Session() as session:
        collection_ob = await get_collection(collection, session, current_user.id)
        await check_doc_access(current_user, doc_id, collection_ob)
        fragment_query = (
            select(Fragment.text)
            .filter(
                Fragment.doc_id == doc_id, Fragment.scale == fragment_type.paragraph
            )
            .order_by(Fragment.position)
        )
        r = await session.execute(fragment_query)
        for (para,) in r:
            for prompt, completion in as_prompts(para):
                io.write(json.dumps(dict(prompt=prompt, completion=completion)))
                io.write("\n")
    io.flush()
    io.seek(0)
    io = BytesIO(io.read())  # WHY IS THIS NECESSARY?
    return send_as_attachment(io, "application/json-l", f"prompts_{doc_id}.jsonl")


async def get_generated_claims_by_para(
    session, paras_by_id
) -> Dict[int, List[Tuple[Statement, Analysis]]]:
    if not paras_by_id:
        return {}
    doc_id = next(iter(paras_by_id.values())).doc_id
    claim = aliased(Statement, name="claim")
    para = aliased(Fragment, name="para")
    results = await session.execute(
        select(claim, Analysis)
        .join(Analysis, claim.from_analyses)
        .join(para, Analysis.target)
        .filter_by(doc_id=doc_id)
        .options(joinedload(Analysis.analyzer))
    )
    generated_claim_by_para = defaultdict(set)
    for claim, analysis in results:
        if analysis.analyzer.name == "extract_claims":
            for r in (claim.generation_data or {}).get("origin", ()):
                if (claim_source := r.get("source", None)) in paras_by_id:
                    generated_claim_by_para[claim_source].add((claim, Analysis))
        else:
            for r in analysis.results or []:
                if not isinstance(r, dict):
                    continue
                if r.get("fragment_id") == claim:
                    for p in r.get("sources", ()):
                        if p in paras_by_id:
                            generated_claim_by_para[claim_source].add((claim, Analysis))
    generated_claim_by_para: Dict[int, List[Tuple[Statement, Analysis]]] = {
        id: list(g) for (id, g) in generated_claim_by_para.items()
    }
    for cl in generated_claim_by_para.values():
        cl.sort(key=lambda c: c[0].id)

    return generated_claim_by_para


@app_router.get("/document/{doc_id}")
@app_router.get("/c/{collection}/document/{doc_id}")
async def get_doc_info(
    request: Request,
    current_user: user_with_coll_permission_c_dep("access"),
    doc_id: int,
    collection: Optional[str] = None,
    order: str = "para",
):
    inverted = order[0] == "-"
    if inverted:
        order = order[1:]
    # logger.debug(order, inverted)
    async with Session() as session:
        base_vars = await get_base_template_vars(
            request, current_user, collection, session
        )
        collection_ob: Collection = base_vars["collection"]
        await check_doc_access(current_user, doc_id, collection_ob)
        doc = await session.get(Document, doc_id)
        # TODO: Add the claims
        if doc is None:
            raise NotFound()
        tasks = []
        async for task_data in TaskRegistry.task_registry.all_task_status(
            session, doc, collection
        ):
            tasks.append(task_data)
            base_vars |= await task_data[0].task_form_args(session)
        public_contents = doc.public_contents or current_user.can("admin")
        # Count available embeddings
        num_embeddings = dict()
        for model, Embedding in embed_db_model_by_name.items():
            frag_embedding = aliased(Embedding)
            doc_embedding = aliased(Embedding)
            q = (
                select(
                    Document.id,
                    count(frag_embedding.fragment_id)
                    + count(doc_embedding.doc_id.distinct()),
                )
                .outerjoin(Fragment, Fragment.doc_id == Document.id)
                .outerjoin(frag_embedding, frag_embedding.fragment_id == Fragment.id)
                .outerjoin(
                    doc_embedding,
                    (doc_embedding.doc_id == Document.id)
                    & (doc_embedding.fragment_id.is_(None)),
                )
                .group_by(Document.id)
                .filter(Document.id == doc_id)
            )
            r = await session.execute(q)
            (id, emb_count) = r.first()
            num_embeddings[model] = emb_count
        has_embedding = bool(sum(num_embeddings.values()))

        # Get the paragraphs themselves
        source = aliased(Fragment)
        key_point = aliased(Fragment)
        key_point_doc = aliased(Document)
        para = aliased(Fragment)
        fragment_query = select(Fragment).filter(
            Fragment.doc_id == doc_id, Fragment.scale == fragment_type.paragraph
        )
        if order == "para":
            fragment_query = fragment_query.order_by(Fragment.position)
            if not public_contents:
                fragment_query = (
                    fragment_query.outerjoin(Analysis, Fragment.id == Analysis.theme_id)
                    .outerjoin(
                        ClaimLink,
                        (ClaimLink.target == Fragment.id)
                        | (ClaimLink.source == Fragment.id),
                    )
                    .filter(
                        (ClaimLink.source.is_not(None))
                        | (Analysis.theme_id.is_not(None))
                    )
                    .distinct()
                )
        else:
            order_col = cast(Analysis.results[order], Float)
            fragment_query = (
                fragment_query.join(Analysis, Fragment.id == Analysis.theme_id)
                .filter(coalesce(cast(Analysis.params[order], Boolean), False))
                .group_by(Fragment.primary_key)
            )
            if inverted:
                fragment_query = fragment_query.order_by(fmin(order_col))
            else:
                fragment_query = fragment_query.order_by(fmax(order_col).desc())
        r = await session.execute(fragment_query)
        paras = [para for (para,) in r]
        num_fragments = len(paras)

        # Get claim quality analysis
        analysis_query = (
            select(Analysis)
            .join(source, source.id == Analysis.theme_id)
            .filter(source.doc_id == doc_id)
        )
        if order != "para":
            analysis_query = analysis_query.filter(
                coalesce(cast(Analysis.params[order], Boolean), False)
            )
        r = await session.execute(analysis_query)
        analyses = defaultdict(list)
        theme_ids = set()
        for (analysis,) in r:
            analyses[analysis.theme_id].append(analysis)
            theme = analysis.params.get("theme")
            if theme:
                theme_ids.add(int(theme))
        r = await session.execute(select(Fragment).filter(Fragment.id.in_(theme_ids)))
        themes = {t.id: t for (t,) in r}

        # Get boundaries
        boundaries_query = (
            select(Fragment)
            .outerjoin(
                ClaimLink,
                (ClaimLink.source == Fragment.id) | (ClaimLink.target == Fragment.id),
            )
            .filter(
                Fragment.doc_id == doc_id,
                Fragment.scale == fragment_type.generated,
                ClaimLink.source.is_(None),
            )
        )
        if order != "para":
            boundaries_query = boundaries_query.join(
                Analysis, Fragment.part_of == Analysis.theme_id
            ).filter(coalesce(cast(Analysis.params[order], Boolean), False))
        r = await session.execute(boundaries_query.order_by(Fragment.part_of))
        boundaries = defaultdict(list)
        for (f,) in r:
            boundaries[f.part_of].append((None, f))
        ids = set(boundaries.keys())
        spans = {id: list(boundaries.get(id, ())) for id in ids}
        renderings = {
            p.id: render_with_spans(p.text, spans.get(p.id, [])) for p in paras
        }

        # Get generated claims
        paras_by_id = {p.id: p for p in paras}
        generated_claim_by_para = await get_generated_claims_by_para(
            session, paras_by_id
        )

        # Get the summaries
        q = (
            select(Statement)
            .filter_by(doc_id=doc_id, scale=fragment_type.summary)
            .options(subqueryload(Statement.from_analyses))
        )
        logger.debug("%s", q)
        summaries = [s for (s,) in await session.execute(q)]
        for summary in summaries:
            await session.refresh(
                summary, ["from_analyses"]
            )  # Why does the subqueryload fail?
            # print(summary.from_analyses)

        return templates.TemplateResponse(
            request,
            "doc_info.html",
            dict(
                doc=doc,
                has_embedding=has_embedding,
                num_fragments=num_fragments,
                order=order,
                tasks=tasks,
                num_frag_embeddings=num_embeddings,
                paras=paras,
                analyses=analyses,
                themes=themes,
                public_contents=public_contents,
                generated_claim_by_para=generated_claim_by_para,
                summary_analyses=[],
                summaries=summaries,
                renderings=renderings,
                **base_vars,
            ),
        )
