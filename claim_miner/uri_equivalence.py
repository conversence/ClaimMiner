"""Utility functions for URI equivalence"""
# Copyright Society Library and Conversence 2022-2024

from logging import getLogger
from urllib.parse import unquote
from collections import defaultdict
from typing import Optional, Dict, Tuple, List

from sqlalchemy import update, select, BigInteger
from sqlalchemy.sql.functions import count, func
from sqlalchemy.orm import joinedload, subqueryload

from .pyd_models import (
    uri_status,
)
from .models import (
    UriEquiv,
    Fragment,
    Document,
    Analysis,
    analysis_context_table,
)
from . import Session, hashfs
from .uri import normalize
from .utils import deduplicate

logger = getLogger(__name__)


def is_archive(url):
    return url.startswith("https://web.archive.org/web/")


def url_from_archive_url(url):
    if is_archive(url):
        return unquote(url.split("/", 5)[-1])


async def add_urls(
    session, urls, equivalences: Optional[Dict[str, UriEquiv]] = None
) -> Tuple[List[UriEquiv], List[UriEquiv]]:
    equivalences = equivalences or {}
    urls = list(
        deduplicate(normalize(url) for url in urls if url.startswith("http"))
    )  # basic sanity check
    existing = list(
        await session.scalars(
            select(UriEquiv).filter(
                UriEquiv.uri.in_(urls).options(joinedload(UriEquiv.canonical))
            )
        )
    )
    existing_uris = {u.uri: u for u in existing}
    existing_canonical = {u.uri: u.canonical or u for u in existing}
    urls = [url for url in urls if url not in existing_uris.keys()]

    new_uris = [
        UriEquiv(
            uri=url,
            status="snapshot" if url in equivalences else "canonical",
            canonical=equivalences.get(url, None),
        )
        for url in urls
    ]
    session.add_all(new_uris)
    return new_uris, existing_uris


async def get_existing_documents(
    session, urls
) -> Tuple[Dict[str, Document], Dict[str, UriEquiv]]:
    urls = list(
        deduplicate(normalize(url) for url in urls if url.startswith("http"))
    )  # basic sanity check
    existing = await session.scalars(
        select(UriEquiv)
        .filter(UriEquiv.uri.in_(urls))
        .options(joinedload(UriEquiv.canonical))
    )
    existing_canonical = {u.uri: u.canonical or u for u in existing}
    documents = await session.scalars(
        select(Document)
        .filter(Document.uri_id.in_(u.id for u in existing_canonical.values()))
        .options(subqueryload(Document.collections), joinedload(Document.uri))
    )
    document_by_uri_id = {d.uri_id: d for d in documents}
    docs_by_uri = {
        k: document_by_uri_id[u.id]
        for k, u in existing_canonical.items()
        if u.id in document_by_uri_id
    }
    remaining_uris = {
        k: v for k, v in existing_canonical.items() if k not in docs_by_uri
    }
    return docs_by_uri, remaining_uris


async def add_documents(session, urls, collections=[]) -> Dict[str, Document]:
    urls = list(
        deduplicate(normalize(url) for url in urls if url.startswith("http"))
    )  # basic sanity check
    existing_docs, remaining_uris = await get_existing_documents(session, urls)
    for c in collections:
        for doc in existing_docs.values():
            if doc:
                if c not in doc.collections:
                    doc.collections = doc.collections + [c]
    new_docs1 = {
        url: Document(uri=uri_equiv, collections=collections)
        for url, uri_equiv in remaining_uris.items()
    }
    # count both the given and canonical urls
    existing_urls = (
        set(existing_docs.keys())
        | set(doc.url for doc in existing_docs.values() if doc)
        | set(remaining_uris.keys())
        | set(uri.uri for uri in remaining_uris.values())
    )
    new_uris = [u for u in urls if u not in existing_urls]
    new_docs2 = {
        uri: Document(
            uri=UriEquiv(uri=uri, status=uri_status.canonical), collections=collections
        )
        for uri in new_uris
    }
    session.add_all(list(new_docs1.values()))
    session.add_all(list(new_docs2.values()))
    return new_docs1 | new_docs2 | existing_docs


async def add_variant(session, new_uri, old_uri_eq, status="unknown"):
    # add a new variant to a UriEquiv group
    # Special cases:
    # Status=canonical: de-canonicalize the original
    if status == "unknown":
        if old_uri_eq.status == "urn":
            # Then we're the only URL
            status = "canonical"
        else:
            # TODO: We have to chose one of them as canonical.
            # How? Shortest? But that disfavors permalinks. Punting in general
            status = "alt"
    new_uri_eq = UriEquiv(uri=new_uri, status=status)
    session.add(new_uri_eq)
    if status == "canonical":
        await merge(session, new_uri_eq, old_uri_eq)
    else:
        new_uri_eq.canonical = old_uri_eq


async def make_uri_group(
    session, base_uri: str, alt_uris: List[Tuple[uri_status, str]] = None
):
    alt_uris = alt_uris or []
    base_uri_db = await session.scalar(select(UriEquiv).filter_by(uri=base_uri))
    if base_uri_db:
        base_uri_db.status = uri_status.canonical
    else:
        base_uri_db = UriEquiv(uri=base_uri, status=uri_status.canonical)
        session.add(base_uri_db)
    if old_canonical := base_uri_db.canonical_id:
        base_uri_db.canonical_id = None
        await session.execute(
            update(UriEquiv)
            .where(UriEquiv.canonical_id == old_canonical)
            .values(canonical_id=base_uri_db.id)
        )
        await session.execute(
            update(UriEquiv)
            .where(UriEquiv.id == old_canonical)
            .values(status=uri_status.alt, canonical_id=base_uri_db.id)
        )
    alt_db_uris = []
    if alt_uris:
        await session.flush()
        alt_uris_ = {uri: status for (status, uri) in alt_uris}
        if repetition := alt_uris_.pop(base_uri, None):
            logger.warning(f"Ignoring repetition of {base_uri} in {alt_uris}")
        r = await session.scalars(
            select(UriEquiv).filter(UriEquiv.uri.in_(list(alt_uris_.keys())))
        )
        existing_uris = {u.uri: u for u in r}
        for uri, status in alt_uris_.items():
            if uri_db := existing_uris.get(uri):
                uri_db.canonical_id = base_uri_db.id
                uri_db.status = status
            else:
                uri_db = UriEquiv(uri=uri, status=status, canonical_id=base_uri_db.id)
                session.add(uri_db)
            alt_db_uris.append(uri_db)
        await session.flush()
    return base_uri_db, alt_db_uris


async def merge(session, canonical_uri_eq, old_uri_eq):
    # two URLs are now known to be identical, merge them.
    canonical_uri_eq.canonical_id = None
    if not canonical_uri_eq.id:
        session.add(canonical_uri_eq)
        await session.flush()
    if old_uri_eq.status == "canonical":
        old_uri_eq.status = "alt"
    old_uri_eq.canonical = canonical_uri_eq
    await session.execute(
        update(UriEquiv)
        .where(UriEquiv.canonical_id == old_uri_eq.id)
        .values(canonical_id=canonical_uri_eq.id)
    )


async def doc_in_use(session, doc_id):
    # TODO: Make this a union query?
    r = await session.scalar(select(count(Analysis.id)).filter_by(target_id=doc_id))
    if r:
        return True
    r = await session.scalar(
        select(count(Analysis.id))
        .join(Fragment, Fragment.id == Analysis.theme_id)
        .filter(Fragment.doc_id == doc_id)
    )
    if r:
        return True
    r = await session.scalar(
        select(count(analysis_context_table.analysis_id))
        .join(Fragment, Fragment.id == analysis_context_table.fragment_id)
        .filter(Fragment.doc_id == doc_id)
    )
    if r:
        return True
    subq = (
        select(
            func.jsonb_array_elements(Fragment.generation_data["sources"]).label("id")
        )
        .filter(Fragment.generation_data.is_not(None))
        .cte("sources")
    )
    r = await session.scalar(
        select(count(Fragment.id))
        .join(subq, subq.columns.id.cast(BigInteger) == Fragment.id)
        .filter(Fragment.doc_id == doc_id)
    )
    return r > 0


async def _migrate():
    # One-time migration function for duplicate documents.
    # Assumes that no duplicates are used for prompts
    # which was checked independently.
    async with Session() as session:
        docs = await session.execute(
            select(Document).order_by(
                Document.file_identity.is_(None),
                Document.retrieved.desc(),
                Document.id.desc(),
            )
        )
        docs = [doc for (doc,) in docs]
        by_norm = defaultdict(list)
        for doc in docs:
            by_norm[normalize(doc.url)].append(doc)
        for docs in by_norm.values():
            latest = docs[0]
            for doc in docs[1:]:
                await session.delete(doc.uri)
                await session.delete(doc)
                if doc.file_identity and doc.file_identity != latest.file_identity:
                    hashfs.delete(doc.file_identity)
                if doc.text_identity and doc.text_identity != latest.file_identity:
                    hashfs.delete(doc.text_identity)
        await session.flush()

        for uri, docs in by_norm.items():
            latest = docs[0]
            latest.uri.uri = uri
            latest.uri.status = "canonical"
        await session.commit()


# TODO: Case of an attempted merge because of DOI collision but the text content is clearly different.
# In that case we want to record the document with a ALT-doi URI to be reviewed by the operator. Has to be stored in document.
# Maybe is_archive could become an enum.
