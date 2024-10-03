from typing import Union, Literal, Optional, List
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from logging import getLogger

from sqlalchemy import select, delete
from sqlalchemy.sql import and_
from sqlalchemy.orm import joinedload, subqueryload
from sqlalchemy.orm.attributes import flag_modified
import pydantic
import httpx
import orjson as json

from .. import hashfs, config, Session
from ..pyd_models import fragment_type, uri_status
from ..models import Document, Analysis, Fragment
from ..uri_equivalence import make_uri_group
from ..utils import run_sync, safe_lang_detect
from .process_html import extract_readable_text_v3
from .process_paragraphs import store_processed_para_data
from .tasks import (
    FeedlyAnalysisModel,
    FeedlyTaskTemplateModel,
    DownloadTask,
    DownloadAnalysisModel,
)
from .kafka import sentry_sdk

logger = getLogger(__name__)


class FeedlyContent(pydantic.BaseModel):
    content: str
    direction: Union[Literal["ltr"], Literal["rtl"], None] = None


class FeedlyOrigin(pydantic.BaseModel):
    streamId: str
    title: Optional[str] = None
    htmlUrl: Optional[str] = None


class FeedlyAltUrl(pydantic.BaseModel):
    href: str
    type: str


class FeedlyArticle(pydantic.BaseModel):
    id: str
    canonicalUrl: Optional[str] = None
    language: Optional[str] = None
    content: Optional[FeedlyContent] = None
    summary: Optional[FeedlyContent] = None
    title: Optional[str] = None
    author: Optional[str] = None
    crawled: datetime
    recrawled: Optional[datetime] = None
    updated: Optional[datetime] = None
    published: Optional[datetime] = None
    origin: FeedlyOrigin
    alternate: List[FeedlyAltUrl] = []


async def process_article(
    raw_article,
    session,
    analysis: Analysis,
    post_process_text=True,
    include_hallucinations=True,
):
    article = FeedlyArticle.model_validate(raw_article)
    url = article.canonicalUrl or (
        f"{article.origin.htmlUrl}#{article.id}" if article.origin.htmlUrl else None
    )
    if not url:
        return
    # Record the ID in a urn
    urn = f"urn:feedly:{article.id}"
    alts = [(uri_status.alt, alt.href) for alt in article.alternate]
    alts.append((uri_status.urn, urn))
    base_url, alt_urls = await make_uri_group(session, url, alts)
    doc = await session.scalar(
        select(Document)
        .filter_by(uri=base_url)
        .options(subqueryload(Document.from_analyses))
    )
    created = article.published.replace(tzinfo=None) if article.published else None
    modified = article.updated.replace(tzinfo=None) if article.updated else None
    crawled = article.recrawled or article.crawled
    retrieved = crawled.replace(tzinfo=None) if crawled else None
    if doc:
        if doc.retrieved >= (modified or retrieved):
            logger.info(f"Not reprocessing {doc.id}")
            return
        doc.title = article.title or doc.title
        doc.language = article.language
        doc.created = created
        doc.modified = modified
        if analysis not in doc.from_analyses:
            doc.from_analyses = doc.from_analyses + [analysis]
    else:
        doc = Document(
            uri=base_url,
            title=article.title,
            language=article.language,
            meta={},
            created=created,
            modified=modified,
            from_analyses=[analysis],
        )
    if article.author:
        doc.meta = doc.meta | dict(author=article.author)
    if doc.meta.pop("summary", None):
        flag_modified(doc, "meta")
    if article.summary:
        summary_text = article.summary.content
        if "<p" in summary_text or "<span" in summary_text:
            _, _, text, _ = await run_sync(extract_readable_text_v3)(summary_text)
            summary_text = text

        if not (summary_language := doc.language):
            summary_language = safe_lang_detect(summary_text)

        if doc.id:
            await session.refresh(doc, ["summary"])
            for summary in doc.summary:
                if summary.text == summary_text:
                    break
            else:
                summary = Fragment(
                    document=doc,
                    text=summary_text,
                    scale=fragment_type.summary,
                    language=summary_language,
                    from_analyses=[analysis],
                )
                session.add(summary)

    if article.content:
        doc.retrieved = retrieved
        doc.return_code = 200
        doc.mimetype = "application/feedly+json"
        file_identity = hashfs.put(BytesIO(json.dumps(raw_article)))
        new_content = file_identity.id != doc.file_identity
        if doc.file_identity and new_content:
            hashfs.delete(doc.file_identity)  # assumed unique
        if new_content:
            doc.file_identity = file_identity.id
            doc.file_size = Path(file_identity.abspath).stat().st_size
        content = article.content.content
        soup, main, text, paras = await run_sync(extract_readable_text_v3)(content)
        paragraphs = []
        if not paragraphs:
            char_pos = 0
            for pos, para in enumerate(paras):
                paragraphs.append((pos, char_pos, para, []))
                char_pos += len(para) + 2
        text_identity = hashfs.put(BytesIO(text.encode("utf-8")))
        new_text = text_identity.id != doc.text_identity
        doc.language = article.language or safe_lang_detect(text)

        if doc.text_identity and new_text:
            hashfs.delete(doc.text_identity)  # assumed unique
            await session.execute(
                delete(Fragment).where(
                    and_(
                        Fragment.doc_id == doc.id,
                        Fragment.scale == fragment_type.paragraph,
                    )
                )
            )
            # TODO: maybe cascade to generated claims?
        if new_text:
            doc.text_identity = text_identity.id
            doc.text_size = Path(text_identity.abspath).stat().st_size
            doc.text_analyzer_id = analysis.analyzer_id
            await store_processed_para_data(session, doc, paragraphs)
    else:
        # Do not fill the content, we'll have a download request
        # Annoying: We're not keeping the feedly json in that case. Not vital.
        # Repair previous code
        if doc.mimetype == "application/feedly+json":
            doc.mimetype = None
            hashfs.delete(doc.text_identity)
            hashfs.delete(doc.file_identity)
            doc.file_identity = None
            doc.text_identity = None
            doc.return_code = None
            await session.execute(
                delete(Fragment).where(
                    and_(
                        Fragment.doc_id == doc.id,
                        Fragment.scale == fragment_type.paragraph,
                    )
                )
            )
            if doc.id:
                await session.commit()
                logger.info(f"Re-sending doc {doc.id} for download")
                task = DownloadTask(DownloadAnalysisModel(target_id=doc.id))
                await task.schedule()

    await session.commit()


async def do_feedly_feed(analysis_id: int, unread_only=False, keep_going=True):
    token = config.get("feedly", "token")
    assert token
    async with Session() as session:
        analysis = await session.get(
            Analysis, analysis_id, options=[joinedload(Analysis.task_template)]
        )
        assert analysis
        analysis_model: FeedlyAnalysisModel = analysis.as_model(session)
        analysis_template: FeedlyTaskTemplateModel = analysis_model.task_template
        newerThan = analysis_model.completed
        if newerThan and (
            datetime.utcnow() - datetime.utcfromtimestamp(newerThan)
        ).days > timedelta(days=31):
            newerThan = None
        params = dict(streamId=analysis_template.stream_id, unreadOnly=unread_only)
        continuation = None
        nextNewerThan = datetime.utcnow()
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}
        ) as client:
            while True:
                if continuation:
                    params["continuation"] = continuation
                    params.pop("newerThan", None)
                elif newerThan:
                    params["newerThan"] = int(newerThan.timestamp())
                r = await client.get(
                    "https://feedly.com/v3/streams/contents", params=params
                )
                assert r.status_code == 200
                results = r.json()
                for article in results["items"]:
                    await process_article(
                        article,
                        session,
                        analysis,
                        post_process_text=analysis_template.post_process_text,
                        include_hallucinations=analysis_template.include_hallucinations,
                    )
                if not (keep_going and (continuation := results.get("continuation"))):
                    break

    return nextNewerThan
