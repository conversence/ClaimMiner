"""
Copyright Society Library and Conversence 2022-2024
"""

from pathlib import Path
import re
from io import BytesIO
from logging import getLogger

import orjson as json

from .. import Session, hashfs
from ..models import Document
from ..utils import safe_lang_detect
from .kafka import sentry_sdk
from .tasks import ProcessTextTask
from .process_paragraphs import store_processed_para_data

logger = getLogger(__name__)

MIN_PARAGRAPH_LENGTH = 120


def collapse_whitespace_and_paras(s):
    s = re.sub(r"[ \t\f\v\xa0]+", " ", s)
    return re.sub(r"([ \t\f\v\xa0]?\n[ \t\f\v\xa0]?)+", "\n", s).strip()


def collapse_whitespace_with_paras(s):
    return re.sub(r"\s+", " ", s).strip()


async def do_process_text(
    doc_id, post_process_text: bool = True, include_hallucinations: bool = True
):
    analyzer_id = await ProcessTextTask.get_analyzer_id()
    async with Session() as session:
        doc = await session.get(Document, doc_id)
        if not doc:
            logger.error(f"Missing document {doc_id}")
            return False, []
        if doc.text_identity:
            # If you want reparsing, delete the data
            return True
        with hashfs.open(doc.file_identity, "r") as f:
            text = f.read()
        identical = True
        title = None
        # Special case for old data that was miscategorized
        if text.startswith("{"):
            try:
                data = json.loads(text)
                doc.mimetype = "application/json"
                sub_text = data.get("text", "")
                title = data.get("title", None)
                comments = data.get("comments", [])
                if comments:
                    ctext = "\n\n".join([c.get("text", "") for c in comments])
                    sub_text = "\n\nComments:\n\n".join((sub_text, ctext))
                if len(sub_text) / len(text) > 0.4 or len(text) > 40:
                    text = sub_text
                    identical = False
            except Exception:
                pass
        doc.language = safe_lang_detect(text)
        paragraphs = []
        paras = text.split("\n\n")
        if len(paras) < 3:
            paras = text.split("\n")
        char_pos = 0
        for pos, para in enumerate(paras):
            if len(para) >= MIN_PARAGRAPH_LENGTH:
                paragraphs.append((pos, char_pos, para, []))
            char_pos += len(para) + 1
        if identical:
            doc.text_identity = doc.file_identity
            doc.text_size = doc.file_size
        else:
            text_address = hashfs.put(BytesIO(text.encode("utf-8")))
            doc.text_identity = text_address.id
            doc.text_size = Path(text_address.abspath).stat().st_size
        await store_processed_para_data(session, doc, paras=paragraphs)
        doc.text_analyzer_id = analyzer_id
        doc.title = title
        await session.commit()
    return True
