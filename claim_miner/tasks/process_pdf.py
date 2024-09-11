"""
Copyright Society Library and Conversence 2022-2024
"""

import re
from logging import getLogger
from io import BytesIO
from pathlib import Path

from pypdf import PdfReader

from .. import Session, hashfs, run_sync
from ..models import Document
from ..utils import safe_lang_detect
from .kafka import sentry_sdk
from .tasks import ProcessPdfTask
from .process_paragraphs import store_processed_para_data

logger = getLogger(__name__)

MIN_PARAGRAPH_LENGTH = 120


def extract_text(path):
    with open(path, "rb") as f:
        reader = PdfReader(f)
        return reader.metadata, "\n".join(
            reader.pages[i].extract_text() for i in range(len(reader.pages))
        )


async def do_process_pdf(
    doc_id, post_process_text: bool = True, include_hallucinations: bool = True
):
    new_data = False
    analyzer_id = await ProcessPdfTask.get_analyzer_id()
    async with Session() as session:
        doc = await session.get(Document, doc_id)
        if not doc:
            logger.error(f"Missing document {doc_id}")
            return False
        if doc.text_identity:
            # If you want reparsing, delete the data
            return True
        f = hashfs.get(doc.file_identity)

        def do_extract():
            return extract_text(f.abspath)

        metadata, text = await run_sync(do_extract)()
        metadata = dict(metadata or {})
        doc.meta |= metadata
        if "\Title" in metadata:
            doc.title = metadata["\Title"]
        doc.language = safe_lang_detect(text)
        paras = re.split(r"\n\s*\n+", text)
        # Convert single newlines to spaces
        paras = [
            re.sub(r"\s\s+", " ", re.sub(r"\s*\n\s*", " ", p.strip()))
            for p in paras
            if p.strip()
        ]
        text = "\n".join(paras)
        paragraphs = []
        char_pos = 0
        for pos, p in enumerate(paras):
            if len(p) > MIN_PARAGRAPH_LENGTH:
                paragraphs.append((pos, char_pos, p, []))
            char_pos += len(p) + 1
        text_address = hashfs.put(BytesIO(text.encode("utf-8")))
        doc.text_identity = text_address.id
        doc.text_size = Path(text_address.abspath).stat().st_size
        doc.text_analyzer_id = analyzer_id
        session.add(doc)
        await store_processed_para_data(session, doc, paras=paragraphs)
        await session.commit()
    return True
