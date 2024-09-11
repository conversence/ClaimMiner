"""
Copyright Society Library and Conversence 2022-2024
"""

from io import BytesIO
from pathlib import Path
import re
from itertools import chain
from logging import getLogger

from bs4 import BeautifulSoup, NavigableString

from .. import Session, hashfs, run_sync
from ..models import Document
from ..utils import safe_lang_detect
from .kafka import sentry_sdk
from .tasks import ProcessHtmlTask
from .process_paragraphs import store_processed_para_data


def collapse_whitespace_and_paras(s):
    s = re.sub(r"[ \t\f\v\xa0]+", " ", s)
    return re.sub(r"([ \t\f\v\xa0]?\n[ \t\f\v\xa0]?)+", "\n", s).strip()


def collapse_whitespace_with_paras(s):
    return re.sub(r"\s+", " ", s).strip()


MIN_PARAGRAPH_LENGTH = 120
logger = getLogger(__name__)


def extract_readable_text_v1(content):
    soup = BeautifulSoup(content, "lxml")
    main = soup.find_all("article") or soup.find_all("main") or [soup]
    text = " ".join(x.get_text() for x in main)
    # TODO: What if  a lot of text is in <li> which does not nest with <p>?
    # What about blockquote etc?
    paras = list(chain(*[x.find_all("p") for x in main]))
    paras = [p.get_text() for p in paras]
    # heuristics
    use_paras = sum(len(p) for p in paras) >= 0.8 * len(text)
    if use_paras:
        paras = [collapse_whitespace_with_paras(p) for p in paras]
    else:
        text = collapse_whitespace_and_paras(text)
        paras = text.split("\n")
    text = "\n\n".join(paras)

    return soup, main, text, paras


unwanted_tags = ["script", "style", "header", "footer", "nav", "form"]
content_tags = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div", "span"]
content_tags_set = set(content_tags)


def extract_readable_text_v3_rec(soup, in_div=False):
    tag = getattr(soup, "name", None)
    if tag == "div":
        # Some divs are used as spans. Heuristic
        divlen = 0
        for sub in getattr(soup, "contents", ()):
            if isinstance(sub, NavigableString):
                yield sub
                divlen += len(sub)
            else:
                subt = list(extract_readable_text_v3_rec(sub, True))
                divlen += sum(len(t) for t in subt)
                yield from subt
        if divlen > 75 or not in_div:
            yield "\n"
    elif (tag := getattr(soup, "name", None)) in content_tags_set:
        yield from soup.stripped_strings
        if tag != "span":
            yield "\n"
    else:
        for sub in getattr(soup, "contents", ()):
            yield from extract_readable_text_v3_rec(sub, in_div)


def deduplicate_cr(g):
    was_cr = False
    for i in g:
        is_cr = i == "\n"
        if not (was_cr and is_cr):
            yield i
        was_cr = is_cr


def extract_readable_text_v3(content):
    soup = BeautifulSoup(content, "lxml")
    # Filter out script and style elements
    for unwanted in soup(unwanted_tags):
        unwanted.decompose()
    main = soup.find_all("article") or soup.find_all("main") or [soup]
    blocks = []
    for subsoup in soup:
        blocks.extend(list(extract_readable_text_v3_rec(subsoup)))
    blocks = list(deduplicate_cr(blocks))
    text = "".join(blocks)
    paras = text.split("\n")
    return soup, main, text, paras


async def do_process_html(
    doc_id, post_process_text: bool = True, include_hallucinations: bool = True
):
    analyzer_id = await ProcessHtmlTask.get_analyzer_id()
    fragments = []
    new_data = False
    async with Session() as session:
        doc = await session.get(Document, doc_id)
        if not doc:
            logger.error(f"Missing document {doc_id}")
            return False, []
        if doc.text_identity:
            # If you want reparsing, delete the data
            return True

        file = hashfs.get(doc.file_identity)

        with open(file.abspath, "r") as f:
            content = f.read()

        soup, main, text, paras = await run_sync(extract_readable_text_v3)(content)

        if tel := soup.find("title"):
            doc.title = tel.text

        doc.language = safe_lang_detect(text)
        paragraphs = []
        char_pos = 0
        if not paras:
            soup, main, text, paras = await run_sync(extract_readable_text_v1)(content)
        for pos, para in enumerate(paras):
            if len(para) >= MIN_PARAGRAPH_LENGTH:
                paragraphs.append((pos, char_pos, para, []))
            char_pos += len(para) + 2
        await store_processed_para_data(session, doc, paragraphs)
        text_address = hashfs.put(BytesIO(text.encode("utf-8")))

        doc.text_identity = text_address.id
        doc.text_size = Path(text_address.abspath).stat().st_size
        doc.text_analyzer_id = analyzer_id
        await session.commit()
    return True
