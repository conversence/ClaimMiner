"""
Copyright Society Library and Conversence 2022-2024
"""

from datetime import datetime
from email.utils import parsedate_to_datetime
from io import BytesIO, StringIO
from pathlib import Path
import atexit
from concurrent.futures import ThreadPoolExecutor
import threading
import asyncio
from logging import getLogger

import httpx
from pytz import utc
from sqlalchemy.future import select
from sqlalchemy.orm import joinedload
from selenium import webdriver
from selenium.common.exceptions import WebDriverException

from .. import Session, hashfs, config
from ..models import Document, UriEquiv, uri_status
from .kafka import sentry_sdk
from .tasks import DownloadTask

logger = getLogger(__name__)


def parse_date(date):
    return parsedate_to_datetime(date).astimezone(utc).replace(tzinfo=None)


_drivers = set()
local_drivers = threading.local()


def make_driver():
    service = None
    if driver_path := config.get("base", "chromedriver", fallback=None):
        from selenium.webdriver.chrome.service import Service as ChromeService

        service = ChromeService(executable_path=driver_path)
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    return webdriver.Chrome(service=service, options=options)


def get_local_driver(reset=False):
    global local_drivers, _drivers
    local_driver = getattr(local_drivers, "driver", None)
    if reset or local_driver is None:
        try:
            if local_driver is not None:
                local_driver.quit()
                local_drivers.driver = None
                _drivers.remove(local_driver)
            local_drivers.driver = make_driver()
            _drivers.add(local_drivers.driver)
        except WebDriverException as e:
            if sentry_sdk:
                sentry_sdk.capture_exception(e)
        return local_drivers.driver


num_drivers = config.get("base", "num_download_drivers", fallback=3)

driver_pool = ThreadPoolExecutor(num_drivers, initializer=get_local_driver)


def end_pool():
    global driver_pool, _drivers
    driver_pool.shutdown()
    for driver in _drivers:
        try:
            driver.quit()
        except Exception as e:
            pass


atexit.register(end_pool)


def driver_download_inner(url):
    try:
        reset = False
        for i in range(3):
            driver = get_local_driver(reset)
            if driver:
                break
            reset = True
        else:
            raise RuntimeError("Could not make driver")
        driver.get(url)
        c = driver.page_source
        driver.get("about:blank")  # clear memory
        driver.delete_all_cookies()
    except Exception as e:
        logger.error("Error downloading %s: %s", url, e)
        get_local_driver(reset=True)
        c = None
    return c


async def driver_download(url):
    loop = asyncio.get_running_loop()
    r = await loop.run_in_executor(driver_pool, driver_download_inner, url)
    return r


async def do_download(doc_id):
    analyzer_id = await DownloadTask.get_analyzer_id()
    new_data = False
    async with Session() as session:
        doc = await session.get(Document, doc_id, options=[joinedload(Document.uri)])
        if not doc:
            logger.error("Missing document %d", doc_id)
            return None
        if doc.file_identity:
            logger.warning("Already downloaded %d", doc_id)
            return None
            # If you want to redownload, delete the file_identity and the fragments
        url = doc.url
        content = None
        async with httpx.AsyncClient() as client:
            try:
                r = await client.head(url, follow_redirects=True)
            except httpx.TimeoutException:
                r = httpx.Response(503)
            except httpx.RequestError:
                r = httpx.Response(400)
            except httpx.HTTPStatusError:
                r = httpx.Response(500)
            if r.status_code == 200 and r.request.url != httpx.URL(doc.url):
                url = str(r.request.url)
                original_uri = doc.uri
                new_uri = await UriEquiv.ensure(url, session)
                original_uri.status = uri_status.alt
                original_uri.canonical_id = new_uri.canonical_id or new_uri.id
                if new_uri.id:
                    # Preexisting uri, there must be a document that goes with it
                    new_doc = await session.scalar(
                        select(Document).filter_by(
                            uri_id=new_uri.canonical_id or new_uri.id
                        )
                    )
                    assert new_doc
                    await session.delete(doc)
                    doc = new_doc
                    if doc.file_identity:
                        return
                else:
                    doc.uri = new_uri
            if (
                r.status_code == 200
                and r.headers.get("Content-Type", "").split(";")[0] != "text/html"
            ):
                r = await client.get(url)
                if r.status_code == 200:
                    content = BytesIO(r.content)
        doc.return_code = r.status_code
        if r.status_code == 404 or r.status_code >= 500:
            await session.commit()
            return None
        doc.retrieved = datetime.utcnow()
        if last_modified := r.headers.get("Last-Modified", None):
            doc.modified = parse_date(last_modified)
        doc.mimetype = r.headers.get("Content-Type", "text/html")
        doc.language = r.headers.get(
            "Content-Language", "en"
        )  # may be overriden upon parsing
        doc.etag = r.headers.get("ETag", None)
        if r.status_code in (200, 403) and r.request.method != "GET":
            content = await driver_download(url)
            if content and len(content) > 300:  # eliminate 404, 403 etc.
                content = StringIO(content)
                doc.status = 200
            else:
                content = None
                doc.status = 460  # Not a real status code
        if content:
            # no point streaming because hashfs is not async
            address = hashfs.put(content)
            doc.file_identity = address.id
            doc.file_size = Path(address.abspath).stat().st_size
            if doc.file_size > 1000:
                # Don't play equivalence with stubs
                uri_eq = await session.scalar(
                    select(Document)
                    .join(UriEquiv, Document.uri)
                    .filter(Document.file_identity == address.id, Document.id != doc_id)
                    .limit(1)
                )
                if uri_eq:
                    # TODO: The first one is arbitrarily canonical. How to choose?
                    uri = doc.uri
                    uri.canonical = uri_eq
                    uri.status = uri_status.alt
                    doc.delete()
                    logger.warning(
                        f"Document with this file already exists at URL {uri_eq.uri}"
                    )
        await session.commit()
        base_type = doc.base_type
        return base_type if new_data else None
