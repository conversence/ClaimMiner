"""
Copyright Society Library and Conversence 2022-2024
"""

from pathlib import Path
from datetime import datetime
from logging import getLogger

from google.cloud import bigquery
from sqlalchemy import cast, ARRAY, Float
from sqlalchemy.future import select
from google.auth import load_credentials_from_file

from .. import Session, config, run_sync
from ..pyd_models import fragment_type, embedding_model
from ..models import embed_db_model_by_name, Statement, Document, UriEquiv, Analysis
from .kafka import sentry_sdk
from .tasks import GdeltTask, GdeltAnalysisModel

logger = getLogger(__name__)

client = None
Embedding = embed_db_model_by_name[embedding_model.universal_sentence_encoder_4.name]

if credential_filename := config.get("base", "google_credentials", fallback=None):
    credentials, project_id = load_credentials_from_file(
        Path(__file__).parent.parent.parent.joinpath(credential_filename)
    )
    # see https://cloud.google.com/bigquery/docs/reference/libraries
    # assumes GOOGLE_APPLICATION_CREDENTIALS points in the right place.
    client = bigquery.Client(credentials=credentials)


query_data = {
    "news": {
        "table": "gdelt-bq.gdeltv2.gsg_iatvsentembed",
        "embed_col": "sentEmbed",
        "url_col": "previewUrl",
        "where_": "",
    },
    "docs": {
        "table": "gdelt-bq.gdeltv2.gsg_docembed",
        "embed_col": "docEmbed",
        "url_col": "url",
        "where_": " AND lang='ENGLISH' ",
    },
}


query_template = """
WITH query AS (
select {embed} as sentEmbed
)
SELECT COSINE_DISTANCE(t.embed, query.sentEmbed) sim, t.date, t.url, t.embed
FROM
(
    SELECT {embed_col} embed, date, {url_col} url, ROW_NUMBER() OVER (PARTITION BY url ORDER BY date desc) rn
    FROM `gdelt-bq.gdeltv2.gsg_docembed`
    WHERE model='USEv4' {where_}
) t, query
WHERE rn = 1
order by sim desc limit {limit}
"""


async def do_gdelt(analysis_id: int):
    analyzer_id = await GdeltTask.get_analyzer_id()
    document_ids = []
    async with Session() as session:
        analysis = await session.get(Analysis, analysis_id)
        analysis_model: GdeltAnalysisModel = analysis.as_model(session)
        embed = await session.scalar(
            select(Embedding.embedding)
            .join(Statement, Embedding.fragment_id == Statement.id)
            .filter(Statement.id == analysis_model.target_id)
            .limit(1)
        )
        if embed is None:
            raise RuntimeError("Embedding is missing!")
        terms = dict(**query_data[analysis_model.source])
        if analysis_model.date is not None:
            terms["where_"] += f' AND date >= "{analysis_model.date}"'
        query_text = query_template.format(
            embed=list(embed), limit=analysis_model.limit, **terms
        )

        def analyze():
            return client.query(query_text)

        query_time = datetime.utcnow()
        query_job = await run_sync(analyze)()
        for row in query_job:
            try:
                url = row.url
                r = await session.scalar(
                    select(Document.id).filter(Document.url == url).limit(1)
                )
                if r:
                    continue
                # Assume the gdelt URLs are canonical for now...
                uri = UriEquiv(uri=url, status="canonical")
                document = Document(uri=uri, language="en", from_analyses=[analysis])
                embedding = Embedding(
                    document=document,
                    embedding=cast(row.embed, ARRAY(Float)),
                    scale=fragment_type.document,
                )
                session.add(document)
                session.add(embedding)
                await session.commit()
                document_ids.append(document.id)
            except Exception as e:
                if sentry_sdk:
                    sentry_sdk.capture_exception(e)
                logger.exception("", exc_info=e)
                await session.rollback()
    return query_time, document_ids
