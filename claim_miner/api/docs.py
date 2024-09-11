from typing import Optional, List
from io import StringIO

from sqlalchemy import select, String, cast
from sqlalchemy.orm import joinedload, subqueryload
from fastapi.responses import FileResponse, ORJSONResponse, StreamingResponse

from .. import Session, hashfs
from ..app import NotFound, Forbidden, BadRequest
from ..auth import user_with_coll_permission_t_dep
from ..models import Document, UriEquiv, Analysis, Fragment, Statement
from ..uri import normalize
from ..pyd_models import DocumentModel, FragmentModel, AnalysisModel
from ..task_registry import TaskRegistry
from . import api_router, get_collection


async def get_document_object(session, doc_id, collection=None, options=None):
    document = await session.get(Document, doc_id, options=options)
    if not document:
        raise NotFound()
    if collection:
        await session.refresh(document, ["collections"])
        for coll in document.collections:
            if coll.name == collection:
                break
        else:
            raise Forbidden()
    return document


@api_router.get("/document/{doc_id}")
@api_router.get("/c/{collection}/document/{doc_id}")
async def get_document(
    doc_id: int,
    current_user: user_with_coll_permission_t_dep("access"),
    collection: Optional[str] = None,
) -> DocumentModel:
    async with Session() as session:
        document = await get_document_object(session, doc_id, collection)
        return document.as_model(session)


@api_router.get("/document/{doc_id}/paragraphs")
@api_router.get("/c/{collection}/document/{doc_id}/paragraphs")
async def get_document_paragraphs(
    doc_id: int,
    current_user: user_with_coll_permission_t_dep("access"),
    collection: Optional[str] = None,
) -> List[FragmentModel]:
    async with Session() as session:
        document = await get_document_object(
            session,
            doc_id,
            collection,
            [subqueryload(Document.paragraphs).subqueryload(Fragment.sub_parts)],
        )
        return [p.as_model(session) for p in document.paragraphs]


@api_router.get("/document/{doc_id}/quotes")
@api_router.get("/c/{collection}/document/{doc_id}/quotes")
async def get_document_quotes(
    doc_id: int,
    current_user: user_with_coll_permission_t_dep("access"),
    collection: Optional[str] = None,
) -> List[FragmentModel]:
    async with Session() as session:
        document = await get_document_object(
            session, doc_id, collection, [subqueryload(Document.quotes)]
        )
        return [p.as_model(session) for p in document.quotes]


@api_router.get("/document/{doc_id}/generated")
@api_router.get("/c/{collection}/document/{doc_id}/generated")
async def get_document_generated_claims(
    doc_id: int,
    current_user: user_with_coll_permission_t_dep("access"),
    collection: Optional[str] = None,
) -> List[AnalysisModel]:
    registry = TaskRegistry.get_registry()
    # TODO: Generalize this code, use recursive queries
    analyzer = registry.analyzer_by_name["extract_claims"]
    async with Session() as session:
        q = select(Analysis.id).filter_by(target_id=doc_id, analyzer_id=analyzer.id)
        if collection:
            q = q.join(Analysis.collection).filter_by(name=collection)
        doc_analysis = list(await session.scalars(q))
        if not doc_analysis:
            return []
        chunk_analyses = list(
            await session.scalars(
                select(Analysis.id).filter(Analysis.part_of_id.in_(doc_analysis))
            )
        )
        if not chunk_analyses:
            return []
        analyses = list(
            await session.scalars(
                select(Analysis)
                .filter(
                    Analysis.part_of_id.in_(chunk_analyses)
                    # ).filter(cast(Analysis.params['scale'], String)=='result'
                )
                .options(
                    joinedload(Analysis.target.of_type(Fragment)),
                    subqueryload(Analysis.generated_topics.of_type(Statement)),
                )
            )
        )
    return [a.as_model(session) for a in analyses]


@api_router.get("/document")
@api_router.get("/c/{collection}/document")
async def get_documents(
    current_user: user_with_coll_permission_t_dep("access"),
    offset: int = 0,
    limit: int = 20,
    collection: Optional[str] = None,
) -> List[DocumentModel]:
    async with Session() as session:
        collection_ob = await get_collection(collection, session, current_user.id)
        if collection and not collection_ob:
            raise NotFound()
        base_query = select(Document.id).order_by(Document.id)
        if collection:
            base_query = base_query.join(Document.collections).filter_by(
                name=collection
            )
        base_query = base_query.offset(offset).limit(limit)
        r = await session.execute(
            select(Document).filter(Document.id.in_(base_query.subquery()))
        )
        # TODO: Add related objects.
        return [f.as_model(session) for (f,) in r]


@api_router.get("/document/{doc_id}/raw")
async def get_doc_raw(
    current_user: user_with_coll_permission_t_dep("access"),
    doc_id: int,
    collection: Optional[str] = None,
) -> FileResponse:
    async with Session() as session:
        document = await get_document_object(session, id, collection)
        if not document.file_identity:
            raise NotFound()
        if not (document.public_contents or current_user.can("admin")):
            raise Forbidden("Copyrighted content")
        file_info = hashfs.get(document.file_identity)
        extension = document.mimetype.split("/")[1]
        return FileResponse(
            file_info.abspath,
            media_type=document.mimetype,
            filename=f"{doc_id}.{extension}",
        )


@api_router.get("/document/{doc_id}/text")
async def get_doc_text(
    current_user: user_with_coll_permission_t_dep("access"),
    doc_id: int,
    collection: Optional[str] = None,
) -> FileResponse:
    async with Session() as session:
        document = await get_document_object(session, doc_id, collection)
        if not document.text_identity:
            raise NotFound()
        if not (document.public_contents or current_user.can("admin")):
            raise Forbidden("Copyrighted content")
        file_info = hashfs.get(document.text_identity)
        return FileResponse(
            file_info.abspath, media_type="plain/text", filename=f"{doc_id}.txt"
        )


@api_router.get("/document/{doc_id}/processed_text")
async def get_doc_processed_text(
    current_user: user_with_coll_permission_t_dep("access"),
    doc_id: int,
    collection: Optional[str] = None,
) -> FileResponse:
    async with Session() as session:
        document = await get_document_object(
            session,
            doc_id,
            collection,
            [subqueryload(Document.paragraphs).subqueryload(Fragment.sub_parts)],
        )
        if not document.text_identity:
            raise NotFound()
        if not (document.public_contents or current_user.can("admin")):
            raise Forbidden("Copyrighted content")
        text = "\n\n".join(
            [
                "\n".join([p2.text for p2 in p.sub_parts]) if p.sub_parts else p.text
                for p in document.paragraphs
            ]
        )

        return StreamingResponse(StringIO(text), 200, media_type="plain/text")


@api_router.post("/c/{collection}/document", status_code=201)
async def add_doc(
    current_user: user_with_coll_permission_t_dep("add_document"),
    collection: str,
    document: DocumentModel,
) -> DocumentModel:
    # TODO: Allow upload
    async with Session() as session:
        collection_ob = await get_collection(collection, session, current_user.id)
        url = normalize(document.url)
        r = await session.scalar(select(UriEquiv.id).filter_by(uri=url).limit(1))
        # TODO: UX to load a new snapshot of an existing document
        if r is not None:
            raise BadRequest("Document with this URL already exists")
        uri = UriEquiv(uri=url)  # tentatively canonical?
        doc = await Document.from_model(
            session,
            document,
            uri=uri,
            created_by=current_user.id,
            collections=[collection_ob],
        )
        session.add(doc)
        await session.commit()
        document = doc.as_model(session)
    location = f"/api/c/{collection}/document/{doc.id}"
    return ORJSONResponse(
        document.model_dump(mode="json"),
        status_code=201,
        headers=dict(location=location),
    )
