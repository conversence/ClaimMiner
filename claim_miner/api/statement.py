from typing import Optional, List, Annotated, Union

import orjson
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import subqueryload, aliased, joinedload
from fastapi import Request, Query, Form, UploadFile
from fastapi.responses import ORJSONResponse

from .. import Session, dispatcher
from ..app import NotFound, BadRequest, Forbidden
from ..auth import user_with_coll_permission_t_dep
from ..pyd_models import (
    fragment_type,
    StatementModel,
    ClaimLinkModel,
    AnalysisModel,
    to_optional,
)
from ..models import Statement, Fragment, Topic, ClaimLink, Analysis
from ..utils import safe_lang_detect, decode_uuid
from . import api_router, get_collection


@api_router.get("/statement/{id}")
@api_router.get("/c/{collection}/statement/{id}")
async def get_statement(
    id: int,
    current_user: user_with_coll_permission_t_dep("access"),
    collection: Optional[str] = None,
) -> StatementModel:
    # TODO: Replace with fastapi
    async with Session() as session:
        statement = await session.get(Statement, id)
        if not statement:
            raise NotFound()
        if collection:
            await session.refresh(statement, ["collections"])
            for coll in statement.collections:
                if coll.name == collection:
                    break
            else:
                raise Forbidden()
        if isinstance(statement, Fragment):
            # Should I just refuse?
            await session.refresh(statement, ["document"])
        return statement.as_model(session)


@api_router.get("/statement/{id}/outgoing")
@api_router.get("/c/{collection}/statement/{id}/outgoing")
async def get_statement_outgoing_links(
    id: int,
    current_user: user_with_coll_permission_t_dep("access"),
    collection: Optional[str] = None,
) -> List[ClaimLinkModel]:
    # TODO: Replace with fastapi
    async with Session() as session:
        statement = await session.get(Statement, id)
        if not statement:
            raise NotFound()
        if collection:
            await session.refresh(statement, ["collections"])
            for coll in statement.collections:
                if coll.name == collection:
                    break
            else:
                raise Forbidden()
        await session.refresh(statement, ["outgoing_links"])
        return [link.as_model(session) for link in statement.outgoing_links]


@api_router.get("/statement/{id}/incoming")
@api_router.get("/c/{collection}/statement/{id}/incoming")
async def get_statement_incoming_links(
    id: int,
    current_user: user_with_coll_permission_t_dep("access"),
    collection: Optional[str] = None,
) -> List[ClaimLinkModel]:
    # TODO: Replace with fastapi
    async with Session() as session:
        statement = await session.get(Statement, id)
        if not statement:
            raise NotFound()
        if collection:
            await session.refresh(statement, ["collections"])
            for coll in statement.collections:
                if coll.name == collection:
                    break
            else:
                raise Forbidden()
        await session.refresh(statement, ["incoming_links"])
        return [link.as_model(session) for link in statement.incoming_links]


@api_router.post("/statement/{id}/outgoing", status_code=201)
@api_router.post("/c/{collection}/statement/{id}/outgoing", status_code=201)
async def add_statement_outgoing_link(
    id: int,
    current_user: user_with_coll_permission_t_dep("add_claim"),
    link: to_optional(ClaimLinkModel),
    collection: Optional[str] = None,
) -> ClaimLinkModel:
    # TODO: Replace with fastapi
    async with Session() as session:
        statement = await session.get(Statement, id)
        if not statement:
            raise NotFound()
        if collection:
            await session.refresh(statement, ["collections"])
            for coll in statement.collections:
                if coll.name == collection:
                    break
            else:
                raise Forbidden()
        link.source = id
        link.created_by = current_user.id
        claim_link = await ClaimLink.from_model(session, link)
        session.add(claim_link)
        await session.commit()
        return claim_link.as_model(session)


@api_router.post("/statement/{id}/incoming", status_code=201)
@api_router.post("/c/{collection}/statement/{id}/incoming", status_code=201)
async def add_statement_incoming_link(
    id: int,
    current_user: user_with_coll_permission_t_dep("add_claim"),
    link: to_optional(ClaimLinkModel),
    collection: Optional[str] = None,
) -> ClaimLinkModel:
    # TODO: Replace with fastapi
    async with Session() as session:
        statement = await session.get(Statement, id)
        if not statement:
            raise NotFound()
        if collection:
            await session.refresh(statement, ["collections"])
            for coll in statement.collections:
                if coll.name == collection:
                    break
            else:
                raise Forbidden()
        link.target = id
        link.created_by = current_user.id
        claim_link = await ClaimLink.from_model(session, link)
        session.add(claim_link)
        await session.commit()
        return claim_link.as_model(session)


@api_router.get("/statement")
@api_router.get("/c/{collection}/statement")
async def get_statements(
    current_user: user_with_coll_permission_t_dep("access"),
    offset: int = 0,
    limit: int = 20,
    collection: Optional[str] = None,
    scales: List[fragment_type] = Query(None),
    alphabetical: bool = False,
) -> List[StatementModel]:
    async with Session() as session:
        collection_ob = await get_collection(collection, session, current_user.id)
        if collection and not collection_ob:
            raise NotFound()
        base_query = select(Statement.id).order_by(
            Statement.text if alphabetical else Statement.id
        )
        if scales:
            # TODO: exclude document, etc. What of hyperedge?
            base_query = base_query.filter(Statement.scale.in_(scales))
        if collection:
            base_query = base_query.join(Statement.collections).filter_by(
                name=collection
            )
        base_query = base_query.offset(offset).limit(limit)
        r = await session.execute(
            select(Statement).filter(Statement.id.in_(base_query.subquery()))
        )
        # TODO: Add related objects. Analysis mostly?
        return [f.as_model(session) for (f,) in r]


@api_router.post("/c/{collection}/statement", status_code=201)
async def add_statement(
    request: Request,
    current_user: user_with_coll_permission_t_dep("add_claim"),
    collection: str,
    statement: StatementModel,
) -> StatementModel:
    async with Session() as session:
        collection_ob = await get_collection(collection, session, current_user.id)
        if not collection_ob:
            raise NotFound()
        statement.confirmed = statement.confirmed & collection_ob.user_can(
            current_user, "confirm_claim"
        )
        statement.created_by = current_user.id
        statement_db = await session.scalar(
            select(Statement).filter_by(text=statement.text).limit(1)
        )
        if statement_db:
            # TODO check for type mismatch
            await session.refresh(statement_db, ["collections"])
            if collection_ob not in statement_db.collections:
                statement_db.collections = statement_db.collections + [collection_ob]
            status_code = 200
        else:
            if not statement.language:
                statement.language = safe_lang_detect(statement.text)
            statement_db = await Statement.from_model(
                session, statement, collections=[collection_ob]
            )
            session.add(statement_db)
            try:
                await session.commit()
            except IntegrityError as e:
                await session.rollback()
                raise BadRequest("Duplicate text")
            status_code = 201
        # Get the defaults back
        statement = statement_db.as_model(session)
    location = f"/api/c/{collection}/statement/{statement.id}"
    return ORJSONResponse(
        statement.model_dump(mode="json"),
        status_code=status_code,
        headers=dict(location=location),
    )


@api_router.get("/statement/check/")
async def check_statement(
    current_user: user_with_coll_permission_t_dep("access"), statement: str
) -> List[StatementModel]:
    async with Session() as session:
        statements = await session.execute(
            select(Statement)
            .filter_by(text=statement)
            .options(subqueryload(Statement.collections))
        )
        statements = [s for (s,) in statements]
        statement_models = []
        for s in statements:
            sm = s.as_model(session)
            statement_models.append(sm.model_dump(mode="json"))
        return ORJSONResponse(statement_models)
