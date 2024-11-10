from typing import Optional, List

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from fastapi import Request
from fastapi.responses import ORJSONResponse

from .. import Session
from ..app import NotFound, BadRequest
from ..auth import user_with_coll_permission_t_dep
from ..pyd_models import AbstractStructuredIdeaModel, ClaimLinkModel
from ..models import Topic, StructureTypes, StructuredIdea, ClaimLink, poly_type_clause
from ..ontology import Ontology
from . import api_router, get_collection

FullStructuredType = StructureTypes
FullStructuredIdeaModelType = Ontology.ontology.as_union_type('cm:idea_bundle', 'hk:predicate')
# UseStructuredType = FullStructuredType
# UseStructuredModelType = FullStructuredIdeaModelType
# BUG: the as_model for claimlink is a ClaimLinkModel, not CmClaimLink, so the above fails

UseStructuredType = StructuredIdea
UseStructuredModelType = Ontology.ontology.as_union_type('cm:idea_bundle')


@api_router.get("/idea/{id}")
@api_router.get("/c/{collection}/idea/{id}")
async def get_structure(
        id:int,
        current_user: user_with_coll_permission_t_dep('access'),
        collection:Optional[str]=None
        ) -> UseStructuredModelType:
    # TODO: Replace with fastapi
    async with Session() as session:
        idea = await session.scalar(select(UseStructuredType).filter_by(id=id))
        if not idea:
            raise NotFound()
        if collection:
            await session.refresh(idea, ['collections'])
            for coll in idea.collections:
                if coll.name == collection:
                    break
            else:
                raise NotFound()
        if isinstance(idea, StructuredIdea):
            await idea.preload_substructures(session)
        return idea.as_model(session)


@api_router.get("/idea")
@api_router.get("/c/{collection}/idea")
async def get_structures(
        current_user: user_with_coll_permission_t_dep('access'),
        offset: int=0,
        limit: int=20,
        term_curie: Optional[str]=None,
        collection:Optional[str]=None,
        ) -> List[UseStructuredModelType]:
    async with Session() as session:
        collection_ob = await get_collection(collection, session, current_user.id)
        if collection and not collection_ob:
            raise NotFound()
        if term_curie:
            target, base_query = Topic.select_by_schema(term_curie)
            base_query = base_query.with_entities(target.id).order_by(target.id)
        else:
            target = UseStructuredType
            base_query = select(target.id).filter(poly_type_clause(target)).order_by(target.id)
        if collection:
            base_query = base_query.join(target.collections).filter_by(name=collection)
        base_query = base_query.offset(offset).limit(limit)
        query = select(target).filter(target.id.in_(base_query.subquery()))
        r = (await session.scalars(query)).all()
        cache = {}
        for f in r:
            if isinstance(f, StructuredIdea):
                cache.update(await f.preload_substructures(session))
        # TODO: Add other related objects. Analysis mostly?
        return [f.as_model(session) for f in r]


@api_router.post("/c/{collection}/idea", status_code=201)
async def add_structure(
        request: Request,
        current_user: user_with_coll_permission_t_dep('add_claim'),
        collection:str,
        idea: UseStructuredModelType,
        ) -> UseStructuredModelType:
    async with Session() as session:
        collection_ob = await get_collection(collection, session, current_user.id)
        if not collection_ob:
            raise NotFound()
        idea.created_by = current_user.id
        if isinstance(idea, AbstractStructuredIdeaModel):
            idea_db = StructuredIdea.from_model(idea, collections=[collection_ob])
        elif isinstance(idea, ClaimLinkModel):
            idea_db = ClaimLink.form_model(idea, collections=[collection_ob])
        session.add(idea_db)
        try:
            await session.commit()
        except IntegrityError as e:
            await session.rollback()
            raise BadRequest(e)
        # Get the defaults back
        idea = idea_db.as_model(session)
    location = f"/api/c/{collection}/idea/{idea_db.id}"
    return ORJSONResponse(idea.model_dump(), status_code=201, headers=dict(location=location))
