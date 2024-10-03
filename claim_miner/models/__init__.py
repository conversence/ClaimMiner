# Copyright Society Library and Conversence 2022-2024
from __future__ import annotations
from logging import getLogger
from typing import Dict, Type, Union

from pydantic import BaseModel
from sqlalchemy import null, text, delete, inspect
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.orm import Mapper, with_polymorphic, aliased as sqla_aliased
from sqlalchemy.orm.util import AliasedInsp, AliasedClass
from sqlalchemy.sql import select, func
from sqlalchemy.sql.visitors import traverse

from .. import Session
from ..pyd_models import EmbeddingModel, topic_type

logger = getLogger(__name__)


def poly_type_clause(poly):
    mappers = inspect(poly).with_polymorphic_mappers or (poly.__mapper__,)
    identities = set(m.polymorphic_identity for m in mappers)
    identities.discard(None)
    if not len(identities):
        return True  # bogus
    if len(identities) > 1:
        return poly.type.in_(list(identities))
    return poly.type == identities.pop()


def aliased(cls, **kwargs):
    insp = inspect(cls)
    if insp.with_polymorphic_mappers:
        kwargs.pop("name", None)
        kwargs.pop("aliased", None)
        return with_polymorphic(
            insp.class_,
            [
                m.class_
                for m in insp.with_polymorphic_mappers
                if m.class_ is not insp.class_
            ],
            aliased=True,
            **kwargs,
        )
    else:
        return sqla_aliased(cls, **kwargs)


async def ensure_dynamic_enum_values(session, enum: ENUM):
    "Update the postgres enum to match the values of the ENUM"
    db_names = [
        n
        for (n,) in await session.execute(
            select(func.unnest(func.enum_range(func.cast(null(), enum))))
        )
    ]
    db_names = set(db_names)
    value_names = set(enum.enums)
    if value_names != db_names:
        # Check no element was removed. If needed, introduce tombstones to enums.
        removed = set(db_names) - set(value_names)
        if removed:
            logger.warn(
                f"Some enum values were removed from type {enum.name}: {', '.join(removed)}"
            )
            db_names = db_names - removed
        for name in value_names - db_names:
            await session.execute(
                text("ALTER TYPE %s ADD VALUE '%s' " % (enum.name, name))
            )
        await session.commit()


def classes_of_selectable(sel):
    # fragile introspection
    entity = sel._annotations.get("parententity")
    if not entity:
        return []
    if isinstance(entity, Mapper):
        return [entity.class_]
    if isinstance(entity, (AliasedInsp, AliasedClass)):
        return [m.class_ for m in entity.with_polymorphic_mappers]
    if isinstance(entity, Base):
        return [entity]
    return []


async def batch_lambda_query(session, lambda_query, large_list, batch_size=20000):
    # Allow batching very large queries
    acc = []
    large_list = large_list if isinstance(large_list, list) else list(large_list)
    for i in range(0, len(large_list), batch_size):
        acc += list(await session.execute(lambda_query(large_list[i : i + batch_size])))
    return acc


def columns_of_selectable(sel):
    cols = set()
    traverse(sel, {}, {"column": lambda x: cols.add(x)})
    return cols


db_class_from_pyd_class: Dict[Type[BaseModel], Type[Base]] = {}


from .base import Base, Topic, topic_type_db, created_objects


def entity_of_selectable(sel):
    # fragile introspection
    entity = sel._annotations.get("parententity")
    if not entity:
        return None
    if isinstance(entity, Mapper):
        return entity.class_
    elif isinstance(entity, (AliasedInsp, AliasedClass)):
        return entity._weak_entity()
    if isinstance(entity, Base):
        return entity


from .auth import User, permission_db
from .collections import (
    Collection,
    TopicCollection,
    CollectionPermissions,
    CollectionScope,
    collection_filter,
)

# After set in collections
from .base import globalScope
from .tasks import (
    Analyzer,
    Analysis,
    TaskTrigger,
    TaskTemplate,
    process_status_db,
    analysis_context_table,
    analysis_output_table,
)
from .content import (
    Document,
    Statement,
    Fragment,
    fragment_type_db,
    ClusterData,
    InClusterData,
    UriEquiv,
    ClaimLink,
    HyperEdge,
    StatementAlone,
    StatementOrFragment,
    AnyClaimOrLink,
    AnyClaimOrHyperedge,
    VisibleClaim,
    search_target_class,
    claim_neighbourhood,
    graph_subquery,
)
from .embedding import (
    Embedding,
    embedding_model_db,
    model_names,
    embed_db_model_by_name,
    model_names_s,
    ensure_embedding_db_tables,
)

PolyTopic = with_polymorphic(Topic, "*", flat=True)
PolyTopicType = Union[Topic, ClaimLink, Statement, Fragment, Document, HyperEdge]

model_by_topic_type: Dict[topic_type, Type[Base]] = {
    topic_type.collection: Collection,
    topic_type.document: Document,
    topic_type.standalone: Statement,
    topic_type.fragment: Fragment,
    topic_type.hyperedge: HyperEdge,
    topic_type.link: ClaimLink,
}


async def finalize_db_models():
    async with Session() as session:
        await ensure_dynamic_enum_values(session, permission_db)
        await ensure_dynamic_enum_values(session, embedding_model_db)
        await ensure_embedding_db_tables(session)
    db_class_from_pyd_class.update(
        {cls.pyd_model: cls for cls in Base.__subclasses__() if cls.pyd_model}
    )
    db_class_from_pyd_class.pop(EmbeddingModel)


async def delete_data(session):
    await session.execute(delete(Analysis))
    await session.execute(delete(ClusterData))
    await session.execute(
        delete(Topic.__table__).where(Topic.__table__.c.type != topic_type.analyzer)
    )
    await session.execute(delete(UriEquiv))
    await session.execute(delete(Collection))


__all__ = [
    "Analysis",
    "Analyzer",
    "AnyClaimOrLink",
    "AnyClaimOrHyperedge",
    "Base",
    "ClaimLink",
    "ClusterData",
    "Collection",
    "CollectionPermissions",
    "CollectionScope",
    "Document",
    "Embedding",
    "Fragment",
    "HyperEdge",
    "InClusterData",
    "PolyTopic",
    "PolyTopicType",
    "Statement",
    "StatementAlone",
    "StatementOrFragment",
    "TaskTemplate",
    "TaskTrigger",
    "Topic",
    "TopicCollection",
    "UriEquiv",
    "User",
    "VisibleClaim",
    "analysis_context_table",
    "analysis_output_table",
    "claim_neighbourhood",
    "collection_filter",
    "created_objects",
    "db_class_from_pyd_class",
    "delete_data",
    "embed_db_model_by_name",
    "embedding_model_db",
    "ensure_embedding_db_tables",
    "finalize_db_models",
    "fragment_type_db",
    "globalScope",
    "graph_subquery",
    "model_by_topic_type",
    "model_names",
    "model_names_s",
    "permission_db",
    "process_status_db",
    "search_target_class",
    "topic_type",
    "topic_type_db",
]
