from __future__ import annotations

from typing import Union, List, Dict, Any, ForwardRef

from sqlalchemy import BigInteger, ForeignKey, select, String
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..pyd_models import (
    TopicCollectionModel,
    UserModel,
    permission,
    BASE_EMBED_MODEL,
    CollectionModel,
    topic_type,
    embedding_model,
    CollectionPermissionsModel,
)
from . import (
    columns_of_selectable,
    classes_of_selectable,
    entity_of_selectable,
    aliased,
    logger,
)
from .auth import User, permission_db
from .base import Base, Topic
from . import base


class TopicCollection(Base):
    """Join table between Topics and Collections"""

    __tablename__ = "topic_collection"
    pyd_model = TopicCollectionModel
    topic_id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey("topic.id", onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
    )
    collection_id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey("collection.id", onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
    )
    topic: Mapped[Topic] = relationship(Topic, foreign_keys=[topic_id])
    collection: Mapped[Collection] = relationship(
        "Collection", foreign_keys=[collection_id]
    )


class CollectionScope:
    """An abstract class defining default behaviour for Collections"""

    @property
    def path(self):
        return ""

    def user_can(
        self, user: Union[User, UserModel], permission: Union[str, permission]
    ):
        if user:
            return user.can(permission)

    def embed_model(self):
        return BASE_EMBED_MODEL

    def embed_models_names(self) -> List[str]:
        from .embedding import embed_db_model_by_name

        return list(embed_db_model_by_name.keys())

    @staticmethod
    async def get_collection(name, session=None, user_id=None):
        if not name:
            return base.globalScope
        if isinstance(name, CollectionScope):
            return name
        if not session:
            from . import Session

            async with Session() as session:
                return await CollectionScope.get_collection(name, session, user_id)
        q = select(Collection).filter_by(name=name).limit(1)
        if user_id:
            q = q.outerjoin(
                CollectionPermissions,
                (CollectionPermissions.collection_id == Collection.id)
                & (CollectionPermissions.user_id == user_id),
            ).add_columns(CollectionPermissions)
        r = await session.execute(q)
        r = r.first()
        if not r:
            raise ValueError("Unknown collection: ", name)
        collection = r[0]
        if user_id:
            collection.user_permissions = r[1]
        return collection

    @staticmethod
    def collection_path(name):
        return f"/c/{name}" if name else ""

    @staticmethod
    async def get_collection_names(session):
        r = await session.execute(select(Collection.name))
        return [n for (n,) in r]


class Collection(Base, CollectionScope):
    """A named collection of Documents and Claims"""

    __tablename__ = "collection"
    pyd_model = CollectionModel
    type = topic_type.collection  # Not a Topic subclass, but still has a type
    id: Mapped[BigInteger] = mapped_column(BigInteger, primary_key=True)  #: Primary key
    name: Mapped[String] = mapped_column(
        String, nullable=False
    )  #: name (should be a slug)
    params: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, server_default="{}"
    )  #: Supplemental information
    topic_collections: Mapped[List[TopicCollection]] = relationship(
        TopicCollection,
        foreign_keys=[TopicCollection.collection_id],
        overlaps="topics",
        cascade="all, delete",
        passive_deletes=True,
    )

    topics: Mapped[List[Topic]] = relationship(
        Topic, secondary=TopicCollection.__table__, back_populates="collections"
    )

    documents: Mapped[List[ForwardRef("Document")]] = relationship(
        "Document", secondary=TopicCollection.__table__, viewonly=True
    )
    "The documents in the collection"

    statements: Mapped[List[ForwardRef("Statement")]] = relationship(
        "Statement", secondary=TopicCollection.__table__, viewonly=True
    )
    "The statements explicitly in the collection"

    permissions: Mapped[List[CollectionPermissions]] = relationship(
        "CollectionPermissions", back_populates="collection", passive_deletes=True
    )
    "Collection-specific permissions"

    @property
    def path(self):
        return CollectionScope.collection_path(self.name)

    def web_path(self, collection=None):
        assert collection is None or collection is self
        return self.path

    def api_path(self, collection=None):
        assert collection is None or collection is self
        return f"/api{self.path}"

    def embed_models_names(self) -> List[str]:
        return self.params.get("embeddings", []) + [BASE_EMBED_MODEL.name]

    def embed_model(self) -> embedding_model:
        embeddings = self.params.get("embeddings", [])
        for embedding in embeddings:
            if embedding == BASE_EMBED_MODEL.name:
                continue
            return embedding_model[embedding]
        return BASE_EMBED_MODEL

    def user_can(self, user: Union[User, UserModel], perm: Union[str, permission]):
        perm = permission[perm] if isinstance(perm, str) else perm
        if super().user_can(user, perm):
            return True
        if cp := getattr(self, "user_permissions", None):
            permissions = cp.permissions or []
            return (perm in permissions) or (permission.admin in permissions)


class GlobalScope(CollectionScope):
    """The global scope: All documents and fragments, belonging to any collection."""

    params: Dict[str, Any] = {}
    id = None

    def __bool__(self):
        return False

    def web_path(self, collection=None):
        return ""


base.globalScope = GlobalScope()


class CollectionPermissions(Base):
    """Collection-specific permissions that a user has in the scope of a specific collection"""

    __tablename__ = "collection_permissions"
    pyd_model = CollectionPermissionsModel
    user_id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(User.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
    )
    collection_id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(Collection.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
    )
    permissions: Mapped[List[permission]] = mapped_column(ARRAY(permission_db))

    collection: Mapped[Collection] = relationship(
        Collection, back_populates="permissions"
    )
    user: Mapped[User] = relationship(User)


Topic.collections: Mapped[List[Collection]] = relationship(
    Collection,
    secondary=TopicCollection.__table__,
    back_populates="documents",
    overlaps="collections",
)
"The collections this document belongs to"


def collection_filter(
    query,
    collection_name,
    include_claims=False,
    include_paragraphs=False,
    target=None,
    doc_target=None,
):
    from .content import Document, Statement, Fragment

    if target is None or (include_paragraphs and doc_target is None):
        has_doc = False
        # black magic to get the selectable
        for col in columns_of_selectable(query.selectable):
            classes = classes_of_selectable(col)
            if Document in classes:
                has_doc = True
                if doc_target is None:
                    doc_target = entity_of_selectable(col) or doc_target
            if Fragment in classes or Statement in classes:
                if Fragment in classes:
                    include_paragraphs = True
                if Statement in classes:
                    include_claims = True
                if target is None:
                    target = entity_of_selectable(col) or target
        if target and include_paragraphs and not has_doc:
            if include_claims:
                query = query.outerjoin(Document, target.Fragment.document)
            else:
                query = query.join(Document, target.document)
            doc_target = Document
    doc_coll = aliased(Collection, name="doc_coll")
    claim_coll = aliased(Collection, name="claim_coll")
    if include_claims:
        if include_paragraphs:
            query = (
                query.outerjoin(claim_coll, target.collections)
                .outerjoin(doc_coll, doc_target.collections)
                .filter(
                    (doc_coll.name == collection_name)
                    | (claim_coll.name == collection_name)
                )
            )
        else:
            query = query.join(claim_coll, target.collections).filter(
                claim_coll.name == collection_name
            )
    elif doc_target:
        query = query.join(doc_coll, doc_target.collections).filter(
            doc_coll.name == collection_name
        )
    else:
        logger.warn("Could not join on collection")
    return query
