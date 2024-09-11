"""
The SQLAlchemy ORM models for ClaimMiner
"""

# Copyright Society Library and Conversence 2022-2024
from __future__ import annotations
from typing import (
    List,
    Union,
    Dict,
    Any,
    Optional,
    Mapping,
    Type,
    Tuple,
    Iterable,
    TypedDict,
    Set,
    ForwardRef,
)
from io import BytesIO, StringIO
from uuid import UUID
from collections import defaultdict
from itertools import chain
from logging import getLogger

from sqlalchemy import (
    Table,
    Integer,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    SmallInteger,
    Float,
    Text,
    BigInteger,
    Index,
    case,
    literal_column,
    select,
    delete,
    inspect,
    Column,
    event,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, ENUM, UUID as UUID_db, array
from sqlalchemy.schema import FetchedValue
from sqlalchemy.orm import (
    DeclarativeBase,
    relationship,
    declared_attr,
    joinedload,
    backref,
    subqueryload,
    with_polymorphic,
    aliased as sqla_aliased,
    Mapper,
    Mapped,
    mapped_column,
    LoaderCallableStatus,
)
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy.orm.util import AliasedInsp, AliasedClass
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.schema import CreateTable, CreateIndex
from sqlalchemy.sql import and_, or_, null, text
from sqlalchemy.sql.functions import coalesce, func
from sqlalchemy.sql.elements import ClauseList
from sqlalchemy.sql.visitors import traverse
from pgvector.sqlalchemy import Vector
from frozendict import frozendict
from fastapi.encoders import jsonable_encoder

from . import hashfs, Session, sync_maker, config
from .pyd_models import (
    BaseModel,
    topic_type,
    permission,
    process_status,
    fragment_type,
    link_type,
    uri_status,
    relevance_type,
    embedding_model,
    BASE_EMBED_MODEL,
    BASE_DOC_EMBED_MODEL,
    AnalysisModel,
    AnalyzerModel,
    UserModel,
    DocumentModel,
    TopicCollectionModel,
    DocumentLinkModel,
    ClaimLinkModel,
    HyperEdgeModel,
    CollectionModel,
    CollectionPermissionsModel,
    UriEquivModel,
    StatementModel,
    FragmentModel,
    TaskTriggerModel,
    TaskTemplateModel,
    InClusterDataModel,
    ClusterDataModel,
    EmbeddingModel,
    AnalysisWithTemplateModel,
)
from .utils import encode_uuid, decode_uuid, filter_dict, to_optional, safe_lang_detect
from .embed import embedder_registry
from .task_registry import TaskRegistry

logger = getLogger(__name__)

flushed_objects_by_session: Mapping[int, Set[Tuple[topic_type, int]]] = defaultdict(set)
created_objects: Set[Tuple[topic_type, int]] = set()


@event.listens_for(sync_maker, "pending_to_persistent")
def intercept_pending_to_persistent(session, object_):
    if isinstance(object_, Topic):
        flushed_objects_by_session[session.hash_key].add((object_.type, object_.id))


@event.listens_for(sync_maker, "after_commit")
def receive_after_commit(session):
    "listen for the 'after_commit' event"
    created_objects.update(flushed_objects_by_session[session.hash_key])
    flushed_objects_by_session[session.hash_key].clear()


@event.listens_for(sync_maker, "after_rollback")
def receive_after_rollback(session):
    logger.debug("got the rollback")
    flushed_objects_by_session[session.hash_key].clear()


class Base(DeclarativeBase):
    """Declarative base class"""

    pyd_model: Optional[Type[BaseModel]] = None

    @hybrid_property
    def primary_key(self):
        insp = inspect(self.__class__)
        primary_keys = [c for c in insp.selectable.c if c.primary_key]
        base_keys = set(
            chain(*[(x.column for x in c.foreign_keys) for c in primary_keys])
        )
        return tuple(getattr(self, c.name) for c in primary_keys if c not in base_keys)

    @primary_key.inplace.expression
    def primary_key(cls):
        insp = inspect(cls)
        primary_keys = [c for c in insp.selectable.c if c.primary_key]
        base_keys = set(
            chain(*[(x.column for x in c.foreign_keys) for c in primary_keys])
        )
        return ClauseList(*[c for c in primary_keys if c not in base_keys])

    @staticmethod
    async def from_model_base(
        session, model: BaseModel, ignore: Optional[List[str]] = None, **extra
    ):
        return db_class_from_pyd_class(model.__class__).from_model(
            session, model, ignore=ignore, **extra
        )

    @classmethod
    async def from_model(
        cls, session, model: BaseModel, ignore: Optional[List[str]] = None, **extra
    ):
        if cls.pyd_model:
            assert isinstance(model, cls.pyd_model)
        model_data = model.model_dump()
        if ignore:
            for k in ignore:
                del model_data[k]
        # Avoid failure when you give a foreign key but not the corresponding object
        for k, v in list(model_data.items()):
            if v is not None:
                continue
            if rel := cls.__mapper__.relationships.get(k):
                for c in rel.local_columns:
                    if model_data.get(c.name) is not None:
                        del model_data[k]
        return cls(**(extra | filter_dict(model_data)))

    def as_model(
        self,
        session,
        model_cls: Optional[Type[BaseModel]] = None,
        ignore_keys: Optional[List[str]] = None,
        recursion: Optional[Set[int]] = None,
        **extra,
    ):
        model_cls = model_cls or self.pyd_model
        assert model_cls
        if self.pyd_model:
            assert issubclass(model_cls, self.pyd_model)
        attributes = dict(self.loaded_attributes())
        recursion = recursion or set()
        recursion.add(self.primary_key)
        for k in self.__mapper__.relationships.keys():
            if k in attributes and k in model_cls.model_fields:
                v = attributes.get(k)
                if not v:
                    continue
                if isinstance(v, list):
                    attributes[k] = [
                        m.as_model(session, recursion=recursion) for m in v
                    ]
                elif isinstance(v, Base):
                    if v.primary_key not in recursion:
                        attributes[k] = v.as_model(session, recursion=recursion)
                else:
                    assert False
        attributes |= extra
        for k in ignore_keys or []:
            attributes.pop(k, None)
        return model_cls.model_validate(filter_dict(attributes))

    def loaded_attributes(self):
        return {
            name: state.loaded_value
            for name, state in inspect(self).attrs.items()
            if state.loaded_value != LoaderCallableStatus.NO_VALUE
        }

    async def ensure_loaded(self, attributes: List[str], session):
        attrs = inspect(self).attrs
        to_load = [
            a
            for a in attributes
            if attrs[a].loaded_value == LoaderCallableStatus.NO_VALUE
        ]
        if to_load:
            await session.refresh(self, to_load)


topic_type_db = ENUM(topic_type, name="topic_type")
permission_db = ENUM(permission, name="permission")
process_status_db = ENUM(process_status, name="process_status")
fragment_type_db = ENUM(fragment_type, name="fragment_type")
link_type_db = ENUM(link_type, name="link_type")
uri_status_db = ENUM(uri_status, name="uri_status")
relevance_type_db = ENUM(relevance_type, name="relevance")


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


globalScope: ForwardRef("CollectionScope") = None


class Topic(Base):
    __mapper_args__ = {"polymorphic_abstract": True, "polymorphic_on": "type"}
    __tablename__ = "topic"
    id: Mapped[BigInteger] = mapped_column(BigInteger, primary_key=True)
    type: Mapped[topic_type] = mapped_column(
        topic_type_db, nullable=False
    )  # Type of topic, connects to other tables
    created_by: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(id, onupdate="CASCADE", ondelete="SET NULL")
    )  # Who created the topic?

    from_analyses: Mapped[List[Analysis]] = relationship(
        "Analysis", secondary="analysis_output", back_populates="generated_topics"
    )
    target_of_analyses: Mapped[List[Analysis]] = relationship("Analysis", viewonly=True)
    topic_collections: Mapped[List[ForwardRef("TopicCollection")]] = relationship(
        "TopicCollection",
        foreign_keys="TopicCollection.topic_id",
        overlaps="collections",
        cascade="all, delete",
        passive_deletes=True,
    )
    # collections: Mapped[List[ForwardRef("Collection")]] = relationship("Collection", secondary="TopicCollection.__table__", back_populates='documents', overlaps='collections')

    # outgoing_links: Mapped[List[ClaimLink]] = relationship("ClaimLink", primaryjoin="Topic.id == ClaimLink.source", back_populates="source_topic")
    # incoming_links: Mapped[List[ClaimLink]] = relationship("ClaimLink", primaryjoin="Topic.id == ClaimLink.target", back_populates="target_topic")

    def paths_to(self, topic):
        for link in self.outgoing_links:
            if link.target_topic == topic:
                yield link

    def paths_from(self, topic):
        for link in self.incoming_links:
            if link.source_topic == topic:
                yield link

    def web_path(self, collection=globalScope):
        return f"{collection.path}/{self.type.name}/{self.id}"

    def api_path(self, collection=globalScope):
        return f"/api{collection.path}/{self.type.name}/{self.id}"


class User(Topic):
    """ClaimMiner users."""

    __tablename__ = "user"
    pyd_model = UserModel

    __mapper_args__ = {
        "polymorphic_load": "inline",
        "polymorphic_identity": topic_type.agent,
    }

    id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(Topic.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
    )  #: Primary key
    handle: Mapped[String] = mapped_column(String)  #: username
    passwd: Mapped[String] = mapped_column(String)  #: password (scram hash)
    email: Mapped[String] = mapped_column(String)  #: email
    name: Mapped[String] = mapped_column(String)  #: name
    confirmed: Mapped[Boolean] = mapped_column(
        Boolean, server_default="false"
    )  #: account confirmed by admin
    created: Mapped[DateTime] = mapped_column(
        DateTime, server_default="now()"
    )  #: date of creation
    external_id: Mapped[String] = mapped_column(
        String, unique=True
    )  #: external id (e.g. google id)
    picture_url: Mapped[String] = mapped_column(String)  #: picture url
    permissions: Mapped[List[permission]] = mapped_column(
        ARRAY(permission_db)
    )  #: User's global permissions

    def can(self, perm: Union[str, permission]):
        "Does the user have this permission?"
        permissions = self.permissions or []
        perm = permission[perm] if isinstance(perm, str) else perm
        return (perm in permissions) or (permission.admin in permissions)


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


class Analyzer(Topic):
    """A versioned computation process.
    Computed values keep a reference to the analyzer that created them.
    The versioning system is not being used yet.
    """

    __tablename__ = "analyzer"
    pyd_model = AnalyzerModel
    __mapper_args__ = {
        "polymorphic_load": "inline",
        "polymorphic_identity": topic_type.analyzer,
    }

    id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(Topic.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
    )  #: Primary key
    name: Mapped[String] = mapped_column(String)  #: the type of analyzer
    version: Mapped[BigInteger] = mapped_column(BigInteger)  #: the version number
    analyses: Mapped[List[Analysis]] = relationship(
        "Analysis", back_populates="analyzer", passive_deletes=True
    )


class TaskTemplate(Base):
    """A coherent set of parameters for an analysis task."""

    __tablename__ = "task_template"
    pyd_model = TaskTemplateModel

    id: Mapped[BigInteger] = mapped_column(BigInteger, primary_key=True)  #: Primary key
    analyzer_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(Analyzer.id), nullable=False
    )
    collection_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey("collection.id")
    )
    nickname: Mapped[String] = mapped_column(
        String, unique=True
    )  #: User-readable subtype, used for prompt names
    params: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, server_default="{}"
    )  #: Prompt logic is here
    draft: Mapped[Boolean] = mapped_column(Boolean, server_default="false")
    """True while editing a prompt, false when it has been used.
    Avoid editing an analyzer that is tied to an existing analysis."""
    analyses: Mapped[List[Analysis]] = relationship(
        "Analysis", back_populates="task_template", passive_deletes=True
    )
    analyzer: Mapped[Analyzer] = relationship(Analyzer)
    collection: Mapped[Collection] = relationship("Collection")

    @property
    def analyzer_name(self):
        if (
            inspect(self).attrs["analyzer"].loaded_value
            != LoaderCallableStatus.NO_VALUE
        ):
            return self.analyzer.name
        analyzer = TaskRegistry.get_registry().analyzer_by_id.get(self.analyzer_id)
        return analyzer.name if analyzer else None

    @analyzer_name.setter
    def analyzer_name(self, name: str):
        analyzer = TaskRegistry.get_registry().analyzer_by_name.get(name)
        if analyzer:
            self.analyzer_id = analyzer.id

    def as_model(
        self,
        session,
        model_cls: Optional[Type[BaseModel]] = None,
        recursion: Optional[Set[int]] = None,
        **extra,
    ):
        model_cls = (
            model_cls
            or TaskRegistry.get_registry()
            .get_task_cls_by_name(self.analyzer_name)
            .task_template_model
            or TaskTemplateModel
        )
        if self.draft:
            model_cls = to_optional(model_cls)
        ignore_keys = ["params", "analyzer"]
        model = super(TaskTemplate, self).as_model(
            session,
            model_cls,
            ignore_keys=ignore_keys,
            recursion=recursion,
            **(extra | self.params),
        )
        return model

    @classmethod
    async def from_model(
        cls, session, model: BaseModel, ignore: Optional[List[str]] = None, **extra
    ):
        assert isinstance(model, TaskTemplateModel)
        args = extra | model.model_dump()
        rel_names = inspect(cls).relationships.keys()
        col_names = inspect(cls).c.keys()
        for k in ("analyzer_name", "analyzer", "collection_name"):
            args.pop(k, None)
        col_args = {k: v for (k, v) in args.items() if k in col_names}
        col_args["params"] = jsonable_encoder(
            {k: v for (k, v) in args.items() if k not in col_names}
        )
        return cls(**col_args)

    def web_path(self, collection=globalScope):
        collection_path = ""
        if collection and (self.collection_id == collection.id):
            collection_path = collection.path
        return (
            f"{collection_path}/analyzer/{self.analyzer_name}/template/{self.nickname}"
        )


class TopicCollection(Base):
    """Join table between Document and Collection"""

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
        return list(embed_db_model_by_name.keys())

    @staticmethod
    async def get_collection(name, session=None, user_id=None):
        if not name:
            return globalScope
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
    documents: Mapped[List[Document]] = relationship(
        "Document", secondary=TopicCollection.__table__, viewonly=True
    )
    "The documents in the collection"
    statements: Mapped[List[Statement]] = relationship(
        "Statement", secondary=TopicCollection.__table__, viewonly=True
    )
    "The statements explicitly in the collection"
    permissions: Mapped[List[CollectionPermissions]] = relationship(
        "CollectionPermissions", back_populates="collection", passive_deletes=True
    )
    "Collection-specific permissions"
    task_triggers: Mapped[List[TaskTrigger]] = relationship(
        "TaskTrigger", back_populates="collection", passive_deletes=True
    )

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


Topic.collections: Mapped[List[Collection]] = relationship(
    Collection,
    secondary=TopicCollection.__table__,
    back_populates="documents",
    overlaps="collections",
)
"The collections this document belongs to"


class GlobalScope(CollectionScope):
    """The global scope: All documents and fragments, belonging to any collection."""

    params: Dict[str, Any] = {}
    id = None

    def __bool__(self):
        return False

    def web_path(self, collection=None):
        return ""


globalScope = GlobalScope()


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


class TaskTrigger(Base):
    """Triggers the execution of an analysis task."""

    __tablename__ = "task_trigger"
    pyd_model = TaskTriggerModel

    id: Mapped[BigInteger] = mapped_column(Integer, primary_key=True)
    target_analyzer_id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(Analyzer.id, onupdate="CASCADE", ondelete="CASCADE"),
        nullable=False,
    )
    collection_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(Collection.id, onupdate="CASCADE", ondelete="CASCADE")
    )
    analyzer_trigger_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(Analyzer.id, onupdate="CASCADE", ondelete="CASCADE")
    )
    task_template_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(TaskTemplate.id, onupdate="CASCADE", ondelete="CASCADE")
    )
    creation_trigger_id: Mapped[topic_type] = mapped_column(topic_type_db)
    automatic: Mapped[Boolean] = mapped_column(Boolean, server_default="false")
    conditions: Mapped[Dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    params: Mapped[Dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    creator_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(User.id, onupdate="CASCADE", ondelete="SET NULL")
    )  # Who created the trigger?

    target_analyzer: Mapped[Analyzer] = relationship(
        Analyzer, foreign_keys=[target_analyzer_id]
    )
    task_template: Mapped[TaskTemplate] = relationship(
        TaskTemplate, foreign_keys=[task_template_id]
    )
    analyzer_trigger: Mapped[Analyzer] = relationship(
        Analyzer, foreign_keys=[analyzer_trigger_id]
    )
    collection: Mapped[Collection] = relationship(
        Collection, foreign_keys=[collection_id], back_populates="task_triggers"
    )
    creator: Mapped[User] = relationship(User, foreign_keys=[creator_id])

    def signature(self):
        return (
            self.target_analyzer_id,
            self.collection_id,
            self.creation_trigger_id or self.analyzer_trigger_id,
            self.task_template_id,
            frozendict(self.conditions),
            frozendict(self.params),
        )

    def web_path(self, collection=globalScope):
        return f"{collection.web_path()}/task_trigger/{self.id}"


class UriEquiv(Base):
    """Equivalence classes of URIs"""

    __tablename__ = "uri_equiv"
    pyd_model = UriEquivModel
    id: Mapped[BigInteger] = mapped_column(BigInteger, primary_key=True)  #: Primary key
    status: Mapped[uri_status] = mapped_column(
        uri_status_db, nullable=False, server_default="unknown"
    )
    canonical_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey("uri_equiv.id", onupdate="CASCADE", ondelete="SET NULL")
    )
    uri: Mapped[String] = mapped_column(String, nullable=False, unique=True)
    canonical: Mapped[UriEquiv] = relationship(
        "UriEquiv", remote_side=[id], backref=backref("equivalents")
    )
    referencing_links: Mapped[List[DocumentLink]] = relationship(
        "DocumentLink", foreign_keys="DocumentLink.target_id", passive_deletes=True
    )

    @classmethod
    async def ensure(cls, uri: str, session=None):
        if session is None:
            async with Session() as session:
                return cls.ensure(uri, session)
        existing = await session.scalar(select(UriEquiv).filter_by(uri=uri))
        if not existing:
            existing = cls(uri=uri)
            session.add(existing)
        return existing


class DocumentLink(Base):
    __tablename__ = "document_link"
    pyd_model = DocumentLinkModel
    source_id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey("document.id", onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    target_id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey("uri_equiv.id", onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
        nullable=False,
    )
    analyzer_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey("analyzer.id", onupdate="CASCADE", ondelete="SET NULL")
    )

    source: Mapped[Document] = relationship(
        "Document", foreign_keys=[source_id], back_populates="href_links"
    )
    target: Mapped[UriEquiv] = relationship(
        UriEquiv, foreign_keys=[target_id], back_populates="referencing_links"
    )
    analyzer: Mapped[Analyzer] = relationship(Analyzer, foreign_keys=[analyzer_id])


class Document(Topic):
    """Represents a document that was requested, uploaded or downloaded"""

    __tablename__ = "document"
    pyd_model = DocumentModel

    __mapper_args__ = {
        "polymorphic_load": "inline",
        "polymorphic_identity": topic_type.document,
    }

    def __init__(self, *args, **kwargs):
        super(Document, self).__init__(*args, **kwargs)
        if file_content := kwargs.pop("file_content", None):
            self.file_content = file_content
        if text_content := kwargs.pop("text_content", None):
            self.text_content = text_content

    id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(Topic.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
    )  #: Primary key
    uri_id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(UriEquiv.id, onupdate="CASCADE", ondelete="CASCADE"),
        nullable=False,
    )  #: Reference to URI, unique for non-archive
    is_archive: Mapped[Boolean] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )  #: For multiple snapshots of same document
    requested: Mapped[DateTime] = mapped_column(
        DateTime(True), nullable=False, server_default="now()"
    )  #: When was the document requested
    return_code: Mapped[BigInteger] = mapped_column(
        SmallInteger
    )  #: What was the return code when asking for the document
    retrieved: Mapped[DateTime] = mapped_column(
        DateTime
    )  #: When was the document retrieved
    created: Mapped[DateTime] = mapped_column(
        DateTime
    )  #: When was the document created (according to HTTP headers)
    modified: Mapped[DateTime] = mapped_column(
        DateTime
    )  #: When was the document last modified (according to HTTP headers)
    mimetype: Mapped[String] = mapped_column(
        String
    )  #: MIME type (according to HTTP headers)
    language: Mapped[String] = mapped_column(
        String
    )  #: Document language (according to HTTP headers, and langdetect)
    text_analyzer_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey("analyzer.id", onupdate="CASCADE", ondelete="SET NULL")
    )  #: What analyzer extracted the text from this document if any
    etag: Mapped[String] = mapped_column(
        String
    )  #: Document etag (according to HTTP headers)
    file_identity: Mapped[String] = mapped_column(
        String
    )  #: Hash of document content, refers to HashFS
    file_size: Mapped[Integer] = mapped_column(Integer)  #: Document size (in bytes)
    text_identity: Mapped[String] = mapped_column(
        String
    )  #: Hash of text extracted from document, refers to HashFS
    text_size: Mapped[Integer] = mapped_column(Integer)  # Text length (in bytes)
    title: Mapped[Text] = mapped_column(
        Text
    )  # Title extracted from the document (HTML or PDF...)
    process_params: Mapped[Dict[str, Any]] = mapped_column(
        JSONB
    )  # Paramaters that were used to extract and segment the text
    meta: Mapped[Dict[str, Any]] = mapped_column(
        "metadata", JSONB, server_default="{}"
    )  #: Metadata column
    public_contents: Mapped[Boolean] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )  # Whether the contents are protected by copyright
    uri: Mapped[UriEquiv] = relationship(
        UriEquiv,
        lazy="joined",
        innerjoin=False,
        foreign_keys=[uri_id],
        backref=backref("document"),
    )  #: The canonical URI of this document
    href_links: Mapped[List[DocumentLink]] = relationship(
        DocumentLink, foreign_keys=DocumentLink.source_id, passive_deletes=True
    )
    href_uri: Mapped[List[UriEquiv]] = relationship(
        UriEquiv, secondary=DocumentLink.__table__, viewonly=True
    )

    @property
    def url(self):
        return self.uri.uri

    @property
    def file_content(self):
        if self.file_identity:
            with open(hashfs.get(self.file_identity).abspath, "rb") as f:
                return f.read()

    @file_content.setter
    def file_content(self, content):
        # Should we delete the old one?
        self.file_identity = hashfs.put(BytesIO(content)).id

    @property
    def text_content(self):
        if self.text_identity:
            with open(hashfs.get(self.text_identity).abspath) as f:
                return f.read()

    @text_content.setter
    def text_content(self, content):
        # Should we delete the old one?
        self.text_identity = hashfs.put(StringIO(content)).id

    async def clean_text_content(self, session):
        await self.ensure_loaded(["paragraphs"], session)
        return "\n".join([p.text for p in self.paragraphs])

    @hybrid_property
    def base_type(self):
        return self.mimetype.split(";")[0] if self.mimetype else None

    @base_type.inplace.expression
    def base_type_expr(cls):
        return func.split_part(cls.mimetype, ";", 1)

    @hybrid_property
    def load_status(self):
        return case(
            {
                literal_column("200"): literal_column("'loaded'"),
                literal_column("0"): literal_column("'not_loaded'"),
            },
            value=coalesce(self.return_code, literal_column("0")),
            else_=literal_column("'error'"),
        ).label("load_status")

    @classmethod
    async def from_model(
        cls, session, model: BaseModel, ignore: Optional[List[str]] = None, **extra
    ):
        ignore = (ignore or []) + ["url"]
        if (
            not model.uri
            and not model.uri_id
            and "uri" not in extra
            and "uri_id" not in extra
            and model.url
        ):
            from .uri_equivalence import normalize

            url = normalize(model.url)
            uri = None
            uri_id = await session.scalar(
                select(UriEquiv.id).filter_by(uri=url).limit(1)
            )
            if uri_id:
                extra["uri_id"] = uri_id
            else:
                extra["uri"] = UriEquiv(uri=url)

        return await super(Document, cls).from_model(session, model, ignore, **extra)


UriEquiv.referencing_documents: Mapped[List[Document]] = relationship(
    Document, secondary=DocumentLink.__table__, viewonly=True
)


analysis_context_table = Table(
    "analysis_context",
    Base.metadata,
    Column(
        "analysis_id", ForeignKey("analysis.id", onupdate="CASCADE", ondelete="CASCADE")
    ),
    Column(
        "fragment_id", ForeignKey("fragment.id", onupdate="CASCADE", ondelete="CASCADE")
    ),
)

analysis_output_table = Table(
    "analysis_output",
    Base.metadata,
    Column(
        "analysis_id", ForeignKey("analysis.id", onupdate="CASCADE", ondelete="CASCADE")
    ),
    Column("topic_id", ForeignKey("topic.id", onupdate="CASCADE", ondelete="CASCADE")),
)


class Statement(Topic):
    """A fragment of text representing a standalone claim, question or category."""

    __tablename__ = "fragment"
    __mapper_args__ = {
        "polymorphic_load": "inline",
        "polymorphic_identity": topic_type.standalone,
    }
    pyd_model = StatementModel
    id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(Topic.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
    )  #: Primary key
    text: Mapped[Text] = mapped_column(
        Text, nullable=False
    )  #: What is the text content of the document
    scale: Mapped[fragment_type] = mapped_column(
        fragment_type_db, nullable=False
    )  #: What type of fragment?
    language: Mapped[String] = mapped_column(
        String, nullable=False
    )  # What is the language of the fragment? Inferred from document language or langdetect.
    generation_data: Mapped[Dict[str, Any]] = mapped_column(
        JSONB
    )  #: Data indicating the generation process
    confirmed: Mapped[Boolean] = mapped_column(
        Boolean, nullable=False, server_default="true"
    )  # Confirmed vs Draft
    doc_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(Document.id, onupdate="CASCADE", ondelete="CASCADE")
    )  #: Which document is this fragment part of (if any) (Defined here for summaries)
    context_of_analyses: Mapped[List[Analysis]] = relationship(
        "Analysis", secondary=analysis_context_table, back_populates="context"
    )
    theme_of_analyses: Mapped[List[Analysis]] = relationship(
        "Analysis", foreign_keys="Analysis.theme_id", back_populates="theme"
    )
    in_cluster_rels: Mapped[List[InClusterData]] = relationship(
        "InClusterData",
        back_populates="fragment",
        cascade="all, delete",
        passive_deletes=True,
    )

    @classmethod
    def ptmatch(cls, language=None):
        "For text search"
        vect = (
            func.to_tsvector(language, cls.text)
            if language
            else func.to_tsvector(cls.text)
        )
        return vect.op("@@", return_type=Boolean)

    async def load_sources(self, session):
        sources = []
        if source_ids := (self.generation_data or {}).get("sources"):
            sources = list(
                await session.scalars(
                    select(Fragment)
                    .filter(Fragment.id.in_(source_ids))
                    .options(
                        joinedload(Fragment.document)
                        .joinedload(Document.uri)
                        .subqueryload(UriEquiv.equivalents)
                    )
                )
            )
        claim_link = aliased(ClaimLink, flat=True)
        quotes = await session.scalars(
            select(Fragment)
            .join(claim_link, Fragment.incoming_links)
            .filter_by(link_type=link_type.quote, source=self.id)
            .options(
                joinedload(Fragment.document)
                .joinedload(Document.uri)
                .subqueryload(UriEquiv.equivalents)
            )
        )
        sources += list(quotes)
        for source in sources:
            if source.doc_id and not source.document:
                # BUG in SQLAlchemy
                document = await session.get(Document, source.doc_id)
                if document:
                    await session.refresh(document, ["uri"])
                    await session.refresh(document.uri, ["equivalents"])
                    source.document = document
        return sources

    @classmethod
    async def get_by_text(
        cls,
        session,
        txt: str,
        lang: Optional[str] = None,
        scale: Optional[fragment_type] = None,
        new_within_parent: Optional[Statement] = None,
    ) -> Statement:
        query = select(cls).filter_by(text=txt, doc_id=None)
        if new_within_parent:
            query = query.join(ClaimLink, ClaimLink.target == cls.id).filter_by(
                source=new_within_parent.id
            )
        existing = await session.scalar(query.limit(1))
        if existing:
            if scale and existing.scale == fragment_type.standalone:
                # Change if more precise
                logger.info(
                    f"Changing scale of statement {existing.id} from generic to {scale.name}"
                )
                existing.scale = scale
            return existing
        lang = lang or safe_lang_detect(txt)
        return Statement(text=txt, language=lang, scale=scale)

    def web_path(self, collection=globalScope):
        return f"{collection.path}/claim/{self.id}"

    def api_path(self, collection=globalScope):
        return f"/api{collection.path}/statement/{self.id}"


class Fragment(Statement):
    """A fragment of text. Can be part of a document, or even part of another fragment. It can be a standalone claim."""

    __mapper_args__ = {
        "polymorphic_load": "inline",
        "polymorphic_identity": topic_type.fragment,
    }
    pyd_model = FragmentModel
    position: Mapped[Integer] = mapped_column(
        Integer
    )  #: What is the relative position in the sequence of paragraphs
    char_position: Mapped[Integer] = mapped_column(
        Integer
    )  #: What is the character start position of this paragraph in the text
    part_of: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey("fragment.id", onupdate="CASCADE", ondelete="CASCADE")
    )  # Is this part of another fragment? (E.g. a sentence or quote in a paragraph)

    part_of_fragment: Mapped[Fragment] = relationship(
        "Fragment",
        foreign_keys=[part_of],
        remote_side=lambda: [Fragment.id],
        back_populates="sub_parts",
    )
    sub_parts: Mapped[List[Fragment]] = relationship(
        "Fragment", foreign_keys=[part_of], remote_side=[part_of], order_by=position
    )
    document: Mapped[Document] = relationship(
        Document,
        primaryjoin=and_(
            Statement.doc_id == Document.id, Statement.scale == fragment_type.paragraph
        ),
        back_populates="paragraphs",
    )

    def web_path(self, collection=globalScope):
        if self.scale == fragment_type.paragraph:
            return f"{collection.path}/document/{self.doc_id}#p_{self.id}"
        elif self.scale == fragment_type.summary:
            return f"{collection.path}/document/{self.doc_id}#s_{self.id}"
        else:
            return f"{collection.path}/document/{self.doc_id}#p_{self.part_of}"

    # TODO: Api path?


Document.summary: Mapped[List[Statement]] = relationship(
    Statement,
    primaryjoin=and_(
        Statement.doc_id == Document.id, Statement.scale == fragment_type.summary
    ),
)

Document.paragraphs: Mapped[List[Fragment]] = relationship(
    Fragment,
    primaryjoin=and_(
        Fragment.doc_id == Document.id, Fragment.scale == fragment_type.paragraph
    ),
    back_populates="document",
    order_by=Fragment.position,
    passive_deletes=True,
)

Document.quotes: Mapped[List[Fragment]] = relationship(
    Fragment,
    primaryjoin=and_(
        Fragment.doc_id == Document.id, Fragment.scale == fragment_type.quote
    ),
    back_populates="document",
    order_by=Fragment.position,
    passive_deletes=True,
)


Collection.claim_roots: Mapped[List[Statement]] = relationship(
    Statement,
    secondary=TopicCollection.__table__,
    secondaryjoin=(TopicCollection.__table__.c.topic_id == Statement.id)
    & (Statement.scale == fragment_type.standalone_root),
    overlaps="fragments",
)


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


def columns_of_selectable(sel):
    cols = set()
    traverse(sel, {}, {"column": lambda x: cols.add(x)})
    return cols


def collection_filter(
    query,
    collection_name,
    include_claims=False,
    include_paragraphs=False,
    target=None,
    doc_target=None,
):
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


class Analysis(Base):
    __tablename__ = "analysis"
    pyd_model = AnalysisModel
    id: Mapped[BigInteger] = mapped_column(BigInteger, primary_key=True)
    analyzer_id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(Analyzer.id, onupdate="CASCADE", ondelete="CASCADE"),
        nullable=False,
    )
    task_template_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(TaskTemplate.id, onupdate="CASCADE", ondelete="CASCADE")
    )
    target_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(Topic.id, onupdate="CASCADE", ondelete="SET NULL")
    )
    theme_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(Statement.id, onupdate="CASCADE", ondelete="SET NULL")
    )
    params: Mapped[Dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    results: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created: Mapped[DateTime] = mapped_column(
        DateTime, server_default="now()", nullable=False
    )
    completed: Mapped[DateTime] = mapped_column(DateTime)
    collection_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(Collection.id, onupdate="CASCADE", ondelete="CASCADE")
    )
    part_of_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey("analysis.id", onupdate="CASCADE", ondelete="CASCADE")
    )
    triggered_by_analysis_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey("analysis.id", onupdate="CASCADE", ondelete="SET NULL")
    )
    status: Mapped[process_status] = mapped_column(
        process_status_db, server_default="'complete'"
    )
    creator_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(User.id, onupdate="CASCADE", ondelete="SET NULL")
    )

    analyzer: Mapped[Analyzer] = relationship(
        Analyzer,
        foreign_keys=[analyzer_id],
        remote_side=[Analyzer.id],
        back_populates="analyses",
    )
    task_template: Mapped[Analyzer] = relationship(
        TaskTemplate, back_populates="analyses"
    )
    theme: Mapped[Statement] = relationship(
        Statement, foreign_keys=[theme_id], back_populates="theme_of_analyses"
    )
    generated_topics: Mapped[List[Topic]] = relationship(
        Topic, secondary=analysis_output_table, back_populates="from_analyses"
    )
    context: Mapped[List[Fragment]] = relationship(
        Fragment, secondary=analysis_context_table, back_populates="context_of_analyses"
    )
    collection: Mapped[Collection] = relationship(
        Collection, foreign_keys=[collection_id]
    )
    part_of: Mapped[Analysis] = relationship(
        "Analysis", foreign_keys=[part_of_id], remote_side=[id], backref="has_parts"
    )
    triggered_by_analysis: Mapped[Analysis] = relationship(
        "Analysis",
        foreign_keys=[triggered_by_analysis_id],
        remote_side=[id],
        backref="triggered_analyses",
    )
    target: Mapped[Topic] = relationship(Topic, back_populates="target_of_analyses")
    creator: Mapped[User] = relationship(User, foreign_keys=[creator_id])
    clusters: Mapped[List[ClusterData]] = relationship(
        "ClusterData", back_populates="analysis", passive_deletes=True
    )

    def web_path(self, collection=globalScope):
        analyzer_name = self.analyzer_name
        if self.id:
            return f"{collection.path}/analysis/{analyzer_name}/{self.id}"
        elif self.target_id:
            task = TaskRegistry.get_registry().get_task_cls_by_name(analyzer_name)
            if task.task_scale:
                return f"{collection.path}/{task.task_scale[0].name}/{self.target_id}/analysis/{analyzer_name}"
        # if self.task_template_id:
        #     return f"{collection.path}/analysis/{self.task_template_nickname}"
        return f"{collection.path}/analysis/{self.analyzer_name}"

    def api_path(self, collection=globalScope):
        analyzer_name = self.analyzer_name
        if self.id:
            return f"/api{collection.path}/analysis/{self.id}"
        else:
            # TODO: Improve this path
            return f"/api{collection.path}/analysis/type/{analyzer_name}"

    @property
    def task_template_nickname(self):
        if self.task_template_id:
            task_template_model = TaskRegistry.get_registry().task_template_by_id[
                self.task_template_id
            ]
            return task_template_model.nickname

    @property
    def collection_name(self):
        if not self.collection_id:
            return None
        if (
            inspect(self).attrs["collection"].loaded_value
            != LoaderCallableStatus.NO_VALUE
        ):
            return self.collection.name
        # Missing

    @property
    def analyzer_name(self):
        if (
            inspect(self).attrs["analyzer"].loaded_value
            != LoaderCallableStatus.NO_VALUE
        ):
            return self.analyzer.name
        analyzer = TaskRegistry.get_registry().analyzer_by_id.get(self.analyzer_id)
        return analyzer.name if analyzer else None

    @analyzer_name.setter
    def analyzer_name(self, name: str):
        analyzer = TaskRegistry.get_registry().analyzer_by_name.get(name)
        if analyzer:
            self.analyzer_id = analyzer.id

    def as_model(
        self,
        session,
        model_cls: Optional[Type[BaseModel]] = None,
        recursion: Optional[Set[int]] = None,
        **extra,
    ):
        model_cls = (
            model_cls
            or TaskRegistry.get_registry().analysis_model_by_name[self.analyzer_name]
        )
        if self.collection_id:
            # If this becomes async, I could load it...
            c = self.loaded_attributes().get("collection")
            if c:
                extra["collection_name"] = c.name
        ignore_keys = ["collection", "analyzer_id", "params", "analyzer"]
        if not issubclass(model_cls, AnalysisWithTemplateModel):
            ignore_keys.extend(["task_template_id", "task_template"])
        model = super(Analysis, self).as_model(
            session,
            model_cls,
            ignore_keys=ignore_keys,
            recursion=recursion,
            **(extra | self.params),
        )
        return model

    @classmethod
    async def from_model(
        cls, session, model: BaseModel, ignore: Optional[List[str]] = None, **extra
    ):
        assert isinstance(model, AnalysisModel)
        args = extra | model.model_dump()
        if model.collection_name and not model.collection_id:
            args["collection_id"] = await session.scalar(
                select(Collection.id).filter_by(name=model.collection_name)
            )
        rel_names = inspect(cls).relationships.keys()
        col_names = inspect(cls).c.keys()
        for k in (
            ["task_template_nickname", "analyzer_name", "collection_name"]
            + rel_names
            + (ignore or [])
        ):
            args.pop(k, None)
        col_args = {k: v for (k, v) in args.items() if k in col_names}
        col_args["params"] = jsonable_encoder(
            {k: v for (k, v) in args.items() if k not in col_names}
        )
        instance = cls(**col_args)
        if (
            getattr(model, "task_template", None) is not None
            and not model.task_template.id
        ):
            template_data = model.task_template.model_dump()
            for k in chain(
                inspect(TaskTemplate).c.keys(),
                inspect(cls).relationships.keys(),
                ("analyzer_name",),
            ):
                template_data.pop(k, None)
            template_model = await TaskRegistry.get_registry().ensure_task_template(
                session,
                model.analyzer_name,
                model.task_template_nickname,
                template_data,
            )
            instance.task_template_id = template_model.id
        if getattr(model, "task_template_id", None) is not None:
            instance.task_template = await session.get(
                TaskTemplate, model.task_template_id
            )
        if source_ids := getattr(model, "source_ids", []):
            instance.context = list(
                await session.scalars(
                    select(Fragment).filter(Fragment.id.in_(source_ids))
                )
            )
        return instance


class InClusterData(Base):
    __tablename__ = "in_cluster_data"
    pyd_model = InClusterDataModel
    cluster_id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey("cluster_data.id", onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
    )
    fragment_id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(Statement.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
    )
    confirmed_by_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(User.id, onupdate="CASCADE", ondelete="SET NULL")
    )
    manual: Mapped[Boolean] = mapped_column(Boolean, server_default="false")

    cluster: Mapped[ClusterData] = relationship(
        "ClusterData", foreign_keys=[cluster_id], back_populates="has_cluster_rels"
    )
    fragment: Mapped[Statement] = relationship(
        Statement, foreign_keys=[fragment_id], back_populates="in_cluster_rels"
    )
    confirmed_by: Mapped[User] = relationship(User, foreign_keys=[confirmed_by_id])


class ClusterData(Base):
    __tablename__ = "cluster_data"
    pyd_model = ClusterDataModel
    id: Mapped[BigInteger] = mapped_column(BigInteger, primary_key=True)
    cluster_size: Mapped[BigInteger] = mapped_column(BigInteger, server_default="1")
    analysis_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(Analysis.id, onupdate="CASCADE", ondelete="CASCADE")
    )
    distinguished_claim_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(Statement.id, onupdate="CASCADE", ondelete="SET NULL")
    )
    relevant: Mapped[relevance_type] = mapped_column(
        relevance_type_db, server_default="'unknown'"
    )
    relevance_checker_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(User.id, onupdate="CASCADE", ondelete="SET NULL")
    )
    auto_include_diameter: Mapped[Float] = mapped_column(Float)

    analysis: Mapped[Analysis] = relationship(Analysis, back_populates="clusters")
    distinguished_claim: Mapped[Statement] = relationship(
        Statement, foreign_keys=[distinguished_claim_id]
    )
    fragments: Mapped[List[Statement]] = relationship(
        Statement, secondary=InClusterData.__table__, back_populates="in_clusters"
    )
    relevance_checker: Mapped[User] = relationship(
        User, foreign_keys=[relevance_checker_id]
    )
    has_cluster_rels: Mapped[List[InClusterData]] = relationship(
        InClusterData,
        back_populates="cluster",
        cascade="all, delete",
        passive_deletes=True,
    )

    def web_path(self, collection=globalScope):
        return f"{collection.path}/analysis/cluster/{self.analysis_id}/{self.id}"

    def api_path(self, collection=globalScope):
        raise NotImplementedError()
        # return f"/api{collection.path}/analysis/{self.analysis_id}/cluster/{self.id}"


Statement.in_clusters: Mapped[List[ClusterData]] = relationship(
    ClusterData, secondary=InClusterData.__table__, back_populates="fragments"
)


class Embedding:
    """The vector embedding of a fragment's text. Abstract class."""

    dimensionality: int
    embedding_model_name: embedding_model
    pyd_model: Optional[Type[BaseModel]] = EmbeddingModel
    use_hnsw: bool = False
    scale: Mapped[fragment_type] = mapped_column(fragment_type_db, nullable=False)

    @declared_attr
    def __tablename__(cls) -> str:
        return f"embedding_{cls.embedding_model_name.name}"

    @declared_attr
    def fragment_id(cls) -> Mapped[BigInteger]:
        return mapped_column(
            BigInteger,
            ForeignKey("fragment.id", onupdate="CASCADE", ondelete="CASCADE"),
            nullable=True,
        )

    @declared_attr
    def doc_id(cls) -> Mapped[BigInteger]:
        return mapped_column(
            BigInteger,
            ForeignKey("document.id", onupdate="CASCADE", ondelete="CASCADE"),
            nullable=True,
            index=True,
        )

    @declared_attr
    def document(cls) -> Mapped[Document]:
        return relationship(
            Document,
            primaryjoin=(cls.doc_id == Document.id) & (cls.fragment_id.is_(None)),
        )

    @declared_attr
    def fragment(cls) -> Mapped[Statement]:
        return relationship(Statement, primaryjoin=(Statement.id == cls.fragment_id))

    @declared_attr
    def embedding(cls) -> Mapped[Vector]:
        return mapped_column(Vector(cls.dimensionality), nullable=False)

    @classmethod
    def txt_index(cls) -> Index:
        return Index(
            f"embedding_{cls.embedding_model_name.name}_cosidx",
            cls.embedding,
            postgresql_using="hnsw" if cls.use_hnsw else "ivfflat",
            postgresql_ops=dict(embedding="vector_cosine_ops"),
        )

    @classmethod
    def pseudo_pkey_index(cls) -> Index:
        return Index(
            f"embedding_{cls.embedding_model_name.name}_fragment_doc_idx",
            cls.fragment_id,
            cls.doc_id,
            unique=True,
        )

    @declared_attr
    def __table_args__(cls):
        return (cls.txt_index(), cls.pseudo_pkey_index())

    @declared_attr
    def __mapper_args__(cls):
        return dict(primary_key=[cls.fragment_id, cls.doc_id])

    @classmethod
    def distance(cls):
        return cls.embedding.cosine_distance

    @classmethod
    async def tf_embed(cls, txt):
        from .embed import tf_embed

        return await tf_embed(txt, cls.embedding_model_name.name)

    @classmethod
    async def makeEmbedding(cls, txt):
        embedding = await cls.tf_embed(txt)
        return cls(embedding=embedding)


def declare_embedding(model: embedding_model, dimension: int) -> Type[Embedding]:
    return type(
        f"Embedding_{model.name}",
        (Embedding, Base),
        dict(dimensionality=dimension, embedding_model_name=model),
    )


embedding_model_db = ENUM(embedding_model, name="embedding_model")

model_dimensionality: Dict[embedding_model, int] = {
    embedding_model[k]: cls.dimensionality for k, cls in embedder_registry.items()
}
model_names: Dict[embedding_model, str] = {
    embedding_model[k]: cls.display_name for k, cls in embedder_registry.items()
}
model_names_s = {k.name: v for k, v in model_names.items()}


all_embed_db_models: List[Type[Embedding]] = [
    declare_embedding(model, model_dimensionality[model]) for model in embedding_model
]

embed_db_model_by_name: Dict[str, Type[Embedding]] = {
    cls.embedding_model_name.name: cls for cls in all_embed_db_models
}


async def ensure_embedding_db_table(session, cls):
    await session.execute(CreateTable(cls.__table__, if_not_exists=True))
    for idx in cls.__table__.indexes:
        await session.execute(CreateIndex(idx, if_not_exists=True))


async def ensure_embedding_db_tables(session):
    for cls in all_embed_db_models:
        await ensure_embedding_db_table(session, cls)
    await session.commit()


class HyperEdge(Topic):
    """A link materialized as a node, but without content."""

    __mapper_args__ = {
        "polymorphic_load": "inline",
        "polymorphic_identity": topic_type.hyperedge,
    }
    pyd_model = HyperEdgeModel

    # Temporary
    scale = fragment_type.reified_arg_link


class ClaimLink(Topic):
    """A typed link between two standalone claims."""

    __tablename__ = "claim_link"
    pyd_model = ClaimLinkModel
    id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(Topic.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
    )
    __mapper_args__ = {
        "polymorphic_load": "inline",
        "polymorphic_identity": topic_type.link,
        "inherit_condition": id == Topic.id,
    }
    source: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(Topic.id, onupdate="CASCADE", ondelete="CASCADE"),
        nullable=False,
    )
    target: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(Topic.id, onupdate="CASCADE", ondelete="CASCADE"),
        nullable=False,
    )
    link_type: Mapped[link_type] = mapped_column(
        link_type_db, primary_key=True, nullable=False
    )
    score: Mapped[Float] = mapped_column(Float)

    source_topic: Mapped[Topic] = relationship(
        Topic,
        foreign_keys=[source],
        backref=backref("outgoing_links", passive_deletes=True),
    )
    target_topic: Mapped[Topic] = relationship(
        Topic,
        foreign_keys=[target],
        backref=backref("incoming_links", passive_deletes=True),
    )


StatementOrFragment = with_polymorphic(Statement, [Fragment], flat=True)
StatementAlone = with_polymorphic(Topic, [Statement], flat=True)
AnyClaimOrLink = with_polymorphic(Topic, [ClaimLink, Statement, HyperEdge], flat=True)
AnyClaimOrHyperedge = with_polymorphic(Topic, [Statement, HyperEdge], flat=True)
VisibleClaim = with_polymorphic(Statement, [], flat=True)
PolyTopic = with_polymorphic(Topic, "*", flat=True)
PolyTopicType = Union[Topic, ClaimLink, Statement, Fragment, Document, HyperEdge]

db_class_from_pyd_class: Dict[Type[BaseModel], Type[Base]] = {
    cls.pyd_model: cls for cls in Base.__subclasses__() if cls.pyd_model
}
db_class_from_pyd_class.pop(EmbeddingModel)


def search_target_class(include_claims, include_paragraphs):
    if include_claims:
        if include_paragraphs:
            return StatementOrFragment
        else:
            return Statement
    else:
        return Fragment


model_by_topic_type: Dict[topic_type, Type[Base]] = {
    topic_type.collection: Collection,
    topic_type.document: Document,
    topic_type.standalone: Statement,
    topic_type.fragment: Fragment,
    topic_type.hyperedge: HyperEdge,
    topic_type.link: ClaimLink,
}


async def claim_neighbourhood(nid: int, session) -> TypedDict(
    "ClaimNghd",
    node=Statement,
    children=List[
        Union[
            Tuple[Statement, ClaimLink],
            Tuple[Statement, ClaimLink, HyperEdge, ClaimLink],
        ]
    ],
    parents=List[
        Union[
            Tuple[Statement, ClaimLink],
            Tuple[Statement, ClaimLink, HyperEdge, ClaimLink],
        ]
    ],
):
    # All claim Fragments related to this one, including through a hyperedge
    flat_topic = with_polymorphic(Topic, [], flat=True, aliased=True)
    children = (
        select(
            ClaimLink.target.label("id"),
            flat_topic.type,
            literal_column("'child'").label("level"),
        )
        .join(flat_topic, ClaimLink.target_topic)
        .filter(ClaimLink.source == nid)
        .cte("children")
    )
    grandchildren = (
        select(
            ClaimLink.target.label("id"),
            flat_topic.type,
            literal_column("'grandchild'").label("level"),
        )
        .join(flat_topic, ClaimLink.target_topic)
        .join(
            children,
            (ClaimLink.source == children.c.id) & (children.c.type == "hyperedge"),
        )
        .cte("grandchildren")
    )
    parents = (
        select(
            ClaimLink.source.label("id"),
            flat_topic.type,
            literal_column("'parent'").label("level"),
        )
        .join(flat_topic, ClaimLink.source_topic)
        .filter(ClaimLink.target == nid)
        .cte("parents")
    )
    grandparents = (
        select(
            ClaimLink.source.label("id"),
            flat_topic.type,
            literal_column("'grandparent'").label("level"),
        )
        .join(flat_topic, ClaimLink.source_topic)
        .join(
            parents,
            (ClaimLink.target == parents.c.id) & (parents.c.type == "hyperedge"),
        )
        .cte("grandparents")
    )
    all_ids = (
        select(
            literal_column(str(nid), BigInteger).label("id"),
            literal_column("'self'").label("level"),
        )
        .union_all(
            select(children.c.id, children.c.level),
            select(grandchildren.c.id, grandchildren.c.level),
            select(parents.c.id, parents.c.level),
            select(grandparents.c.id, grandparents.c.level),
        )
        .cte("all_ids")
    )
    fragment_or_edge = with_polymorphic(
        Topic, [Statement, HyperEdge, Fragment], flat=True
    )
    q = (
        select(fragment_or_edge, all_ids.c.level)
        .join(all_ids, all_ids.c.id == fragment_or_edge.id)
        .order_by(all_ids.c.level)
        .options(
            subqueryload(fragment_or_edge.outgoing_links),
            subqueryload(fragment_or_edge.incoming_links),
        )
    )
    nodes = await session.execute(q)
    nodes = list(nodes)
    target = [n for n, l in nodes if l == "self"][0]
    by_id = {n.id: n for n, l in nodes}

    def get_paths(
        direction: bool,
    ) -> Iterable[
        Union[
            Tuple[Statement, Fragment, ClaimLink],
            Tuple[Statement, Fragment, ClaimLink, HyperEdge, ClaimLink],
        ]
    ]:
        for link in target.outgoing_links if direction else target.incoming_links:
            # direct_node = link.target_topic if direction else link.source_topic
            direct_node = by_id[link.target if direction else link.source]
            if direct_node.type != "hyperedge":
                yield (direct_node, link)
            else:
                for l2 in (
                    direct_node.outgoing_links
                    if direction
                    else direct_node.incoming_links
                ):
                    # indirect_node = l2.target_topic if direction else l2.source_topic
                    indirect_node = by_id[l2.target if direction else l2.source]
                    yield (indirect_node, l2, direct_node, link)

    return dict(
        node=target, children=list(get_paths(True)), parents=list(get_paths(False))
    )


def graph_subquery(
    root_id: int, graph_link_types=None, graph_statement_types=None, forward=True
):
    graph_link_types = graph_link_types or (
        link_type.answers_question,
        link_type.freeform,
        link_type.supported_by,
        link_type.opposed_by,
        link_type.subclaim,
        link_type.subcategory,
    )
    graph_statement_types = graph_statement_types or (
        fragment_type.standalone_question,
        fragment_type.standalone_claim,
        fragment_type.standalone_category,
        fragment_type.standalone_argument,
    )
    link_table = sqla_aliased(ClaimLink.__table__)
    lt1 = sqla_aliased(ClaimLink.__table__)
    topic_table = Topic.__table__
    statement_table = Statement.__table__
    if forward:
        base = select(lt1.c.id, lt1.c.source, lt1.c.target).filter(
            lt1.c.link_type.in_(graph_link_types), lt1.c.source == root_id
        )
    else:
        base = select(lt1.c.id, lt1.c.source, lt1.c.target).filter(
            lt1.c.link_type.in_(graph_link_types), lt1.c.target == root_id
        )
    base_cte = base.cte(recursive=True)
    base_ctea = sqla_aliased(base_cte, name="base")
    if forward:
        rec_q = base_ctea.union_all(
            select(link_table.c.id, link_table.c.source, link_table.c.target)
            .filter(
                link_table.c.link_type.in_(graph_link_types),
                link_table.c.source == base_ctea.c.target,
            )
            .join(topic_table, link_table.c.target == topic_table.c.id)
            .outerjoin(statement_table, link_table.c.target == statement_table.c.id)
            .filter(
                or_(
                    topic_table.c.type == topic_type.hyperedge,
                    statement_table.c.scale.in_(graph_statement_types),
                ),
            )
        )
    else:
        rec_q = base_ctea.union_all(
            select(link_table.c.id, link_table.c.source, link_table.c.target)
            .filter(
                link_table.c.link_type.in_(graph_link_types),
                link_table.c.target == base_ctea.c.source,
            )
            .join(topic_table, link_table.c.source == topic_table.c.id)
            .outerjoin(statement_table, link_table.c.source == statement_table.c.id)
            .filter(
                or_(
                    topic_table.c.type == topic_type.hyperedge,
                    statement_table.c.scale.in_(graph_statement_types),
                ),
            )
        )
    return rec_q


async def batch_lambda_query(session, lambda_query, large_list, batch_size=20000):
    # Allow batching very large queries
    acc = []
    large_list = large_list if isinstance(large_list, list) else list(large_list)
    for i in range(0, len(large_list), batch_size):
        acc += list(await session.execute(lambda_query(large_list[i : i + batch_size])))
    return acc


async def finalize_db_models():
    async with Session() as session:
        await ensure_dynamic_enum_values(session, permission_db)
        await ensure_dynamic_enum_values(session, embedding_model_db)
        await ensure_embedding_db_tables(session)


async def delete_data(session):
    await session.execute(delete(Analysis))
    await session.execute(delete(ClusterData))
    await session.execute(
        delete(Topic.__table__).where(Topic.__table__.c.type != topic_type.analyzer)
    )
    await session.execute(delete(UriEquiv))
    await session.execute(delete(Collection))
