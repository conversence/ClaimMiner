from __future__ import annotations

from collections import defaultdict
from itertools import chain
from typing import Mapping, Set, Tuple, Optional, Type, List, ForwardRef

from pydantic import BaseModel
from sqlalchemy import event, inspect, ClauseList, BigInteger, ForeignKey
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import (
    DeclarativeBase,
    LoaderCallableStatus,
    Mapped,
    mapped_column,
    relationship,
)

from .. import sync_maker
from ..pyd_models import topic_type
from ..utils import filter_dict
from . import logger, db_class_from_pyd_class

flushed_objects_by_session: Mapping[int, Set[Tuple[topic_type, int]]] = defaultdict(set)
created_objects: Set[Tuple[topic_type, int]] = set()

globalScope: ForwardRef("CollectionScope") = None


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
