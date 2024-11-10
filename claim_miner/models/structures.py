from __future__ import annotations

from typing import Union, List, Dict, Any, Mapping, Optional, Type, Set
from enum import Enum

from sqlalchemy import BigInteger, ForeignKey, select, String, Boolean, FetchedValue, DateTime
from sqlalchemy.sql import any_, func
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, array
from sqlalchemy.orm import Mapped, mapped_column, relationship, foreign, remote, validates
from rdflib import URIRef, Namespace

from ..pyd_models import (
    topic_type,
    SchemaDefModel,
    AbstractStructuredIdeaModel,
)
from . import (
    BaseModel
)
from .base import Base, Topic



class NamespaceDef(Base):
    __tablename__ = 'namespace'
    prefix: Mapped[String] = mapped_column(String, nullable=False, primary_key=True)
    uri: Mapped[String] = mapped_column(String, nullable=False, unique=True)
    is_base: Mapped[Boolean] = mapped_column(Boolean, default=False, server_default='false')


class SchemaDef(Topic):
    """A schema definition"""
    __tablename__ = 'schema_def'
    id: Mapped[BigInteger] = mapped_column(BigInteger, ForeignKey(Topic.id, onupdate='CASCADE', ondelete='CASCADE'), primary_key=True)  #: Primary key
    __mapper_args__ = {
        "polymorphic_load": "inline",
        "polymorphic_identity": topic_type.schema_def,
        "inherit_condition": id==Topic.id,
    }
    pyd_model = SchemaDefModel
    prefix: Mapped[String] = mapped_column(String, ForeignKey(NamespaceDef.prefix, onupdate='CASCADE', ondelete='CASCADE'), nullable=False)
    term: Mapped[String] = mapped_column(String, nullable=False)
    data: Mapped[Dict] = mapped_column(JSONB, nullable=False)
    parent_id: Mapped[BigInteger] = mapped_column(BigInteger, ForeignKey(id, onupdate='CASCADE', ondelete='SET NULL'))
    ancestors_id: Mapped[List[BigInteger]] = mapped_column(ARRAY(BigInteger), server_default=FetchedValue(), server_onupdate=FetchedValue())

    parent: Mapped[SchemaDef] = relationship("SchemaDef", primaryjoin=foreign(parent_id)==remote(id), uselist=False)
    ancestors: Mapped[List[SchemaDef]] = relationship("SchemaDef", primaryjoin=remote(id)==any_(foreign(ancestors_id)), uselist=True)
    namespace: Mapped[NamespaceDef] = relationship(NamespaceDef, lazy='joined')

    @property
    def full_term(self) -> URIRef:
        return Namespace(self.namespace.uri)[self.term]

    @property
    def curie(self) -> str:
        return f"{self.prefix}:{self.term}"

    def range_types(self):
        existing = set()
        for role in self.data.get("roles", {}).values():
            range = role.get('range', [])
            if range and isinstance(range, list):
                for r in range:
                    if r not in existing:
                        existing.add(r)
                        yield r
            elif range:
                if range not in existing:
                    existing.add(range)
                    yield range

    def ancestor_chain(self):
        # assumes that parents are loaded.
        while self:
            yield self
            self = self.parent

    def is_subschema(self, term: URIRef):
        term = str(term)
        for schema in self.ancestor_chain():
            if schema.term == term:
                return schema

    def inherited_data(self, aspect='attributes'):
        base = {}
        schema = self
        while schema:
            local_aspect = schema.data.get(aspect, {})
            base = local_aspect | base
            schema = schema.parent
        return base

    @property
    def all_attributes(self):
        return self.inherited_data('attributes')

    @property
    def all_roles(self):
        return self.inherited_data('roles')


Topic.schema_def: Mapped[SchemaDef] = relationship(SchemaDef, foreign_keys=Topic.schema_def_id, remote_side=SchemaDef.id)


class StructuredIdea(Topic):
    """A structured idea"""
    __tablename__ = 'structured_idea'
    id: Mapped[BigInteger] = mapped_column(BigInteger, ForeignKey(Topic.id, onupdate='CASCADE', ondelete='CASCADE'), primary_key=True)  #: Primary key
    __mapper_args__ = {
        "polymorphic_load": "inline",
        "polymorphic_identity": topic_type.structured_idea,
        "inherit_condition": id==Topic.id
    }
    pyd_model = AbstractStructuredIdeaModel
    ref_structure: Mapped[Dict[str, Union[int, List[int]]]] = mapped_column(JSONB, server_default='{}')
    literal_structure: Mapped[Dict[str, Any]] = mapped_column(JSONB, server_default='{}')
    modified_at: Mapped[DateTime] = mapped_column(DateTime, nullable=False, server_default='now()')
    refs: Mapped[List[BigInteger]] = mapped_column(ARRAY(BigInteger), nullable=False, server_onupdate=FetchedValue(), server_default=FetchedValue())

    references: Mapped[List[Topic]] = relationship(Topic, primaryjoin=foreign(refs).contains(array([remote(Topic.id)])), viewonly=True, uselist=True, back_populates="in_structures")
    references_rec: Mapped[List[Topic]] = relationship(Topic, primaryjoin=remote(Topic.id)==foreign(any_(func.sub_structures_rec(id))), viewonly=True, uselist=True) # back_populates="in_structures"

    @property
    def term(self):
        return self.schema.term

    @classmethod
    def from_model(cls, model: BaseModel, **extra):
        assert isinstance(model, AbstractStructuredIdeaModel)
        # The models have a flatter structure, convert to db structure
        # Assume ontology was loaded
        from .ontology import Ontology
        ontology = Ontology.ontology
        schema = ontology.terms[ontology.as_term(model.schema_def_term)]
        assert schema
        literal_structure = {att: getattr(model, att) for att in schema.all_attributes}
        literal_structure = {k: v.name if isinstance(v, Enum) else v for (k, v) in literal_structure.items()}
        # assuming everything was flushed...
        def get_id_of_role(rolename):
            ob = getattr(model, rolename, None)
            if ob:
                if isinstance(ob, list):
                    return [o.id for o in ob]
                else:
                    return ob.id
            else:
                plural = "s" if schema.all_roles[rolename].get('multiple', False) else ''
                role_id_name = f"{rolename}_id{plural}"
                return getattr(model, role_id_name, None)

        ref_structure = {r: get_id_of_role(r) for r in schema.all_roles}
        # ref_structure = {k: v for (k, v) in ref_structure.items() if v}
        return cls(schema_def_id=schema.id, created_by=get_id_of_role('created_by'), ref_structure=ref_structure, literal_structure=literal_structure, **extra)

    async def preload_substructures(self, session):
        f = func.sub_structures_rec(self.id).table_valued('id')
        r = await session.execute(select(Topic).join(f, Topic.id==f.c.id))
        topics = {x.id: x for (x,) in r}
        return topics


    def as_model(self, session, model_cls: Optional[Type[BaseModel]]=None, recursion: Optional[Set[int]]=None):
        # The models have a flatter structure, convert from db structure
        if (not model_cls) and self.schema_def_id:
            from .ontology import Ontology
            ontology = Ontology.ontology
            schema = ontology.schemas_by_id[self.schema_def_id]
            model_cls = ontology.term_models.get(schema.term)
        model_cls = model_cls or self.pyd_model
        assert model_cls
        assert issubclass(model_cls, self.pyd_model)
        base = self.loaded_attributes()
        instances = session._proxied.identity_map
        recursion = recursion or set()
        recursion.add(self.id)
        for role, ids in self.ref_structure.items():
            if isinstance(ids, list):
                id_name = f"{role}_ids"
                base[id_name] = ids
                vals = filter(None, [instances.get((Topic, (i,), None)) for i in ids if i not in recursion])
                if vals:
                    base[role] = [v.as_model(session, recursion=recursion) for v in vals]
            else:
                id_name = f"{role}_id"
                base[id_name] = ids
                ob = instances.get((Topic, (ids,), None))
                if ob is not None and ob.id not in recursion:
                    base[role] = ob.as_model(session, recursion=recursion)
        literal_structure = base.get('literal_structure', {})
        for k, v in literal_structure.items():
            if v is not None:
                v_type = model_cls.model_fields[k].annotation
                if v_type and isinstance(v_type, type) and issubclass(v_type, Enum) and not isinstance(v, v_type):
                    literal_structure[k] = v_type[v]
        base = base | literal_structure
        model = model_cls.model_validate(base)
        return(model)

    async def set_actors(self, session, ref_objects: Mapping[str, Union[Topic, List[Topic]]], adding=True):
        # TODO: Make this happen automatically upon flush, with ref_objects as an unmapped instance variable. (Hybrid accessor?)
        needs_flush = False
        for k, v in ref_objects.items():
            if isinstance(v, list):
                for ob in v:
                    if not ob.id:
                        session.add(ob)
                        needs_flush = True
            else:
                if not v.id:
                    session.add(v)
                    needs_flush = True
        if needs_flush:
            await session.flush()
        ref_structure = dict(self.ref_structure) if adding else {}
        for k, v in ref_objects.items():
            if isinstance(v, list):
                ref_structure[k] = [ob.id for ob in v]
                ref_structure[k].sort()
            else:
                ref_structure[k] = v.id
        if needs_flush or not adding or self.ref_structure != ref_structure:
            self.ref_structure = ref_structure

    @validates('ref_structure')
    def validate_ref_structure(self, key, ref_structure):
        for v in ref_structure.values():
            if isinstance(v, list):
                v.sort()
        return ref_structure


Topic.in_structures: Mapped[List[StructuredIdea]] = relationship(StructuredIdea, primaryjoin=foreign(StructuredIdea.refs).contains(array([remote(Topic.id)])), viewonly=True, uselist=True, back_populates="references")
# TODO: Include ClaimLinks
Topic.in_structures_rec: Mapped[List[StructuredIdea]] = relationship(StructuredIdea, primaryjoin=remote(StructuredIdea.id)==any_(foreign(func.in_structures_rec(Topic.id))), viewonly=True, uselist=True)
