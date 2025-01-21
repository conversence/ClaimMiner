# Copyright Conversence 2023
# license: Apache 2.0

from __future__ import annotations

from typing import Optional, Any, Dict, List, Literal, Union, Tuple, ForwardRef, Set, Type, Mapping
from datetime import datetime, timedelta, time, date
from itertools import chain
from collections import defaultdict
from enum import Enum
from pathlib import Path

import yaml
from rdflib import URIRef, Namespace
from rdflib.namespace import RDF, XSD
from sqlalchemy import select, delete
from sqlalchemy.sql.functions import count
from pydantic import Field, create_model, ConfigDict

from . import Session
from .models import NamespaceDef, Ontology, SchemaTerm, model_by_topic_type, Base, Topic, aliased
from .pyd_models import TopicModel, SchemaTermModel, topic_type, pyd_model_by_topic_type, HK
from .linkml_pydantic import PydanticModelGenerator

scalar_field_types: Dict[URIRef, Type] = {
    XSD.anyURI: URIRef,
    XSD.base64Binary: bytes,
    XSD.boolean: bool,
    XSD.date: date,
    XSD.dateTime: datetime,
    XSD.decimal: int,
    XSD.double: float,
    XSD.duration: timedelta,
    XSD.float: float,
    XSD.hexBinary: bytes,
    XSD.gDay: int,
    XSD.gMonth: int,
    XSD.gMonthDay: int,
    XSD.gYear: int,
    XSD.gYearMonth: int,
    # XSD.NOTATION: None,
    XSD.QName: str,  # PydanticURIRef
    XSD.string: str,
    XSD.time: time,
    XSD.language: str,  # langcodes?
    RDF.langString: str,  # LangStringModel
}


base_topic_types = {
    HK.document: topic_type.document,
    HK.fragment: topic_type.fragment,
    HK.statement: topic_type.standalone,
    HK.predicate: topic_type.link,
    HK.structure: topic_type.structured_idea,
}

base_db_models: Dict[URIRef, Type[Base]] = {k: model_by_topic_type[v] for k,v in base_topic_types.items()}

base_enums: Dict[URIRef, Type[Enum]] = {
    # DM.polarity: dm_polarity,
    # DM.claim_form: fragment_type,
    # DM.node_phrasing: dm_node_phrasing,
}


def as_field(range: URIRef) -> Tuple[Type, None]:
    ftype = scalar_field_types.get(range, Any)
    return (ftype, None)


class DynamicBaseSchema(TopicModel):
    model_config = ConfigDict(from_attributes=True, extra='forbid')
    type: Literal[topic_type.structured_idea] = topic_type.structured_idea
    # collections_id: Optional[List[int]] = None   # TODO
    # collections: Optional[List[CollectionModel]]

class OntologyRegistry():
    registry: OntologyRegistry

    def __init__(self):
        assert not getattr(self.__class__, 'registry', None), "Do not declare twice"
        self.__class__.registry = self
        self.prefixes: Dict[str, Namespace] = {}
        self.ontologies_by_prefix: Dict[str, Ontology] = {}
        self.terms: Dict[URIRef, SchemaTerm] = {}
        self.schema_by_curie: Dict[str, SchemaTerm] = {}
        self.term_models: Dict[URIRef, Type[SchemaTermModel]] = {}
        self.descendants: Dict[URIRef, Set[int]] = defaultdict(set)
        self.term_types: Dict[URIRef, topic_type] = {}
        self.schemas_by_id: Dict[int, SchemaTerm] = {}

    def read_ontology(self, fname: Path) -> Dict:
        with open(fname) as f:
            self.onto_data: Dict = yaml.safe_load(f)
        return self.onto_data

    @classmethod
    async def ensure_onto(cls, load_file=False, schema_file:str='schemas/core.linkml.yaml', delete_others:bool=False) -> OntologyRegistry:
        registry = OntologyRegistry()
        if not load_file:
            load_file = not await registry.load_onto_from_db()
        if load_file:
            await registry.load_onto_from_file(Path(schema_file), delete_others)
        return registry

    async def load_onto_from_db(self):
        async with Session() as session:
            r = await session.execute(select(NamespaceDef))
            namespaces = [n for (n,) in r]
            self.prefixes = {n.prefix: Namespace(n.uri) for n in namespaces}
            if not self.prefixes:
                return False
            base = [n for n in namespaces if n.is_base]
            assert len(base) == 1
            self.base_prefix = base[0].prefix
            self.base = self.prefixes[self.base_prefix]

            r = await session.execute(select(SchemaTerm))
            self.schemas_by_id = {s.id: s for (s,) in r}
            if not self.schemas_by_id:
                return False
            for schema_term in self.schemas_by_id.values():
                if schema_term.parent_id:
                    schema_term.parent = self.schemas_by_id[schema_term.parent_id]
            self.schema_by_curie = {s.prefix: s for s in self.schemas_by_id.values()}
            self.terms = {s.full_term: s for s in self.schemas_by_id.values()}
            self.generate_models()  # as a sanity check
            self.calc_descendants()
            return True

    async def load_onto_from_file(self, schema_file: Path, delete_others:bool=False):
        onto_data = self.read_ontology(schema_file)
        async with Session() as session:
            r = await session.execute(select(NamespaceDef))
            namespace_defs = {n.prefix: n for (n,) in r}
            # Clear old base value
            for ns in namespace_defs.values():
                ns.is_base = False
            for prefix, ns in onto_data['prefixes'].items():
                if nsdef := namespace_defs.get(prefix):
                    nsdef.uri = ns
                else:
                    nsdef = NamespaceDef(prefix=prefix, uri=ns)
                    namespace_defs[prefix] = nsdef
                    session.add(nsdef)
                self.prefixes[prefix] = Namespace(nsdef.uri)
            self.base = Namespace(onto_data['base'])
            for prefix, url in self.prefixes.items():
                if self.base == url:
                    self.base_prefix = prefix
                    namespace_defs[prefix].is_base = True
                    break
            else:
                raise RuntimeError("Declare the namespace of the base")
            await self.ensure_subclasses(session, onto_data['types'], namespace_defs)
            self.generate_models()  # as a sanity check
            self.calc_descendants()
            SchemaTermA = aliased(SchemaTerm, flat=True)
            all_terms = await session.execute(select(SchemaTermA.term, count(Topic.id)).outerjoin(Topic, Topic.schema_term_id==SchemaTermA.id).group_by(SchemaTermA.term))
            all_terms = {URIRef(t): n for (t, n) in all_terms}
            if others := set(all_terms) - set(self.terms):
                print ("Database terms missing in the ontologies:\n", "\n".join(others))
                if delete_others:
                    if others_empty := [o for o in others if not all_terms[o]]:
                        await session.execute(delete(SchemaTerm).where(SchemaTerm.term.in_(list(others_empty))))
                        print(f"Deleted {len(others_empty)} terms")
                    if others_used := [o for o in others if all_terms[o]]:
                        print("Did not delete those terms, which are used: \n", "\n".join(others_used))
            await session.commit()
            self.schemas_by_id = {s.id: s for s in self.terms.values()}

    def as_term(self, curie: Union[str, URIRef]) -> URIRef:
        if isinstance(curie, URIRef):
            return curie
        curiet = curie.split(':', 1)
        if len(curiet) == 1:
            return self.base[curie]
        else:
            prefix, short_term = curiet
            if prefix in self.prefixes:
                return self.prefixes[prefix][short_term]
            # Maybe already a URL?
            return URIRef(curie)

    def as_term_and_curie(self, term_or_curie: Union[str, URIRef]) -> Tuple[URIRef, Optional[str]]:
        if ':' not in term_or_curie:
            return self.base[term_or_curie], f"{self.base_prefix}:{term_or_curie}"
        for prefix, ns in self.prefixes.items():
            if term_or_curie.startswith(f"{prefix}:"):
                return ns[term_or_curie.split(':',1)[1]], term_or_curie
            if term_or_curie.startswith(ns):
                end = term_or_curie[len(ns):]
                return ns[end], f"{prefix}:{end}"
        return URIRef(term_or_curie), None

    async def ensure_subclasses(self, session, classes, namespace_defs: Mapping[str, NamespaceDef], parent: Optional[SchemaTerm]=None):
        for sclass, info in classes.items():
            info = info or {}
            term, curie = self.as_term_and_curie(sclass)
            prefix, short_term = self.curie_elements(term)
            assert prefix in self.prefixes
            assert term not in self.terms, f"Repeated term: {term}"
            assert curie
            assert curie not in self.schema_by_curie, f"Repeated curie: {curie}"
            subclasses = info.pop('subClasses', {})
            term_db = await session.scalar(select(SchemaTerm).filter_by(term=short_term).join(SchemaTerm.ontology).filter_by(prefix=prefix))
            if term_db:
                term_db.parent = parent
                term_db.data = info
            else:
                term_db = SchemaTerm(term=short_term, prefix=prefix, data=info, parent=parent, namespace=namespace_defs[prefix])
                session.add(term_db)
            self.terms[term] = term_db
            self.schema_by_curie[curie] = term_db
            await self.ensure_subclasses(session, subclasses, namespace_defs, term_db)

    def curie_elements(self, term) -> Tuple[str, str]:
        for prefix, t in self.prefixes.items():
            if term.startswith(t):
                short_term = str(term[len(t):])
                return prefix, short_term
        assert False, "Missing namespace"

    def as_curie(self, term: URIRef) -> str:
        prefix, short_term = self.curie_elements(term)
        return f"{prefix}:{short_term}"

    def model_classname(self, short_term, prefix=None) -> str:
        if not prefix:
            prefix, short_term = self.curie_elements(short_term)
        return f"{str(prefix).title()}{''.join(short_term.title().split('_'))}"

    def model_for_term_(self, term_s:str) -> Union[Type[TopicModel], ForwardRef]:
        term = self.as_term(term_s)
        if term in base_topic_types:
            return pyd_model_by_topic_type[base_topic_types[term]]
        if term in self.term_models:
            return self.term_models[term]
        return ForwardRef(self.model_classname(term))

    def db_model_for_term(self, term: URIRef) -> Type[Topic]:
        schema = self.terms.get(term)
        assert schema
        for s in schema.ancestor_chain():
            if model := base_db_models.get(URIRef(s.term)):
                return model
        return Topic

    def generate_model(self, db_model: SchemaTerm) -> Type[SchemaTermModel]:
        schema = db_model.data
        curie = db_model.curie
        classname = self.model_classname(db_model.term, db_model.prefix)
        attributes = dict(schema_term_term=(Literal[curie], Field(curie)))
        parent: Type[SchemaTermModel] = DynamicBaseSchema
        if db_model.parent_id:
            parent = self.term_models[db_model.parent.full_term]
        for att_name, info in schema.get('attributes', {}).items():
            default:Any = None
            info = info or {}
            range_t = self.as_term(info.get('range')) if 'range' in info else None
            att_type: Optional[Type] = scalar_field_types.get(range_t) or base_enums.get(range_t)
            if not att_type:
                print(f"Missing! {att_name}: {att_type}")
                continue
            if info.get('multiple', False):
                att_type = List[att_type]
                default = []
            if info.get('optional', False):
                att_type = Optional[att_type]
                default = None
            attributes[att_name] = (att_type, default)

        for att_name, info in schema.get('roles', {}).items():
            default = None
            info = info or {}
            range_t: Union[str, List[str]] = info.get('range')
            att_type = int
            if not att_type:
                print(f"Missing! {att_name}: {att_type}")
                continue
            if info.get('multiple', False):
                att_name_id = f'{att_name}_ids'
                att_type = List[att_type]
                default = []
            else:
                att_name_id = f'{att_name}_id'
            if info.get('optional', False):
                att_type = Optional[att_type]
                default = None
            attributes[att_name_id] = (att_type, default)
            if range_t:
                if isinstance(range_t, list):
                    types: Tuple[Type] = tuple(self.model_for_term_(t) for t in range_t)
                    if not types:
                        print("Empty type list:", att_name, info)
                        continue
                    elif len(types) == 1:
                        att_model = types[0]
                    else:
                        att_model = Union[types]  # type: ignore
                else:
                    att_model: type[TopicModel] = self.model_for_term_(range_t)
                if info.get('multiple', False):
                    att_model = List[att_model]
                attributes[att_name] = (Optional[att_model], None)
        return create_model(
            classname,
            __base__=parent,
            __validators__= {},
            **attributes)

    def generate_models(self):
        incompletes = {}
        known = set(base_topic_types.keys())
        for db_model in self.terms.values():
            ancestors = list(db_model.ancestor_chain())
            ancestors.reverse()
            broken = False
            for dbm in ancestors:
                term = dbm.full_term
                if term in self.term_models:
                    continue
                if broken or (set(self.as_term(t) for t in dbm.range_types()) - known):
                    incompletes[term] = dbm
                    broken = True
                else:
                    self.term_models[term] = self.generate_model(dbm)
                    known.add(term)
        while incompletes:
            for term, db_model in list(incompletes.items()):
                if db_model.parent and db_model.parent.full_term not in self.term_models:
                    continue
                if (set(self.as_term(t) for t in db_model.range_types()) - known):
                    continue
                print('rebuilding', term)
                self.term_models[term] = self.generate_model(db_model)
                known.add(term)
                del incompletes[term]

    def as_union_type(self, *subclasses_of_types: List[str]):
        subclasses_of = [self.term_models.get(self.as_term(subclasses_of_type)) for subclasses_of_type in subclasses_of_types]
        subclasses_of = tuple(filter(None, subclasses_of))
        return Union[tuple(v for v in self.term_models.values() if not subclasses_of or issubclass(v, subclasses_of))]

    def as_field_type(self, *subclasses_of: List[str]):
        return Field(self.as_union_type(subclasses_of), discriminator='schema_term_term')

    def calc_descendants(self):
        for term, db_model in self.terms.items():
            parent_type = None
            for parent in db_model.ancestor_chain():
                parent_term = URIRef(parent.term)
                self.descendants[parent_term].add(db_model.id)
                if parent_term in base_topic_types and parent_type is None:
                    parent_type = base_topic_types[parent_term]
                    self.term_types[term] = parent_type

    def subtype_constraints(self, query, entity: Type[Topic], term: URIRef, exact=False):
        query = query.filter(entity.type == self.term_types[term])
        if exact:
            query = query.filter(entity.schema_term_id == self.term_models[term].id)
        else:
            descendents = self.descendants[term]
            if len(descendents) > 1:
                query = query.filter(entity.schema_term_id.in_(list(descendents)))
        return query

    def subtype_constraints_multiple(self, query, entity: Type[Topic], terms: List[URIRef], exact=False):
        types = {self.term_types[t] for t in terms}
        if len(types) == 1:
            query = query.filter(entity.type == list(types)[0])
        else:
            query = query.filter(entity.type.in_(list(types)))
        if exact:
            schema_ids: Set[int] = {self.term_models[term].id for term in terms}
        else:
            schema_ids = set(chain(*(self.descendants[term] for term in terms)))
        if len(schema_ids) > 1:
            query = query.filter(entity.schema_term_id.in_(list(schema_ids)))
        return query

if __name__ == '__main__':
    from argparse import ArgumentParser
    from asyncio import run
    parser = ArgumentParser()
    parser.add_argument('--load_file', action='store_true')
    parser.add_argument('--schema', type=Path, default='schemas/core.yaml')
    parser.add_argument('--delete-others', default=False, action='store_true')
    args = parser.parse_args()
    run(OntologyRegistry.ensure_onto(args.load_file, args.schema, args.delete_others))
