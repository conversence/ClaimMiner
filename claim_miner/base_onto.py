from __future__ import annotations
from typing import Dict, Any, Type, Union, Tuple, Optional, Iterable, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field as dcfield
from collections import defaultdict


from pydantic import BaseModel
from rdflib import URIRef

from claim_miner.pyd_models import ontology_status

# what do I expect from ontologies
# idea: not try to have a single context, or at least punt it. Each ontology defines its own context.
# Issue: Note that a single request may return structures from many ontologies; so per-object context.

class BaseOntologyRegistry():
    registry: BaseOntologyRegistry = None  # Singleton

    prefixes: Dict[str, URIRef]
    ontologies_by_prefix: Dict[str, BaseOntologyData]
    ontology_handlers: Dict[str, OntologyHandler]
    local_url_prefix: URIRef
    schema_by_term: Dict[URIRef, BaseModel]
    subclasses: Dict[URIRef, List[BaseModel]]

    @classmethod
    def instantiate(cls, local_url: URIRef=None, local_url_prefix: str='local'):
        assert not cls.registry, "Singleton"
        return cls(local_url, local_url_prefix)

    def __init__(self, local_url: URIRef=None, local_url_prefix: str='local'):
        assert not getattr(self.__class__, 'registry', None), "Do not declare twice"
        self.prefixes = {}
        if local_url:
            self.prefixes[local_url_prefix] = local_url
        else:
            local_url_prefix = None
        self.local_url_prefix = local_url_prefix
        self.ontologies_by_prefix = {}
        self.ontology_handlers = {}
        self.schema_by_term = {}
        self.subclasses = defaultdict(list)
        self.__class__.registry = self

    def register_handler(self, handler: OntologyHandler):
        assert handler.ontology_language not in self.ontology_handlers, f"{handler.ontology_language} registered twice"
        self.ontology_handlers[handler.ontology_language] = handler

    def register_ontology(self, ontology: BaseOntologyData):
        self.ontologies_by_prefix[ontology.base_prefix] = ontology

    def register_class(self, term: URIRef, schema: Type[BaseModel], ontology: BaseOntologyData):
        self.schema_by_term[term] = schema
        for c in schema.mro():
            if c is BaseModel or c is schema:
                continue
            # TODO: Define this in an appropriate subclass of BaseModel, or create a reverse lookup in BaseOntologyData
            superclass_term = c.schema_term_term
            self.subclasses[superclass_term].append(schema)

    def prepare_ontology(self, data: Any, ontology_language: str = 'linkml') -> BaseOntologyData:
        if handler := self.ontology_handlers.get(ontology_language):
            ontology = handler.prepare(self, data)
            self.ontologies_by_prefix[ontology.base_prefix] = ontology
            # Here assuming no prefix collision...
            # Actually should give priority to ontologies' base prefixes
            self.prefixes.update(ontology.prefixes)
            return ontology

    def from_file(self, fname, ontology_language: str = 'linkml') -> BaseOntologyData:
        if handler := self.ontology_handlers.get(ontology_language):
            if ontology := handler.from_file(fname):
                self.register_ontology(ontology)
                return ontology

    def from_db_schema(self, db_schema, ontology_language: str = 'linkml') -> BaseOntologyData:
        if handler := self.ontology_handlers.get(ontology_language):
            if ontology := handler.from_db_schema(db_schema):
                self.register_ontology(ontology)
                return ontology

    def get_pydantic_schema(self, term_or_curie: Union[str, URIRef]) -> Type[BaseModel]:
        term, curie = self.as_term_and_curie(term_or_curie)
        if not curie:
            return None
        prefix, short_term = curie.split(':')
        ontology = self.ontologies_by_prefix.get(prefix)
        if not ontology:
            return None
        return ontology.get_pydantic_schema(short_term)
        # MAYBE TODO: add root cases from the schema

    def get_union_of_pydantic_schemas(self, term_or_curies: Iterable[Union[str, URIRef]]) -> Type:
        schemas = [self.get_pydantic_schema(term_or_curie) for term_or_curie in term_or_curies]
        schemas = tuple(filter(None, schemas))
        if not schemas:
            return None
        if len(schemas) == 1:
            return schemas[0]
        return Union[*schemas]

    def as_union_type(self, *subclasses_of_types: List[Union[str, URIRef]]):
        # This function and previous should do the same.
        subclasses_of = [self.get_pydantic_schema(subclasses_of_type) for subclasses_of_type in subclasses_of_types]
        subclasses_of = tuple(filter(None, subclasses_of))
        # Issue: here I should ask the source ontology for the list... but subclasses could be cross-ontology! Where would that be stored?
        return Union[tuple(v for v in self.term_models.values() if not subclasses_of or issubclass(v, subclasses_of))]

    def as_term(self, curie: Union[str, URIRef]) -> URIRef:
        if isinstance(curie, URIRef):
            return curie
        curiet = curie.split(':', 1)
        if len(curiet) == 1:
            assert self.local_url_prefix
            return self.local_url[curiet[1]]
        else:
            prefix, short_term = curiet
            if prefix in self.prefixes:
                return self.prefixes[prefix][short_term]
            # Maybe already a URL?
            return URIRef(curie)

    def as_curie(self, uri: URIRef) -> str:
        for prefix, term in self.prefixes.values():
            if uri.startswith(term):
                return f"{prefix}:{term[len(term):]}"

    def as_term_and_curie(self, term_or_curie: Union[str, URIRef]) -> Tuple[URIRef, Optional[str]]:
        if isinstance(term_or_curie, URIRef):
            return (term_or_curie, self.as_curie(term_or_curie))
        for prefix, ns in self.prefixes.items():
            if term_or_curie.startswith(f"{prefix}:"):
                return ns[term_or_curie.split(':',1)[1]], term_or_curie
            if term_or_curie.startswith(ns):
                end = term_or_curie[len(ns):]
                return ns[end], f"{prefix}:{end}"
        if self.local_url_prefix and ':' not in term_or_curie:
            return self.local_url[term_or_curie], f"{self.local_url_prefix}:{term_or_curie}"
        return URIRef(term_or_curie), None

    @property
    def base_prefix(self):
        return self.prefixes.get(self.local_url_prefix)


@dataclass
class OntologyHandler(ABC):
    ontology_language: str = dcfield()

    @abstractmethod
    def from_file(self, fname) -> BaseOntologyData:
        pass

    @abstractmethod
    def from_db_schema(self, db_schema) -> BaseOntologyData:
        pass


@dataclass
class BaseOntologyData(ABC):
    ontology_language: str = dcfield()
    base_prefix: str = dcfield()
    prefixes: Dict[str, URIRef] = dcfield(default_factory=dict)
    classes: Dict[str, OntologyClassData] = dcfield(default_factory=dict)
    status: ontology_status = dcfield(default=ontology_status.draft)

    @property
    def base_url(self):
        return self.prefixes[self.base_prefix]

    @abstractmethod
    def as_context(self) -> Any:
        # No jsonld context type?
        pass

    def as_term(self, curie: Union[str, URIRef]) -> URIRef:
        if isinstance(curie, URIRef):
            return curie
        curiet = curie.split(':', 1)
        if len(curiet) == 1:
            return self.base_url[curie]
        else:
            prefix, short_term = curiet
            if prefix in self.prefixes:
                return self.prefixes[prefix][short_term]
            # Maybe already a URL?
            return URIRef(curie)

    def as_term_and_curie(self, term_or_curie: Union[str, URIRef]) -> Tuple[URIRef, Optional[str]]:
        if ':' not in term_or_curie:
            return self.base_url[term_or_curie], f"{self.base_prefix}:{term_or_curie}"
        for prefix, ns in self.prefixes.items():
            if term_or_curie.startswith(f"{prefix}:"):
                return ns[term_or_curie.split(':',1)[1]], term_or_curie
            if term_or_curie.startswith(ns):
                end = term_or_curie[len(ns):]
                return ns[end], f"{prefix}:{end}"
        return URIRef(term_or_curie), None

    @abstractmethod
    def get_pydantic_schema(self, short_term: str) -> BaseModel:
        pass

@dataclass
class OntologyClassData(ABC):
    deprecated: bool = dcfield()

    @abstractmethod
    def as_pydantic(self) -> Type[BaseModel]:
        pass

    @abstractmethod
    def parse(self, data: Any) -> BaseModel:
        pass

