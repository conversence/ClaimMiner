from __future__ import annotations
from typing import Dict, Any, Type
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field as dcfield


from pydantic import BaseModel
from rdflib import URIRef

from claim_miner.pyd_models import ontology_status

# what do I expect from ontologies
# idea: not try to have a single context, or at least punt it. Each ontology defines its own context.
# Issue: Note that a single request may return structures from many ontologies; so per-object context.


class BaseOntologyRegistry(ABC):
    prefixes: Dict[str, URIRef] = dcfield(default_factory=dict)
    ontologies_by_prefix: Dict[str, OntologyData] = dcfield(default_factory=dict)
    ontology_handlers: Dict[str, OntologyHandler] = dcfield(default_factory=dict)
    local_url_prefix: URIRef = dcfield()

    def prepare_ontology(self, data: Any, ontology_language: str = 'linkml') -> OntologyData:
        if handler := self.ontology_handlers.get(ontology_language):
            ontology = handler.prepare(self, data)
            self.ontologies_by_prefix[ontology.base_prefix] = ontology
            return ontology

    def as_uri(self, curie: str) -> URIRef:
        pass  # TODO

    def as_curie(self, uri: URIRef) -> str:
        pass  # TODO


@dataclass
class OntologyHandler(ABC):
    ontology_language: str = dcfield()

    @abstractmethod
    async def prepare(self, data) -> OntologyData:
        pass


@dataclass
class OntologyData(ABC):
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

@dataclass
class OntologyClassData(ABC):
    deprecated: bool = dcfield()

    @abstractmethod
    def as_pydantic(self) -> Type[BaseModel]:
        pass

    @abstractmethod
    def parse(self, data: Any) -> BaseModel:
        pass

