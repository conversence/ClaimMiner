from copy import deepcopy

from dataclasses import dataclass, field as dcfield
from typing import List, Optional, Dict, Type, Union, Generator, Tuple, Set, Any, ForwardRef
from types import new_class
from datetime import datetime, date, time, timedelta
import os
from enum import Enum, StrEnum

from linkml_runtime.linkml_model.meta import (
    ClassDefinition,
    SchemaDefinition,
    SlotDefinition,
    SubsetDefinition,
    TypeDefinition,
    EnumDefinition,
    AnonymousSlotExpression,
)
from linkml.utils.generator import Generator as LMGenerator
from pydantic import BaseModel, Field, create_model
from rdflib.namespace import XSD, RDF, RDFS, Namespace
from rdflib import URIRef

from .pydantic_adapters import PydanticURIRef
from .pyd_models import ontology_status
from .base_onto import BaseOntologyRegistry, OntologyData, OntologyClassData, OntologyHandler

DEBUG = True
if DEBUG:
    import pdb

# This is wrong, because it treats the ontology as an isolate. I need an ontology status prefix.
# What if a class is defined in two ontologies? How does linkml do cross-refs?
# What if a class is only defined in ontologies where it's not main?
# What if there is prefix conflict (Flatly refuse imho)

# I considered multiple version of ontology at same URL, it's a nightmare.
# Each ontology has a unique prefix/URL.
# Lifecycle is draft, published, deprecated, obsolete. Unidirectional.
# A published ontology should not change substantially, besides correcting typos in comments.
# Though we may allow additions, like minor versions? What about class-level deprecations?
# A published ontology should not refer to a draft ontology.
# Structures based on draft ontologies should not be public. (not sure how to enforce that)
# Structures based on deprecated ontology can be upgraded with a lens.
# When a deprecated ontology has no live structures, it can be made obsolete, and taken out of API.


SHEX = Namespace("http://www.w3.org/ns/shex#")
LINKML = Namespace("https://w3id.org/linkml/")

# LinkML types to RDF types
linkml_types: Dict[str, URIRef] = dict(
    boolean=XSD.boolean,
    curie=XSD.string,
    date_or_datetime=LINKML.DateOrDatetime,
    date=XSD.date,
    datetime=XSD.dateTime,
    decimal=XSD.decimal,
    double=XSD.double,
    float=XSD.float,
    integer=XSD.integer,
    jsonpath=XSD.string,
    jsonpointer=XSD.string,
    ncname=XSD.string,
    nodeidentifier=SHEX.nonliteral,  # aka iri or blank
    objectidentifier=SHEX.iri,
    sparqlpath=XSD.string,
    string=XSD.string,
    time=XSD.time,
    uri=XSD.anyURI,
    uriorcurie=XSD.anyURI,
)
# TODO: Nothing to notify langstring? Should I add a type? C'mon!

# RDF Types to python types
py_types: Dict[URIRef, Type] = {
    XSD.anyURI: PydanticURIRef,
    XSD.base64Binary: bytes,
    XSD.boolean: bool,
    XSD.date: date,
    XSD.dateTime: datetime,
    XSD.decimal: int,
    XSD.integer: int,
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
    XSD.QName: str,
    XSD.string: str,
    XSD.time: time,
    XSD.language: str,
    RDFS.Resource: PydanticURIRef,
    RDF.langString: str,
    SHEX.iri: PydanticURIRef,
    SHEX.nonliteral: str,
    SHEX.nonLiteral: str,  # alternative spelling
    LINKML.DateOrDatetime: datetime,
}

class field_type(Enum):
    literal = 0
    langstr = 1
    enum = 2
    object = 3

class LinkMLOntologyHandler(OntologyHandler):
    ontology_language = 'linkml'
    def prepare(self, data):
        registry = None  # TODO: Inject.
        generator = PydanticModelGenerator(data, ontology_registry=registry)
        generator.serialize()
        return generator.ontology_data

@dataclass
class LinkMLOntologyData(OntologyData):
    registry: BaseOntologyRegistry = dcfield(default=None)
    schema: SchemaDefinition = dcfield(default=None)
    ontology_language = 'linkml'

    def as_context(self):
        pass  # TODO

@dataclass
class LinkMLClassData(OntologyClassData):
    pydantic_cls: Type[BaseModel] = dcfield()
    schema: ClassDefinition = dcfield(default=None)

    def as_pydantic(self) -> Type[BaseModel]:
        return self.pydantic_cls

    def parse(self, data: Any) -> BaseModel:
        return self.pydantic_cls.from_json(data)

@dataclass
class FieldInfo:
    slot: Optional[SlotDefinition|AnonymousSlotExpression] = dcfield()
    type_uri: URIRef | List[URIRef] = dcfield()
    name: str = dcfield()
    type: Type = dcfield()
    default: Any = dcfield()
    optional: bool = dcfield(default=True)
    primary_key: bool = dcfield(default=False)
    ftype: field_type = dcfield(default=field_type.literal)
    description: Optional[str] = dcfield(default=None)

    def as_pyd_field(self):
        return (self.type, self.default)

    # @property
    # def field_type(self) -> field_type:
    #     if self.type_uri == XSD.langString:
    #         return field_type.langstr
    #     if issubclass(self.type, Enum):
    #         return field_type.enum
    #     if issubclass(self.type, BaseModel):
    #         return field_type.object
    #     if issubclass(self.type, object):
    #         print("Strange field type:", self.type)
    #         return field_type.object
    #     return field_type.literal


@dataclass
class PydanticModelGenerator(LMGenerator):
    # ObjectVars
    # original_schema: SchemaDefinition
    class_defs: Dict[str, ClassDefinition] = dcfield(default_factory=dict)
    class_data: Dict[str, LinkMLClassData] = dcfield(default_factory=dict)
    class_names: Dict[str, str] = dcfield(default_factory=dict)
    enums: Dict[str, Type[Enum]] = dcfield(default_factory=dict)
    types: Dict[str, Type[Any]] = dcfield(default_factory=dict)  # TODO: what structure for a type alias?
    current_fields: Dict[str, FieldInfo] = dcfield(default_factory=dict)
    generic_slots_fields: Dict[str, FieldInfo] = dcfield(default_factory=dict)
    current_pkeys: Set[str] = dcfield(default_factory=set)
    base_model_cls: Type = dcfield(default=BaseModel)
    delayed_classes: List = dcfield(default_factory=list)

    # ClassVars
    ontology_registry: BaseOntologyRegistry = None
    generatorname = os.path.basename(__file__)
    generatorversion = "0.0.1"
    valid_formats = ["py"]
    visit_all_class_slots = True
    uses_schemaloader = True
    requires_metamodel = True
    file_extension = "py"


    def visit_schema(self, **kwargs) -> Optional[str]:
        """Visited once at the beginning of generation

        @param kwargs: Arguments passed through from CLI -- implementation dependent
        """
        self.default_uri = URIRef(self.schema.prefixes[self.schema.default_prefix].prefix_reference)

    def end_schema(self, **kwargs) -> Optional[str]:
        """Visited once at the end of generation

        @param kwargs: Arguments passed through from CLI -- implementation dependent
        """
        while self.delayed_classes:
            still_delayed = []
            for (cls, classname, attributes) in self.delayed_classes:
                if self.class_is_ready(cls):
                    self.finish_class(cls, classname, attributes)
                else:
                    still_delayed.append((cls, classname, attributes))
            assert len(still_delayed) < len(self.delayed_classes)
            self.delayed_classes = still_delayed

        classes = {n: cd.pydantic_cls for (n, cd) in self.class_data.items()}
        locals().update(classes)
        for cinfo in self.class_data.values():
            cinfo.pydantic_cls.model_rebuild()
        self.ontology_data = LinkMLOntologyData(
            'linkml',
            base_prefix=self.schema.default_prefix,
            prefixes={k: URIRef(v) for (k, v) in self.schema.prefixes.items()},
            classes=self.class_data,
            # enums and types TODO
            status = ontology_status.draft, # TODO
            schema = self.schema
        )


    def visit_class(self, cls: ClassDefinition) -> Optional[Union[str, bool]]:
        """Visited once per schema class

        @param cls: class being visited
        @return: Visit slots and end class.  False means skip and go on
        """
        if cls.from_schema != self.schema.id:
            return False
        if cls.mixin or cls.abstract:
            return False
        all_attributes = [self.schema.slots[s] for s in cls.slots]
        for a in all_attributes:
            if a.identifier:
                self.current_pkeys = {a.name}
                break
        else:
            for sup_cls in self.cls_def_mro(cls):
                if main_pkeys := sup_cls.unique_keys.get("main", None):
                    pkeys = set(main_pkeys.unique_key_slots)
                    pseudo_slot_names = {
                        f"{sup_cls.name.lower()}__{k.lower()}" for k in pkeys
                    }
                    pkeys |= {s for s in cls.slots if s.lower() in pseudo_slot_names}
                    self.current_pkeys = pkeys
        self.class_defs[cls.name] = cls
        self.current_fields = {}
        return True

    def curie_elements(self, term) -> Tuple[str, str]:
        for prefix, t in self.schema.prefixes.items():
            if term.startswith(t):
                short_term = str(term[len(t):])
                return prefix, short_term
        assert False, "Missing namespace"

    def as_curie(self, term: URIRef) -> str:
        prefix, short_term = self.curie_elements(term)
        return f"{prefix}:{short_term}"

    def class_name(self, name:str) -> str:
        # Could it be a full URL?
        model = self.schema.classes.get(name, None)
        if model and model.class_uri:
            name = model.class_uri
        if ':' in name:
            prefix, name = name.split(':', 1)
        else:
            prefix = self.schema.default_prefix
        if '_' in name:
            name = ''.join([f.title() for f in name.split('_')])
        elif name[0] != name[0].capitalize():
            name = name.title()
        return f"{str(prefix).title()}{name}"

    def get_type_info(self, slot: Optional[SlotDefinition|AnonymousSlotExpression]) -> Tuple[URIRef, Type, field_type]:
        ftype = None
        py_type = None
        rangeSchemaTypeUri = None
        slot = slot or self.schema.default_range
        name = str(slot.range)
        if rangeSchemaType := self.schema.types.get(name):
            rangeSchemaTypeUri = rangeSchemaType.uri.as_uri(self.namespaces)
            py_type = py_types.get(rangeSchemaTypeUri)
            ftype = field_type.langstr if rangeSchemaTypeUri == RDF.langString else field_type.literal
        elif rangeSchemaType := self.enums.get(name):
            ftype = field_type.enum
            py_type = rangeSchemaType
            # rangeSchemaType =
        else:
            rangeSchemaTypeUri = URIRef(self.default_uri + name)
            cls = None
            if self.schema.classes.get(name):
                cls = self.class_data.get(name)
                cls = cls.as_pydantic if cls else None
                ftype = field_type.object
            elif self.schema.enums.get(name):
                # TODO: Pre-defined enums? Inject beforehand?
                cls = self.enums.get(name)
                cls = cls.as_pydantic if cls else None
                ftype = field_type.enum
            py_type = cls or ForwardRef(self.class_name(name))
        return (rangeSchemaTypeUri, py_type, ftype)



    def make_field_info(
        self, aliased_slot_name: str, slot: SlotDefinition
    ) -> Generator[FieldInfo, None, None]:

        # Ok, question: What if the class is defined in two ontologies? What if the definitions conflicts?
        # Also... having versions is all well and good, but does that mean we append them to class definitions?
        # Decision: Don't do that. Version the Ontology, with a new prefix.
        # There _may_ be a process where the new ontology gets the canonical prefix at publish time.
        # There could be a scheme with past ontologies having a version number, and future ontologies having a _wip_branch_N suffix

        # Note on versioning. Do not confuse history and log replay, unless you also store procedures (and even then.)
        # MAYBE a log replay should create a new history stream? Makes sense. We _could_ also decide the old history is purged.
        # BUT ideally, the replay will have a different URL.

        (rangeSchemaTypeUri, py_type, ftype) = self.get_type_info(slot)
        finfo = FieldInfo(
            slot,
            rangeSchemaTypeUri,
            slot.name,
            py_type,
            slot.ifabsent,
            not slot.required,
            slot.identifier or (slot.name in self.current_pkeys),
            ftype,
            slot.description
        )

        if slot.any_of:
            # many cases: is it all enums, all classes, all literals... Can it be mixed? That last case need not be handled now.
            # TODO: handle enum case
            type_infos = [self.get_type_info(s) for s in slot.any_of if s.range]
            ftypes = list({ti[2] for ti in type_infos})
            assert len(ftypes) == 1 and ftypes[0]
            finfo.ftype = ftypes[0]
            if len(ftypes) > 1:
                finfo.type = Union[*[fi[1] for fi in type_infos]]
                finfo.type_uri = list({fi[0] for fi in type_infos})
            else:
                finfo.type = type_infos[0][1]
                finfo.type_uri = type_infos[0][0]

        if slot.multivalued and slot.inlined_as_list:
            finfo.type = List[finfo.type]
        print(aliased_slot_name, slot.range, finfo.type)
        # TODO: Foreign keys and other check constraints
        # Case of resources vs ids?
        # Question: How to distinguish projection types from generic RDF types here?
        match finfo.ftype:
            case field_type.langstr:
                # TODO: Maybe allow union of str and langstr?
                assert not slot.multivalued
                lang_slot_name = f"{aliased_slot_name}_lang"
                yield FieldInfo(slot, XSD.language, lang_slot_name, str, "und", not slot.required, description=f"Language of {aliased_slot_name}")
            case field_type.object:
                # if classes, we could give up to 3 fields: One for the class, one for the num id, one for the URL. The last one should ideally be computed.
                # I'll start with class + URI, punt the int.
                # Rejected: Give union of URI or structure.
                # Rejected: Give union of URI or int
                # Q: If I allow URL and receive an external URL... what do I do? Fetch? Subscribe? Certainly create a local snapshot.
                id_slot_name = f"{aliased_slot_name}_id"
                yield FieldInfo(slot, RDFS.Resource, id_slot_name,
                                List[PydanticURIRef] if slot.multivalued and slot.inlined_as_list else PydanticURIRef,
                                None,  not slot.required, False, field_type.literal, description=f"URI of {aliased_slot_name}")
                finfo.optional = True
            case _:
                pass
        yield finfo


    def visit_class_slot(
        self, cls: ClassDefinition, aliased_slot_name: str, slot: SlotDefinition
    ) -> Optional[str]:
        """Visited for each slot in a class.  If class level visit_all_slots is true, this is visited once
        for any class that is inherited (class itself, is_a, mixin, apply_to).  Otherwise, just the own slots.

        @param cls: containing class
        @param aliased_slot_name: Aliased slot name.  May not be unique across all class slots
        @param slot: being visited
        """
        # if DEBUG and cls.name == 'Analysis':
        #     pdb.set_trace()
        for field in self.make_field_info(aliased_slot_name, slot):
            self.current_fields[field.name] = field

    def visit_slot(self, aliased_slot_name: str, slot: SlotDefinition) -> Optional[str]:
        """Visited once for every slot definition in the schema.

        @param aliased_slot_name: Aliased name of the slot.  May not be unique
        @param slot: visited slot
        """
        for field in self.make_field_info(aliased_slot_name, slot):
            self.generic_slots_fields[field.name] = field

    def class_is_ready(self, cls) -> bool:
        if cls.is_a and not self.class_names.get(cls.is_a):
            return False
        if cls.mixins:
            return all((self.class_names.get(m) for m in cls.mixins))
        return True


    def end_class(self, cls: ClassDefinition) -> Optional[str]:
        """Visited after visit_class_slots (if visit_class returned true)

        @param cls: class being visited
        """
        delay = False
        # classname = cls.name
        classname = self.class_name(cls.name)
        attributes = {
            name: (finfo.type, finfo.default)
                #    Field(
                #        default=finfo.default, description=finfo.description,
                #        required=not finfo.optional))
            for (name, finfo) in self.current_fields.items()
        }
        if self.class_is_ready(cls):
            self.finish_class(cls, classname, attributes)
        else:
            self.delayed_classes.append((cls, classname, attributes))

    def finish_class(self, cls: ClassDefinition, classname: str, attributes: Dict) -> None:

        base_class = self.base_model_cls
        if cls.is_a:
            base_class_name = self.class_names.get(cls.is_a)
            base_class_data = self.class_data[base_class_name]
            base_class = base_class_data.as_pydantic()
        if cls.mixins:
            base_class = [base_class]
            for mixin_name in cls.mixins:
                mixin_cname = self.class_names.get(mixin_name)
                mixin_data = self.class_data[mixin_cname]
                base_class.append(mixin_data.as_pydantic())
            base_class = tuple(base_class)

        attributes['__base__'] = base_class
        self.class_names[cls.name] = classname
        # print(attributes)
        # TODO: connect to predefined base classes such as ProjectionMixin or EventBase
        # TODO: other unique indices
        # TODO: Handle mixins
        try:
            pyd_cls = create_model(classname, **attributes)
        except Exception as e:
            print(e)
            if DEBUG:
                pdb.post_mortem()
        self.class_data[classname] = LinkMLClassData(cls.deprecated, pyd_cls, cls)

    def visit_type(self, typ: TypeDefinition) -> Optional[str]:
        """Visited once for every type definition in the schema

        @param typ: Type definition
        """
        # TODO
        pass

    def visit_subset(self, subset: SubsetDefinition) -> Optional[str]:
        """Visited once for every subset definition in the schema

        #param subset: Subset definition
        """
        # Tagging system, pass
        pass

    def visit_enum(self, enum_def: EnumDefinition) -> Optional[str]:
        """Visited once for every enum definition in the schema

        @param enum: Enum definition
        """
        enum_options = [(str(k), str(k)) for k in enum_def.permissible_values]
        enum_name = self.class_name(enum_def.name)
        enum = StrEnum(enum_name, enum_options)
        self.enums[enum_def.name] = enum

    def cls_def_mro(
        self, cls: ClassDefinition, with_mixins=True
    ) -> Generator[ClassDefinition, None, None]:
        yield cls
        # yield the superclass
        if sup_cls := cls.is_a:
            yield from self.cls_def_mro(self.schema.classes[sup_cls], with_mixins)
        if not with_mixins:
            return
        # yield the mixins
        for sup_cls in cls.mixins:
            yield from self.cls_def_mro(self.schema.classes[sup_cls])
        # TODO: Maybe handle the case where a mixin is repeated

    @classmethod
    def from_schema(cls, schema):
        generator = PydanticModelGenerator(schema=schema)
        generator.serialize()
        return generator

    @classmethod
    def from_file(cls, fname):
        from linkml_runtime.utils.schemaview import load_schema_wrap
        schema = load_schema_wrap(fname)
        return cls.from_schema(schema)


if __name__ == "__main__":
    import sys
    g = PydanticModelGenerator.from_file(sys.argv[1])
    print(g.class_data)
    # pdb.set_trace()
