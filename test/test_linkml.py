import pytest
import pdb

pytestmark = pytest.mark.anyio


async def test_ontology_uses_subclassses(linkml_ontology):
    for cdata in linkml_ontology.class_data.values():
        pyd_cls = cdata.as_pydantic()
        cls_def = cdata.schema
        if cls_def.is_a:
            sup_cls_name = linkml_ontology.class_names[cls_def.is_a]
            sup_class = linkml_ontology.class_data[sup_cls_name]
            if not sup_class:
                pdb.set_trace()
            assert issubclass(pyd_cls, sup_class.as_pydantic())
            print(f"{pyd_cls.__name__} < {sup_class.as_pydantic().__name__}")


async def test_ontology_loads_enum(linkml_ontology):
    pyd_cls = linkml_ontology.class_data['HkStatement'].as_pydantic()
    pyd_cls(text='text', language='en', language_level='abc')
