from __future__ import annotations

from datetime import datetime
from enum import Enum, StrEnum
from typing import Annotated, Optional, List, Union, Dict, Any, Literal, Type
from uuid import UUID
from logging import getLogger

from pydantic import BaseModel, ConfigDict, model_validator, Discriminator, field_serializer
from frozendict import frozendict
from rdflib import Namespace, URIRef

from .utils import to_optional

logger = getLogger(__name__)

analysis_extras = "ignore"


class topic_type(Enum):
    """The basic types of entity in the database."""

    fragment = "fragment"
    standalone = "standalone"
    link = "link"
    hyperedge = "hyperedge"
    document = "document"
    analyzer = "analyzer"
    analysis = "analysis"
    collection = "collection"
    agent = "agent"
    cluster = "cluster"
    ontology = "ontology"
    schema_term = "schema_term"
    structured_idea = "structured_idea"


class TopicModel(BaseModel):
    """ABC for most other models"""

    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    type: topic_type
    created_by: Optional[int] = None
    schema_term_id: Optional[int] = None

    # creator: TopicModel
    from_analyses: Optional[List["AnalysisModel"]] = None
    # target_of_analyses: List[AnalysisModel]

    # outgoing_links: List[ClaimLinkModel]
    # incoming_links: List[ClaimLinkModel]

    @model_validator(mode="before")
    @classmethod
    def guess_schema_term(cls, data: Any) -> Any:
        if isinstance(data, dict):
            schema_term_term = data.pop('schema_term_term', None)
        elif isinstance(data, object):
            schema_term_term = getattr(data, 'schema_term_term', None)
        if not schema_term_term:
            sdt_field = cls.model_fields.get('schema_term_term')
            if sdt_field:
                schema_term_term = sdt_field.default
        if schema_term_term:
            from .ontology_registry import OntologyRegistry
            schema_term = OntologyRegistry.registry.terms.get(schema_term_term)
            if schema_term:
                data['schema_term_id'] = schema_term.id

        # Ensure enums
        if isinstance(data, dict):
            if isinstance(data.get("type"), str):
                data["type"] = topic_type[data["type"]]
        return data


HK = Namespace("https://hyperknowledge.org/schemas/v0#")
CM = Namespace("https://claimminer.info/schemas/v0#")


class UserModel(TopicModel):
    """ClaimMiner users."""

    model_config = ConfigDict(from_attributes=True)
    type: Literal[topic_type.agent] = topic_type.agent
    schema_term_term: Literal["hk:agent"] = "hk:agent"

    handle: str
    id: Optional[int] = None
    email: Optional[str] = None
    name: Optional[str] = None
    confirmed: bool = False
    created: Optional[datetime] = None
    picture_url: Optional[str] = None
    permissions: List["permission"] = []

    def can(self, perm: Union[str, permission]) -> bool:
        perm = permission[perm] if isinstance(perm, str) else perm
        return perm in self.permissions or permission.admin in self.permissions

    @model_validator(mode="before")
    @classmethod
    def ensure_enums(cls, data: Any) -> Any:
        data = super().guess_schema_term(data)
        if isinstance(data, dict):
            permissions = data.get("permissions", [])
            if permissions and isinstance(permissions[0], str):
                data["permissions"] = [permission[p] for p in permissions]
        return data


class UserModelWithPw(UserModel):
    external_id: Optional[str] = None
    passwd: str


UserModelOptional = to_optional(UserModelWithPw)

base_permissions = [
    "access",  #: Read access to the collection's data
    "add_document",  #: Add a document to the collection
    "add_claim",  #: Add a claim to the collection
    "confirm_claim",  # You can set a claim to non-draft state
    "admin",  # Full access
]
permission_values = set(base_permissions)


class OrderedEnum(Enum):
    def __lt__(self, other):
        assert self.__class__ is other.__class__
        return self.__class__._member_names_.index(
            self.name
        ) < self.__class__._member_names_.index(other.name)

    def __le__(self, other):
        assert self.__class__ is other.__class__
        return self.__class__._member_names_.index(
            self.name
        ) <= self.__class__._member_names_.index(other.name)


class process_status(OrderedEnum):
    inapplicable = "inapplicable"
    not_ready = "not_ready"
    not_requested = "not_requested"
    pending = "pending"
    ongoing = "ongoing"
    complete = "complete"
    error = "error"

class ontology_status(OrderedEnum):
    draft = 'draft'
    published = 'published'
    vintage = 'deprecated'
    obsolete = 'obsolete'


class fragment_type(Enum):
    """The set of all fragment subtypes"""

    document = "document"  # Represents a document as a whole.
    paragraph = "paragraph"  # Represents a paragraph in the document.
    sentence = "sentence"  # Represents a sentence in a paragraph. Not used yet
    phrase = "phrase"  # Represents a phrase in a sentence. Not used yet
    quote = "quote"  # Represents a quote from a document. May span paragraphs. Not used yet.
    summary = "summary"  # A summary of a document
    reified_arg_link = "reified_arg_link"  #: Argument wrapper
    standalone = "standalone"  #: standalone statement of unspecified subtype
    generated = (
        "generated"  #: Used for boundaries found by claim analyzers. To be removed.
    )
    standalone_root = "standalone_root"  #: Connected to a root claim, for importation.
    standalone_category = "standalone_category"  #: generic grouping category
    standalone_question = "standalone_question"  #: A multi-answer question
    standalone_claim = "standalone_claim"  #: A claim (also represents its associated yes-no question in DebateMap)
    standalone_argument = "standalone_argument"  #: An argument that a (set of) Claims makes another Claim more or less plausible. Reified connection.


fragment_type_names = {
    fragment_type.paragraph: "Paragraph",
    fragment_type.sentence: "Sentence",
    fragment_type.phrase: "Phrase",
    fragment_type.quote: "Quote",
    fragment_type.summary: "Summary",
    fragment_type.reified_arg_link: "Empty Argument",  #: Argument wrapper
    fragment_type.standalone: "Generic",  #: standalone statement of unspecified subtype
    fragment_type.generated: "Generated",  #: Used for boundaries found by claim analyzers. To be removed.
    fragment_type.standalone_root: "Import root",  #: Connected to a root claim, for importation.
    fragment_type.standalone_category: "Category",  #: generic grouping category
    fragment_type.standalone_question: "Question",  #: A multi-answer question
    fragment_type.standalone_claim: "Claim",  #: A claim (also represents its associated yes-no question in DebateMap)
    fragment_type.standalone_argument: "Argument",  #: An argument that a (set of) Claims makes another Claim more or less plausible. Reified connection.
}
# Fragment types for standalone fragments (not part of a document) and their user-readable names.

statement_types = list(fragment_type_names)

visible_statement_types = [
    fragment_type.standalone,
    fragment_type.standalone_category,
    fragment_type.standalone_question,
    fragment_type.standalone_claim,
    fragment_type.standalone_argument,
]
"""The subset of standalone fragment types that can be chosen in prompt construction, and that are displayed to the user."""


class link_type(Enum):
    """The set of all types of link between claims"""

    freeform = "freeform"
    key_point = "key_point"
    supported_by = "supported_by"
    opposed_by = "opposed_by"
    implied = "implied"
    implicit = "implicit"
    derived = "derived"
    answers_question = "answers_question"
    has_premise = "has_premise"
    irrelevant = "irrelevant"
    relevant = "relevant"
    subcategory = "subcategory"
    subclaim = "subclaim"
    subquestion = "subquestion"
    quote = "quote"


link_type_names = {
    link_type.freeform: "Freeform",
    link_type.supported_by: "Supported by",
    link_type.opposed_by: "Opposed by",
    link_type.answers_question: "Answers question",
    link_type.has_premise: "Has premise",
    link_type.irrelevant: "Irrelevant",
    link_type.relevant: "Relevant",
    link_type.subcategory: "Sub-category",
    link_type.subclaim: "Sub-claim",
    link_type.subquestion: "Sub-question",
    link_type.quote: "Quote",
}
"""User-readable names for link types"""

visible_statement_type_names = {
    k: v for (k, v) in fragment_type_names.items() if k in visible_statement_types
}


class uri_status(Enum):
    """Status of URIs in an equivalence class"""

    canonical = "canonical"  #: Canonical member of the equivalence class
    urn = "urn"  #: Distinguished non-URL member of the equivalence class
    snapshot = "snapshot"  #: Reference to a snapshot URL, eg archive.org
    alt = "alt"  #: Alternate (non-canonical) URL/URI with the same content
    unknown = "unknown"  #: Undetermined canonicality


class relevance_type(Enum):
    """Whether a claim is relevant to a given collection's main question"""

    irrelevant = "irrelevant"
    unknown = "unknown"
    relevant = "relevant"


class AnalyzerModel(TopicModel):
    """A versioned computation process.
    Computed values keep a reference to the analyzer that created them.
    The versioning system is not being used yet.
    """

    model_config = ConfigDict(from_attributes=True)

    type: Literal[topic_type.analyzer] = topic_type.analyzer
    schema_term_term: Literal["cm:analyzer"] = "cm:analyzer"
    id: Optional[int] = None
    name: str
    version: int

    # analyses: List[AnalysisModel]


class TaskTemplateModel(BaseModel):
    """A coherent set of parameters for an analysis task."""

    model_config = ConfigDict(from_attributes=True, extra=analysis_extras)

    id: Optional[int] = None
    analyzer_id: int
    analyzer_name: str
    nickname: Optional[str] = None
    draft: bool = False
    """True while editing a prompt, false when it has been used.
    Avoid editing an analyzer that is tied to an existing analysis."""
    collection_id: Optional[int] = None
    collection_name: Optional[str] = None

    analyzer: Optional[AnalyzerModel] = None

    @model_validator(mode="before")
    @classmethod
    def get_analyzer_name(cls, data: Any) -> Any:
        from .task_registry import TaskRegistry

        registry = TaskRegistry.get_registry()
        if isinstance(data, dict):
            if (
                data.get("analyzer_name", None) is None
                and data.get("analyzer_id", None) is not None
            ):
                data["analyzer_name"] = registry.analyzer_by_id[
                    data["analyzer_id"]
                ].name
            elif (
                data.get("analyzer_name", None) is not None
                and data.get("analyzer_id", None) is None
            ):
                data["analyzer_id"] = registry.analyzer_by_name[
                    data["analyzer_name"]
                ].id
        else:
            if getattr(data, "analyzer_name", None) is None and getattr(
                data, "analyzer_id", None
            ):
                data.analyzer_name = registry.analyzer_by_id[data.analyzer_id].name
            elif getattr(data, "analyzer_name", None) and (
                getattr(data, "analyzer_id", None) is None
            ):
                data.analyzer_id = registry.analyzer_by_name[data.analyzer_name].id
        return data

    def web_path(self, collection=None):
        collection_path = ""
        if collection and (self.collection_id == collection.id):
            collection_path = collection.path
        return (
            f"{collection_path}/analyzer/{self.analyzer_name}/template/{self.nickname}"
        )


class TopicCollectionModel(BaseModel):
    """Join table between Document and Collection"""

    model_config = ConfigDict(from_attributes=True)
    topic_id: int
    collection_id: int

    # topic: TopicModel
    # collection: CollectionModel


class CollectionModel(BaseModel):
    """A named collection of Documents and Claims"""

    model_config = ConfigDict(from_attributes=True)
    id: Optional[int] = None
    name: str
    params: Dict[str, Any] = {}

    # documents: List[DocumentModel]
    # "The documents in the collection"
    # topic_collections: List[TopicCollectionModel]
    # fragments: List[StatementModel]
    # "The fragments in the collection"
    # permissions: List[CollectionPermissionsModel]
    # "CollectionModel-specific permissions"
    # claim_roots: List[StatementModel]


class CollectionPermissionsModel(BaseModel):
    """CollectionModel-specific permissions that a user has in the scope of a specific collection"""

    model_config = ConfigDict(from_attributes=True)
    user_id: int
    collection_id: int
    permissions: List["permission"] = []


PartialCollectionModel = to_optional(CollectionModel)


class TaskTriggerModel(BaseModel):
    """Triggers the execution of an analysis task."""

    model_config = ConfigDict(from_attributes=True)

    id: Optional[int] = None
    target_analyzer_id: int
    collection_id: Optional[int] = None
    analyzer_trigger_id: Optional[int] = None
    task_template_id: Optional[int] = None
    creation_trigger_id: Optional[topic_type] = None
    automatic: bool = False
    conditions: Dict[str, Any] = {}
    params: Dict[str, Any] = {}
    creator_id: Optional[int] = None

    target_analyzer: Optional[AnalyzerModel] = None
    analyzer_trigger: Optional[AnalyzerModel] = None
    task_template: Optional[TaskTemplateModel] = None
    collection: Optional[CollectionModel] = None

    def signature(self):
        return (
            self.target_analyzer_id,
            self.collection_id,
            self.creation_trigger_id or self.analyzer_trigger_id,
            self.task_template_id,
            frozendict(self.conditions or {}),
            frozendict(self.params or {}),
        )


class UriEquivModel(BaseModel):
    """Equivalence classes of URIs"""

    model_config = ConfigDict(from_attributes=True)
    schema_term_term: Literal["cm:uri_equivalence"] = "cm:uri_equivalence"
    id: Optional[int] = None
    status: uri_status = uri_status.unknown
    canonical_id: Optional[int] = None
    uri: str

    canonical: Optional[UriEquivModel] = None
    # referencing_links: List[DocumentLinkModel]
    # referencing_documents: List[DocumentModel]


class DocumentLinkModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    source_id: int
    target_id: int
    analyzer_id: Optional[int] = None

    # source: DocumentModel
    # target: UriEquivModel
    # analyzer: AnalyzerModel


class DocumentModel(TopicModel):
    """Represents a document that was requested, uploaded or downloaded"""

    model_config = ConfigDict(from_attributes=True)
    type: Literal[topic_type.document] = topic_type.document
    schema_term_term: Literal["hk:document"] = "hk:document"
    uri_id: Optional[int] = None
    is_archive: bool = False
    public_contents: bool = True
    requested: datetime = datetime.utcnow()
    return_code: Optional[int] = None
    retrieved: Optional[datetime] = None
    created: Optional[datetime] = None
    modified: Optional[datetime] = None
    mimetype: Optional[str] = None
    language: Optional[str] = None
    text_analyzer_id: Optional[int] = None
    etag: Optional[str] = None
    # file_identity: Optional[str] = None
    file_size: Optional[int] = None
    # text_identity: Optional[str] = None
    text_size: Optional[int] = None
    title: Optional[str] = None
    process_params: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    uri: Optional[UriEquivModel] = None
    url: str

    # doc_collections: List[TopicCollectionModel]
    # "The collections this document belongs to"
    # href_links: List[DocumentLinkModel]
    # href_uri: List[UriEquivModel]
    # summary: StatementModel
    # paragraphs: List[FragmentModel]

    # @property
    # def url(self):
    #     return self.uri.uri

    # @property
    # def base_type(self):
    #     return self.mimetype.split(';')[0]

    @model_validator(mode="before")
    @classmethod
    def ensure_enums(cls, data: Any) -> Any:
        data = super().guess_schema_term(data)
        if isinstance(data, dict):
            if data.get("uri", None) is not None and data.get("url", None) is None:
                data["url"] = data["uri"].uri
        else:
            if (
                getattr(data, "uri", None) is not None
                and getattr(data, "url", None) is None
            ):
                data.url = data.uri.uri

        return data


class StatementModel(TopicModel):
    """A fragment of text representing a standalone claim, question or category."""

    model_config = ConfigDict(from_attributes=True)
    type: Literal[topic_type.standalone] = topic_type.standalone
    schema_term_term: Literal["hk:statement"] = "hk:statement"
    text: str
    scale: fragment_type
    language: str = "en"
    generation_data: Optional[Dict[str, Any]] = None
    confirmed: bool = True
    doc_id: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def ensure_enums(cls, data: Any) -> Any:
        data = super().guess_schema_term(data)
        if isinstance(data, dict) and isinstance(data.get("scale"), str):
            data["scale"] = fragment_type[data["scale"]]
        return data

    # context_of_analyses: List[AnalysisModel]
    # theme_of_analyses: List[AnalysisModel]
    # in_clusters: List[ClusterDataModel]
    # in_cluster_rels: List[InClusterDataModel]


class FragmentModel(StatementModel):
    """A fragment of text. Can be part of a document, or even part of another fragment. It can be a standalone claim."""

    model_config = ConfigDict(from_attributes=True)
    type: Literal[topic_type.fragment] = topic_type.fragment
    schema_term_term: Literal["hk:fragment"] = "hk:fragment"
    position: Optional[int] = None
    char_position: Optional[int] = None
    part_of: Optional[int] = None
    doc_id: int
    sub_parts: Optional[List[FragmentModel]] = None

    # part_of_fragment: FragmentModel
    document: Optional[DocumentModel] = None


class AnalysisModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, extra=analysis_extras)

    schema_term_term: Literal["cm:analysis"] = "cm:analysis"
    id: Optional[int] = None
    analyzer_name: str
    analyzer_id: int
    created: datetime = datetime.utcnow()
    completed: Optional[datetime] = None
    status: Optional[process_status] = None
    results: Any = {}
    collection_id: Optional[int] = None
    collection_name: Optional[str] = None
    part_of_id: Optional[int] = None
    triggered_by_analysis_id: Optional[int] = None
    creator_id: Optional[int] = None
    generated_topics: Optional[List[AnyTopicModel]] = None
    # part_of: Optional[AnalysisModel] = None
    # has_parts: Optional[List[AnalysisModel]] = None
    # context: List[FragmentModel]
    clusters: Optional[List[ClusterDataModel]] = None

    @model_validator(mode="before")
    @classmethod
    def get_analyzer_name(cls, data: Any) -> Any:
        from .task_registry import TaskRegistry

        registry = TaskRegistry.get_registry()
        analyzer_name_ = getattr(
            cls.__annotations__.get("analyzer_name", object()), "__args__", (None,)
        )[0]
        if isinstance(data, dict):
            if analyzer_name_ and data.get("analyzer_name", None) is None:
                data["analyzer_name"] = analyzer_name_
            if (
                data.get("analyzer_name", None) is None
                and data.get("analyzer_id", None) is not None
            ):
                data["analyzer_name"] = registry.analyzer_by_id[
                    data["analyzer_id"]
                ].name
            elif (
                data.get("analyzer_name", None) is not None
                and data.get("analyzer_id", None) is None
            ):
                data["analyzer_id"] = registry.analyzer_by_name[
                    data["analyzer_name"]
                ].id
        else:
            if analyzer_name_ and getattr(data, "analyzer_name", None) is None:
                data.analyzer_name = analyzer_name_
            if getattr(data, "analyzer_name", None) is None and getattr(
                data, "analyzer_id", None
            ):
                data.analyzer_name = registry.analyzer_by_id[data.analyzer_id].name
            elif getattr(data, "analyzer_name", None) and (
                getattr(data, "analyzer_id", None) is None
            ):
                data.analyzer_id = registry.analyzer_by_name[data.analyzer_name].id
        return data

    def web_path(self, collection=None):
        collection_path = collection.path if collection else ""
        analyzer_name = self.analyzer_name
        if self.id:
            return f"{collection_path}/analysis/{analyzer_name}/{self.id}"
        elif hasattr(self, "target_id"):
            # theme_id also?
            from .task_registry import TaskRegistry

            task = TaskRegistry.get_registry().get_task_cls_by_name(analyzer_name)
            if task.task_scale:
                return f"{collection_path}/{task.task_scale[0].name}/{self.target_id}/analysis/{analyzer_name}"
        return f"{collection_path}/analysis/{analyzer_name}"

    def api_path(self, collection=None):
        collection_path = collection.path if collection else ""
        analyzer_name = self.analyzer_name
        if self.id:
            return f"/api{collection_path}/analysis/{self.id}"
        else:
            # TODO: Improve this path
            return f"/api{collection_path}/analysis/type/{analyzer_name}"


class AnalysisWithTemplateModel(AnalysisModel):
    task_template_id: Optional[int] = None
    task_template_nickname: str
    task_template: Optional[TaskTemplateModel] = (
        None  # override with TaskTemplate subclass
    )

    @model_validator(mode="before")
    @classmethod
    def get_template_name(cls, data: Any) -> Any:
        from .task_registry import TaskRegistry

        registry = TaskRegistry.get_registry()
        data = cls.get_analyzer_name(data)
        if isinstance(data, dict):
            if (
                data.get("task_template_id", None) is None
                and data.get("task_template_nickname", None) is not None
            ):
                template = registry.task_template_by_nickname.get(
                    data["task_template_nickname"]
                )
                if template:
                    data["task_template_id"] = template.id
            elif (
                data.get("task_template_id", None) is not None
                and data.get("task_template_nickname", None) is None
            ):
                data["task_template_nickname"] = registry.task_template_by_id[
                    data["task_template_id"]
                ].nickname
            elif (
                data.get("task_template", None) is not None
                and data.get("task_template_nickname", None) is None
            ):
                data["task_template_nickname"] = data["task_template"].get("nickname")
        else:
            if (
                getattr(data, "task_template_id", None) is None
                and getattr(data, "task_template_nickname", None) is not None
            ):
                template = registry.task_template_by_nickname.get(
                    data.task_template_nickname
                )
                if template:
                    data.task_template_id = template.id
            elif (
                getattr(data, "task_template_id", None) is not None
                and getattr(data, "task_template_nickname", None) is None
            ):
                data.task_template_nickname = registry.task_template_by_id[
                    data.task_template_id
                ].nickname
            elif (
                getattr(data, "task_template", None) is not None
                and getattr(data, "task_template_nickname", None) is None
            ):
                data.task_template_nickname = data.task_template.nickname
        return data

    def get_task_template(self):
        if not self.task_template:
            from .task_registry import TaskRegistry

            self.task_template = TaskRegistry.get_registry().task_template_by_id[
                self.task_template_id
            ]
        return self.task_template


class InClusterDataModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    cluster_id: int
    fragment_id: int
    confirmed_by_id: Optional[int] = None
    manual: bool = False

    # cluster: ClusterDataModel
    # fragment: StatementModel
    # confirmed_by: UserModel


class ClusterDataModel(BaseModel):  # Not a TopicModel yet
    model_config = ConfigDict(from_attributes=True)
    type: Literal[topic_type.cluster] = topic_type.cluster
    schema_term_term: Literal["cm:cluster"] = "cm:cluster"
    id: Optional[int] = None
    analysis_id: int
    cluster_size: int
    relevant: relevance_type = relevance_type.unknown
    distinguished_claim_id: Optional[int] = None
    relevance_checker_id: Optional[int] = None
    auto_include_diameter: Optional[float] = None

    # analysis: AnalysisModel
    # distinguished_claim: StatementModel
    fragments: Optional[List[StatementModel]] = None
    # relevance_checker: UserModel
    # has_cluster_rels: List[InClusterDataModel]


# TODO: Make this extensible like permissions
class embedding_model(Enum):
    universal_sentence_encoder_4 = "universal_sentence_encoder_4"
    all_minilm_l6_v2 = "all_minilm_l6_v2"

    txt_embed_ada_2 = "txt_embed_ada_2"
    txt3_small_1536 = "txt3_small_1536"
    txt3_small_512 = "txt3_small_512"
    txt3_large_256 = "txt3_large_256"
    txt3_large_1024 = "txt3_large_1024"
    uae_l_v1 = "uae_l_v1"


BASE_EMBED_MODEL = embedding_model.all_minilm_l6_v2
BASE_DOC_EMBED_MODEL = embedding_model.universal_sentence_encoder_4


class EmbeddingModel(BaseModel):
    """The vector embedding of a fragment's text. Abstract class."""

    model_config = ConfigDict(from_attributes=True)
    schema_term_term: Literal["cm:embedding"] = "cm:embedding"
    analyzer_id: int
    scale: fragment_type
    doc_id: Optional[int] = None
    fragment_id: Optional[int] = None
    embedding: List[float]
    dimensionality: int
    embedding_model_name: embedding_model

    # document: Optional[DocumentModel] = None
    # fragment: Optional[Union[StatementModel, FragmentModel]] = None
    # analyzer: AnalyzerModel


class HyperEdgeModel(TopicModel):
    """A link materialized as a node, but without content."""

    model_config = ConfigDict(from_attributes=True)
    schema_term_term: Literal["hk:hyperedge"] = "hk:hyperedge"
    type: Literal[topic_type.hyperedge] = topic_type.hyperedge
    # Temporary
    scale: Literal[fragment_type.reified_arg_link]

    @model_validator(mode="before")
    @classmethod
    def ensure_enums(cls, data: Any) -> Any:
        data = super().guess_schema_term(data)
        if isinstance(data, dict) and isinstance(data.get("scale"), str):
            data["scale"] = fragment_type[data["scale"]]
        return data


class ClaimLinkModel(TopicModel):
    """A typed link between two standalone claims."""

    model_config = ConfigDict(from_attributes=True)
    type: Literal[topic_type.link] = topic_type.link
    schema_term_term: Literal["cm:claim_link"] = "cm:claim_link"
    id: Optional[int] = None
    source: int
    target: int
    link_type: link_type
    analyzer: Optional[int] = None
    score: Optional[float] = None
    created_by: Optional[int] = None

    # source_topic: TopicModel = Field(discriminator="type")
    # target_topic: TopicModel = Field(discriminator="type")

    @model_validator(mode="before")
    @classmethod
    def ensure_enums(cls, data: Any) -> Any:
        data = super().guess_schema_term(data)
        if isinstance(data, dict) and isinstance(data.get("link_type"), str):
            data["link_type"] = link_type[data["link_type"]]
        return data



class OntologyModel(BaseModel):
    """A schema definition"""
    model_config = ConfigDict(from_attributes=True)
    schema_term_term: Literal["hk:ontology"] = "hk:ontology"
    ontology_language: str = "linkml"
    prefix: str
    # TODO: full_term: rdflib.URIRef
    data: Dict  # TODO: Schema structure

class SchemaTermModel(TopicModel):
    """A schema definition"""
    model_config = ConfigDict(from_attributes=True)
    schema_term_term: Literal["hk:schema_term"] = "hk:schema_term"
    parent_id: Optional[int] = None
    ancestors_id: Optional[List[int]] = None
    term: str
    public_term: Optional[str]
    ontology_prefix: str
    ontology: Optional[OntologyModel]
    # TODO: full_term: rdflib.URIRef
    data: Dict  # TODO: Schema structure

    @property
    def full_term(self) -> URIRef:
        from .ontology_registry import OntologyRegistry
        ontology = self.ontology or OntologyRegistry.registry.ontologies_by_prefix[self.ontology_prefix]
        return OntologyRegistry.registry.as_term(f"{ontology.prefix}:{self.term}")

class AbstractStructuredIdeaModel(TopicModel):
    pass
class GenericStructuredIdeaModel(AbstractStructuredIdeaModel):
    model_config = ConfigDict(from_attributes=True)
    ref_structure: Optional[Dict[str, Union[int, List[int]]]] = None
    literal_structure: Optional[Dict[str, Any]] = None
    refs: Optional[List[int]] = None
    schema_term: Optional[SchemaTermModel] = None
    references: Optional[List[TopicModel]] = None

    @property
    def term(self):
        return self.schema_term.term

    # TODO: implement type constraints on creation

DocumentOrFragmentModel = Annotated[
    Union[DocumentModel, FragmentModel], Discriminator("schema_term_term")
]
StatementOrFragmentModel = Annotated[
    Union[StatementModel, FragmentModel], Discriminator("schema_term_term")
]
AnyClaimOrLinkModel = Annotated[
    Union[ClaimLinkModel, StatementModel, HyperEdgeModel], Discriminator("schema_term_term")
]
AnyClaimOrHyperedgeModel = Annotated[
    Union[StatementModel, HyperEdgeModel], Discriminator("schema_term_term")
]
AnyTopicModel = Annotated[
    Union[StatementModel, HyperEdgeModel, ClaimLinkModel, DocumentModel, FragmentModel],
    Discriminator("schema_term_term"),
]


class CollectionExtendedModel(CollectionModel):
    user_permissions: Optional[CollectionPermissionsModel] = None
    num_documents: Optional[int] = 0
    num_statements: Optional[int] = 0


class CollectionExtendedAdminModel(CollectionExtendedModel):
    permissions: Optional[List[CollectionPermissionsModel]] = []


class search_mode(Enum):
    text = "text"
    semantic = "semantic"
    mmr = "mmr"


permission: Type[Enum] = None

pyd_model_by_topic_type: Dict[topic_type, type[BaseModel]] = {
    topic_type.collection: CollectionModel,
    topic_type.document: DocumentModel,
    topic_type.standalone: StatementModel,
    topic_type.fragment: FragmentModel,
    topic_type.hyperedge: HyperEdgeModel,
    topic_type.link: ClaimLinkModel,
    topic_type.agent: UserModel,
    topic_type.cluster: ClusterDataModel,
    topic_type.analyzer: AnalyzerModel,
    topic_type.schema_term: SchemaTermModel,
    topic_type.structured_idea: AbstractStructuredIdeaModel,
}


def finalize_permissions():
    global permission, AllAnalysisModels, AnalysisModelUnion
    if permission is None:
        from .task_registry import TaskRegistry

        logger.debug("Finalizing")
        # Maintain order of base_permissions
        permissions = base_permissions + list(permission_values - set(base_permissions))
        permission = StrEnum("permission", permissions)
        for model in pyd_model_by_topic_type.values():
            if not model.__pydantic_complete__:
                model.model_rebuild()
        UserModel.model_rebuild()
        UserModelWithPw.model_rebuild()
        UserModelOptional.model_rebuild()
        CollectionPermissionsModel.model_rebuild()
        AnalysisModel.model_rebuild()
        registry = TaskRegistry.get_registry()
        for model in registry.analysis_model_by_name.values():
            model.model_rebuild()
