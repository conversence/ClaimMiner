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
    Type,
    Tuple,
    Iterable,
    TypedDict,
)
from io import BytesIO, StringIO

from sqlalchemy import (
    Integer,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    SmallInteger,
    Float,
    Text,
    BigInteger,
    case,
    literal_column,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB, ENUM
from sqlalchemy.orm import (
    relationship,
    joinedload,
    backref,
    subqueryload,
    with_polymorphic,
    aliased as sqla_aliased,
    Mapped,
    mapped_column,
)
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.sql import and_, or_
from sqlalchemy.sql.functions import coalesce, func

from .. import hashfs, Session
from ..pyd_models import (
    BaseModel,
    topic_type,
    fragment_type,
    link_type,
    uri_status,
    relevance_type,
    DocumentModel,
    DocumentLinkModel,
    ClaimLinkModel,
    HyperEdgeModel,
    UriEquivModel,
    StatementModel,
    FragmentModel,
    InClusterDataModel,
    ClusterDataModel,
    HK,
)
from ..utils import safe_lang_detect
from . import logger, aliased
from .auth import User
from .base import Base, Topic, globalScope
from .collections import TopicCollection, Collection
from .tasks import Analyzer, analysis_context_table, Analysis

fragment_type_db = ENUM(fragment_type, name="fragment_type")
link_type_db = ENUM(link_type, name="link_type")
uri_status_db = ENUM(uri_status, name="uri_status")
relevance_type_db = ENUM(relevance_type, name="relevance")


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
            from ..uri_equivalence import normalize

            url = normalize(model.url)
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
        Analysis,
        foreign_keys=Analysis.theme_id,
        back_populates="theme",
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
        schema_term_id: Optional[int] = None,
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
        if not schema_term_id:
            from .ontology_registry import OntologyRegistry
            schema_term_id = OntologyRegistry.registry.terms[HK.statement].id
        assert schema_term_id
        return Statement(text=txt, language=lang, scale=scale, schema_term_id=schema_term_id)

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

Analysis.clusters: Mapped[List[ClusterData]] = relationship(
    ClusterData, back_populates="analysis", passive_deletes=True
)


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


def search_target_class(include_claims, include_paragraphs):
    if include_claims:
        if include_paragraphs:
            return StatementOrFragment
        else:
            return Statement
    else:
        return Fragment


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
    target = [node for (node, link) in nodes if link == "self"][0]
    by_id = {node.id: node for (node, _) in nodes}

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
