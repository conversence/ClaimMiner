from __future__ import annotations

from typing import Optional, Type, Dict, List

from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
from sqlalchemy import BigInteger, ForeignKey, Index
from sqlalchemy.dialects.postgresql import ENUM
from sqlalchemy.orm import Mapped, mapped_column, declared_attr, relationship
from sqlalchemy.sql.ddl import CreateTable, CreateIndex

from ..embed import embedder_registry
from ..pyd_models import embedding_model, EmbeddingModel, fragment_type
from . import Base
from .content import fragment_type_db, Document, Statement


embedding_model_db = ENUM(embedding_model, name="embedding_model")
model_dimensionality: Dict[embedding_model, int] = {
    embedding_model[k]: cls.dimensionality for k, cls in embedder_registry.items()
}
model_names: Dict[embedding_model, str] = {
    embedding_model[k]: cls.display_name for k, cls in embedder_registry.items()
}
model_names_s = {k.name: v for k, v in model_names.items()}


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
        from ..embed import tf_embed

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
