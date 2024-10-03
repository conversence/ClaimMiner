from __future__ import annotations

from itertools import chain
from typing import List, Dict, Any, Optional, Type, Set, ForwardRef

from fastapi.encoders import jsonable_encoder
from frozendict import frozendict
from pydantic import BaseModel

from sqlalchemy import (
    BigInteger,
    ForeignKey,
    String,
    Boolean,
    inspect,
    Integer,
    Table,
    Column,
    DateTime,
    select,
)
from sqlalchemy.dialects.postgresql import ENUM, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship, LoaderCallableStatus

from .base import Topic, Base, topic_type_db, globalScope
from .auth import User
from .collections import Collection

from claim_miner.pyd_models import (
    process_status,
    AnalyzerModel,
    topic_type,
    TaskTemplateModel,
    TaskTriggerModel,
    AnalysisModel,
    AnalysisWithTemplateModel,
)
from claim_miner.task_registry import TaskRegistry
from claim_miner.utils import to_optional

process_status_db = ENUM(process_status, name="process_status")


class Analyzer(Topic):
    """A versioned computation process.
    Computed values keep a reference to the analyzer that created them.
    The versioning system is not being used yet.
    """

    __tablename__ = "analyzer"
    pyd_model = AnalyzerModel
    __mapper_args__ = {
        "polymorphic_load": "inline",
        "polymorphic_identity": topic_type.analyzer,
    }

    id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(Topic.id, onupdate="CASCADE", ondelete="CASCADE"),
        primary_key=True,
    )  #: Primary key
    name: Mapped[String] = mapped_column(String)  #: the type of analyzer
    version: Mapped[BigInteger] = mapped_column(BigInteger)  #: the version number
    analyses: Mapped[List[Analysis]] = relationship(
        "Analysis", back_populates="analyzer", passive_deletes=True
    )


class TaskTemplate(Base):
    """A coherent set of parameters for an analysis task."""

    __tablename__ = "task_template"
    pyd_model = TaskTemplateModel

    id: Mapped[BigInteger] = mapped_column(BigInteger, primary_key=True)  #: Primary key
    analyzer_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(Analyzer.id), nullable=False
    )
    collection_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey("collection.id")
    )
    nickname: Mapped[String] = mapped_column(
        String, unique=True
    )  #: User-readable subtype, used for prompt names
    params: Mapped[Dict[str, Any]] = mapped_column(
        JSONB, server_default="{}"
    )  #: Prompt logic is here
    draft: Mapped[Boolean] = mapped_column(Boolean, server_default="false")
    """True while editing a prompt, false when it has been used.
    Avoid editing an analyzer that is tied to an existing analysis."""
    analyses: Mapped[List[Analysis]] = relationship(
        "Analysis", back_populates="task_template", passive_deletes=True
    )
    analyzer: Mapped[Analyzer] = relationship(Analyzer)
    collection: Mapped[Collection] = relationship("Collection")

    @property
    def analyzer_name(self):
        if (
            inspect(self).attrs["analyzer"].loaded_value
            != LoaderCallableStatus.NO_VALUE
        ):
            return self.analyzer.name
        analyzer = TaskRegistry.get_registry().analyzer_by_id.get(self.analyzer_id)
        return analyzer.name if analyzer else None

    @analyzer_name.setter
    def analyzer_name(self, name: str):
        analyzer = TaskRegistry.get_registry().analyzer_by_name.get(name)
        if analyzer:
            self.analyzer_id = analyzer.id

    def as_model(
        self,
        session,
        model_cls: Optional[Type[BaseModel]] = None,
        recursion: Optional[Set[int]] = None,
        **extra,
    ):
        model_cls = (
            model_cls
            or TaskRegistry.get_registry()
            .get_task_cls_by_name(self.analyzer_name)
            .task_template_model
            or TaskTemplateModel
        )
        if self.draft:
            model_cls = to_optional(model_cls)
        ignore_keys = ["params", "analyzer"]
        model = super(TaskTemplate, self).as_model(
            session,
            model_cls,
            ignore_keys=ignore_keys,
            recursion=recursion,
            **(extra | self.params),
        )
        return model

    @classmethod
    async def from_model(
        cls, session, model: BaseModel, ignore: Optional[List[str]] = None, **extra
    ):
        assert isinstance(model, TaskTemplateModel)
        args = extra | model.model_dump()
        rel_names = inspect(cls).relationships.keys()
        col_names = inspect(cls).c.keys()
        for k in ("analyzer_name", "analyzer", "collection_name"):
            args.pop(k, None)
        col_args = {k: v for (k, v) in args.items() if k in col_names}
        col_args["params"] = jsonable_encoder(
            {k: v for (k, v) in args.items() if k not in col_names}
        )
        return cls(**col_args)

    def web_path(self, collection=globalScope):
        collection_path = ""
        if collection and (self.collection_id == collection.id):
            collection_path = collection.path
        return (
            f"{collection_path}/analyzer/{self.analyzer_name}/template/{self.nickname}"
        )


class TaskTrigger(Base):
    """Triggers the execution of an analysis task."""

    __tablename__ = "task_trigger"
    pyd_model = TaskTriggerModel

    id: Mapped[BigInteger] = mapped_column(Integer, primary_key=True)
    target_analyzer_id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(Analyzer.id, onupdate="CASCADE", ondelete="CASCADE"),
        nullable=False,
    )
    collection_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(Collection.id, onupdate="CASCADE", ondelete="CASCADE")
    )
    analyzer_trigger_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(Analyzer.id, onupdate="CASCADE", ondelete="CASCADE")
    )
    task_template_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(TaskTemplate.id, onupdate="CASCADE", ondelete="CASCADE")
    )
    creation_trigger_id: Mapped[topic_type] = mapped_column(topic_type_db)
    automatic: Mapped[Boolean] = mapped_column(Boolean, server_default="false")
    conditions: Mapped[Dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    params: Mapped[Dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    creator_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(User.id, onupdate="CASCADE", ondelete="SET NULL")
    )  # Who created the trigger?

    target_analyzer: Mapped[Analyzer] = relationship(
        Analyzer, foreign_keys=[target_analyzer_id]
    )
    task_template: Mapped[TaskTemplate] = relationship(
        TaskTemplate, foreign_keys=[task_template_id]
    )
    analyzer_trigger: Mapped[Analyzer] = relationship(
        Analyzer, foreign_keys=[analyzer_trigger_id]
    )
    collection: Mapped[Collection] = relationship(
        Collection, foreign_keys=[collection_id], back_populates="task_triggers"
    )
    creator: Mapped[User] = relationship(User, foreign_keys=[creator_id])

    def signature(self):
        return (
            self.target_analyzer_id,
            self.collection_id,
            self.creation_trigger_id or self.analyzer_trigger_id,
            self.task_template_id,
            frozendict(self.conditions),
            frozendict(self.params),
        )

    def web_path(self, collection=globalScope):
        return f"{collection.web_path()}/task_trigger/{self.id}"


Collection.task_triggers: Mapped[List[TaskTrigger]] = relationship(
    "TaskTrigger", back_populates="collection", passive_deletes=True
)


analysis_context_table = Table(
    "analysis_context",
    Base.metadata,
    Column(
        "analysis_id", ForeignKey("analysis.id", onupdate="CASCADE", ondelete="CASCADE")
    ),
    Column(
        "fragment_id", ForeignKey("fragment.id", onupdate="CASCADE", ondelete="CASCADE")
    ),
)
analysis_output_table = Table(
    "analysis_output",
    Base.metadata,
    Column(
        "analysis_id", ForeignKey("analysis.id", onupdate="CASCADE", ondelete="CASCADE")
    ),
    Column("topic_id", ForeignKey("topic.id", onupdate="CASCADE", ondelete="CASCADE")),
)


class Analysis(Base):
    __tablename__ = "analysis"
    pyd_model = AnalysisModel
    id: Mapped[BigInteger] = mapped_column(BigInteger, primary_key=True)
    analyzer_id: Mapped[BigInteger] = mapped_column(
        BigInteger,
        ForeignKey(Analyzer.id, onupdate="CASCADE", ondelete="CASCADE"),
        nullable=False,
    )
    task_template_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(TaskTemplate.id, onupdate="CASCADE", ondelete="CASCADE")
    )
    target_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(Topic.id, onupdate="CASCADE", ondelete="SET NULL")
    )
    theme_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey("fragment.id", onupdate="CASCADE", ondelete="SET NULL")
    )
    params: Mapped[Dict[str, Any]] = mapped_column(JSONB, server_default="{}")
    results: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created: Mapped[DateTime] = mapped_column(
        DateTime, server_default="now()", nullable=False
    )
    completed: Mapped[DateTime] = mapped_column(DateTime)
    collection_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(Collection.id, onupdate="CASCADE", ondelete="CASCADE")
    )
    part_of_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey("analysis.id", onupdate="CASCADE", ondelete="CASCADE")
    )
    triggered_by_analysis_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey("analysis.id", onupdate="CASCADE", ondelete="SET NULL")
    )
    status: Mapped[process_status] = mapped_column(
        process_status_db, server_default="'complete'"
    )
    creator_id: Mapped[BigInteger] = mapped_column(
        BigInteger, ForeignKey(User.id, onupdate="CASCADE", ondelete="SET NULL")
    )

    analyzer: Mapped[Analyzer] = relationship(
        Analyzer,
        foreign_keys=[analyzer_id],
        remote_side=[Analyzer.id],
        back_populates="analyses",
    )
    task_template: Mapped[Analyzer] = relationship(
        TaskTemplate, back_populates="analyses"
    )
    # TODO: Can I constrain this to be a statement without circular imports?
    theme: Mapped[ForwardRef("Statement")] = relationship(
        "Statement",
        foreign_keys=[theme_id],
        back_populates="theme_of_analyses",
    )
    generated_topics: Mapped[List[Topic]] = relationship(
        Topic, secondary=analysis_output_table, back_populates="from_analyses"
    )
    context: Mapped[List[ForwardRef("Fragment")]] = relationship(
        "Fragment",
        secondary=analysis_context_table,
        back_populates="context_of_analyses",
    )
    collection: Mapped[Collection] = relationship(
        Collection, foreign_keys=[collection_id]
    )
    part_of: Mapped[Analysis] = relationship(
        "Analysis", foreign_keys=[part_of_id], remote_side=[id], backref="has_parts"
    )
    triggered_by_analysis: Mapped[Analysis] = relationship(
        "Analysis",
        foreign_keys=[triggered_by_analysis_id],
        remote_side=[id],
        backref="triggered_analyses",
    )
    target: Mapped[Topic] = relationship(Topic, back_populates="target_of_analyses")
    creator: Mapped[User] = relationship(User, foreign_keys=[creator_id])

    def web_path(self, collection=globalScope):
        analyzer_name = self.analyzer_name
        if self.id:
            return f"{collection.path}/analysis/{analyzer_name}/{self.id}"
        elif self.target_id:
            task = TaskRegistry.get_registry().get_task_cls_by_name(analyzer_name)
            if task.task_scale:
                return f"{collection.path}/{task.task_scale[0].name}/{self.target_id}/analysis/{analyzer_name}"
        # if self.task_template_id:
        #     return f"{collection.path}/analysis/{self.task_template_nickname}"
        return f"{collection.path}/analysis/{self.analyzer_name}"

    def api_path(self, collection=globalScope):
        analyzer_name = self.analyzer_name
        if self.id:
            return f"/api{collection.path}/analysis/{self.id}"
        else:
            # TODO: Improve this path
            return f"/api{collection.path}/analysis/type/{analyzer_name}"

    @property
    def task_template_nickname(self):
        if self.task_template_id:
            task_template_model = TaskRegistry.get_registry().task_template_by_id[
                self.task_template_id
            ]
            return task_template_model.nickname

    @property
    def collection_name(self):
        if not self.collection_id:
            return None
        if (
            inspect(self).attrs["collection"].loaded_value
            != LoaderCallableStatus.NO_VALUE
        ):
            return self.collection.name
        # Missing

    @property
    def analyzer_name(self):
        if (
            inspect(self).attrs["analyzer"].loaded_value
            != LoaderCallableStatus.NO_VALUE
        ):
            return self.analyzer.name
        analyzer = TaskRegistry.get_registry().analyzer_by_id.get(self.analyzer_id)
        return analyzer.name if analyzer else None

    @analyzer_name.setter
    def analyzer_name(self, name: str):
        analyzer = TaskRegistry.get_registry().analyzer_by_name.get(name)
        if analyzer:
            self.analyzer_id = analyzer.id

    def as_model(
        self,
        session,
        model_cls: Optional[Type[BaseModel]] = None,
        recursion: Optional[Set[int]] = None,
        **extra,
    ):
        model_cls = (
            model_cls
            or TaskRegistry.get_registry().analysis_model_by_name[self.analyzer_name]
        )
        if self.collection_id:
            # If this becomes async, I could load it...
            c = self.loaded_attributes().get("collection")
            if c:
                extra["collection_name"] = c.name
        ignore_keys = ["collection", "analyzer_id", "params", "analyzer"]
        if not issubclass(model_cls, AnalysisWithTemplateModel):
            ignore_keys.extend(["task_template_id", "task_template"])
        model = super(Analysis, self).as_model(
            session,
            model_cls,
            ignore_keys=ignore_keys,
            recursion=recursion,
            **(extra | self.params),
        )
        return model

    @classmethod
    async def from_model(
        cls, session, model: BaseModel, ignore: Optional[List[str]] = None, **extra
    ):
        from .content import Fragment

        assert isinstance(model, AnalysisModel)
        args = extra | model.model_dump()
        if model.collection_name and not model.collection_id:
            args["collection_id"] = await session.scalar(
                select(Collection.id).filter_by(name=model.collection_name)
            )
        rel_names = inspect(cls).relationships.keys()
        col_names = inspect(cls).c.keys()
        for k in (
            ["task_template_nickname", "analyzer_name", "collection_name"]
            + rel_names
            + (ignore or [])
        ):
            args.pop(k, None)
        col_args = {k: v for (k, v) in args.items() if k in col_names}
        col_args["params"] = jsonable_encoder(
            {k: v for (k, v) in args.items() if k not in col_names}
        )
        instance = cls(**col_args)
        if (
            getattr(model, "task_template", None) is not None
            and not model.task_template.id
        ):
            template_data = model.task_template.model_dump()
            for k in chain(
                inspect(TaskTemplate).c.keys(),
                inspect(cls).relationships.keys(),
                ("analyzer_name",),
            ):
                template_data.pop(k, None)
            template_model = await TaskRegistry.get_registry().ensure_task_template(
                session,
                model.analyzer_name,
                model.task_template_nickname,
                template_data,
            )
            instance.task_template_id = template_model.id
        if getattr(model, "task_template_id", None) is not None:
            instance.task_template = await session.get(
                TaskTemplate, model.task_template_id
            )
        if source_ids := getattr(model, "source_ids", []):
            instance.context = list(
                await session.scalars(
                    select(Fragment).filter(Fragment.id.in_(source_ids))
                )
            )
        return instance


Topic.from_analyses: Mapped[List[Analysis]] = relationship(
    "Analysis", secondary="analysis_output", back_populates="generated_topics"
)
Topic.target_of_analyses: Mapped[List[Analysis]] = relationship(
    "Analysis", viewonly=True
)
