from enum import Enum
from typing import Any
import uuid

from sqlalchemy import Column, Computed, Index, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlmodel import Field, SQLModel


class HotpotQASplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class HotpotQA(SQLModel, table=True):
    __tablename__ = "hotpot_qa"  # pyright: ignore[reportAssignmentType]

    id: str = Field(primary_key=True)
    question: str
    answer: str
    type: str
    level: str
    supporting_facts: dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))
    context: dict[str, Any] = Field(sa_column=Column(JSONB, nullable=False))
    split: HotpotQASplit = Field(nullable=False)


class SourcePassage(SQLModel, table=True):
    __tablename__ = "source_passage"  # pyright: ignore[reportAssignmentType]
    __table_args__ = (
        UniqueConstraint(
            "normalized_title",
            "content_hash",
            name="uq_source_passage_identity",
        ),
        Index("ix_source_passage_content_hash", "content_hash"),
        Index(
            "ix_source_passage_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid7, primary_key=True)
    title: str
    normalized_title: str
    sentences: list[str] = Field(
        sa_column=Column(JSONB, nullable=False)
    )
    text: str
    search_vector: Any = Field(
        default=None,
        sa_column=Column(
            TSVECTOR,
            Computed("to_tsvector('english', text)", persisted=True),
        ),
    )
    content_hash: str


class HotpotQAContext(SQLModel, table=True):
    __tablename__ = "hotpot_qa_context"  # pyright: ignore[reportAssignmentType]
    __table_args__ = (
        Index(
            "ix_hotpot_qa_context_retrieval",
            "hotpot_qa_id",
            "source_passage_id",
        ),
    )

    hotpot_qa_id: str = Field(
        foreign_key="hotpot_qa.id",
        primary_key=True,
    )
    position: int = Field(primary_key=True)
    source_passage_id: uuid.UUID = Field(
        foreign_key="source_passage.id",
        nullable=False,
    )
