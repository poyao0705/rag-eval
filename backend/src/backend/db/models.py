from enum import Enum
from typing import Any

from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel


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
