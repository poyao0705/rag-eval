from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query: str
    top_k: int = 10

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be blank")
        if self.top_k <= 0:
            raise ValueError("top_k must be greater than zero")


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    passage_id: UUID
    title: str
    text: str
    rank: int
    score: float | None
    retriever: str


class QueryEmbedder(Protocol):
    async def embed_query(self, query: str) -> Sequence[float]: ...


class Retriever(Protocol):
    name: str

    async def retrieve(
        self,
        request: RetrievalRequest,
        session: AsyncSession,
    ) -> Sequence[RetrievedPassage]: ...
