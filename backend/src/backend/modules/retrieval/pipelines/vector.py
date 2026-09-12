from collections.abc import Sequence

from sqlalchemy import literal, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import SourcePassage
from backend.modules.retrieval.contracts import (
    QueryEmbedder,
    RetrievalRequest,
    RetrievedPassage,
)
from backend.modules.retrieval.embeddings import EMBEDDING_DIMENSIONS
from backend.modules.retrieval.pipelines._common import ranked_passages


class VectorRetriever:
    name: str = "vector"

    def __init__(self, embedder: QueryEmbedder) -> None:
        self._embedder = embedder

    async def retrieve(
        self,
        request: RetrievalRequest,
        session: AsyncSession,
    ) -> Sequence[RetrievedPassage]:
        embedding = list(await self._embedder.embed_query(request.query))
        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"embedding dimension must be {EMBEDDING_DIMENSIONS}")

        table = SourcePassage.__table__  # pyright: ignore[reportAttributeAccessIssue]
        distance = table.c.embedding.cosine_distance(embedding)
        score = (literal(1.0) - distance).label("score")
        statement = (
            select(SourcePassage, score)
            .where(table.c.embedding.is_not(None))
            .order_by(distance.asc(), table.c.id.asc())
            .limit(request.top_k)
        )
        result = await session.execute(statement)
        rows = [(row[0], float(row[1])) for row in result.all()]
        return ranked_passages(rows, retriever=self.name)
