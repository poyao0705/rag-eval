from collections.abc import Sequence
from dataclasses import replace

from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import DEFAULT_RAG_CONFIG
from backend.modules.retrieval.contracts import (
    QueryEmbedder,
    RetrievalRequest,
    RetrievedPassage,
)
from backend.modules.retrieval.pipelines.tsvector import TSVectorRetriever
from backend.modules.retrieval.pipelines.vector import VectorRetriever
from backend.modules.retrieval.utils import reciprocal_rank_fusion


class HybridTSVectorRetriever:
    """A combination of TSVector and dense retrieval."""

    name: str = "hybrid_tsvector"

    def __init__(
        self,
        embedder: QueryEmbedder,
        *,
        candidate_top_k: int = DEFAULT_RAG_CONFIG.hybrid_candidate_top_k,
    ) -> None:
        if (
            not isinstance(candidate_top_k, int)
            or isinstance(candidate_top_k, bool)
            or candidate_top_k <= 0
        ):
            raise ValueError("candidate_top_k must be a positive integer")
        self.candidate_top_k = candidate_top_k
        self.tsvector = TSVectorRetriever()
        self.vector = VectorRetriever(embedder)

    async def retrieve(
        self,
        request: RetrievalRequest,
        session: AsyncSession,
    ) -> Sequence[RetrievedPassage]:
        candidate_request = replace(
            request,
            top_k=max(request.top_k, self.candidate_top_k),
        )
        lexical = await self.tsvector.retrieve(candidate_request, session)
        vector = await self.vector.retrieve(candidate_request, session)
        return reciprocal_rank_fusion(
            [lexical, vector], retriever=self.name
        )[: request.top_k]
