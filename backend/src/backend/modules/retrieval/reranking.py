from collections.abc import Sequence
from dataclasses import dataclass, replace

from cohere import AsyncClientV2

from backend.core.config import DEFAULT_RAG_CONFIG
from backend.modules.retrieval.contracts import RetrievedPassage


@dataclass(slots=True)
class CohereReranker:
    client: AsyncClientV2
    model: str = DEFAULT_RAG_CONFIG.rerank_model

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedPassage],
        limit: int,
    ) -> list[RetrievedPassage]:
        if not candidates or limit <= 0:
            return []

        response = await self.client.rerank(
            query=query,
            documents=[candidate.text for candidate in candidates],
            top_n=min(limit, len(candidates)),
            model=self.model,
        )
        passages = []
        for rank, result in enumerate(response.results, start=1):
            index = result.index
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(candidates)
            ):
                raise ValueError(f"Cohere returned an invalid document index: {index}")
            passages.append(replace(
                candidates[index], rank=rank, relevance_score=result.relevance_score
            ))
        return passages
