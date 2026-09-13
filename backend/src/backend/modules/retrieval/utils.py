from collections.abc import Iterable
from dataclasses import replace
from uuid import UUID

from backend.db.models import SourcePassage
from backend.modules.retrieval.contracts import RetrievedPassage


def ranked_passages(
    rows: Iterable[tuple[SourcePassage, float]],
    retriever: str,
) -> list[RetrievedPassage]:
    return [
        RetrievedPassage(
            passage_id=passage.id,
            title=passage.title,
            text=passage.text,
            rank=rank,
            score=score,
            retriever=retriever,
        )
        for rank, (passage, score) in enumerate(rows, start=1)
    ]


def reciprocal_rank_fusion(
    lists: Iterable[Iterable[RetrievedPassage]],
    rank_constant: int = 60,
    retriever: str = "hybrid",
) -> list[RetrievedPassage]:
    if (
        not isinstance(rank_constant, int)
        or isinstance(rank_constant, bool)
        or rank_constant < 0
    ):
        raise ValueError("rank_constant must be a nonnegative integer")

    candidates: dict[UUID, tuple[RetrievedPassage, float]] = {}
    for results in lists:
        seen: set[UUID] = set()
        for rank, candidate in enumerate(results, start=1):
            if candidate.passage_id in seen:
                continue
            seen.add(candidate.passage_id)
            contribution = 1 / (rank_constant + rank)
            existing = candidates.get(candidate.passage_id)
            if existing is None:
                candidates[candidate.passage_id] = (candidate, contribution)
            else:
                candidates[candidate.passage_id] = (
                    existing[0],
                    existing[1] + contribution,
                )

    ranked = sorted(
        candidates.values(),
        key=lambda item: (-item[1], str(item[0].passage_id)),
    )
    return [
        replace(candidate, rank=rank, score=fused_score, retriever=retriever)
        for rank, (candidate, fused_score) in enumerate(ranked, start=1)
    ]
