from collections.abc import Iterable

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
            score=float(score),
            retriever=retriever,
        )
        for rank, (passage, score) in enumerate(rows, start=1)
    ]
