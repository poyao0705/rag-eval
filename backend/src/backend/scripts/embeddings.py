import argparse
import asyncio
from dataclasses import dataclass
import sys
from typing import Any

from openai import AsyncOpenAI
from sqlalchemy import select, update

from backend.db.models import SourcePassage

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
DEFAULT_BATCH_SIZE = 100


@dataclass
class EmbeddingStats:
    selected_passages: int = 0
    embedded_passages: int = 0


def _validated_embeddings(response: Any, expected_count: int) -> list[list[float]]:
    data = list(response.data)
    if len(data) != expected_count:
        raise ValueError(
            "embedding response count does not match requested passage count"
        )

    try:
        ordered_data = sorted(data, key=lambda item: item.index)
        indexes = [item.index for item in ordered_data]
    except AttributeError as error:
        raise ValueError("embedding response items must include an index") from error

    if indexes != list(range(expected_count)):
        raise ValueError("embedding response indexes are not contiguous and ordered")

    embeddings = [item.embedding for item in ordered_data]
    for embedding in embeddings:
        if len(embedding) != EMBEDDING_DIMENSIONS:
            raise ValueError(f"embedding dimension must be {EMBEDDING_DIMENSIONS}")
    return embeddings


async def embed_passages(
    client: Any,
    session_factory: Any | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> EmbeddingStats:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if session_factory is None:
        from backend.db.database import SessionFactory

        session_factory = SessionFactory

    stats = EmbeddingStats()
    batch_number = 0
    print(
        f"embedding model={EMBEDDING_MODEL} batch_size={batch_size}",
        flush=True,
    )
    while True:
        source_passage_table = SourcePassage.__table__  # pyright: ignore[reportAttributeAccessIssue]
        statement = (
            select(SourcePassage)
            .where(source_passage_table.c.embedding.is_(None))
            .order_by(source_passage_table.c.id)
            .limit(batch_size)
        )
        async with session_factory() as session:
            passages = list((await session.scalars(statement)).all())

        if not passages:
            break

        batch_number += 1
        stats.selected_passages += len(passages)
        print(
            f"batch={batch_number} selected={len(passages)} "
            f"total_embedded={stats.embedded_passages} status=requesting",
            flush=True,
        )
        try:
            response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[passage.text for passage in passages],
            )
            embeddings = _validated_embeddings(response, len(passages))

            async with session_factory.begin() as session:
                for passage, embedding in zip(passages, embeddings, strict=True):
                    await session.execute(
                        update(SourcePassage)
                        .where(
                            source_passage_table.c.id == passage.id,
                            source_passage_table.c.embedding.is_(None),
                        )
                        .values(embedding=embedding)
                    )
        except Exception:
            print(
                f"batch={batch_number} selected={len(passages)} "
                f"total_embedded={stats.embedded_passages} "
                "status=failed current_batch_not_committed",
                file=sys.stderr,
                flush=True,
            )
            raise

        stats.embedded_passages += len(passages)
        print(
            f"batch={batch_number} selected={len(passages)} "
            f"total_embedded={stats.embedded_passages} status=committed",
            flush=True,
        )

    print(
        f"embedding complete selected={stats.selected_passages} "
        f"embedded={stats.embedded_passages}",
        flush=True,
    )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")

    from backend.core.config import Settings

    settings = Settings()  # pyright: ignore[reportCallIssue]
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY.get_secret_value())
    stats = asyncio.run(embed_passages(client, batch_size=args.batch_size))
    print(f"selected_passages={stats.selected_passages}")
    print(f"embedded_passages={stats.embedded_passages}")
