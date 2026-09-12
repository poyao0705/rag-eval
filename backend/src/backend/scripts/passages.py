import argparse
import asyncio
from collections.abc import Iterable
from dataclasses import dataclass
import uuid

from sqlalchemy import select, tuple_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.db.database import SessionFactory
from backend.db.models import HotpotQA, HotpotQAContext, SourcePassage
from backend.passage import extract_context


@dataclass
class MaterializationStats:
    qa_rows: int = 0
    candidate_passages: int = 0
    inserted_passages: int = 0
    candidate_links: int = 0
    inserted_links: int = 0


def build_passage_rows(
    rows: Iterable[HotpotQA],
) -> tuple[list[dict], list[dict], int]:
    passage_rows: list[dict] = []
    link_rows: list[dict] = []
    candidate_ids: dict[tuple[str, str], uuid.UUID] = {}
    candidate_count = 0

    for row in rows:
        candidates = extract_context(row.context)
        candidate_count += len(candidates)
        for position, candidate in enumerate(candidates):
            passage_id = candidate_ids.get(candidate.identity)
            if passage_id is None:
                passage_id = uuid.uuid7()
                candidate_ids[candidate.identity] = passage_id
                passage_rows.append(
                    {
                        "id": passage_id,
                        "title": candidate.title,
                        "normalized_title": candidate.normalized_title,
                        "sentences": list(candidate.sentences),
                        "text": candidate.text,
                        "content_hash": candidate.content_hash,
                    }
                )
            link_rows.append(
                {
                    "hotpot_qa_id": row.id,
                    "position": position,
                    "source_passage_id": passage_id,
                }
            )

    return passage_rows, link_rows, candidate_count


async def materialize_passages(
    session_factory: async_sessionmaker[AsyncSession] = SessionFactory,
    batch_size: int = 1_000,
) -> MaterializationStats:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    stats = MaterializationStats()
    last_id: str | None = None

    while True:
        async with session_factory.begin() as session:
            statement = select(HotpotQA).order_by(HotpotQA.id).limit(batch_size)
            if last_id is not None:
                statement = statement.where(HotpotQA.id > last_id)
            rows = list((await session.scalars(statement)).all())
            if not rows:
                break

            passage_rows, link_rows, candidate_count = build_passage_rows(rows)

            inserted_passages = 0
            if passage_rows:
                passage_statement = (
                    insert(SourcePassage)
                    .values(passage_rows)
                    .on_conflict_do_nothing(
                        index_elements=["normalized_title", "content_hash"]
                    )
                    .returning(SourcePassage.id)
                )
                inserted_passages = len(
                    (await session.scalars(passage_statement)).all()
                )

                identities = [
                    (row["normalized_title"], row["content_hash"])
                    for row in passage_rows
                ]
                source_rows = await session.execute(
                    select(
                        SourcePassage.normalized_title,
                        SourcePassage.content_hash,
                        SourcePassage.id,
                    ).where(
                        tuple_(
                            SourcePassage.normalized_title,
                            SourcePassage.content_hash,
                        ).in_(identities)
                    )
                )
                identity_to_id = {
                    (normalized_title, content_hash): source_id
                    for normalized_title, content_hash, source_id in source_rows
                }
                candidate_identities = {
                    row["id"]: (row["normalized_title"], row["content_hash"])
                    for row in passage_rows
                }
                for link_row in link_rows:
                    link_row["source_passage_id"] = identity_to_id[
                        candidate_identities[link_row["source_passage_id"]]
                    ]

            inserted_links = 0
            if link_rows:
                link_statement = (
                    insert(HotpotQAContext)
                    .values(link_rows)
                    .on_conflict_do_nothing(
                        index_elements=["hotpot_qa_id", "position"]
                    )
                    .returning(
                        HotpotQAContext.hotpot_qa_id,
                        HotpotQAContext.position,
                    )
                )
                inserted_links = len((await session.execute(link_statement)).all())

            stats.qa_rows += len(rows)
            stats.candidate_passages += candidate_count
            stats.inserted_passages += inserted_passages
            stats.candidate_links += len(link_rows)
            stats.inserted_links += inserted_links
            last_id = rows[-1].id

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1_000)
    args = parser.parse_args()
    stats = asyncio.run(materialize_passages(batch_size=args.batch_size))
    print(f"qa_rows={stats.qa_rows}")
    print(f"candidate_passages={stats.candidate_passages}")
    print(f"inserted_passages={stats.inserted_passages}")
    print(f"candidate_links={stats.candidate_links}")
    print(f"inserted_links={stats.inserted_links}")
