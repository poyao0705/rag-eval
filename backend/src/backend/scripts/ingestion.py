import asyncio
from itertools import batched

from datasets import IterableDatasetDict, load_dataset
from sqlalchemy.dialects.postgresql import insert

from backend.db.database import SessionFactory
from backend.db.models import HotpotQA, HotpotQASplit

DEFAULT_DATASET_PATH = "hotpotqa/hotpot_qa"
DEFAULT_DATASET_NAME = "distractor"
BATCH_SIZE = 1_000


def load_data_from_huggingface(
    dataset_path: str,
    dataset_name: str,
) -> IterableDatasetDict:
    return load_dataset(dataset_path, dataset_name, streaming=True)


async def ingest_data(dataset: IterableDatasetDict) -> None:

    for split, data_rows in dataset.items():
        count = 0
        for batch in batched(data_rows, BATCH_SIZE):
            print(
                f"Processing {split} batch {count}, this batch contains {len(batch)} rows"
            )
            records = [
                HotpotQA.model_validate(
                    {**row, "split": HotpotQASplit(split)}
                ).model_dump()
                for row in batch
            ]
            statement = insert(HotpotQA).values(records)
            statement = statement.on_conflict_do_nothing(index_elements=["id"])

            async with SessionFactory.begin() as session:
                await session.execute(statement)

            count += 1


def main() -> None:
    print("Ingesting data...")
    dataset = load_data_from_huggingface(
        dataset_path=DEFAULT_DATASET_PATH,
        dataset_name=DEFAULT_DATASET_NAME,
    )
    asyncio.run(ingest_data(dataset))
