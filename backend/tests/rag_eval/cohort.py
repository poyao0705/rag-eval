from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import DEFAULT_RAG_CONFIG, RAGConfig

COHORT_SQL = """
SELECT q.id, q.question, q.answer
FROM hotpot_qa AS q
WHERE q.split = 'VALIDATION'
  AND btrim(q.question) <> ''
  AND btrim(q.answer) <> ''
  AND EXISTS (
    SELECT 1 FROM hotpot_qa_context AS c WHERE c.hotpot_qa_id = q.id
  )
ORDER BY md5(CAST(:seed AS text) || ':' || q.id), q.id
LIMIT :sample_size
"""


@dataclass(frozen=True, slots=True)
class QAExample:
    id: str
    question: str
    answer: str


async def load_cohort(
    session: AsyncSession, config: RAGConfig = DEFAULT_RAG_CONFIG
) -> list[QAExample]:
    result = await session.execute(
        text(COHORT_SQL),
        {
            "seed": config.evaluation_seed,
            "sample_size": config.evaluation_sample_size,
        },
    )
    cohort = [QAExample(**row) for row in result.mappings().all()]
    if (
        len(cohort) != config.evaluation_sample_size
        or len({qa.id for qa in cohort}) != config.evaluation_sample_size
    ):
        raise ValueError(
            f"expected {config.evaluation_sample_size} distinct eligible validation questions"
        )
    if any(not qa.question.strip() or not qa.answer.strip() for qa in cohort):
        raise ValueError("cohort contains a blank question or answer")
    return cohort
