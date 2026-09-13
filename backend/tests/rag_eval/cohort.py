from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


SEED = 42
SAMPLE_SIZE = 20
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


async def load_cohort(session: AsyncSession) -> list[QAExample]:
    result = await session.execute(
        text(COHORT_SQL), {"seed": SEED, "sample_size": SAMPLE_SIZE}
    )
    cohort = [QAExample(**row) for row in result.mappings().all()]
    if len(cohort) != SAMPLE_SIZE or len({qa.id for qa in cohort}) != SAMPLE_SIZE:
        raise ValueError("expected 20 distinct eligible validation questions")
    if any(not qa.question.strip() or not qa.answer.strip() for qa in cohort):
        raise ValueError("cohort contains a blank question or answer")
    return cohort
