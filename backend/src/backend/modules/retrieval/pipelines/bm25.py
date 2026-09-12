from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import SourcePassage
from backend.modules.retrieval.contracts import RetrievalRequest, RetrievedPassage
from backend.modules.retrieval.pipelines._common import ranked_passages


class BM25Retriever:
    name: str = "bm25"

    async def retrieve(
        self,
        request: RetrievalRequest,
        session: AsyncSession,
    ) -> Sequence[RetrievedPassage]:
        table = SourcePassage.__table__  # pyright: ignore[reportAttributeAccessIssue]
        score = func.pdb.score(table.c.id).label("score")
        statement = (
            select(SourcePassage, score)
            .where(table.c.text.op("|||")(request.query))
            .order_by(score.desc(), table.c.id.asc())
            .limit(request.top_k)
        )
        result = await session.execute(statement)
        rows = [(row[0], float(row[1])) for row in result.all()]
        return ranked_passages(rows, retriever=self.name)
