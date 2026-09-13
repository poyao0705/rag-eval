from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import SourcePassage
from backend.modules.retrieval.contracts import RetrievalRequest, RetrievedPassage
from backend.modules.retrieval.utils import ranked_passages


class TSVectorRetriever:
    name: str = "tsvector"

    async def retrieve(
        self,
        request: RetrievalRequest,
        session: AsyncSession,
    ) -> Sequence[RetrievedPassage]:
        table = SourcePassage.__table__  # pyright: ignore[reportAttributeAccessIssue]
        parsed_query = func.websearch_to_tsquery("english", request.query)
        score = func.ts_rank_cd(table.c.search_vector, parsed_query).label("score")
        statement = (
            select(SourcePassage, score)
            .where(table.c.search_vector.bool_op("@@")(parsed_query))
            .order_by(score.desc(), table.c.id.asc())
            .limit(request.top_k)
        )
        result = await session.execute(statement)
        rows = [(row[0], float(row[1])) for row in result.all()]
        return ranked_passages(rows, retriever=self.name)
