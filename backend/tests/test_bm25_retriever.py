from typing import Any, cast
import unittest

from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import SourcePassage
from backend.modules.retrieval.contracts import RetrievalRequest
from backend.modules.retrieval.pipelines.bm25 import BM25Retriever


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.statement: Any | None = None

    async def execute(self, statement):
        self.statement = statement
        return FakeResult(self.rows)


class BM25RetrieverTests(unittest.IsolatedAsyncioTestCase):
    async def test_compiles_bm25_query_and_maps_ranked_results(self):
        passage = SourcePassage(
            title="Distributed Systems",
            normalized_title="distributed systems",
            sentences=["Consensus coordinates replicas."],
            text="Consensus coordinates replicas.",
            content_hash="bm25",
        )
        session = FakeSession([(passage, 7.5)])
        request = RetrievalRequest(query="distributed systems", top_k=3)

        results = await BM25Retriever().retrieve(
            request,
            cast(AsyncSession, session),
        )

        self.assertIsNotNone(session.statement)
        assert session.statement is not None
        compiled = session.statement.compile(dialect=postgresql.dialect())
        sql = " ".join(str(compiled).split())
        self.assertIn("source_passage.text |||", sql)
        self.assertIn("pdb.score(source_passage.id) AS score", sql)
        self.assertIn("ORDER BY score DESC, source_passage.id ASC", sql)
        self.assertIn("distributed systems", compiled.params.values())
        self.assertIn(3, compiled.params.values())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].passage_id, passage.id)
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[0].score, 7.5)
        self.assertEqual(results[0].retriever, "bm25")
