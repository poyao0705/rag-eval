from typing import Any, cast
import unittest

from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import SourcePassage
from backend.modules.retrieval.contracts import RetrievalRequest
from backend.modules.retrieval.embeddings import EMBEDDING_DIMENSIONS
from backend.modules.retrieval.pipelines.vector import VectorRetriever


class FakeEmbedder:
    def __init__(self, embedding):
        self.embedding = embedding
        self.queries = []

    async def embed_query(self, query: str):
        self.queries.append(query)
        return self.embedding


class FakeResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.statement: Any | None = None
        self.execute_count = 0

    async def execute(self, statement):
        self.execute_count += 1
        self.statement = statement
        return FakeResult(self.rows)


class VectorRetrieverTests(unittest.IsolatedAsyncioTestCase):
    async def test_embeds_once_compiles_cosine_query_and_maps_similarity(self):
        passage = SourcePassage(
            title="Distributed Systems",
            normalized_title="distributed systems",
            sentences=["Consensus coordinates replicas."],
            text="Consensus coordinates replicas.",
            content_hash="vector",
        )
        query_embedding = [0.5] * EMBEDDING_DIMENSIONS
        embedder = FakeEmbedder(query_embedding)
        session = FakeSession([(passage, 0.875)])
        request = RetrievalRequest(query="distributed systems", top_k=5)
        retriever = VectorRetriever(embedder)

        results = await retriever.retrieve(
            request,
            cast(AsyncSession, session),
        )

        self.assertEqual(embedder.queries, ["distributed systems"])
        self.assertEqual(session.execute_count, 1)
        self.assertIsNotNone(session.statement)
        assert session.statement is not None
        compiled = session.statement.compile(dialect=postgresql.dialect())
        sql = " ".join(str(compiled).split())
        self.assertIn("source_passage.embedding <=>", sql)
        self.assertIn("source_passage.embedding IS NOT NULL", sql)
        self.assertIn("AS score", sql)
        self.assertIn("ORDER BY (source_passage.embedding <=>", sql)
        self.assertIn("ASC, source_passage.id ASC", sql)
        self.assertIn(query_embedding, compiled.params.values())
        self.assertIn(5, compiled.params.values())
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].passage_id, passage.id)
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[0].score, 0.875)
        self.assertEqual(results[0].retriever, "vector")

    async def test_rejects_wrong_query_embedding_dimension_before_sql(self):
        embedder = FakeEmbedder([0.0] * 1535)
        session = FakeSession([])
        retriever = VectorRetriever(embedder)

        with self.assertRaisesRegex(ValueError, "1536"):
            await retriever.retrieve(
                RetrievalRequest(query="distributed systems"),
                cast(AsyncSession, session),
            )

        self.assertEqual(embedder.queries, ["distributed systems"])
        self.assertEqual(session.execute_count, 0)
        self.assertIsNone(session.statement)
