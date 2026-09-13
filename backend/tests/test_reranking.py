import unittest
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from uuid import UUID

from cohere import AsyncClientV2

from backend.modules.retrieval.contracts import RetrievedPassage
from backend.modules.retrieval.reranking import CohereReranker


class RerankingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.client = SimpleNamespace(rerank=AsyncMock())
        self.reranker = CohereReranker(cast(AsyncClientV2, self.client))
        self.candidates = [
            RetrievedPassage(UUID(int=i), f"Title {i}", f"Text {i}", i,
                             1 / (60 + i), "hybrid_bm25")
            for i in (1, 2, 3)
        ]

    async def test_maps_response_order_and_preserves_originals(self):
        original = list(self.candidates)
        self.client.rerank.return_value = SimpleNamespace(results=[
            SimpleNamespace(index=2, relevance_score=0.9),
            SimpleNamespace(index=0, relevance_score=0.7),
        ])
        results = await self.reranker.rerank("focused query", self.candidates, 2)
        self.client.rerank.assert_awaited_once_with(
            query="focused query", documents=["Text 1", "Text 2", "Text 3"],
            top_n=2, model="rerank-english-v3.0",
        )
        self.assertEqual(results, [
            replace(original[2], rank=1, relevance_score=0.9),
            replace(original[0], rank=2, relevance_score=0.7),
        ])
        self.assertEqual(self.candidates, original)
        self.assertTrue(all(p.relevance_score is None for p in original))

    async def test_empty_or_nonpositive_limit_does_not_call_api(self):
        for candidates, limit in (([], 2), (self.candidates, 0),
                                  (self.candidates, -1)):
            with self.subTest(limit=limit, count=len(candidates)):
                self.assertEqual(await self.reranker.rerank("q", candidates, limit), [])
        self.client.rerank.assert_not_awaited()

    async def test_caps_request_and_passes_model_override(self):
        self.client.rerank.return_value = SimpleNamespace(results=[
            SimpleNamespace(index=0, relevance_score=0.8),
        ])
        reranker = CohereReranker(
            cast(AsyncClientV2, self.client), model="custom-model"
        )
        results = await reranker.rerank("q", self.candidates[:1], 10)
        self.client.rerank.assert_awaited_once_with(
            query="q", documents=["Text 1"], top_n=1, model="custom-model",
        )
        self.assertEqual(len(results), 1)

    async def test_rejects_invalid_indices(self):
        for index in (-1, 3, True, 1.5, "0"):
            with self.subTest(index=index):
                self.client.rerank.return_value = SimpleNamespace(results=[
                    SimpleNamespace(index=index, relevance_score=0.5),
                ])
                with self.assertRaisesRegex(ValueError, "invalid document index"):
                    await self.reranker.rerank("q", self.candidates, 2)

    async def test_propagates_api_failure(self):
        failure = RuntimeError("provider unavailable")
        self.client.rerank.side_effect = failure
        with self.assertRaises(RuntimeError) as caught:
            await self.reranker.rerank("q", self.candidates, 2)
        self.assertIs(caught.exception, failure)
