import unittest
from typing import cast
from unittest.mock import AsyncMock, patch
from uuid import UUID

from backend.core.config import RAGConfig
from backend.modules.retrieval.contracts import (
    QueryEmbedder,
    RetrievalRequest,
    RetrievedPassage,
)
from backend.modules.retrieval.pipelines.hybrid_bm25 import HybridBM25Retriever
from backend.modules.retrieval.pipelines.hybrid_tsvector import (
    HybridTSVectorRetriever,
)


class HybridRetrieverTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.embedder = cast(QueryEmbedder, object())
        self.session = object()

    def make_passage(self, number: int, retriever: str) -> RetrievedPassage:
        return RetrievedPassage(
            UUID(int=number),
            f"Title {number}",
            f"Text {number}",
            1,
            float(number),
            retriever,
        )

    async def assert_retrieves_expanded_candidates(self, retriever, lexical_name):
        lexical_results = [
            self.make_passage(1, "lexical"),
            self.make_passage(2, "lexical"),
            self.make_passage(3, "lexical"),
        ]
        vector_results = [
            self.make_passage(2, "vector"),
            self.make_passage(4, "vector"),
        ]
        request = RetrievalRequest(query="find this", top_k=2)
        lexical = getattr(retriever, lexical_name)

        with (
            patch.object(lexical, "retrieve", new_callable=AsyncMock) as lexical_call,
            patch.object(
                retriever.vector, "retrieve", new_callable=AsyncMock
            ) as vector_call,
        ):
            lexical_call.return_value = lexical_results
            vector_call.return_value = vector_results
            results = await retriever.retrieve(request, self.session)

        expected_request = RetrievalRequest(query="find this", top_k=3)
        lexical_call.assert_awaited_once_with(expected_request, self.session)
        vector_call.assert_awaited_once_with(expected_request, self.session)
        lexical_args = lexical_call.await_args
        assert lexical_args is not None
        self.assertIsNot(lexical_args.args[0], request)
        self.assertEqual(request, RetrievalRequest(query="find this", top_k=2))
        self.assertEqual([passage.passage_id.int for passage in results], [2, 1])
        self.assertEqual([passage.rank for passage in results], [1, 2])
        self.assertEqual([passage.retriever for passage in results], [retriever.name] * 2)

    async def assert_handles_empty_results(self, retriever, lexical_name):
        lexical = getattr(retriever, lexical_name)
        with (
            patch.object(lexical, "retrieve", new_callable=AsyncMock) as lexical_call,
            patch.object(
                retriever.vector, "retrieve", new_callable=AsyncMock
            ) as vector_call,
        ):
            lexical_call.return_value = []
            vector_call.return_value = []
            results = await retriever.retrieve(
                RetrievalRequest(query="empty", top_k=2), self.session
            )

        self.assertEqual(results, [])

    async def assert_expands_to_final_limit(self, retriever, lexical_name):
        lexical = getattr(retriever, lexical_name)
        with (
            patch.object(lexical, "retrieve", new_callable=AsyncMock) as lexical_call,
            patch.object(
                retriever.vector, "retrieve", new_callable=AsyncMock
            ) as vector_call,
        ):
            lexical_call.return_value = [self.make_passage(1, "lexical")]
            vector_call.return_value = [self.make_passage(2, "vector")]
            results = await retriever.retrieve(
                RetrievalRequest(query="wide", top_k=5), self.session
            )

        expected_request = RetrievalRequest(query="wide", top_k=5)
        lexical_call.assert_awaited_once_with(expected_request, self.session)
        vector_call.assert_awaited_once_with(expected_request, self.session)
        self.assertEqual(len(results), 2)

    async def test_bm25_hybrid_fuses_and_caps_results(self):
        await self.assert_retrieves_expanded_candidates(
            HybridBM25Retriever(self.embedder, candidate_top_k=3), "bm25"
        )

    async def test_tsvector_hybrid_fuses_and_caps_results(self):
        await self.assert_retrieves_expanded_candidates(
            HybridTSVectorRetriever(self.embedder, candidate_top_k=3), "tsvector"
        )

    async def test_both_hybrids_handle_empty_component_results(self):
        for retriever, lexical_name in (
            (HybridBM25Retriever(self.embedder, candidate_top_k=3), "bm25"),
            (HybridTSVectorRetriever(self.embedder, candidate_top_k=3), "tsvector"),
        ):
            with self.subTest(retriever=retriever.name):
                await self.assert_handles_empty_results(retriever, lexical_name)

    async def test_both_hybrids_expand_when_final_limit_is_larger(self):
        for retriever, lexical_name in (
            (HybridBM25Retriever(self.embedder, candidate_top_k=2), "bm25"),
            (HybridTSVectorRetriever(self.embedder, candidate_top_k=2), "tsvector"),
        ):
            with self.subTest(retriever=retriever.name):
                await self.assert_expands_to_final_limit(retriever, lexical_name)

    def test_both_hybrids_reject_nonpositive_candidate_limits(self):
        for factory in (HybridBM25Retriever, HybridTSVectorRetriever):
            with self.subTest(factory=factory.__name__):
                with self.assertRaisesRegex(ValueError, "candidate_top_k"):
                    factory(self.embedder, candidate_top_k=0)

    def test_default_candidate_limit_matches_rag_config(self):
        self.assertEqual(RAGConfig().hybrid_candidate_top_k, 50)
        self.assertEqual(
            HybridBM25Retriever(self.embedder).candidate_top_k,
            RAGConfig().hybrid_candidate_top_k,
        )
        self.assertEqual(
            HybridTSVectorRetriever(self.embedder).candidate_top_k,
            RAGConfig().hybrid_candidate_top_k,
        )


if __name__ == "__main__":
    unittest.main()
