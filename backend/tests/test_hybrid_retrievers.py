import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch
from uuid import UUID

from cohere import AsyncClientV2
from sqlalchemy.ext.asyncio import AsyncSession

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
from backend.modules.retrieval.reranking import CohereReranker
from backend.modules.retrieval.utils import reciprocal_rank_fusion


class HybridRetrieverTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.embedder = cast(QueryEmbedder, object())
        self.session = cast(AsyncSession, object())
        self.client = SimpleNamespace(rerank=AsyncMock())
        self.reranker = CohereReranker(cast(AsyncClientV2, self.client))
        self.client.rerank.side_effect = lambda **kwargs: SimpleNamespace(results=[
            SimpleNamespace(index=i, relevance_score=0.9 - i * 0.1)
            for i in range(kwargs["top_n"])
        ])

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
        self.client.rerank.reset_mock()
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
        self.client.rerank.assert_not_awaited()

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
            HybridBM25Retriever(
                self.embedder, reranker=self.reranker, candidate_top_k=3
            ), "bm25"
        )

    async def test_tsvector_hybrid_fuses_and_caps_results(self):
        await self.assert_retrieves_expanded_candidates(
            HybridTSVectorRetriever(
                self.embedder, reranker=self.reranker, candidate_top_k=3
            ), "tsvector"
        )

    async def test_both_hybrids_handle_empty_component_results(self):
        for retriever, lexical_name in (
            (
                HybridBM25Retriever(
                    self.embedder, reranker=self.reranker, candidate_top_k=3
                ),
                "bm25",
            ),
            (
                HybridTSVectorRetriever(
                    self.embedder, reranker=self.reranker, candidate_top_k=3
                ),
                "tsvector",
            ),
        ):
            with self.subTest(retriever=retriever.name):
                await self.assert_handles_empty_results(retriever, lexical_name)

    async def test_both_hybrids_expand_when_final_limit_is_larger(self):
        for retriever, lexical_name in (
            (
                HybridBM25Retriever(
                    self.embedder, reranker=self.reranker, candidate_top_k=2
                ),
                "bm25",
            ),
            (
                HybridTSVectorRetriever(
                    self.embedder, reranker=self.reranker, candidate_top_k=2
                ),
                "tsvector",
            ),
        ):
            with self.subTest(retriever=retriever.name):
                await self.assert_expands_to_final_limit(retriever, lexical_name)

    async def test_both_hybrids_rerank_full_fused_pool_before_truncation(self):
        for factory, lexical_name in (
            (HybridBM25Retriever, "bm25"),
            (HybridTSVectorRetriever, "tsvector"),
        ):
            with self.subTest(factory=factory.__name__):
                client = SimpleNamespace(rerank=AsyncMock(return_value=SimpleNamespace(
                    results=[SimpleNamespace(index=2, relevance_score=0.95)]
                )))
                retriever = factory(
                    self.embedder,
                    reranker=CohereReranker(cast(AsyncClientV2, client)),
                    candidate_top_k=3,
                )
                lexical = [self.make_passage(i, "lexical") for i in (1, 2, 3)]
                vector = [self.make_passage(1, "vector")]
                fused = reciprocal_rank_fusion([lexical, vector], retriever=retriever.name)
                with (
                    patch.object(getattr(retriever, lexical_name), "retrieve",
                                 new_callable=AsyncMock, return_value=lexical),
                    patch.object(retriever.vector, "retrieve",
                                 new_callable=AsyncMock, return_value=vector),
                ):
                    results = await retriever.retrieve(
                        RetrievalRequest(query="focused query", top_k=1),
                        self.session,
                    )
                client.rerank.assert_awaited_once_with(
                    query="focused query", documents=[p.text for p in fused],
                    top_n=1, model="rerank-english-v3.0",
                )
                self.assertEqual(len(fused), 3)
                self.assertEqual(len(results), 1)
                self.assertEqual(results[0].passage_id, fused[2].passage_id)
                self.assertEqual(results[0].score, fused[2].score)
                self.assertEqual(results[0].retriever, retriever.name)
                self.assertEqual(results[0].rank, 1)
                self.assertEqual(results[0].relevance_score, 0.95)

    async def test_both_hybrids_propagate_rerank_failure(self):
        for factory, lexical_name in (
            (HybridBM25Retriever, "bm25"),
            (HybridTSVectorRetriever, "tsvector"),
        ):
            with self.subTest(factory=factory.__name__):
                failure = RuntimeError("provider unavailable")
                client = SimpleNamespace(rerank=AsyncMock(side_effect=failure))
                retriever = factory(
                    self.embedder,
                    reranker=CohereReranker(cast(AsyncClientV2, client)),
                )
                with (
                    patch.object(getattr(retriever, lexical_name), "retrieve",
                                 new_callable=AsyncMock,
                                 return_value=[self.make_passage(1, "lexical")]),
                    patch.object(retriever.vector, "retrieve",
                                 new_callable=AsyncMock, return_value=[]),
                ):
                    with self.assertRaises(RuntimeError) as caught:
                        await retriever.retrieve(
                            RetrievalRequest(query="q"), cast(AsyncSession, self.session)
                        )
                self.assertIs(caught.exception, failure)

    def test_both_hybrids_reject_nonpositive_candidate_limits(self):
        for factory in (HybridBM25Retriever, HybridTSVectorRetriever):
            with self.subTest(factory=factory.__name__):
                with self.assertRaisesRegex(ValueError, "candidate_top_k"):
                    factory(self.embedder, reranker=self.reranker, candidate_top_k=0)

    def test_default_candidate_limit_matches_rag_config(self):
        self.assertEqual(RAGConfig().hybrid_candidate_top_k, 50)
        self.assertEqual(
            HybridBM25Retriever(self.embedder, reranker=self.reranker).candidate_top_k,
            RAGConfig().hybrid_candidate_top_k,
        )
        self.assertEqual(
            HybridTSVectorRetriever(self.embedder, reranker=self.reranker).candidate_top_k,
            RAGConfig().hybrid_candidate_top_k,
        )


if __name__ == "__main__":
    unittest.main()
