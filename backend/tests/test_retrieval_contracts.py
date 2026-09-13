import unittest

from backend.modules.retrieval.contracts import RetrievalRequest


class RetrievalRequestTests(unittest.TestCase):
    def test_accepts_nonblank_query_and_default_limit(self):
        request = RetrievalRequest(query="distributed systems")

        self.assertEqual(request.query, "distributed systems")
        self.assertEqual(request.top_k, 10)

    def test_rejects_blank_query(self):
        for query in ("", "   ", "\t\n"):
            with self.subTest(query=query):
                with self.assertRaisesRegex(ValueError, "query must not be blank"):
                    RetrievalRequest(query=query)

    def test_rejects_non_positive_top_k(self):
        for top_k in (0, -1):
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(
                    ValueError,
                    "top_k must be greater than zero",
                ):
                    RetrievalRequest(query="distributed systems", top_k=top_k)
