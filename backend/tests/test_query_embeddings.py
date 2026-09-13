from types import SimpleNamespace
import unittest

from backend.modules.retrieval.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    OpenAIQueryEmbedder,
    validated_embeddings,
)


class FakeEmbeddings:
    def __init__(self):
        self.requests = []

    async def create(self, *, model, input):
        self.requests.append({"model": model, "input": input})
        return SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=0,
                    embedding=[0.25] * EMBEDDING_DIMENSIONS,
                )
            ]
        )


class FakeClient:
    def __init__(self):
        self.embeddings = FakeEmbeddings()


class EmbeddingValidationTests(unittest.TestCase):
    def test_restores_embedding_response_order(self):
        response = SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=1,
                    embedding=[1.0] * EMBEDDING_DIMENSIONS,
                ),
                SimpleNamespace(
                    index=0,
                    embedding=[0.0] * EMBEDDING_DIMENSIONS,
                ),
            ]
        )

        embeddings = validated_embeddings(response, expected_count=2)

        self.assertEqual(embeddings[0][0], 0.0)
        self.assertEqual(embeddings[1][0], 1.0)

    def test_rejects_missing_data(self):
        with self.assertRaisesRegex(ValueError, "response must include data"):
            validated_embeddings(SimpleNamespace(), expected_count=1)

    def test_rejects_wrong_response_count(self):
        response = SimpleNamespace(data=[])

        with self.assertRaisesRegex(ValueError, "response count"):
            validated_embeddings(response, expected_count=1)

    def test_rejects_non_contiguous_indexes(self):
        response = SimpleNamespace(
            data=[
                SimpleNamespace(
                    index=0,
                    embedding=[0.0] * EMBEDDING_DIMENSIONS,
                ),
                SimpleNamespace(
                    index=2,
                    embedding=[2.0] * EMBEDDING_DIMENSIONS,
                ),
            ]
        )

        with self.assertRaisesRegex(ValueError, "indexes are not contiguous"):
            validated_embeddings(response, expected_count=2)

    def test_rejects_wrong_dimension(self):
        response = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[0.0] * 1535)]
        )

        with self.assertRaisesRegex(ValueError, "1536"):
            validated_embeddings(response, expected_count=1)


class OpenAIQueryEmbedderTests(unittest.IsolatedAsyncioTestCase):
    async def test_embeds_one_query_with_shared_model(self):
        client = FakeClient()
        embedder = OpenAIQueryEmbedder(client)

        embedding = await embedder.embed_query("distributed systems")

        self.assertEqual(embedding, [0.25] * EMBEDDING_DIMENSIONS)
        self.assertEqual(
            client.embeddings.requests,
            [
                {
                    "model": EMBEDDING_MODEL,
                    "input": ["distributed systems"],
                }
            ],
        )
