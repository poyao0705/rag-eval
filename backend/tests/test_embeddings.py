from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO
from types import SimpleNamespace
import unittest
import uuid

from backend.scripts.embeddings import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    embed_passages,
)


@dataclass
class PassageStub:
    id: uuid.UUID
    text: str


class FakeScalarResult:
    def __init__(self, rows):
        self.rows = rows

    def all(self):
        return self.rows


class FakeReadSession:
    def __init__(self, factory):
        self.factory = factory
        self.closed = False

    async def __aenter__(self):
        self.factory.read_sessions.append(self)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        self.closed = True

    async def scalars(self, statement):
        self.factory.read_statements.append(statement)
        return FakeScalarResult(self.factory.read_batches.pop(0))


class FakeWriteSession:
    def __init__(self, factory):
        self.factory = factory

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    async def execute(self, statement):
        self.factory.write_statements.append(statement)


class FakeTransaction:
    def __init__(self, factory):
        self.factory = factory

    async def __aenter__(self):
        self.factory.write_sessions += 1
        return FakeWriteSession(self.factory)

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class FakeSessionFactory:
    def __init__(self, read_batches):
        self.read_batches = list(read_batches)
        self.read_sessions = []
        self.read_statements = []
        self.write_sessions = 0
        self.write_statements = []

    def __call__(self):
        return FakeReadSession(self)

    def begin(self):
        return FakeTransaction(self)


class FakeEmbeddings:
    def __init__(self, factory):
        self.factory = factory
        self.requests = []

    async def create(self, *, model, input):
        self.requests.append(
            {
                "model": model,
                "input": input,
                "read_session_closed": self.factory.read_sessions[-1].closed,
            }
        )
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index)] * EMBEDDING_DIMENSIONS)
                for index in reversed(range(len(input)))
            ]
        )


class FakeClient:
    def __init__(self, factory):
        self.embeddings = FakeEmbeddings(factory)


class EmbeddingPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_embed_passages_sends_texts_and_updates_vectors(self):
        first = PassageStub(uuid.uuid4(), "first passage")
        second = PassageStub(uuid.uuid4(), "second passage")
        factory = FakeSessionFactory([[first, second], []])
        client = FakeClient(factory)

        output = StringIO()
        with redirect_stdout(output):
            stats = await embed_passages(
                client, session_factory=factory, batch_size=2
            )

        self.assertIn("embedding model=text-embedding-3-small batch_size=2", output.getvalue())
        self.assertIn(
            "batch=1 selected=2 total_embedded=0 status=requesting",
            output.getvalue(),
        )
        self.assertIn(
            "batch=1 selected=2 total_embedded=2 status=committed",
            output.getvalue(),
        )
        self.assertIn("embedding complete selected=2 embedded=2", output.getvalue())
        self.assertEqual(stats.selected_passages, 2)
        self.assertEqual(stats.embedded_passages, 2)
        self.assertEqual(
            client.embeddings.requests,
            [
                {
                    "model": EMBEDDING_MODEL,
                    "input": ["first passage", "second passage"],
                    "read_session_closed": True,
                }
            ],
        )
        self.assertEqual(factory.write_sessions, 1)
        self.assertEqual(len(factory.write_statements), 2)
        params = [statement.compile().params for statement in factory.write_statements]
        self.assertIn(first.id, params[0].values())
        self.assertIn(second.id, params[1].values())
        self.assertIn([0.0] * EMBEDDING_DIMENSIONS, params[0].values())
        self.assertIn([1.0] * EMBEDDING_DIMENSIONS, params[1].values())

    async def test_wrong_response_dimension_does_not_start_write_transaction(self):
        passage = PassageStub(uuid.uuid4(), "passage")
        factory = FakeSessionFactory([[passage]])

        class WrongDimensionClient:
            class Embeddings:
                async def create(self, *, model, input):
                    return SimpleNamespace(
                        data=[SimpleNamespace(index=0, embedding=[0.0] * 1535)]
                    )

            embeddings = Embeddings()

        error_output = StringIO()
        with redirect_stderr(error_output):
            with self.assertRaisesRegex(ValueError, "1536"):
                await embed_passages(
                    WrongDimensionClient(), session_factory=factory, batch_size=1
                )

        self.assertIn(
            "batch=1 selected=1 total_embedded=0 status=failed current_batch_not_committed",
            error_output.getvalue(),
        )
        self.assertEqual(factory.write_sessions, 0)
        self.assertEqual(factory.write_statements, [])

    async def test_embed_passages_processes_each_committed_batch_then_resumes(self):
        first = PassageStub(uuid.uuid4(), "first passage")
        second = PassageStub(uuid.uuid4(), "second passage")
        factory = FakeSessionFactory([[first], [second], []])
        client = FakeClient(factory)

        stats = await embed_passages(client, session_factory=factory, batch_size=1)

        self.assertEqual(stats.selected_passages, 2)
        self.assertEqual(stats.embedded_passages, 2)
        self.assertEqual(len(client.embeddings.requests), 2)
        self.assertEqual(factory.write_sessions, 2)
