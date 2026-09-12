import json
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import UUID

from backend.modules.rag.generation import OpenAIAnswerGenerator
from backend.modules.rag.graph import build_rag_graph
from backend.modules.retrieval.contracts import RetrievedPassage


def _build_graph(retriever: object, session: object, generator: object) -> Any:
    return build_rag_graph(
        cast(Any, retriever), cast(Any, session), cast(Any, generator)
    )


class RAGGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieve_once_and_preserve_context(self):
        passages = [
            RetrievedPassage(UUID(int=2), "Second title", "First ranked text", 1, 0.2, "bm25"),
            RetrievedPassage(UUID(int=1), "First title", "Second ranked text", 2, 0.1, "bm25"),
        ]
        retriever = SimpleNamespace(name="bm25", retrieve=AsyncMock(return_value=passages))
        generator = SimpleNamespace(answer=AsyncMock(return_value="Grounded answer"))
        session = object()
        graph = _build_graph(retriever, session, generator)

        state = await graph.ainvoke({"question": "Which answer?"})

        request, used_session = retriever.retrieve.await_args.args
        self.assertEqual((request.query, request.top_k), ("Which answer?", 5))
        self.assertIs(used_session, session)
        retriever.retrieve.assert_awaited_once()
        generator.answer.assert_awaited_once_with(
            "Which answer?",
            ["Second title\nFirst ranked text", "First title\nSecond ranked text"],
        )
        self.assertEqual(state["retrieved_passages"], passages)
        self.assertEqual(state["retrieval_context"], generator.answer.await_args.args[1])
        self.assertEqual(state["answer"], "Grounded answer")
        self.assertNotIn("expected_output", state)
        self.assertNotIn("gold_answer", state)

    async def test_empty_context_is_not_replaced(self):
        retriever = SimpleNamespace(name="vector", retrieve=AsyncMock(return_value=[]))
        generator = SimpleNamespace(answer=AsyncMock(return_value="Insufficient evidence."))

        state = await _build_graph(retriever, object(), generator).ainvoke({"question": "Q?"})

        self.assertEqual(state["retrieval_context"], [])
        generator.answer.assert_awaited_once_with("Q?", [])

    async def test_invocations_do_not_retain_previous_context(self):
        passage = RetrievedPassage(UUID(int=1), "Title", "Text", 1, 0.1, "bm25")
        retriever = SimpleNamespace(
            name="bm25", retrieve=AsyncMock(side_effect=[[passage], []])
        )
        generator = SimpleNamespace(answer=AsyncMock(side_effect=["First", "Second"]))
        graph = _build_graph(retriever, object(), generator)

        first = await graph.ainvoke({"question": "First question"})
        second = await graph.ainvoke({"question": "Second question"})

        self.assertEqual(first["retrieval_context"], ["Title\nText"])
        self.assertEqual(second["retrieval_context"], [])
        generator.answer.assert_any_await("First question", ["Title\nText"])
        generator.answer.assert_awaited_with("Second question", [])
        self.assertEqual(retriever.retrieve.await_count, 2)

    async def test_retrieval_failure_does_not_generate(self):
        retriever = SimpleNamespace(
            name="bm25", retrieve=AsyncMock(side_effect=RuntimeError("retrieval failed"))
        )
        generator = SimpleNamespace(answer=AsyncMock(return_value="not used"))
        graph = _build_graph(retriever, object(), generator)

        with self.assertRaisesRegex(RuntimeError, "retrieval failed"):
            await graph.ainvoke({"question": "Q?"})

        generator.answer.assert_not_awaited()

    async def test_empty_generator_answer_is_an_error(self):
        retriever = SimpleNamespace(name="bm25", retrieve=AsyncMock(return_value=[]))
        generator = SimpleNamespace(answer=AsyncMock(return_value=""))
        graph = _build_graph(retriever, object(), generator)

        with self.assertRaisesRegex(ValueError, "empty or non-text"):
            await graph.ainvoke({"question": "Q?"})

    async def test_compiled_graph_only_invokes_its_retriever(self):
        selected = SimpleNamespace(name="selected", retrieve=AsyncMock(return_value=[]))
        unused_one = SimpleNamespace(name="unused-one", retrieve=AsyncMock(return_value=[]))
        unused_two = SimpleNamespace(name="unused-two", retrieve=AsyncMock(return_value=[]))
        generator = SimpleNamespace(answer=AsyncMock(return_value="Answer"))
        graph = _build_graph(selected, object(), generator)

        await graph.ainvoke({"question": "Q?"})

        selected.retrieve.assert_awaited_once()
        unused_one.retrieve.assert_not_awaited()
        unused_two.retrieve.assert_not_awaited()


class OpenAIAnswerGeneratorTests(unittest.IsolatedAsyncioTestCase):
    async def test_uses_fixed_model_and_json_input(self):
        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=AsyncMock(return_value=SimpleNamespace(output_text="The answer"))
            )
        )
        generator = OpenAIAnswerGenerator(client)

        self.assertEqual(await generator.answer("Q", ["T\nD"]), "The answer")
        client.responses.create.assert_awaited_once()
        kwargs = client.responses.create.await_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-5-mini")
        self.assertNotIn("temperature", kwargs)
        self.assertEqual(
            json.loads(kwargs["input"]), {"question": "Q", "documents": ["T\nD"]}
        )


if __name__ == "__main__":
    unittest.main()
