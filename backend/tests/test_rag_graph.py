import json
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import UUID

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from backend.core.config import RAGConfig
from backend.modules.rag.graph import build_rag_graph
from backend.modules.retrieval.contracts import RetrievedPassage


class ScriptedModel(FakeMessagesListChatModel):
    requests: list[dict[str, Any]] = Field(default_factory=list)

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self.bind(
            tools=[convert_to_openai_tool(tool) for tool in tools],
            tool_choice=tool_choice,
            **kwargs,
        )

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append({"messages": list(messages), **kwargs})
        return super()._generate(messages, stop, run_manager, **kwargs)


def tool_request(**overrides):
    call = {"name": "retrieve", "args": {}, "id": "call-1"}
    call.update(overrides)
    return AIMessage(content="", tool_calls=[call])


def build_graph(model, passages=(), *, error=None, config=None):
    retriever = SimpleNamespace(
        name="selected",
        retrieve=AsyncMock(return_value=list(passages), side_effect=error),
    )
    session = object()
    graph = build_rag_graph(
        cast(Any, retriever), cast(Any, session), model, config or RAGConfig()
    )
    return graph, retriever, session


class RAGGraphTests(unittest.IsolatedAsyncioTestCase):
    async def test_retrieval_uses_original_question_not_model_arguments(self):
        question = "  Who wrote Café?\nKeep punctuation!  "
        for args in [
            {},
            {"query": "rewritten search"},
            {"query": " ", "runtime": {"state": {"question": "forged"}}},
        ]:
            with self.subTest(args=args):
                model = ScriptedModel(responses=[
                    tool_request(args=args), AIMessage(content="Answer")
                ])
                graph, retriever, _ = build_graph(model)
                await graph.ainvoke({"question": question})
                retriever.retrieve.assert_awaited_once()
                self.assertEqual(retriever.retrieve.await_args.args[0].query, question)
                schema = model.requests[0]["tools"][0]["function"]["parameters"]
                self.assertEqual(schema.get("properties", {}), {})
                self.assertEqual(schema.get("required", []), [])

    async def test_zero_retrieval_calls_are_allowed(self):
        model = ScriptedModel(responses=[AIMessage(content="Insufficient evidence.")])
        graph, retriever, _ = build_graph(model)
        state = await graph.ainvoke({"question": "Q?"})
        self.assertEqual(state["answer"], "Insufficient evidence.")
        self.assertEqual(state["retrieval_context"], [])
        self.assertEqual(state["retrieved_passages"], [])
        self.assertEqual(state["retrieval_count"], 0)
        retriever.retrieve.assert_not_awaited()
        self.assertEqual(len(model.requests), 1)

    async def test_second_retrieval_is_blocked_and_agent_continues(self):
        model = ScriptedModel(responses=[
            tool_request(),
            tool_request(id="call-2"),
            AIMessage(content="Insufficient evidence after one search."),
        ])
        graph, retriever, _ = build_graph(model)
        state = await graph.ainvoke({"question": "Q?"})
        self.assertEqual(state["answer"], "Insufficient evidence after one search.")
        self.assertEqual(state["retrieval_count"], 1)
        retriever.retrieve.assert_awaited_once()
        blocked = next(
            message for message in state["messages"]
            if isinstance(message, ToolMessage) and message.tool_call_id == "call-2"
        )
        self.assertEqual(blocked.status, "error")
        self.assertIn("Tool call limit exceeded", blocked.content)
        self.assertEqual(len(model.requests), 3)

    async def test_multiple_retrieval_requests_block_excess_calls_and_continue(self):
        calls = tool_request().tool_calls + tool_request(id="call-2").tool_calls
        model = ScriptedModel(responses=[
            AIMessage(content="", tool_calls=calls),
            AIMessage(content="Insufficient evidence after one search."),
        ])
        graph, retriever, _ = build_graph(model)
        state = await graph.ainvoke({"question": "Q?"})
        self.assertEqual(state["answer"], "Insufficient evidence after one search.")
        self.assertEqual(state["retrieval_count"], 1)
        retriever.retrieve.assert_awaited_once()
        blocked = next(
            message for message in state["messages"]
            if isinstance(message, ToolMessage) and message.tool_call_id == "call-2"
        )
        self.assertEqual(blocked.status, "error")
        self.assertEqual(len(model.requests), 2)

    async def test_tool_result_enters_state_before_final_model_and_preserves_context(
        self,
    ):
        passages = [
            RetrievedPassage(
                UUID(int=2), "Second title", "First ranked text", 1, 0.2, "bm25"
            ),
            RetrievedPassage(
                UUID(int=1), "First title", "Second ranked text", 2, 0.1, "bm25"
            ),
        ]
        model = ScriptedModel(
            responses=[tool_request(), AIMessage(content="Grounded answer")]
        )
        graph, retriever, session = build_graph(
            model, passages, config=RAGConfig(top_k=10)
        )
        snapshots = []
        async for state in graph.astream(
            {"question": "Which answer?"}, stream_mode="values"
        ):
            snapshots.append(state)
        state = snapshots[-1]
        context = ["Second title\nFirst ranked text", "First title\nSecond ranked text"]
        before_final = next(
            s for s in snapshots
            if s["messages"] and isinstance(s["messages"][-1], ToolMessage)
        )
        self.assertIsInstance(before_final["messages"][-1], ToolMessage)
        self.assertEqual(before_final["retrieved_passages"], passages)
        self.assertEqual(before_final["retrieval_context"], context)
        self.assertEqual(before_final["messages"][-1].tool_call_id, "call-1")
        self.assertEqual(json.loads(before_final["messages"][-1].content), context)
        self.assertEqual(state["retrieval_context"], context)
        self.assertEqual(state["retrieved_passages"], passages)
        self.assertEqual(state["answer"], "Grounded answer")
        self.assertEqual(state["retrieval_count"], 1)
        retriever.retrieve.assert_awaited_once()
        request, used_session = retriever.retrieve.await_args.args
        self.assertEqual((request.query, request.top_k), ("Which answer?", 10))
        self.assertIs(used_session, session)
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(
            model.requests[1]["messages"][-1], before_final["messages"][-1]
        )
        self.assertNotIn("gold_answer", state)

    async def test_empty_context_still_consumes_retrieval(self):
        model = ScriptedModel(
            responses=[tool_request(), AIMessage(content="Insufficient evidence.")]
        )
        graph, retriever, _ = build_graph(model)
        state = await graph.ainvoke({"question": "Q?"})
        self.assertEqual(state["retrieval_context"], [])
        self.assertEqual(state["retrieved_passages"], [])
        self.assertEqual(state["retrieval_count"], 1)
        self.assertEqual(json.loads(model.requests[1]["messages"][-1].content), [])
        retriever.retrieve.assert_awaited_once()

    async def test_retrieval_failure_does_not_generate_or_retry(self):
        model = ScriptedModel(responses=[tool_request(), AIMessage(content="not used")])
        for error in [RuntimeError("retrieval failed"), ValueError("retrieval failed")]:
            with self.subTest(error=type(error).__name__):
                model.requests.clear()
                model.i = 0
                graph, retriever, _ = build_graph(model, error=error)
                with self.assertRaisesRegex(type(error), "retrieval failed"):
                    await graph.ainvoke({"question": "Q?"})
                retriever.retrieve.assert_awaited_once()
                self.assertEqual(len(model.requests), 1)

    async def test_invalid_final_answer_fails(self):
        for content in [
            "",
            "   ",
            [{"type": "image_url", "image_url": {"url": "unused"}}],
        ]:
            for retrieve_first in [False, True]:
                with self.subTest(content=content, retrieve_first=retrieve_first):
                    model = ScriptedModel(
                        responses=[tool_request(), AIMessage(content=content)]
                        if retrieve_first else [AIMessage(content=content)]
                    )
                    graph, retriever, _ = build_graph(model)
                    with self.assertRaisesRegex(ValueError, "empty or non-text"):
                        await graph.ainvoke({"question": "Q?"})
                    self.assertEqual(retriever.retrieve.await_count, int(retrieve_first))

    async def test_invocations_reset_budget_and_ignore_supplied_stale_state(self):
        model = ScriptedModel(responses=[tool_request(), AIMessage(content="Answer")])
        passage = RetrievedPassage(UUID(int=1), "Title", "Text", 1, 0.1, "bm25")
        graph, retriever, _ = build_graph(model)
        retriever.retrieve.side_effect = [[passage], []]
        first = await graph.ainvoke({"question": "First?"})
        second = await graph.ainvoke({**first, "question": "Second?"})
        self.assertEqual(first["retrieval_context"], ["Title\nText"])
        self.assertEqual(second["retrieval_context"], [])
        self.assertEqual(second["retrieval_count"], 1)
        self.assertEqual(retriever.retrieve.await_count, 2)
        self.assertEqual(
            [call.args[0].query for call in retriever.retrieve.await_args_list],
            ["First?", "Second?"],
        )
        self.assertEqual(len(model.requests), 4)
        self.assertEqual(model.requests[2]["messages"][-1].content, "Second?")
        self.assertFalse(
            any(isinstance(m, ToolMessage) for m in model.requests[2]["messages"])
        )

        model.responses = [AIMessage(content="Insufficient evidence.")]
        model.i = 0
        third = await graph.ainvoke({**first, "question": "Third?"})
        self.assertEqual(third["retrieval_context"], [])
        self.assertEqual(third["retrieved_passages"], [])
        self.assertEqual(third["retrieval_count"], 0)
        self.assertEqual(retriever.retrieve.await_count, 2)

    async def test_invalid_question_does_not_call_model(self):
        for question in ["", " ", None, 42]:
            with self.subTest(question=question):
                model = ScriptedModel(responses=[tool_request()])
                graph, retriever, _ = build_graph(model)
                with self.assertRaises(ValueError):
                    await graph.ainvoke({"question": question})
                self.assertEqual(model.requests, [])
                retriever.retrieve.assert_not_awaited()


class OpenAIModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_model_consumes_tool_result_without_forced_choice(self):
        import httpx2 as httpx
        from openai import AsyncOpenAI

        from backend.modules.rag.generation import build_answer_model

        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertEqual(str(request.url), "https://provider.invalid/v1/responses")
            self.assertEqual(request.headers["authorization"], "Bearer test-key")
            output = (
                [
                    {"type": "reasoning", "id": "rs-1", "summary": []},
                    {
                        "type": "function_call",
                        "id": "fc-1",
                        "call_id": "call-1",
                        "name": "retrieve",
                        "arguments": "{}",
                        "status": "completed",
                    },
                ]
                if len(requests) == 1
                else [
                    {
                        "type": "message",
                        "id": "msg-1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Grounded answer",
                                "annotations": [],
                            }
                        ],
                    },
                ]
            )
            return httpx.Response(
                200,
                json={
                    "id": f"resp-{len(requests)}",
                    "object": "response",
                    "created_at": 1,
                    "model": "answer-model",
                    "status": "completed",
                    "output": output,
                    "parallel_tool_calls": False,
                },
            )

        async with AsyncOpenAI(
            api_key="test-key",
            base_url="https://provider.invalid/v1",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            config = RAGConfig(answer_model="answer-model", top_k=10)
            model = build_answer_model(client, config)
            passage = RetrievedPassage(UUID(int=1), "Title", "Text", 1, 0.1, "bm25")
            graph, retriever, _ = build_graph(model, [passage], config=config)
            state = await graph.ainvoke({"question": "Q?"})
            self.assertFalse(client.is_closed())

        self.assertEqual(state["answer"], "Grounded answer")
        self.assertEqual(state["retrieval_context"], ["Title\nText"])
        retriever.retrieve.assert_awaited_once()
        self.assertEqual(retriever.retrieve.await_args.args[0].query, "Q?")
        self.assertEqual(len(requests), 2)
        first, final = requests
        self.assertEqual(first["tools"][0]["name"], "retrieve")
        self.assertIn(first.get("tool_choice"), (None, "auto"))
        self.assertEqual(first["tools"][0]["parameters"].get("properties", {}), {})
        self.assertEqual(first["tools"][0]["parameters"].get("required", []), [])
        self.assertIn(final.get("tool_choice"), (None, "auto"))
        self.assertTrue(final.get("tools"))
        tool_result = next(
            item for item in final["input"] if item["type"] == "function_call_output"
        )
        self.assertEqual(tool_result["call_id"], "call-1")
        self.assertEqual(json.loads(tool_result["output"]), ["Title\nText"])
        self.assertTrue(any(item.get("type") == "reasoning" for item in final["input"]))
        for request in requests:
            self.assertEqual(request["model"], "answer-model")
            self.assertNotIn("temperature", request)


if __name__ == "__main__":
    unittest.main()
