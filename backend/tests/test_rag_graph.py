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


def tool_request(query="rewritten search", **overrides):
    call = {"name": "retrieve", "args": {"query": query}, "id": "call-1"}
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
        before_final = next(s for s in snapshots if s.get("retrieval_count") == 1)
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
        self.assertEqual((request.query, request.top_k), ("rewritten search", 10))
        self.assertIs(used_session, session)
        self.assertEqual(len(model.requests), 2)
        self.assertEqual(model.requests[0]["tool_choice"], "retrieve")
        self.assertFalse(model.requests[0]["parallel_tool_calls"])
        self.assertEqual(model.requests[1].get("tools", []), [])
        self.assertEqual(model.requests[1]["tool_choice"], "none")
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

    async def test_invalid_initial_calls_do_not_retrieve(self):
        invalid = [
            AIMessage(content="Skipping retrieval"),
            tool_request(name="unknown"),
            tool_request(id=""),
            tool_request(id=" "),
            tool_request(id=None),
            tool_request(query=" "),
            tool_request(args={}),
            tool_request(args={"query": 123}),
            tool_request(args={"query": "Q", "top_k": 100}),
            AIMessage(content="", tool_calls=tool_request().tool_calls * 2),
            AIMessage(
                content="",
                invalid_tool_calls=[
                    {
                        "name": "retrieve",
                        "args": "{",
                        "id": "bad",
                        "error": "invalid JSON",
                    }
                ],
            ),
        ]
        for response in invalid:
            with self.subTest(response=response):
                model = ScriptedModel(responses=[response])
                graph, retriever, _ = build_graph(model)
                with self.assertRaises(ValueError):
                    await graph.ainvoke({"question": "Q?"})
                retriever.retrieve.assert_not_awaited()
                self.assertEqual(len(model.requests), 1)

    async def test_second_tool_call_is_rejected_even_if_model_ignores_disabled_tools(
        self,
    ):
        model = ScriptedModel(
            responses=[tool_request(), tool_request("second search", id="call-2")]
        )
        graph, retriever, _ = build_graph(model)
        with self.assertRaisesRegex(ValueError, "retrieval|tool"):
            await graph.ainvoke({"question": "Q?"})
        retriever.retrieve.assert_awaited_once()
        self.assertEqual(len(model.requests), 2)

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
            with self.subTest(content=content):
                model = ScriptedModel(
                    responses=[tool_request(), AIMessage(content=content)]
                )
                graph, retriever, _ = build_graph(model)
                with self.assertRaisesRegex(ValueError, "empty or non-text"):
                    await graph.ainvoke({"question": "Q?"})
                retriever.retrieve.assert_awaited_once()

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
        self.assertEqual(len(model.requests), 4)
        self.assertEqual(model.requests[2]["messages"][-1].content, "Second?")
        self.assertFalse(
            any(isinstance(m, ToolMessage) for m in model.requests[2]["messages"])
        )

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
    async def test_real_model_sends_required_tool_and_consumes_state_result(self):
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
                        "arguments": '{"query":"provider search"}',
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
        self.assertEqual(retriever.retrieve.await_args.args[0].query, "provider search")
        self.assertEqual(len(requests), 2)
        first, final = requests
        self.assertEqual(first["tool_choice"], {"type": "function", "name": "retrieve"})
        self.assertFalse(first["parallel_tool_calls"])
        self.assertEqual(set(first["tools"][0]["parameters"]["properties"]), {"query"})
        self.assertEqual(final["tool_choice"], "none")
        self.assertFalse(final.get("tools"))
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
