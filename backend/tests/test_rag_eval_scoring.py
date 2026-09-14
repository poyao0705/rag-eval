"""Offline integration coverage for the stock DeepEval/OpenAI judge adapter."""

import json
import unittest
from typing import Any, cast

import httpx
from deepeval.models import OpenAIModel
from deepeval.test_case import LLMTestCase
from openai import (
    AuthenticationError,
    BadRequestError,
    LengthFinishReasonError,
    NotFoundError,
)
from pydantic import ValidationError

from rag_eval.scoring import (
    JudgeProbe,
    build_judge,
    build_metrics,
    probe_judge,
    score_case,
)


class JudgeAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        self.responses = []
        transport = httpx.MockTransport(self._respond)
        self.sync_client = httpx.Client(transport=transport)
        self.async_client = httpx.AsyncClient(transport=transport)
        # Keep construction, DeepEval generation, and OpenAI schema parsing real.
        # Only HTTP is replaced; no credentials or network are used.
        self.judge = build_judge(
            api_key="offline-placeholder", base_url="https://judge.invalid/v1"
        )
        self.judge.kwargs["http_client"] = self.sync_client
        self.judge.async_http_client = self.async_client

    async def asyncTearDown(self):
        cast(Any, self.judge.model).close()
        self.sync_client.close()
        await self.async_client.aclose()

    def _respond(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("unexpected judge request (possible fallback)")
        status, body = self.responses.pop(0)
        return httpx.Response(status, json=body)

    def _completion(self, content, *, refusal=None, finish_reason="stop"):
        self.responses.append((200, {
            "id": "chatcmpl-offline",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-5.4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant", "content": content, "refusal": refusal,
                },
                "finish_reason": finish_reason,
                "logprobs": None,
            }],
            "usage": {
                "prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20,
            },
        }))

    def _assert_native_requests(self, schema_names):
        self.assertEqual(len(self.requests), len(schema_names))
        for request, name in zip(self.requests, schema_names, strict=True):
            body = json.loads(request.content)
            self.assertEqual(str(request.url), "https://judge.invalid/v1/chat/completions")
            self.assertEqual(body["model"], "gpt-5.4")
            self.assertEqual(body["response_format"]["type"], "json_schema")
            self.assertEqual(body["response_format"]["json_schema"]["name"], name)
            self.assertIs(body["response_format"]["json_schema"]["strict"], True)
            # Stock DeepEval adjusts the temperature for gpt-5.4.
            self.assertEqual(body["temperature"], 1)

    async def test_build_judge_probe_uses_supported_stock_model_and_native_parse(self):
        self._completion('{"ok": true}')

        await probe_judge(self.judge)

        self.assertIs(type(self.judge), OpenAIModel)
        self._assert_native_requests(["JudgeProbe"])

    async def test_metric_schemas_use_native_parse(self):
        self._completion('{"statements": ["Paris is in France."]}')
        self._completion('{"verdicts": [{"verdict": "yes", "reason": null}]}')
        self._completion('{"reason": "The statement answers the question."}')
        metrics = build_metrics(self.judge)
        for _name, metric in metrics:
            self.assertIs(metric.model, self.judge)
            self.assertTrue(metric.using_native_model)

        result = await score_case(
            LLMTestCase(input="Where is Paris?", actual_output="Paris is in France."),
            metrics[:1],
        )

        self.assertEqual(result, [{
            "name": "answer_relevancy", "score": 1.0,
            "reason": "The statement answers the question.", "error": None,
        }])
        self._assert_native_requests(["Statements", "Verdicts", "AnswerRelevancyScoreReason"])

    def test_sync_schema_generation_also_uses_native_parse(self):
        self._completion('{"ok": true}')

        parsed, _cost = self.judge.generate_with_schema("Return ok true", schema=JudgeProbe)

        assert isinstance(parsed, JudgeProbe)
        self.assertIs(parsed.ok, True)
        self._assert_native_requests(["JudgeProbe"])

    async def test_unverified_capability_blocks_probe_before_http(self):
        for capability in (None, False):
            with self.subTest(capability=capability):
                self.judge.model_data.supports_structured_outputs = capability
                with self.assertRaisesRegex(ValueError, "native structured"):
                    await probe_judge(self.judge)
        self.assertEqual(self.requests, [])

    async def test_probe_rejects_false_malformed_missing_and_refused_responses(self):
        cases = [
            ('{"ok": false}', None, "stop", ValueError),
            ('{"ok":', None, "stop", ValidationError),
            ('{}', None, "stop", ValidationError),
            ('{"ok": "not-a-boolean"}', None, "stop", ValidationError),
            (None, "Cannot comply", "stop", ValueError),
            (None, None, "stop", ValueError),
            ('{"ok": true}', None, "length", LengthFinishReasonError),
        ]
        for content, refusal, finish_reason, error in cases:
            with self.subTest(content=content, refusal=refusal, finish_reason=finish_reason):
                self.requests.clear()
                self._completion(content, refusal=refusal, finish_reason=finish_reason)
                with self.assertRaises(error):
                    await probe_judge(self.judge)
                self._assert_native_requests(["JudgeProbe"])

    async def test_metric_rejects_invalid_schema_without_another_request(self):
        self._completion('{"statements": 7}')

        result = await score_case(
            LLMTestCase(input="Q", actual_output="A"), build_metrics(self.judge)[:1]
        )

        self.assertIsNone(result[0]["score"])
        self.assertIsNone(result[0]["reason"])
        self.assertEqual(result[0]["error"], {"type": "ValidationError"})
        self._assert_native_requests(["Statements"])

    async def test_provider_errors_propagate_and_metric_reporting_redacts_details(self):
        for status, error in [(400, BadRequestError), (401, AuthenticationError), (404, NotFoundError)]:
            with self.subTest(status=status):
                self.requests.clear()
                self.responses.append((status, {"error": {
                    "message": "secret-provider-details", "type": "invalid_request_error",
                    "param": None, "code": None,
                }}))
                with self.assertRaises(error):
                    await probe_judge(self.judge)
                self._assert_native_requests(["JudgeProbe"])
        self.requests.clear()
        self.responses.append((401, {"error": {
            "message": "secret-provider-details", "type": "invalid_request_error",
            "param": None, "code": None,
        }}))
        result = await score_case(
            LLMTestCase(input="Q", actual_output="A"), build_metrics(self.judge)[:1]
        )
        self.assertEqual(result[0]["error"], {"type": "AuthenticationError"})
        self.assertNotIn("secret-provider-details", json.dumps(result))
        self._assert_native_requests(["Statements"])
