import json
import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from rag_eval.cohort import QAExample, load_cohort
from rag_eval.scoring import (
    JudgeProbe,
    build_judge,
    build_metrics,
    build_test_case,
    probe_judge,
    score_case,
)


class CohortLoaderTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _session(rows):
        result = SimpleNamespace(
            mappings=lambda: SimpleNamespace(all=lambda: rows)
        )
        return SimpleNamespace(execute=AsyncMock(return_value=result))

    async def test_cohort_query_is_fixed_and_labeled(self):
        rows = [
            dict(id=f"qa-{i}", question=f"Question {i}?", answer=f"Answer {i}")
            for i in range(20)
        ]
        session = self._session(rows)

        cohort = await load_cohort(cast(Any, session))

        statement, params = session.execute.await_args.args
        sql = str(statement)
        self.assertIn("q.split = 'VALIDATION'", sql)
        self.assertIn(
            "ORDER BY md5(CAST(:seed AS text) || ':' || q.id), q.id", sql
        )
        self.assertNotIn("random()", sql.lower())
        self.assertEqual(params, {"seed": 42, "sample_size": 20})
        self.assertEqual(cohort[0].answer, "Answer 0")
        self.assertEqual(len(cohort), 20)

    async def test_rejects_incomplete_duplicate_and_blank_cohorts(self):
        valid_rows = [
            dict(id=f"qa-{i}", question=f"Question {i}?", answer=f"Answer {i}")
            for i in range(20)
        ]
        cases = [
            valid_rows[:19],
            [*valid_rows[:19], {**valid_rows[19], "id": valid_rows[0]["id"]}],
            [*valid_rows[:19], {**valid_rows[19], "answer": "   "}],
        ]

        for rows in cases:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    await load_cohort(cast(Any, self._session(rows)))

    async def test_loader_is_repeatable_and_preserves_answer_whitespace(self):
        rows = [
            dict(id=f"qa-{i}", question=f"Question {i}?", answer=f"Answer {i}")
            for i in range(20)
        ]
        rows[0]["answer"] = "  Answer 0  "
        session = self._session(rows)

        first = await load_cohort(cast(Any, session))
        second = await load_cohort(cast(Any, session))

        self.assertEqual(first, second)
        self.assertEqual(first[0].answer, "  Answer 0  ")
        self.assertIsInstance(first[0], QAExample)
        self.assertEqual(session.execute.await_count, 2)

    def test_qa_example_is_immutable(self):
        example = QAExample("qa-1", "Question?", "Answer")
        with self.assertRaises((AttributeError, TypeError)):
            setattr(example, "answer", "Changed")


class ScoringTests(unittest.IsolatedAsyncioTestCase):
    def test_build_test_case_keeps_gold_out_of_graph_state(self):
        qa = QAExample("qa-1", "Who?", "GOLD_SENTINEL")
        state = {
            "question": "Who?",
            "answer": "Actual",
            "retrieval_context": ["T\nD"],
        }

        case = build_test_case(qa, cast(Any, state))

        self.assertEqual(case.input, "Who?")
        self.assertEqual(case.actual_output, "Actual")
        self.assertEqual(case.expected_output, "GOLD_SENTINEL")
        self.assertEqual(case.retrieval_context, ["T\nD"])
        self.assertNotIn("GOLD_SENTINEL", json.dumps(state))

    def test_build_metrics_uses_five_fresh_explicitly_configured_metrics(self):
        judge = object()
        with (
            patch("rag_eval.scoring.AnswerRelevancyMetric") as answer_relevancy,
            patch("rag_eval.scoring.FaithfulnessMetric") as faithfulness,
            patch("rag_eval.scoring.ContextualPrecisionMetric") as precision,
            patch("rag_eval.scoring.ContextualRecallMetric") as recall,
            patch("rag_eval.scoring.ContextualRelevancyMetric") as relevancy,
        ):
            first = build_metrics(judge)
            second = build_metrics(judge)

        self.assertEqual(
            [name for name, _metric in first],
            [
                "answer_relevancy",
                "faithfulness",
                "contextual_precision",
                "contextual_recall",
                "contextual_relevancy",
            ],
        )
        for constructor in (
            answer_relevancy,
            faithfulness,
            precision,
            recall,
            relevancy,
        ):
            self.assertEqual(constructor.call_count, 2)
            constructor.assert_any_call(
                model=judge, include_reason=True, threshold=0.5
            )

    async def test_low_score_and_metric_error_are_distinct(self):
        low = SimpleNamespace(
            a_measure=AsyncMock(), score=0.1, reason="Low relevance"
        )
        broken = SimpleNamespace(
            a_measure=AsyncMock(side_effect=RuntimeError("secret"))
        )

        results = await score_case(
            cast(Any, object()),
            cast(Any, [("answer_relevancy", low), ("faithfulness", broken)]),
        )

        self.assertEqual(results[0]["score"], 0.1)
        self.assertIsNone(results[0]["error"])
        self.assertIsNone(results[1]["score"])
        self.assertEqual(results[1]["error"], {"type": "RuntimeError"})
        self.assertNotIn("secret", json.dumps(results))
        self.assertTrue(all(set(result) == {"name", "score", "reason", "error"}
                            for result in results))

    async def test_invalid_scores_and_empty_context_are_explicit_errors(self):
        invalid = SimpleNamespace(a_measure=AsyncMock(), score=float("nan"))
        empty_context_failure = SimpleNamespace(
            a_measure=AsyncMock(side_effect=ValueError("empty context"))
        )

        results = await score_case(
            cast(Any, object()),
            cast(
                Any,
                [
                    ("answer_relevancy", invalid),
                    ("faithfulness", empty_context_failure),
                ],
            ),
        )

        self.assertEqual(results[0]["error"], {"type": "InvalidMetricScore"})
        self.assertEqual(results[1]["error"], {"type": "ValueError"})

        qa = QAExample("qa-1", "Q", "A")
        case = build_test_case(
            qa, {"question": "Q", "answer": "Actual", "retrieval_context": []}
        )
        self.assertEqual(case.retrieval_context, [])

    def test_build_judge_uses_only_the_explicit_model(self):
        with patch("deepeval.models.OpenAIModel") as constructor:
            judge = build_judge(api_key="key", base_url="https://provider")

        self.assertIs(judge, constructor.return_value)
        constructor.assert_called_once_with(
            model="gpt-5.6-luna", api_key="key", base_url="https://provider"
        )

    async def test_probe_requires_a_parsed_true_response(self):
        good_judge = SimpleNamespace(
            a_generate=AsyncMock(return_value=(JudgeProbe(ok=True), None))
        )
        await probe_judge(good_judge)
        good_judge.a_generate.assert_awaited_once()
        self.assertIs(
            good_judge.a_generate.await_args.kwargs["schema"], JudgeProbe
        )

        bad_judge = SimpleNamespace(
            a_generate=AsyncMock(return_value=(JudgeProbe(ok=False), None))
        )
        with self.assertRaises(ValueError):
            await probe_judge(bad_judge)

        broken_judge = SimpleNamespace(
            a_generate=AsyncMock(side_effect=RuntimeError("provider failure"))
        )
        with self.assertRaisesRegex(RuntimeError, "provider failure"):
            await probe_judge(broken_judge)


if __name__ == "__main__":
    unittest.main()
