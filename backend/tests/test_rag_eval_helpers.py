import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch
from uuid import UUID

from rag_eval.cohort import QAExample, load_cohort
from rag_eval.report import case_record, new_report, summarize, write_report
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


class ReportTests(unittest.TestCase):
    def test_summarize_keeps_failed_measurements_out_of_the_mean(self):
        cases = [
            {
                "retriever": "bm25",
                "metrics": [
                    {
                        "name": "answer_relevancy",
                        "score": 0.2,
                        "error": None,
                    }
                ],
                "error": None,
            },
            {
                "retriever": "bm25",
                "metrics": [
                    {
                        "name": "answer_relevancy",
                        "score": None,
                        "error": {"type": "TimeoutError"},
                    }
                ],
                "error": None,
            },
        ]

        summary = summarize(cases)

        self.assertEqual(
            summary["bm25"]["answer_relevancy"],
            {"mean": 0.2, "scored_count": 1, "error_count": 1},
        )
        failures_only = summarize([cases[1]])
        self.assertEqual(
            failures_only["bm25"]["answer_relevancy"],
            {"mean": None, "scored_count": 0, "error_count": 1},
        )
        self.assertEqual(
            {
                summary["attempted_case_count"],
                summary["completed_case_count"],
                summary["failed_case_count"],
                summary["expected_case_count"],
            },
            {2, 2, 0, 60},
        )

    def test_write_report_round_trips_safe_partial_case_records(self):
        qa = QAExample("qa-1", "Who?", "GOLD_SENTINEL")
        passage = SimpleNamespace(passage_id=UUID(int=7))
        state = {
            "answer": "Actual",
            "retrieved_passages": [passage],
            "retrieval_context": ["First", "Second"],
            "client": "CREDENTIAL_SENTINEL",
        }
        record = case_record(
            qa,
            "bm25",
            cast(Any, state),
            [
                {
                    "name": "answer_relevancy",
                    "score": 0.2,
                    "reason": "Low",
                    "error": None,
                    "client": "CREDENTIAL_SENTINEL",
                },
                {
                    "name": "faithfulness",
                    "score": None,
                    "reason": None,
                    "error": {"type": "TimeoutError"},
                },
            ],
        )
        report = new_report([qa])
        report["cases"].append(record)

        with TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "results.json"
            write_report(path, report)
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["seed"], 42)
        self.assertEqual(saved["generator_model"], "gpt-5-mini")
        self.assertEqual(saved["judge_model"], "gpt-5.6-luna")
        self.assertEqual(saved["top_k"], 5)
        self.assertEqual(saved["qa_ids"], ["qa-1"])
        self.assertEqual(saved["cases"][0]["passage_ids"], [str(UUID(int=7))])
        self.assertEqual(saved["cases"][0]["retrieval_context"], ["First", "Second"])
        self.assertEqual(saved["cases"][0]["metrics"][1]["error"], {"type": "TimeoutError"})
        self.assertNotIn("CREDENTIAL_SENTINEL", json.dumps(saved))
        self.assertEqual(saved["summary"]["attempted_case_count"], 1)
        self.assertEqual(saved["summary"]["completed_case_count"], 1)


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
