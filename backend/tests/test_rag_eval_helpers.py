import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, call, patch
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


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_cases_uses_fixed_order_and_question_only_inputs(self):
        from test_rag import run_cases

        cohort = [QAExample(f"qa-{i}", f"Question {i}?", f"Answer {i}") for i in range(20)]
        graphs = {
            name: SimpleNamespace(
                ainvoke=AsyncMock(
                    return_value={
                        "answer": "A",
                        "retrieval_context": [],
                        "retrieved_passages": [],
                    }
                )
            )
            for name in ("bm25", "tsvector", "vector")
        }
        scores = [
            {"name": name, "score": 0.1, "reason": "low", "error": None}
            for name in (
                "answer_relevancy",
                "faithfulness",
                "contextual_precision",
                "contextual_recall",
                "contextual_relevancy",
            )
        ]
        with TemporaryDirectory() as directory:
            with (
                patch("test_rag.build_test_case", return_value=object()),
                patch("test_rag.build_metrics", return_value=[]),
                patch("test_rag.score_case", new=AsyncMock(return_value=scores)),
            ):
                report = await run_cases(
                    cohort,
                    graphs,
                    object(),
                    Path(directory) / "results.json",
                )

        expected_calls = [call({"question": qa.question}) for qa in cohort]
        for graph in graphs.values():
            self.assertEqual(graph.ainvoke.await_args_list, expected_calls)
        self.assertEqual(len(report["cases"]), 60)
        self.assertEqual(report["qa_ids"], [qa.id for qa in cohort])
        self.assertEqual(
            [case["qa_id"] for case in report["cases"][:20]],
            [qa.id for qa in cohort],
        )
        self.assertTrue(all(len(case["metrics"]) == 5 for case in report["cases"]))
        self.assertTrue(all("Answer" not in str(graph.ainvoke.call_args) for graph in graphs.values()))

    async def test_run_cases_persists_partial_graph_failure(self):
        from test_rag import run_cases

        cohort = [QAExample("qa-1", "Question?", "Answer")]
        failed = SimpleNamespace(ainvoke=AsyncMock(side_effect=RuntimeError("secret")))
        untouched = SimpleNamespace(ainvoke=AsyncMock())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            with self.assertRaisesRegex(RuntimeError, "bm25/qa-1"):
                await run_cases(
                    cohort,
                    {"bm25": failed, "tsvector": untouched, "vector": untouched},
                    object(),
                    path,
                )
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(len(saved["cases"]), 1)
        self.assertEqual(saved["cases"][0]["error"], {"type": "RuntimeError"})
        self.assertEqual(untouched.ainvoke.await_count, 0)
        self.assertNotIn("secret", json.dumps(saved))

    async def test_run_cases_surfaces_metric_error_after_persistence(self):
        from test_rag import run_cases

        cohort = [QAExample("qa-1", "Question?", "Answer")]
        graph = SimpleNamespace(
            ainvoke=AsyncMock(
                return_value={"answer": "A", "retrieval_context": [], "retrieved_passages": []}
            )
        )
        metric_error = [{"name": "faithfulness", "score": None, "reason": None, "error": {"type": "TimeoutError"}}]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            with (
                patch("test_rag.build_test_case", return_value=object()),
                patch("test_rag.build_metrics", return_value=[]),
                patch("test_rag.score_case", new=AsyncMock(return_value=metric_error)),
            ):
                with self.assertRaisesRegex(RuntimeError, "scoring failed"):
                    await run_cases(
                        cohort,
                        {"bm25": graph, "tsvector": graph, "vector": graph},
                        object(),
                        path,
                    )
            saved = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(saved["cases"][0]["metrics"], metric_error)
        self.assertEqual(saved["summary"]["bm25"]["faithfulness"]["error_count"], 1)
        self.assertEqual(graph.ainvoke.await_count, 1)

    @staticmethod
    def _live_resource_mocks():
        settings = SimpleNamespace(
            DATABASE_URL="postgresql+psycopg://db.example/rag",
            OPENAI_API_KEY=SimpleNamespace(get_secret_value=lambda: "configured-key"),
        )
        engine = SimpleNamespace(dispose=AsyncMock())
        session = SimpleNamespace(execute=AsyncMock())
        session_context = MagicMock()
        session_context.__aenter__ = AsyncMock(return_value=session)
        session_context.__aexit__ = AsyncMock(return_value=None)
        session_factory = MagicMock(return_value=session_context)
        return settings, engine, session, session_factory

    async def test_missing_index_stops_before_paid_calls_and_cleans_engine(self):
        from test_rag import run_live_evaluation

        settings, engine, session, session_factory = self._live_resource_mocks()
        with TemporaryDirectory() as directory:
            with (
                patch("test_rag.DEFAULT_REPORT_PATH", Path(directory) / "results.json"),
                patch("backend.core.config.Settings", return_value=settings),
                patch("sqlalchemy.ext.asyncio.create_async_engine", return_value=engine),
                patch("sqlalchemy.ext.asyncio.async_sessionmaker", return_value=session_factory),
                patch("test_rag._verify_bm25_index", new=AsyncMock(side_effect=RuntimeError("missing index"))),
                patch("test_rag.build_judge") as build_judge,
                patch("openai.AsyncOpenAI") as async_openai,
            ):
                with self.assertRaisesRegex(RuntimeError, "setup failed at database"):
                    await run_live_evaluation()
            saved = json.loads((Path(directory) / "results.json").read_text(encoding="utf-8"))

        build_judge.assert_not_called()
        async_openai.assert_not_called()
        engine.dispose.assert_awaited_once()
        self.assertEqual(saved["qa_ids"], [])
        self.assertEqual(saved["errors"], [{"stage": "database", "type": "RuntimeError"}])
        self.assertNotIn("missing index", json.dumps(saved))

    async def test_invalid_cohort_stops_before_paid_calls_and_cleans_engine(self):
        from test_rag import run_live_evaluation

        settings, engine, session, session_factory = self._live_resource_mocks()
        with TemporaryDirectory() as directory:
            with (
                patch("test_rag.DEFAULT_REPORT_PATH", Path(directory) / "results.json"),
                patch("backend.core.config.Settings", return_value=settings),
                patch("sqlalchemy.ext.asyncio.create_async_engine", return_value=engine),
                patch("sqlalchemy.ext.asyncio.async_sessionmaker", return_value=session_factory),
                patch("test_rag._verify_bm25_index", new=AsyncMock()),
                patch("test_rag.load_cohort", new=AsyncMock(side_effect=ValueError("invalid cohort"))),
                patch("test_rag.build_judge") as build_judge,
                patch("openai.AsyncOpenAI") as async_openai,
            ):
                with self.assertRaisesRegex(RuntimeError, "setup failed at cohort"):
                    await run_live_evaluation()
            saved = json.loads((Path(directory) / "results.json").read_text(encoding="utf-8"))

        build_judge.assert_not_called()
        async_openai.assert_not_called()
        engine.dispose.assert_awaited_once()
        self.assertEqual(saved["qa_ids"], [])
        self.assertEqual(saved["errors"], [{"stage": "cohort", "type": "ValueError"}])
        self.assertNotIn("invalid cohort", json.dumps(saved))

    async def test_evaluation_failure_closes_client_and_disposes_engine(self):
        from test_rag import run_live_evaluation

        settings, engine, session, session_factory = self._live_resource_mocks()
        client = SimpleNamespace()
        client_context = MagicMock()
        client_context.__aenter__ = AsyncMock(return_value=client)
        client_context.__aexit__ = AsyncMock(return_value=None)
        cohort = [QAExample("qa-1", "Question?", "Answer")]
        with TemporaryDirectory() as directory:
            with (
                patch("test_rag.DEFAULT_REPORT_PATH", Path(directory) / "results.json"),
                patch("backend.core.config.Settings", return_value=settings),
                patch("sqlalchemy.ext.asyncio.create_async_engine", return_value=engine),
                patch("sqlalchemy.ext.asyncio.async_sessionmaker", return_value=session_factory),
                patch("test_rag._verify_bm25_index", new=AsyncMock()),
                patch("test_rag.load_cohort", new=AsyncMock(return_value=cohort)),
                patch("test_rag.build_judge", return_value=object()),
                patch("test_rag.probe_judge", new=AsyncMock()),
                patch("openai.AsyncOpenAI", return_value=client_context),
                patch("test_rag.build_rag_graph", return_value=object()),
                patch("test_rag.run_cases", new=AsyncMock(side_effect=RuntimeError("case failure"))),
            ):
                with self.assertRaisesRegex(RuntimeError, "case failure"):
                    await run_live_evaluation()

        client_context.__aenter__.assert_awaited_once()
        client_context.__aexit__.assert_awaited_once()
        engine.dispose.assert_awaited_once()

    def test_live_collection_is_opt_in_and_does_not_import_database(self):
        env = os.environ.copy()
        for key in ("RUN_RAG_EVAL", "OPENAI_API_KEY", "DATABASE_URL"):
            env.pop(key, None)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_rag.py", "-q"],
            cwd=Path(__file__).parent.parent,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("1 skipped", result.stdout)

        imported = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import test_rag; assert 'backend.db.database' not in sys.modules",
            ],
            cwd=Path(__file__).parent,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(imported.returncode, 0, imported.stdout + imported.stderr)


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

    async def test_probe_rejects_unverified_structured_output_before_paid_call(self):
        judge = SimpleNamespace(a_generate=AsyncMock())
        with self.assertRaisesRegex(ValueError, "native structured-output"):
            await probe_judge(judge)
        judge.a_generate.assert_not_awaited()

    async def test_probe_requires_a_parsed_true_response(self):
        good_judge = SimpleNamespace(
            supports_structured_outputs=lambda: True,
            a_generate=AsyncMock(return_value=(JudgeProbe(ok=True), None)),
        )
        await probe_judge(good_judge)
        good_judge.a_generate.assert_awaited_once()
        self.assertIs(
            good_judge.a_generate.await_args.kwargs["schema"], JudgeProbe
        )

        bad_judge = SimpleNamespace(
            supports_structured_outputs=lambda: True,
            a_generate=AsyncMock(return_value=(JudgeProbe(ok=False), None)),
        )
        with self.assertRaises(ValueError):
            await probe_judge(bad_judge)

        broken_judge = SimpleNamespace(
            supports_structured_outputs=lambda: True,
            a_generate=AsyncMock(side_effect=RuntimeError("provider failure")),
        )
        with self.assertRaisesRegex(RuntimeError, "provider failure"):
            await probe_judge(broken_judge)


if __name__ == "__main__":
    unittest.main()
