"""Offline tests for RAG evaluation heatmap rendering."""

from __future__ import annotations

import importlib
import json
import math
import os
import struct
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, patch

visualize = importlib.import_module("rag_eval.visualize")
_matrices = visualize._matrices
build_figures = visualize.build_figures
render_report = visualize.render_report


def sample_report() -> dict:
    def case(qa: str, retriever: str, score: object, error: object = None) -> dict:
        return {
            "qa_id": qa,
            "retriever": retriever,
            "question": f"Question {qa}?",
            "error": None,
            "metrics": [
                {"name": "contextual_recall", "score": score, "error": error}
            ],
        }

    return {
        "schema_version": 1,
        "timestamp": "2026-09-13T00:00:00+00:00",
        "seed": 42,
        "generator_model": "generator",
        "judge_model": "judge",
        "top_k": 10,
        "retrievers": ["vector", "bm25"],
        "qa_ids": ["q2", "q1"],
        "expected_case_count": 4,
        "errors": [],
        "cases": [
            case("q1", "bm25", 0.0),
            case("q1", "vector", 0.5),
            case("q2", "vector", 1.0),
        ],
        "summary": {"stale": True},
    }


class MatrixTests(unittest.TestCase):
    def test_order_means_and_missing_values(self) -> None:
        summary, aggregate, recall = _matrices(sample_report())
        self.assertEqual(aggregate[0][3], 0.75)
        self.assertEqual(aggregate[1][3], 0.0)
        self.assertEqual(summary["vector"]["contextual_recall"]["scored_count"], 2)
        self.assertEqual(recall[0][0], 1.0)
        self.assertTrue(math.isnan(recall[0][1]))
        self.assertEqual(recall[1], [0.5, 0.0])
        self.assertTrue(math.isnan(aggregate[0][0]))

    def test_error_and_invalid_scores_are_missing(self) -> None:
        for score, error in [
            (None, None),
            (True, None),
            (float("nan"), None),
            (float("inf"), None),
            (0.9, {"type": "JudgeError"}),
        ]:
            with self.subTest(score=score, error=error):
                report = sample_report()
                report["cases"][2]["metrics"][0].update(
                    score=score, error=error
                )
                summary, _, recall = _matrices(report)
                self.assertTrue(math.isnan(recall[0][0]))
                self.assertEqual(
                    summary["vector"]["contextual_recall"]["scored_count"], 1
                )

    def test_empty_cases_keep_expected_grid(self) -> None:
        report = sample_report()
        report["cases"] = []
        _, aggregate, recall = _matrices(report)
        self.assertTrue(all(math.isnan(x) for row in aggregate + recall for x in row))

    def test_case_order_does_not_change_matrices(self) -> None:
        report = sample_report()
        reversed_report = {**report, "cases": list(reversed(report["cases"]))}
        first = _matrices(report)
        second = _matrices(reversed_report)
        for first_row, second_row in zip(first[1] + first[2], second[1] + second[2], strict=True):
            for first_value, second_value in zip(first_row, second_row, strict=True):
                if math.isnan(first_value):
                    self.assertTrue(math.isnan(second_value))
                else:
                    self.assertEqual(first_value, second_value)

    def test_rejects_unsupported_or_ambiguous_reports(self) -> None:
        reports = []
        for key, value in [
            ("schema_version", 2),
            ("qa_ids", []),
            ("retrievers", []),
            ("qa_ids", ["q1", "q1"]),
        ]:
            report = sample_report()
            report[key] = value
            reports.append(report)

        duplicate = sample_report()
        duplicate["cases"].append(duplicate["cases"][0])
        reports.append(duplicate)

        unknown_retriever = sample_report()
        unknown_retriever["cases"][0]["retriever"] = "unknown"
        reports.append(unknown_retriever)

        unknown_question = sample_report()
        unknown_question["cases"][0]["qa_id"] = "unknown"
        reports.append(unknown_question)

        duplicate_metric = sample_report()
        duplicate_metric["cases"][0]["metrics"].append(
            duplicate_metric["cases"][0]["metrics"][0].copy()
        )
        reports.append(duplicate_metric)

        for score in (-0.1, 1.1):
            invalid_range = sample_report()
            invalid_range["cases"][0]["metrics"][0]["score"] = score
            reports.append(invalid_range)

        for report in reports:
            with self.subTest(report=report), self.assertRaises(ValueError):
                _matrices(report)


class RenderingTests(unittest.TestCase):
    def test_figure_semantics(self) -> None:
        summary, recall = build_figures(sample_report())
        self.assertEqual(summary.axes[0].images[0].get_clim(), (0.0, 1.0))
        self.assertEqual(recall.axes[0].images[0].get_clim(), (0.0, 1.0))
        self.assertEqual(len(summary.axes[0].get_yticklabels()), 2)
        self.assertEqual(len(summary.axes[0].get_xticklabels()), 5)
        self.assertEqual(len(recall.axes[0].get_yticklabels()), 2)
        self.assertEqual(len(recall.axes[0].get_xticklabels()), 2)
        labels = [text.get_text() for text in summary.axes[0].texts]
        self.assertIn("0.75\nn=2", labels)
        self.assertIn("N/A\nn=0", labels)
        self.assertIn("N/A", [text.get_text() for text in recall.axes[0].texts])

    def test_png_export_preserves_json(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "custom.json"
            source.write_text(json.dumps(sample_report()), encoding="utf-8")
            before = source.read_bytes()
            outputs = render_report(source)
            self.assertEqual(
                [path.name for path in outputs],
                ["custom.summary.png", "custom.contextual_recall.png"],
            )
            for output in outputs:
                self.assertTrue(output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
                width, height = struct.unpack(">II", output.read_bytes()[16:24])
                self.assertGreater(height, 100)
                self.assertGreater(width, 100)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(render_report(source), outputs)

    def test_cli_needs_no_credentials_or_display(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(sample_report()), encoding="utf-8")
            result = self._run_cli(source)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(source.with_suffix(".summary.png").exists())
            self.assertTrue(source.with_suffix(".contextual_recall.png").exists())

    def test_cli_rejects_missing_or_malformed_input(self) -> None:
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            malformed = Path(directory) / "malformed.json"
            malformed.write_text("{", encoding="utf-8")
            for source in (missing, malformed):
                with self.subTest(source=source):
                    result = self._run_cli(source)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(source.with_suffix(".summary.png").exists())
                    self.assertFalse(
                        source.with_suffix(".contextual_recall.png").exists()
                    )

    def test_all_missing_data_still_renders_gray_cells(self) -> None:
        report = sample_report()
        for case in report["cases"]:
            case["metrics"][0]["score"] = None
        with TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(report), encoding="utf-8")
            outputs = render_report(source)
            self.assertEqual(len(outputs), 2)
            self.assertTrue(all(output.exists() for output in outputs))

    def test_titles_mark_partial_and_error_reports(self) -> None:
        report = sample_report()
        report["errors"] = [{"stage": "judge", "type": "RuntimeError"}]
        report["cases"][1]["metrics"][0]["error"] = {"type": "JudgeError"}
        summary, recall = build_figures(report)
        self.assertIn("PARTIAL", summary.axes[0].get_title())
        self.assertIn("ERRORS", summary.axes[0].get_title())
        self.assertIn("PARTIAL", recall.axes[0].get_title())
        self.assertIn("ERRORS", recall.axes[0].get_title())
        self.assertIn("Faithfulness is not correctness", "".join(
            text.get_text() for text in summary.texts
        ))

    def test_one_retriever_report_keeps_both_grids_dynamic(self) -> None:
        report = sample_report()
        report.update(
            retrievers=["vector"],
            qa_ids=["q1"],
            expected_case_count=1,
            cases=[report["cases"][1]],
        )
        summary, recall = build_figures(report)
        self.assertEqual(len(summary.axes[0].get_xticklabels()), 5)
        self.assertEqual(len(recall.axes[0].get_xticklabels()), 1)
        self.assertEqual(len(recall.axes[0].get_yticklabels()), 1)

    def test_save_failure_removes_temporary_file_and_preserves_json(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(sample_report()), encoding="utf-8")
            temporary = source.with_suffix(".summary.tmp")
            temporary.write_bytes(b"stale")
            before = source.read_bytes()
            with patch(
                "rag_eval.visualize.Figure.savefig",
                side_effect=OSError("disk full"),
            ):
                with self.assertRaisesRegex(OSError, "disk full"):
                    render_report(source)
            self.assertEqual(source.read_bytes(), before)
            self.assertFalse(temporary.exists())

    @staticmethod
    def _run_cli(source: Path) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        for key in (
            "OPENAI_API_KEY",
            "COHERE_API_KEY",
            "DATABASE_URL",
            "RUN_RAG_EVAL",
            "DISPLAY",
            "WAYLAND_DISPLAY",
        ):
            env.pop(key, None)
        return subprocess.run(
            [sys.executable, "-m", "rag_eval.visualize", str(source)],
            cwd=Path(__file__).parent,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )


class CollectionTests(unittest.TestCase):
    def test_live_module_import_does_not_load_matplotlib(self) -> None:
        env = os.environ.copy()
        env.pop("RUN_RAG_EVAL", None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import test_rag; assert 'matplotlib' not in sys.modules",
            ],
            cwd=Path(__file__).parent,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


class LiveChartHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_renders_only_after_success(self) -> None:
        from test_rag import DEFAULT_REPORT_PATH, TestRAGEvaluation

        events: list[str] = []

        async def evaluate() -> dict:
            events.append("evaluation")
            return {}

        def render(path: Path) -> tuple[Path, Path]:
            self.assertEqual(path, DEFAULT_REPORT_PATH)
            events.append("render")
            return (path, path)

        with (
            patch("test_rag.run_live_evaluation", side_effect=evaluate),
            patch("rag_eval.visualize.render_report", side_effect=render),
        ):
            await TestRAGEvaluation("test_live_evaluation").test_live_evaluation()
        self.assertEqual(events, ["evaluation", "render"])

    async def test_evaluation_failure_does_not_render(self) -> None:
        from test_rag import TestRAGEvaluation

        with (
            patch(
                "test_rag.run_live_evaluation",
                new=AsyncMock(side_effect=RuntimeError("evaluation failed")),
            ),
            patch("rag_eval.visualize.render_report") as render,
        ):
            with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
                await TestRAGEvaluation("test_live_evaluation").test_live_evaluation()
            render.assert_not_called()

    async def test_render_failure_is_visible_without_retrying_evaluation(self) -> None:
        from test_rag import TestRAGEvaluation

        with patch(
            "test_rag.run_live_evaluation", new=AsyncMock(return_value={})
        ) as evaluate, patch(
            "rag_eval.visualize.render_report", side_effect=OSError("disk full")
        ):
            with self.assertRaisesRegex(OSError, "disk full"):
                await TestRAGEvaluation("test_live_evaluation").test_live_evaluation()
            evaluate.assert_awaited_once()
