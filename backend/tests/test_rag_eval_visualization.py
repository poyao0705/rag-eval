"""Offline tests for RAG evaluation heatmap rendering."""

from __future__ import annotations

import math
import unittest

from rag_eval.visualize import _matrices


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
