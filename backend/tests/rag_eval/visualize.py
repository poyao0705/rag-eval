"""Render saved RAG evaluation reports as static heatmaps."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

from rag_eval.report import METRICS, summarize


def _axis(report: Mapping[str, Any], name: str) -> list[str]:
    values = report.get(name)
    if not isinstance(values, list) or not values or any(
        not isinstance(value, str) or not value for value in values
    ) or len(set(values)) != len(values):
        raise ValueError(f"report {name} must be a non-empty list of unique strings")
    return values


def _valid_score(score: Any) -> float | None:
    if not isinstance(score, Real) or isinstance(score, bool):
        return None
    value = float(score)
    if math.isfinite(value) and not 0 <= value <= 1:
        raise ValueError("metric scores must be between 0 and 1")
    return value if math.isfinite(value) else None


def _matrices(
    report: dict[str, Any],
) -> tuple[dict[str, Any], list[list[float]], list[list[float]]]:
    if not isinstance(report, dict) or report.get("schema_version") != 1:
        raise ValueError("report schema_version must be 1")

    retrievers = _axis(report, "retrievers")
    qa_ids = _axis(report, "qa_ids")
    cases = report.get("cases")
    expected = report.get("expected_case_count")
    if not isinstance(cases, list) or not isinstance(expected, int) or isinstance(expected, bool):
        raise ValueError("report cases and expected_case_count have invalid types")
    if expected < 0:
        raise ValueError("report expected_case_count must not be negative")

    seen: set[tuple[str, str]] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("report cases must contain objects")
        qa_id = case.get("qa_id")
        retriever = case.get("retriever")
        if qa_id not in qa_ids or retriever not in retrievers:
            raise ValueError("case refers to an undeclared question or retriever")
        key = (qa_id, retriever)
        if key in seen:
            raise ValueError("report contains duplicate question/retriever cases")
        seen.add(key)
        metrics = case.get("metrics")
        if not isinstance(metrics, list):
            raise ValueError("case metrics must be a list")
        metric_names: set[str] = set()
        for metric in metrics:
            if not isinstance(metric, Mapping) or not isinstance(metric.get("name"), str):
                raise ValueError("metrics must contain named objects")
            name = metric["name"]
            if name in metric_names:
                raise ValueError("case contains duplicate metrics")
            metric_names.add(name)
            _valid_score(metric.get("score"))

    summary = summarize(
        cases,
        retriever_names=retrievers,
        expected_case_count=expected,
    )
    aggregate = [
        [
            float(summary[retriever][metric]["mean"])
            if summary[retriever][metric]["mean"] is not None
            else math.nan
            for metric in METRICS
        ]
        for retriever in retrievers
    ]
    by_case = {(case["qa_id"], case["retriever"]): case for case in cases}
    recall: list[list[float]] = []
    for qa_id in qa_ids:
        row: list[float] = []
        for retriever in retrievers:
            case = by_case.get((qa_id, retriever), {})
            metric = next(
                (
                    metric
                    for metric in case.get("metrics", [])
                    if metric["name"] == "contextual_recall"
                ),
                {},
            )
            score = _valid_score(metric.get("score"))
            row.append(
                score
                if score is not None and metric.get("error") is None
                else math.nan
            )
        recall.append(row)
    return summary, aggregate, recall
