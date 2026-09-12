"""JSON report construction and persistence for the local RAG evaluation."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path
from typing import Any, Mapping, Sequence

from rag_eval.cohort import QAExample, SEED


GENERATOR_MODEL = "gpt-5-mini"
JUDGE_MODEL = "gpt-5.4"
TOP_K = 5
EXPECTED_CASE_COUNT = 60
RETRIEVERS = ("bm25", "tsvector", "vector")
METRICS = (
    "answer_relevancy",
    "faithfulness",
    "contextual_precision",
    "contextual_recall",
    "contextual_relevancy",
)


def new_report(cohort: Sequence[QAExample]) -> dict[str, Any]:
    """Return an empty report carrying the fixed evaluation metadata."""
    return {
        "schema_version": 1,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "qa_ids": [qa.id for qa in cohort],
        "generator_model": GENERATOR_MODEL,
        "judge_model": JUDGE_MODEL,
        "top_k": TOP_K,
        "cases": [],
        "errors": [],
    }


def _safe_error(error: Mapping[str, Any] | None) -> dict[str, str] | None:
    """Keep only non-sensitive, explicitly supported error fields."""
    if error is None:
        return None
    result: dict[str, str] = {}
    for key in ("type", "stage"):
        value = error.get(key)
        if isinstance(value, str):
            result[key] = value
    return result


def _safe_metric(metric: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize only the public metric record fields."""
    name = metric.get("name")
    score = metric.get("score")
    if not isinstance(name, str):
        raise ValueError("metric record is missing a string name")
    if (
        isinstance(score, Real)
        and not isinstance(score, bool)
        and math.isfinite(float(score))
    ):
        score_value: float | None = float(score)
    else:
        score_value = None
    reason = metric.get("reason")
    return {
        "name": name,
        "score": score_value,
        "reason": reason if isinstance(reason, str) else None,
        "error": _safe_error(metric.get("error")),
    }


def case_record(
    qa: QAExample,
    retriever: str,
    state: Mapping[str, Any] | None,
    metrics: list[dict[str, Any]],
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON-safe record for one completed or failed graph case."""
    state = state or {}
    passages = state.get("retrieved_passages", [])
    context = state.get("retrieval_context", [])
    return {
        "qa_id": qa.id,
        "retriever": retriever,
        "question": qa.question,
        "expected_answer": qa.answer,
        "actual_answer": (
            state.get("answer") if isinstance(state.get("answer"), str) else None
        ),
        "passage_ids": [str(p.passage_id) for p in passages],
        "retrieval_context": [item for item in context if isinstance(item, str)],
        "metrics": [_safe_metric(metric) for metric in metrics],
        "error": _safe_error(error),
    }


def _valid_score(score: Any) -> bool:
    return (
        isinstance(score, Real)
        and not isinstance(score, bool)
        and math.isfinite(float(score))
    )


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize successful scores and make every missing measurement visible."""
    totals: dict[str, dict[str, dict[str, Any]]] = {
        retriever: {
            metric: {"scores": [], "error_count": 0}
            for metric in METRICS
        }
        for retriever in RETRIEVERS
    }
    attempted = len(cases)
    completed = sum(1 for case in cases if case.get("error") is None)

    for case in cases:
        retriever = case.get("retriever")
        if retriever not in totals:
            continue
        by_name = {
            metric.get("name"): metric
            for metric in case.get("metrics", [])
            if isinstance(metric, Mapping)
        }
        for metric_name in METRICS:
            entry = totals[retriever][metric_name]
            metric = by_name.get(metric_name)
            if metric is None:
                # An omitted measurement is a visible failure, whether the
                # graph itself failed or the metric loop stopped early.
                entry["error_count"] += 1
                continue
            if metric.get("error") is not None:
                entry["error_count"] += 1
            elif _valid_score(metric.get("score")):
                entry["scores"].append(metric["score"])
            else:
                entry["error_count"] += 1

    summary: dict[str, Any] = {}
    for retriever in RETRIEVERS:
        summary[retriever] = {}
        for metric in METRICS:
            entry = totals[retriever][metric]
            scores = entry["scores"]
            summary[retriever][metric] = {
                "mean": (sum(scores) / len(scores) if scores else None),
                "scored_count": len(scores),
                "error_count": entry["error_count"],
            }
    summary.update(
        {
            "attempted_case_count": attempted,
            "completed_case_count": completed,
            "failed_case_count": attempted - completed,
            "expected_case_count": EXPECTED_CASE_COUNT,
        }
    )
    return summary


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically write a report after recomputing its summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report["summary"] = summarize(report["cases"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


DEFAULT_REPORT_PATH = Path("backend/.rag-eval/results.json")
