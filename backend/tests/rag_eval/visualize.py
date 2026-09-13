"""Render saved RAG evaluation reports as static heatmaps."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any

from matplotlib import colormaps  # pyright: ignore[reportMissingImports]
from matplotlib.backends.backend_agg import FigureCanvasAgg  # pyright: ignore[reportMissingImports]
from matplotlib.figure import Figure  # pyright: ignore[reportMissingImports]

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


def _status(report: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    flags: list[str] = []
    if summary["attempted_case_count"] != summary["expected_case_count"]:
        flags.append("PARTIAL")
    metric_errors = sum(
        summary[retriever][metric]["error_count"]
        for retriever in report["retrievers"]
        for metric in METRICS
    )
    if report.get("errors") or metric_errors:
        flags.append("ERRORS")
    return ", ".join(flags) or "COMPLETE"


def _metadata(report: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    return (
        f"{_status(report, summary)} | "
        f"cases {summary['attempted_case_count']}/{summary['expected_case_count']} | "
        f"{report.get('timestamp', 'unknown')} | seed {report.get('seed', 'unknown')} | "
        f"top_k {report.get('top_k', 'unknown')}\n"
        f"generator {report.get('generator_model', 'unknown')} | "
        f"judge {report.get('judge_model', 'unknown')}"
    )


def _heatmap(
    values: Sequence[Sequence[float]],
    rows: Sequence[str],
    columns: Sequence[str],
    annotations: Sequence[Sequence[str]],
    title: str,
) -> Figure:
    figure = Figure(
        figsize=(max(9, 1.7 * len(columns)), max(4, 0.48 * len(rows) + 2.5)),
        layout="constrained",
    )
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    image = axes.imshow(
        values,
        cmap=colormaps["viridis"].with_extremes(bad="#dddddd"),
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    axes.set_xticks(range(len(columns)), labels=columns, rotation=25, ha="right")
    axes.set_yticks(range(len(rows)), labels=rows)
    axes.set_title(title, fontsize=10)
    for row_index, row in enumerate(values):
        for column_index, value in enumerate(row):
            axes.text(
                column_index,
                row_index,
                annotations[row_index][column_index],
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if math.isfinite(value) and value < 0.5
                    else "black"
                ),
            )
    figure.colorbar(image, ax=axes, label="Score (0–1; gray = unavailable)")
    return figure


def build_figures(report: dict[str, Any]) -> tuple[Figure, Figure]:
    summary, aggregate, recall = _matrices(report)
    retrievers = report["retrievers"]
    metrics = [metric.replace("_", "\n") for metric in METRICS]
    summary_annotations = [
        [
            (
                f"{summary[retriever][metric]['mean']:.2f}\n"
                f"n={summary[retriever][metric]['scored_count']}"
            )
            if summary[retriever][metric]["mean"] is not None
            else f"N/A\nn={summary[retriever][metric]['scored_count']}"
            for metric in METRICS
        ]
        for retriever in retrievers
    ]
    recall_annotations = [
        [f"{value:.2f}" if math.isfinite(value) else "N/A" for value in row]
        for row in recall
    ]
    metadata = _metadata(report, summary)
    summary_figure = _heatmap(
        aggregate,
        retrievers,
        metrics,
        summary_annotations,
        f"RAG evaluation summary\n{metadata}",
    )
    summary_figure.text(
        0.01,
        0.01,
        "Faithfulness is not correctness",
        fontsize=8,
    )
    recall_figure = _heatmap(
        recall,
        [f"Q{index + 1:02d} {qa_id}" for index, qa_id in enumerate(report["qa_ids"])],
        retrievers,
        recall_annotations,
        f"Contextual recall by question\n{metadata}",
    )
    return summary_figure, recall_figure


def render_report(report_path: Path) -> tuple[Path, Path]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    figures = build_figures(report)
    outputs = (
        report_path.with_suffix(".summary.png"),
        report_path.with_suffix(".contextual_recall.png"),
    )
    for figure, output in zip(figures, outputs, strict=True):
        temporary = output.with_suffix(".tmp")
        try:
            figure.savefig(temporary, format="png", dpi=160)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render saved RAG evaluation heatmaps."
    )
    parser.add_argument(
        "report",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[2] / ".rag-eval" / "results.json",
    )
    args = parser.parse_args()
    try:
        outputs = render_report(args.report)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(
            1,
            f"Heatmap generation failed ({type(error).__name__}); JSON unchanged.\n",
        )
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
