"""DeepEval case conversion, metric construction, and judge gating."""

from __future__ import annotations

import math
from numbers import Real
from typing import Any, Iterable

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel

from backend.modules.rag.graph import RAGState
from rag_eval.cohort import QAExample

# DeepEval 4.2.2's recommended/default judge, pinned explicitly for this harness.
JUDGE_MODEL = "gpt-5.4"


class JudgeProbe(BaseModel):
    ok: bool


def build_test_case(qa: QAExample, state: RAGState) -> LLMTestCase:
    """Convert a graph result while keeping the gold answer out of graph state."""
    answer = state.get("answer")
    retrieval_context = state.get("retrieval_context")
    if answer is None or retrieval_context is None:
        raise ValueError("graph state is missing the final answer or retrieval context")
    return LLMTestCase(
        input=qa.question,
        actual_output=answer,
        expected_output=qa.answer,
        retrieval_context=list(retrieval_context),
    )


def build_metrics(judge: Any) -> list[tuple[str, BaseMetric]]:
    """Create fresh, explicitly configured instances for one test case."""
    metric_types = (
        ("answer_relevancy", AnswerRelevancyMetric),
        ("faithfulness", FaithfulnessMetric),
        ("contextual_precision", ContextualPrecisionMetric),
        ("contextual_recall", ContextualRecallMetric),
        ("contextual_relevancy", ContextualRelevancyMetric),
    )
    return [
        (key, metric_type(model=judge, include_reason=True, threshold=0.5))
        for key, metric_type in metric_types
    ]


def _invalid_score_record(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "score": None,
        "reason": None,
        "error": {"type": "InvalidMetricScore"},
    }


def _valid_score(score: Any) -> bool:
    if not isinstance(score, Real) or isinstance(score, bool):
        return False
    numeric_score = float(score)
    return math.isfinite(numeric_score) and 0 <= numeric_score <= 1


async def score_case(
    test_case: LLMTestCase,
    metrics: Iterable[tuple[str, BaseMetric]],
) -> list[dict[str, Any]]:
    """Measure metrics sequentially, preserving each metric's failure."""
    results: list[dict[str, Any]] = []
    for name, metric in metrics:
        try:
            await metric.a_measure(test_case)
            score = getattr(metric, "score", None)
            if not _valid_score(score):
                results.append(_invalid_score_record(name))
                continue
            results.append(
                {
                    "name": name,
                    "score": score,
                    "reason": getattr(metric, "reason", None),
                    "error": None,
                }
            )
        except Exception as error:
            results.append(
                {
                    "name": name,
                    "score": None,
                    "reason": None,
                    "error": {"type": type(error).__name__},
                }
            )
    return results


def build_judge(*, api_key: str, base_url: str | None = None):
    """Build only the explicitly requested judge; never silently fall back."""
    from deepeval.models import OpenAIModel

    return OpenAIModel(model=JUDGE_MODEL, api_key=api_key, base_url=base_url)


async def probe_judge(judge: Any) -> None:
    """Require native structured output before making the paid probe call."""
    supports_structured_outputs = getattr(judge, "supports_structured_outputs", None)
    if not callable(supports_structured_outputs):
        raise ValueError(
            "judge does not expose verified native structured-output capability"
        )
    if supports_structured_outputs() is not True:
        raise ValueError(
            "judge does not support native structured outputs; refusing fallback JSON parsing"
        )

    parsed, _cost = await judge.a_generate(
        "Return a JSON object with ok set to true.", schema=JudgeProbe
    )
    if not isinstance(parsed, JudgeProbe) or parsed.ok is not True:
        raise ValueError("judge structured-output probe returned an invalid response")
