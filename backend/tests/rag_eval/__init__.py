"""Shared names for the local RAG evaluation harness."""

from typing import Final, Literal

type EvalMode = Literal["llm", "hybrid", "system_one"]

EVAL_MODE_BY_FLAG: Final[dict[str, EvalMode]] = {
    "0": "llm",
    "1": "hybrid",
    "2": "system_one",
}
