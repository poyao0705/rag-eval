"""Opt-in sequential RAG evaluation harness.

Importing this module is intentionally credential-free.  The live setup is
kept inside ``run_live_evaluation`` so ordinary collection remains offline.
"""

from __future__ import annotations

import os
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from rag_eval.cohort import QAExample, load_cohort
from rag_eval.report import case_record, new_report, write_report
from rag_eval.scoring import (
    build_judge,
    build_metrics,
    build_test_case,
    probe_judge,
    score_case,
)
from sqlalchemy import text

from backend.core.config import DEFAULT_RAG_CONFIG, RAGConfig
from backend.modules.rag.generation import build_answer_model
from backend.modules.rag.graph import build_rag_graph
from backend.modules.retrieval.embeddings import OpenAIQueryEmbedder
from backend.modules.retrieval.pipelines.bm25 import BM25Retriever
from backend.modules.retrieval.pipelines.tsvector import TSVectorRetriever
from backend.modules.retrieval.pipelines.vector import VectorRetriever

RETRIEVER_NAMES = ("bm25", "tsvector", "vector")
DEFAULT_REPORT_PATH = Path(__file__).resolve().parents[1] / ".rag-eval" / "results.json"
BM25_INDEX_PREFLIGHT = text(
    """
    SELECT 1
    FROM pg_index AS i
    JOIN pg_class AS index_class ON index_class.oid = i.indexrelid
    JOIN pg_namespace AS index_namespace
      ON index_namespace.oid = index_class.relnamespace
    JOIN pg_class AS table_class ON table_class.oid = i.indrelid
    JOIN pg_namespace AS table_namespace
      ON table_namespace.oid = table_class.relnamespace
    WHERE index_class.relname = 'source_passage_paradedb_idx'
      AND table_class.relname = 'source_passage'
      AND index_namespace.nspname = current_schema()
      AND table_namespace.nspname = current_schema()
      AND i.indisvalid = true
      AND i.indisready = true
    """
)


def _write_setup_error(
    report_path: Path, report: dict[str, Any], stage: str, error: Exception
) -> None:
    """Persist only a safe type/stage pair for a live setup failure."""
    report["errors"].append({"stage": stage, "type": type(error).__name__})
    write_report(report_path, report)


async def run_cases(
    cohort: list[QAExample],
    graphs: Mapping[str, Any],
    judge: Any,
    report_path: Path,
    config: RAGConfig = DEFAULT_RAG_CONFIG,
) -> dict[str, Any]:
    """Run all graph cases in fixed order, persisting after every case."""
    if set(graphs) != set(RETRIEVER_NAMES):
        raise ValueError("graphs must contain exactly bm25, tsvector, and vector")

    report = new_report(cohort, config)
    write_report(report_path, report)
    for name in RETRIEVER_NAMES:
        graph = graphs[name]
        for qa in cohort:
            state: Mapping[str, Any] | None = None
            try:
                state = await graph.ainvoke({"question": qa.question})
                case = build_test_case(qa, cast(Any, state))
                scores = await score_case(case, build_metrics(judge, config))
            except Exception as error:
                report["cases"].append(
                    case_record(qa, name, state, [], {"type": type(error).__name__})
                )
                write_report(report_path, report)
                raise RuntimeError(
                    f"RAG case failed: {name}/{qa.id}; see report"
                ) from None

            report["cases"].append(case_record(qa, name, state, scores))
            write_report(report_path, report)
            if any(item.get("error") is not None for item in scores):
                raise RuntimeError(f"RAG scoring failed: {name}/{qa.id}; see report")
    return report


async def _verify_bm25_index(session: Any) -> None:
    result = await session.execute(BM25_INDEX_PREFLIGHT)
    if result.scalar_one_or_none() is None:
        raise RuntimeError("required BM25 index is missing or invalid")


async def run_live_evaluation() -> dict[str, Any]:
    """Own live resources and execute the explicitly authorized benchmark."""
    report_path = DEFAULT_REPORT_PATH
    report = new_report([], DEFAULT_RAG_CONFIG)
    write_report(report_path, report)
    engine = None
    stage = "settings"
    try:
        # These imports construct settings/clients only after the opt-in test
        # gate has allowed this function to run.
        from openai import AsyncOpenAI
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from backend.core.config import Settings

        settings = Settings()  # pyright: ignore[reportCallIssue]
        config = settings.rag_config
        report = new_report([], config)
        write_report(report_path, report)
        engine = create_async_engine(str(settings.DATABASE_URL))
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        stage = "database"
        async with session_factory() as session:
            await session.execute(text("SET TRANSACTION READ ONLY"))
            await _verify_bm25_index(session)
            stage = "cohort"
            cohort = await load_cohort(session, config)
            report = new_report(cohort, config)
            write_report(report_path, report)

            stage = "judge_construction"
            judge_key = os.environ.get("RAG_JUDGE_API_KEY")
            if not judge_key:
                judge_key = settings.OPENAI_API_KEY.get_secret_value()
            judge = build_judge(
                api_key=judge_key,
                base_url=os.environ.get("RAG_JUDGE_BASE_URL"),
                config=config,
            )
            stage = "judge_probe"
            await probe_judge(judge)

            stage = "generator"
            generator_key = settings.OPENAI_API_KEY.get_secret_value()
            async with AsyncOpenAI(api_key=generator_key, max_retries=0) as client:
                generator = build_answer_model(client, config)
                graphs = {
                    "bm25": build_rag_graph(
                        BM25Retriever(), session, generator, config
                    ),
                    "tsvector": build_rag_graph(
                        TSVectorRetriever(), session, generator, config
                    ),
                    "vector": build_rag_graph(
                        VectorRetriever(OpenAIQueryEmbedder(client)),
                        session,
                        generator,
                        config,
                    ),
                }
                stage = "evaluation"
                return await run_cases(cohort, graphs, judge, report_path, config)
    except Exception as error:
        if stage == "evaluation":
            raise
        _write_setup_error(report_path, report, stage, error)
        raise RuntimeError(
            f"RAG evaluation setup failed at {stage}; see report"
        ) from None
    finally:
        if engine is not None:
            await engine.dispose()


@unittest.skipUnless(
    os.environ.get("RUN_RAG_EVAL") == "1", "paid RAG evaluation is opt-in"
)
class TestRAGEvaluation(unittest.IsolatedAsyncioTestCase):
    async def test_live_evaluation(self) -> None:
        await run_live_evaluation()


if __name__ == "__main__":
    unittest.main()
