import os
import unittest
from typing import cast
from unittest.mock import patch

from pydantic import SecretStr, ValidationError

from backend.core.config import RAGConfig, Settings

DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/test"


class SettingsTests(unittest.TestCase):
    def test_openai_api_key_is_required_and_secret(self):
        settings = Settings(  # pyright: ignore[reportCallIssue]
            _env_file=None,  # pyright: ignore[reportCallIssue]
            DATABASE_URL=DATABASE_URL,
            OPENAI_API_KEY="sk-test",
        )
        api_key = cast(SecretStr, getattr(settings, "OPENAI_API_KEY"))

        self.assertIsInstance(api_key, SecretStr)
        self.assertEqual(api_key.get_secret_value(), "sk-test")

    def test_openai_api_key_missing_fails_validation(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError):
                Settings(  # pyright: ignore[reportCallIssue]
                    _env_file=None,  # pyright: ignore[reportCallIssue]
                    DATABASE_URL=DATABASE_URL,
                )

    def test_openai_api_key_blank_fails_validation(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValidationError) as context:
                Settings(  # pyright: ignore[reportCallIssue]
                    _env_file=None,  # pyright: ignore[reportCallIssue]
                    DATABASE_URL=DATABASE_URL,
                    OPENAI_API_KEY="   ",
                )

        self.assertIn("OPENAI_API_KEY", str(context.exception))

    def test_rag_config_collects_runtime_and_evaluation_settings(self):
        settings = Settings(  # pyright: ignore[reportCallIssue]
            _env_file=None,  # pyright: ignore[reportCallIssue]
            DATABASE_URL=DATABASE_URL,
            OPENAI_API_KEY="sk-test",
            RAG_ANSWER_MODEL="answer-model",
            RAG_TOP_K=10,
            RAG_EVAL_JUDGE_MODEL="judge-model",
            RAG_EVAL_SEED=9,
            RAG_EVAL_SAMPLE_SIZE=3,
            RAG_EVAL_METRIC_THRESHOLD=0.75,
        )

        self.assertEqual(
            settings.rag_config,
            RAGConfig(
                answer_model="answer-model",
                top_k=10,
                hybrid_candidate_top_k=50,
                judge_model="judge-model",
                evaluation_seed=9,
                evaluation_sample_size=3,
                evaluation_metric_threshold=0.75,
            ),
        )

    def test_hybrid_candidate_top_k_reads_from_environment(self):
        with patch.dict(
            os.environ,
            {
                "DATABASE_URL": DATABASE_URL,
                "OPENAI_API_KEY": "sk-test",
                "RAG_HYBRID_CANDIDATE_TOP_K": "25",
            },
            clear=True,
        ):
            settings = Settings(_env_file=None)  # pyright: ignore[reportCallIssue]

        self.assertEqual(settings.rag_config.hybrid_candidate_top_k, 25)

    def test_rag_config_rejects_invalid_numeric_values(self):
        for field, value in (
            ("RAG_TOP_K", 0),
            ("RAG_HYBRID_CANDIDATE_TOP_K", 0),
            ("RAG_HYBRID_CANDIDATE_TOP_K", -1),
            ("RAG_EVAL_SAMPLE_SIZE", 0),
            ("RAG_EVAL_METRIC_THRESHOLD", -0.01),
            ("RAG_EVAL_METRIC_THRESHOLD", 1.01),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValidationError):
                    Settings(  # pyright: ignore[reportCallIssue]
                        _env_file=None,  # pyright: ignore[reportCallIssue]
                        DATABASE_URL=DATABASE_URL,
                        OPENAI_API_KEY="sk-test",
                        **{field: value},
                    )
