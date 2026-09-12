import os
import unittest
from typing import cast
from unittest.mock import patch

from pydantic import SecretStr, ValidationError

from backend.core.config import Settings


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
