from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ = load_dotenv()


@dataclass(frozen=True, slots=True)
class RAGConfig:
    answer_model: str = "gpt-5-mini"
    top_k: int = 10
    judge_model: str = "gpt-5.4"
    evaluation_seed: int = 42
    evaluation_sample_size: int = 20
    evaluation_metric_threshold: float = 0.5


DEFAULT_RAG_CONFIG = RAGConfig()


class Settings(BaseSettings):
    DATABASE_URL: PostgresDsn
    OPENAI_API_KEY: SecretStr
    RAG_ANSWER_MODEL: str = DEFAULT_RAG_CONFIG.answer_model
    RAG_TOP_K: int = DEFAULT_RAG_CONFIG.top_k
    RAG_EVAL_JUDGE_MODEL: str = DEFAULT_RAG_CONFIG.judge_model
    RAG_EVAL_SEED: int = DEFAULT_RAG_CONFIG.evaluation_seed
    RAG_EVAL_SAMPLE_SIZE: int = DEFAULT_RAG_CONFIG.evaluation_sample_size
    RAG_EVAL_METRIC_THRESHOLD: float = DEFAULT_RAG_CONFIG.evaluation_metric_threshold

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_openai_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("OPENAI_API_KEY must not be blank")
        return value

    @field_validator("RAG_ANSWER_MODEL", "RAG_EVAL_JUDGE_MODEL")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model name must not be blank")
        return value

    @field_validator("RAG_TOP_K", "RAG_EVAL_SAMPLE_SIZE")
    @classmethod
    def validate_positive_integer(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be greater than zero")
        return value

    @field_validator("RAG_EVAL_METRIC_THRESHOLD")
    @classmethod
    def validate_metric_threshold(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("metric threshold must be between zero and one")
        return value

    @property
    def rag_config(self) -> RAGConfig:
        return RAGConfig(
            answer_model=self.RAG_ANSWER_MODEL,
            top_k=self.RAG_TOP_K,
            judge_model=self.RAG_EVAL_JUDGE_MODEL,
            evaluation_seed=self.RAG_EVAL_SEED,
            evaluation_sample_size=self.RAG_EVAL_SAMPLE_SIZE,
            evaluation_metric_threshold=self.RAG_EVAL_METRIC_THRESHOLD,
        )

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
