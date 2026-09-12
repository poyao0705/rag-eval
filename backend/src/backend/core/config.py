from dotenv import load_dotenv
from pydantic import PostgresDsn, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_ = load_dotenv()


class Settings(BaseSettings):
    DATABASE_URL: PostgresDsn
    OPENAI_API_KEY: SecretStr

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_openai_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("OPENAI_API_KEY must not be blank")
        return value

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
