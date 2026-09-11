from dotenv import load_dotenv
from pydantic import PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    DATABASE_URL: PostgresDsn
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
