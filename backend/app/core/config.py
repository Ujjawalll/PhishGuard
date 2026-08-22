from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    API_ENV: str = "development"
    SECRET_KEY: str = "changeme_in_production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440
    DATABASE_URL: str = "sqlite+aiosqlite:///./phishguard.db"  # Default to sqlite for local dev if Postgres unavailable
    WORKER_TIMEOUT_SECONDS: int = 10
    WORKER_MAX_REDIRECTS: int = 5

    class Config:
        env_file = ".env"

settings = Settings()
