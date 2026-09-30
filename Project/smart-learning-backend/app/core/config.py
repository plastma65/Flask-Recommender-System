from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Cấu hình ứng dụng – đọc từ biến môi trường hoặc file .env."""

    # ── Server ──
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"

    # ── Database ──
    DATABASE_URL: str = "postgresql+asyncpg://postgres:password@localhost:5432/smart_learning"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10

    # ── CORS ──
    CORS_ORIGINS: list[str] = ["*"]

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
