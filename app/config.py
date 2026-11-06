"""Application configuration for FraudGuard."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to a default."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_origins(name: str, default: str) -> tuple[str, ...]:
    """Read a comma-separated list of allowed browser origins.

    Entries are stripped and empty entries dropped, so a trailing comma in the
    environment variable cannot produce an empty-string origin.
    """
    raw = os.getenv(name, default)
    return tuple(origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    """Settings read from environment variables with safe development defaults."""

    app_name: str = os.getenv("APP_NAME", "FraudGuard")
    app_env: str = os.getenv("APP_ENV", "development")
    #: PostgreSQL connection string. Empty means "persistence disabled" rather
    #: than "broken": the API still scores transactions, it just stores nothing.
    #: Never commit a real value; it contains the database password.
    database_url: str = os.getenv("DATABASE_URL", "")
    database_connect_timeout: int = _env_int("DB_CONNECT_TIMEOUT", 10)
    database_echo: bool = _env_bool("DB_ECHO", False)
    model_path: str = os.getenv("MODEL_PATH", "artifacts/models/fraud_model.joblib")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    #: Browser origins allowed to call this API (CORS). Deliberately an explicit
    #: list, never a wildcard: the dashboard is the only intended browser client,
    #: and an allow-list stays auditable. Defaults to the Vite dev server.
    cors_origins: tuple[str, ...] = _env_origins(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    )


settings = Settings()
