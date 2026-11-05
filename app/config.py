"""Application configuration for FraudGuard."""

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    """Settings read from environment variables with safe development defaults."""

    app_name: str = os.getenv("APP_NAME", "FraudGuard")
    app_env: str = os.getenv("APP_ENV", "development")
    database_url: str = os.getenv("DATABASE_URL", "")
    model_path: str = os.getenv("MODEL_PATH", "artifacts/fraud_model.joblib")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
