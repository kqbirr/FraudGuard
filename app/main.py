"""FastAPI application entry point for FraudGuard."""

from fastapi import FastAPI

from app.config import settings
from app.schemas import HealthResponse, RootResponse

app = FastAPI(title=settings.app_name)


@app.get("/", response_model=RootResponse)
def read_root() -> RootResponse:
    """Return a basic service welcome message."""

    return RootResponse(message="FraudGuard API is running.")


@app.get("/health", response_model=HealthResponse)
def read_health() -> HealthResponse:
    """Report that the API process is healthy."""

    return HealthResponse(status="ok")
