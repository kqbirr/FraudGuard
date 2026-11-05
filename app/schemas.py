"""Pydantic response schemas for the initial API."""

from pydantic import BaseModel


class RootResponse(BaseModel):
    """Response returned by the API root endpoint."""

    message: str


class HealthResponse(BaseModel):
    """Response returned by the service health endpoint."""

    status: str
