"""Pydantic request and response schemas for the FraudGuard API.

The request schema is deliberately explicit rather than generated: 30 named
float fields give Swagger UI a self-documenting form and make every validation
rule visible in one place.

Validation policy (Phase 4)
---------------------------
* exactly the 30 model features are accepted — ``Class`` or anything else is
  rejected rather than ignored, because accepting a label at inference time
  would be a silent data-leak hazard;
* values must be numeric and finite — NaN and infinity are rejected, never
  coerced or replaced with zero;
* ``Time`` and ``Amount`` must be non-negative, matching the training data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Final, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Applied to every model feature: reject NaN and +/-infinity outright.
_FINITE: Final[dict[str, bool]] = {"allow_inf_nan": False}

_PCA_DESCRIPTION: Final[str] = (
    "Anonymised PCA component produced by the dataset authors. It has no "
    "public business meaning; do not attempt to interpret its sign or scale."
)


class RootResponse(BaseModel):
    """Response returned by the API root endpoint."""

    message: str


class TransactionFeatures(BaseModel):
    """One transaction to score.

    Field names and order match the model's persisted ``feature_names``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    Time: float = Field(
        ...,
        **_FINITE,
        ge=0,
        description="Elapsed seconds since the first record of the training window.",
        examples=[0.0, 406.0, 172792.0],
    )
    V1: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V2: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V3: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V4: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V5: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V6: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V7: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V8: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V9: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V10: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V11: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V12: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V13: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V14: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V15: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V16: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V17: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V18: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V19: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V20: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V21: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V22: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V23: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V24: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V25: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V26: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V27: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    V28: float = Field(..., **_FINITE, description=_PCA_DESCRIPTION)
    Amount: float = Field(
        ...,
        **_FINITE,
        ge=0,
        description="Transaction amount.",
        examples=[149.62, 2.69, 2125.87],
    )


class PredictionResponse(BaseModel):
    """Result of scoring a single transaction."""

    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Model score: probability-like, 1.0 = certain fraud."
    )
    decision: Literal["fraud", "legitimate"] = Field(
        ..., description="'fraud' when fraud_probability >= threshold, else 'legitimate'."
    )
    threshold: float = Field(
        ..., description="Decision threshold saved with the model during Phase 3."
    )
    latency_ms: float = Field(
        ..., ge=0.0, description="Time spent scoring this transaction, in milliseconds."
    )


class HealthResponse(BaseModel):
    """Structured health report.

    ``status`` is kept for backwards compatibility with the Phase 1 contract:
    it is still ``"ok"`` when the service is fully operational.
    """

    status: Literal["ok", "degraded"] = Field(
        ..., description="'ok' when the model is loaded, 'degraded' otherwise."
    )
    model_loaded: bool = Field(..., description="Whether a model is resident in memory.")
    model_name: str | None = Field(
        None, description="Display name of the loaded model, or null when unavailable."
    )
    threshold: float | None = Field(
        None,
        description=(
            "Decision threshold read from the loaded model artifact; null when no model "
            "is loaded, because no check against it could be applied. A transaction is "
            "flagged as fraud when fraud_probability >= threshold."
        ),
    )
    model_artifact_available: bool = Field(
        ..., description="Whether the model artifact file exists on disk."
    )
    model_artifact_path: str | None = Field(
        None, description="Project-relative path of the artifact (never an absolute path)."
    )
    database_configured: bool = Field(
        ...,
        description=(
            "Whether DATABASE_URL is set. No connection is attempted, so this is a "
            "configuration check, not a reachability check."
        ),
    )


class MetricsLiteResponse(BaseModel):
    """Process-local, in-memory counters.

    These reset whenever the process (or a serverless instance) restarts, and are
    per-instance rather than aggregated. They are a development and smoke-test
    aid, not a substitute for persistent monitoring.
    """

    total_requests: int = Field(..., ge=0, description="Predictions served since start.")
    fraud_predictions: int = Field(..., ge=0, description="Predictions returned as fraud.")
    legitimate_predictions: int = Field(
        ..., ge=0, description="Predictions returned as legitimate."
    )
    average_latency_ms: float | None = Field(
        None, description="Mean scoring latency, or null when no predictions yet."
    )
    min_latency_ms: float | None = Field(None, description="Fastest observed scoring latency.")
    max_latency_ms: float | None = Field(None, description="Slowest observed scoring latency.")
    note: str = Field(
        default=(
            "In-memory, per-process counters. They reset on restart and are not "
            "aggregated across instances; use a real metrics backend for monitoring."
        ),
        description="Reminder about the ephemeral nature of these numbers.",
    )


class FlaggedTransactionRead(BaseModel):
    """One stored fraud alert.

    Deliberately excludes the 30 model features: see ``app/models/flagged.py``
    for why the alert table stores only these summary fields.
    """

    model_config = ConfigDict(from_attributes=True)

    id: UUID = Field(..., description="Alert identifier (UUID v4).")
    transaction_id: str | None = Field(
        None, description="Optional upstream reference supplied by the caller."
    )
    fraud_probability: float = Field(..., description="Model score at decision time.")
    decision: str = Field(..., description="Verdict recorded for this alert.")
    threshold: float = Field(..., description="Threshold in force when scored.")
    model_name: str = Field(..., description="Model that produced the score.")
    prediction_latency_ms: float | None = Field(
        None, description="Time spent scoring the transaction, in milliseconds."
    )
    transaction_time: float | None = Field(
        None, description="Elapsed-time feature from the request, if supplied."
    )
    transaction_amount: float | None = Field(
        None, description="Transaction amount from the request, if supplied."
    )
    created_at: datetime = Field(..., description="When the alert was stored (UTC).")

    @field_validator("created_at")
    @classmethod
    def _normalise_to_utc(cls, value: datetime) -> datetime:
        """Always emit a timezone-aware UTC timestamp.

        PostgreSQL returns an aware value for ``timestamptz``, but SQLite (used by
        the test suite) drops the offset. Normalising here keeps the API contract
        identical on both backends instead of letting clients see naive strings
        in tests and offset strings in production.
        """
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class FlaggedListResponse(BaseModel):
    """A page of stored fraud alerts, newest first."""

    total: int = Field(..., ge=0, description="Total alerts stored, across all pages.")
    limit: int = Field(..., ge=1, description="Page size actually applied.")
    offset: int = Field(..., ge=0, description="Number of rows skipped.")
    items: list[FlaggedTransactionRead] = Field(
        default_factory=list, description="Alerts in this page, newest first."
    )
