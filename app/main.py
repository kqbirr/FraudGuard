"""FastAPI application entry point for FraudGuard.

Phase 4 wires the Phase 3 XGBoost artifact into the API. The model is loaded
once, during application startup (FastAPI ``lifespan``), and reused by every
request; nothing is loaded or deserialised on the request path.

Routes
------
* ``GET  /``            - service banner (unchanged from Phase 1)
* ``GET  /health``      - structured health, including model status
* ``POST /predict``     - score one transaction with the saved model
* ``GET  /metrics-lite``- in-memory, per-process counters
* ``GET  /flagged``     - stored fraud alerts, newest first, paginated
"""

from __future__ import annotations

import math
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, AsyncIterator, Final

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import settings
from app.database import (
    DatabaseUnavailable,
    configure_from_settings,
    get_db,
    get_db_optional,
    is_database_configured,
    safe_error_message,
)
from app.logging_config import configure_logging, get_logger
from app.metrics import metrics
from app.model_loader import (
    PROJECT_ROOT,
    LoadedModel,
    ModelArtifactError,
    load_model,
    resolve_model_path,
)
from app.schemas import (
    FlaggedListResponse,
    FlaggedTransactionRead,
    HealthResponse,
    MetricsLiteResponse,
    PredictionResponse,
    RootResponse,
    TransactionFeatures,
)
from app.services.alerts import list_alerts, persist_alert

logger = get_logger()

#: Number of features the model expects; used in the 503 message.
EXPECTED_FEATURE_COUNT: Final[int] = 30


def _display_path(path: Path) -> str:
    """Render a path relative to the project root, never as an absolute path."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model exactly once per process, before serving traffic.

    A missing artifact does not crash the app: it starts in a degraded state so
    ``/health`` can report the problem, and ``/predict`` returns 503 until the
    model is available.
    """
    configure_logging(settings.log_level)

    # Build the engine once, at startup, never per request. An empty
    # DATABASE_URL simply disables persistence instead of failing.
    configure_from_settings()

    artifact_path = resolve_model_path()
    app.state.model = None

    try:
        loaded = load_model()
    except ModelArtifactError as exc:
        logger.error(
            "model_unavailable",
            extra={
                "event": "model_unavailable",
                "artifact_path": _display_path(artifact_path),
                "reason": str(exc),
            },
        )
    else:
        app.state.model = loaded
        logger.info(
            "model_loaded",
            extra={
                "event": "model_loaded",
                "model_name": loaded.model_name,
                "display_name": loaded.display_name,
                "threshold": loaded.decision_threshold,
                "feature_count": len(loaded.feature_names),
                "artifact_path": _display_path(artifact_path),
            },
        )

    try:
        yield
    finally:
        app.state.model = None
        logger.info("shutdown", extra={"event": "shutdown"})


app = FastAPI(
    title=settings.app_name,
    description=(
        "Real-Time Fraud Detection Service. Scores a single credit-card "
        "transaction with a trained XGBoost model."
    ),
    lifespan=lifespan,
)

# CORS for the dashboard only. The allowed origins come from CORS_ORIGINS and are
# always an explicit list - never "*" - so the browser surface stays auditable as
# the frontend moves from the Vite dev server to its deployed origin. Credentials
# are disabled because this API has no cookies and no authentication (Phase 5
# boundary); enabling them would be the one combination a wildcard makes unsafe.
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


def _make_json_safe(value: Any) -> Any:
    """Replace non-finite floats so an error body can always be serialised.

    Why this exists: when a client sends ``NaN`` or ``Infinity``, FastAPI's
    default handler echoes the offending value back inside the 422 body. The JSON
    encoder refuses to serialise those values, so the encoder's ``ValueError``
    replaces a clean 422 with a server error. Reporting ``'nan'``/``'inf'`` as a
    string keeps the response valid JSON.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {key: _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return 422 with a JSON-safe detail body for any invalid request."""
    return JSONResponse(
        status_code=422,
        content={"detail": _make_json_safe(jsonable_encoder(exc.errors()))},
    )


@app.exception_handler(DatabaseUnavailable)
async def database_unavailable_handler(
    request: Request, exc: DatabaseUnavailable
) -> JSONResponse:
    """Map "no database configured" to a clean 503.

    The message returned to the client is deliberately generic: the exception
    text and any driver output stay in the logs, where they are redacted.
    """
    logger.error(
        "database_unavailable",
        extra={"event": "database_unavailable", "error": safe_error_message(exc)},
    )
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "The alert database is unavailable."},
    )


def get_model(request: Request) -> LoadedModel:
    """Dependency returning the loaded model, or 503 when unavailable."""
    model: LoadedModel | None = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Model is not available on this instance. The artifact could not be "
                "loaded at startup; check the server logs and /health."
            ),
        )
    return model


ModelDep = Annotated[LoadedModel, Depends(get_model)]
DatabaseSession = Annotated[Session, Depends(get_db)]
OptionalDatabaseSession = Annotated[Session | None, Depends(get_db_optional)]


def _record_fraud_alert(
    db: Session | None,
    *,
    model: LoadedModel,
    probability: float,
    latency_ms: float,
    payload: TransactionFeatures,
) -> None:
    """Persist a fraud alert, following the documented failure policy.

    Policy (deliberate, and the trade-off is stated in the README):

    * **Database not configured** - log a warning and return. Prediction is the
      primary function of the service and must keep working in local development
      or in a deployment that has no database yet.
    * **Configured but the write fails** - raise 503. For a fraud decision the
      endpoint's contract is "the alert was recorded". Returning 200 while the
      alert vanished would let the alerting pipeline silently lose frauds, which
      is the worse failure. The cost is that a database outage also fails fraud
      scoring; that is accepted because an unrecorded fraud is unrecoverable.

    Only the alert id is logged, never the feature values.
    """
    if db is None:
        logger.warning(
            "alert_not_persisted",
            extra={"event": "alert_not_persisted", "reason": "database_not_configured"},
        )
        return

    try:
        alert = persist_alert(
            db,
            fraud_probability=probability,
            threshold=model.decision_threshold,
            model_name=model.model_name,
            prediction_latency_ms=latency_ms,
            transaction_time=payload.Time,
            transaction_amount=payload.Amount,
        )
    except SQLAlchemyError as exc:
        logger.error(
            "alert_persist_failed",
            extra={
                "event": "alert_persist_failed",
                "error": safe_error_message(exc),
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The transaction was scored as fraud but the alert could not be "
                "stored. No alert was recorded; retry the request."
            ),
        ) from exc

    logger.info(
        "alert_persisted",
        extra={"event": "alert_persisted", "alert_id": str(alert.id)},
    )


@app.get("/", response_model=RootResponse, tags=["service"])
def read_root() -> RootResponse:
    """Return a basic service welcome message."""

    return RootResponse(message="FraudGuard API is running.")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["service"],
    responses={503: {"description": "Model not loaded; service is degraded."}},
)
def read_health(request: Request, response: Response) -> HealthResponse:
    """Report API and model health.

    Returns HTTP 200 with ``status="ok"`` when the model is resident, and
    HTTP 503 with ``status="degraded"`` when it is not — a health check that
    reports 200 while unable to score would be misleading.
    """
    model: LoadedModel | None = getattr(request.app.state, "model", None)
    artifact_path = resolve_model_path()
    available = artifact_path.exists()

    if model is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(
            status="degraded",
            model_loaded=False,
            model_name=None,
            # No model, no threshold: reporting one here would invite a client to
            # apply a cutoff the service cannot currently enforce.
            threshold=None,
            model_artifact_available=available,
            model_artifact_path=_display_path(artifact_path),
            database_configured=is_database_configured(),
        )

    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_name=model.display_name,
        # Taken from the artifact loaded at startup, never from a constant here.
        threshold=model.decision_threshold,
        model_artifact_available=available,
        model_artifact_path=_display_path(artifact_path),
        database_configured=is_database_configured(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["prediction"],
    responses={
        422: {"description": "Validation error: missing, unexpected or non-finite feature."},
        503: {"description": "Model not loaded."},
    },
)
def predict(
    payload: TransactionFeatures,
    model: ModelDep,
    db: OptionalDatabaseSession,
) -> PredictionResponse:
    """Score one transaction and return the fraud decision.

    The client may not send ``Class`` (or any other unknown field): the request
    schema forbids extras, so a label cannot leak into inference. Request feature
    values are never logged — only the derived probability, decision and latency.
    """
    features = payload.model_dump()

    started = time.perf_counter()
    probability = model.score_row(features)
    latency_ms = (time.perf_counter() - started) * 1000.0

    is_fraud = probability >= model.decision_threshold
    decision = "fraud" if is_fraud else "legitimate"

    metrics.record(decision_is_fraud=is_fraud, latency_ms=latency_ms)

    logger.info(
        "prediction_served",
        extra={
            "event": "prediction_served",
            "decision": decision,
            "fraud_probability": round(probability, 6),
            "threshold": model.decision_threshold,
            "latency_ms": round(latency_ms, 3),
        },
    )

    # Persistence happens after the latency measurement, so the reported
    # latency_ms stays purely model-scoring time and never includes database IO.
    if is_fraud:
        _record_fraud_alert(
            db,
            model=model,
            probability=probability,
            latency_ms=latency_ms,
            payload=payload,
        )

    return PredictionResponse(
        fraud_probability=probability,
        decision=decision,
        threshold=model.decision_threshold,
        latency_ms=round(latency_ms, 3),
    )


@app.get("/metrics-lite", response_model=MetricsLiteResponse, tags=["service"])
def read_metrics_lite() -> MetricsLiteResponse:
    """Expose lightweight in-memory counters.

    These are per-process and reset on restart (including every cold start of a
    serverless instance). They are not aggregated and are not a monitoring
    substitute — see ``app/metrics.py``.
    """
    return MetricsLiteResponse(**metrics.snapshot())


@app.get(
    "/flagged",
    response_model=FlaggedListResponse,
    tags=["alerts"],
    responses={503: {"description": "Alert database unavailable."}},
)
def read_flagged(
    db: DatabaseSession,
    limit: Annotated[int, Query(ge=1, le=100, description="Page size (1-100).")] = 20,
    offset: Annotated[int, Query(ge=0, description="Rows to skip.")] = 0,
) -> FlaggedListResponse:
    """Return stored fraud alerts, newest first.

    ``total`` counts every stored alert, not just this page, so a dashboard can
    paginate. Ordering is deterministic even when timestamps collide.
    """
    try:
        total, items = list_alerts(db, limit=limit, offset=offset)
    except SQLAlchemyError as exc:
        logger.error(
            "flagged_list_failed",
            extra={"event": "flagged_list_failed", "error": safe_error_message(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The alert store is currently unreachable.",
        ) from exc

    return FlaggedListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[FlaggedTransactionRead.model_validate(item) for item in items],
    )
