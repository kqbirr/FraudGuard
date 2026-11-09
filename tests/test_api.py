"""Phase 4 tests for the FraudGuard prediction API.

The inference logic itself is covered by ``tests/test_model_pipeline.py``; this
module tests the HTTP layer: validation, threshold handling, health reporting and
the in-memory metrics.

Run from the project root::

    python -m pytest tests/test_api.py -v
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.metrics import metrics
from app.model_loader import load_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = PROJECT_ROOT / "artifacts" / "models" / "demo_transactions.csv"
FEATURE_NAMES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]

pytestmark = pytest.mark.skipif(
    not DEMO_PATH.exists(),
    reason="demo sample missing - run 'python scripts/train_models.py' first",
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """A client whose context manager runs the lifespan (loads the model once)."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def demo_frame() -> pd.DataFrame:
    return pd.read_csv(DEMO_PATH)


def _payload(row: pd.Series) -> dict[str, float]:
    return {name: float(row[name]) for name in FEATURE_NAMES}


@pytest.fixture(scope="module")
def legit_payload(demo_frame: pd.DataFrame) -> dict[str, float]:
    return _payload(demo_frame[demo_frame["Class"] == 0].iloc[0])


@pytest.fixture(scope="module")
def fraud_payload(demo_frame: pd.DataFrame) -> dict[str, float]:
    """A fraudulent demo row that the model actually flags.

    The demo sample holds 5 frauds and the saved threshold detects 4 of them, so
    the first fraud row is deliberately avoided here: it scores below threshold
    and would test the legitimate path by accident. Full coverage of the sample
    (misses included) is asserted by ``test_decision_follows_threshold_rule``.
    """
    return _payload(demo_frame[demo_frame["Class"] == 1].iloc[-1])


@pytest.fixture()
def fresh_metrics() -> Iterator[None]:
    """Reset the process-wide counters around a test."""
    metrics.reset()
    yield
    metrics.reset()


@contextmanager
def model_unavailable() -> Iterator[None]:
    """Temporarily pretend the model failed to load, then restore it."""
    saved = getattr(app.state, "model", None)
    app.state.model = None
    try:
        yield
    finally:
        app.state.model = saved


# --------------------------------------------------------------------------- #
# Service endpoints
# --------------------------------------------------------------------------- #


def test_root_endpoint(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "FraudGuard API is running."


def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_name"] == "XGBoost"
    assert body["model_artifact_available"] is True
    # The Phase 1 contract key must survive.
    assert "status" in body

    # The threshold must come from the loaded artifact's metadata, so it is
    # compared against the loader rather than against a literal here.
    assert body["threshold"] == pytest.approx(load_model().decision_threshold)
    assert 0.0 < body["threshold"] <= 1.0


def test_health_does_not_leak_absolute_paths(client: TestClient) -> None:
    body = client.get("/health").json()
    path = body["model_artifact_path"]
    assert path is not None
    assert ":" not in path, "absolute Windows paths must not be exposed"
    assert not path.startswith("/")
    assert path.endswith("fraud_model.ubj")


def test_health_degrades_without_model(client: TestClient) -> None:
    with model_unavailable():
        response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert body["model_name"] is None


def test_health_reports_no_threshold_without_model(client: TestClient) -> None:
    """A degraded service must not advertise a cutoff it cannot apply."""
    with model_unavailable():
        body = client.get("/health").json()

    assert body["threshold"] is None


def test_health_threshold_matches_prediction_threshold(
    client: TestClient, legit_payload: dict[str, float]
) -> None:
    """One artifact, one threshold: /health and /predict cannot disagree."""
    health = client.get("/health").json()
    prediction = client.post("/predict", json=legit_payload).json()

    assert health["threshold"] == pytest.approx(prediction["threshold"], abs=1e-12)


# --------------------------------------------------------------------------- #
# Prediction: happy paths
# --------------------------------------------------------------------------- #


def test_predict_legitimate_transaction(
    client: TestClient, legit_payload: dict[str, float]
) -> None:
    response = client.post("/predict", json=legit_payload)
    assert response.status_code == 200

    body = response.json()
    assert body["decision"] == "legitimate"
    assert 0.0 <= body["fraud_probability"] < body["threshold"]
    assert body["latency_ms"] >= 0.0


def test_predict_fraud_transaction(
    client: TestClient, fraud_payload: dict[str, float]
) -> None:
    response = client.post("/predict", json=fraud_payload)
    assert response.status_code == 200

    body = response.json()
    assert body["decision"] == "fraud"
    assert body["fraud_probability"] >= body["threshold"]


def test_threshold_is_the_saved_artifact_value(
    client: TestClient, legit_payload: dict[str, float]
) -> None:
    body = client.post("/predict", json=legit_payload).json()
    # The Phase 3 artifact recorded this exact threshold.
    assert body["threshold"] == pytest.approx(0.5383065342903137, abs=1e-12)


def test_decision_follows_threshold_rule(
    client: TestClient, demo_frame: pd.DataFrame
) -> None:
    """decision == 'fraud' exactly when probability >= threshold, for every demo row."""
    for _, row in demo_frame.iterrows():
        body = client.post("/predict", json=_payload(row)).json()
        expected = "fraud" if body["fraud_probability"] >= body["threshold"] else "legitimate"
        assert body["decision"] == expected


def test_response_contains_only_documented_fields(
    client: TestClient, legit_payload: dict[str, float]
) -> None:
    body = client.post("/predict", json=legit_payload).json()
    assert set(body) == {"fraud_probability", "decision", "threshold", "latency_ms"}
    assert isinstance(body["fraud_probability"], float)
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_feature_order_does_not_affect_the_result(
    client: TestClient, fraud_payload: dict[str, float]
) -> None:
    """JSON object key order is irrelevant: the server enforces model order."""
    forward = client.post("/predict", json=fraud_payload).json()["fraud_probability"]
    reversed_payload = dict(reversed(list(fraud_payload.items())))
    backward = client.post("/predict", json=reversed_payload).json()["fraud_probability"]
    assert forward == pytest.approx(backward, abs=1e-12)


def test_predict_is_deterministic(
    client: TestClient, fraud_payload: dict[str, float]
) -> None:
    first = client.post("/predict", json=fraud_payload).json()["fraud_probability"]
    second = client.post("/predict", json=fraud_payload).json()["fraud_probability"]
    assert first == second


# --------------------------------------------------------------------------- #
# Prediction: validation failures
# --------------------------------------------------------------------------- #


def test_missing_feature_is_rejected(
    client: TestClient, legit_payload: dict[str, float]
) -> None:
    incomplete = dict(legit_payload)
    incomplete.pop("Amount")

    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422
    assert "Amount" in json.dumps(response.json())


def test_extra_feature_is_rejected(
    client: TestClient, legit_payload: dict[str, float]
) -> None:
    with_class = {**legit_payload, "Class": 1}
    response = client.post("/predict", json=with_class)
    assert response.status_code == 422
    assert "Class" in json.dumps(response.json())


def test_class_label_is_never_accepted(
    client: TestClient, fraud_payload: dict[str, float]
) -> None:
    """A client cannot tell the API 'this is fraud' via the label."""
    response = client.post("/predict", json={**fraud_payload, "Class": 0})
    assert response.status_code == 422


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_values_are_rejected(
    client: TestClient, legit_payload: dict[str, float], bad_value: float
) -> None:
    payload = {**legit_payload, "Time": bad_value}
    # json.dumps emits the NaN/Infinity literals that a careless client may send.
    body = json.dumps(payload)

    response = client.post(
        "/predict", content=body, headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422, response.text


def test_non_numeric_value_is_rejected(
    client: TestClient, legit_payload: dict[str, float]
) -> None:
    response = client.post("/predict", json={**legit_payload, "V1": "not-a-number"})
    assert response.status_code == 422


def test_negative_amount_is_rejected(
    client: TestClient, legit_payload: dict[str, float]
) -> None:
    response = client.post("/predict", json={**legit_payload, "Amount": -1.0})
    assert response.status_code == 422


def test_negative_time_is_rejected(
    client: TestClient, legit_payload: dict[str, float]
) -> None:
    """Elapsed dataset time cannot run backwards, so ``Time`` has the same bound."""
    response = client.post("/predict", json={**legit_payload, "Time": -1.0})
    assert response.status_code == 422

    body = json.dumps(response.json())
    assert "Time" in body


def test_empty_body_is_rejected(client: TestClient) -> None:
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_returns_503_without_model(
    client: TestClient, legit_payload: dict[str, float]
) -> None:
    with model_unavailable():
        response = client.post("/predict", json=legit_payload)

    assert response.status_code == 503
    assert "not available" in response.json()["detail"].lower()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_metrics_empty_state(client: TestClient, fresh_metrics: None) -> None:
    body = client.get("/metrics-lite").json()

    assert body["total_requests"] == 0
    assert body["fraud_predictions"] == 0
    assert body["legitimate_predictions"] == 0
    assert body["average_latency_ms"] is None
    assert body["min_latency_ms"] is None
    assert body["max_latency_ms"] is None
    assert "reset" in body["note"].lower()


def test_metrics_count_predictions(
    client: TestClient,
    fresh_metrics: None,
    legit_payload: dict[str, float],
    fraud_payload: dict[str, float],
) -> None:
    client.post("/predict", json=legit_payload)
    client.post("/predict", json=legit_payload)
    client.post("/predict", json=fraud_payload)

    body = client.get("/metrics-lite").json()
    assert body["total_requests"] == 3
    assert body["fraud_predictions"] == 1
    assert body["legitimate_predictions"] == 2


def test_metrics_latency_bounds_are_consistent(
    client: TestClient,
    fresh_metrics: None,
    legit_payload: dict[str, float],
    fraud_payload: dict[str, float],
) -> None:
    client.post("/predict", json=legit_payload)
    client.post("/predict", json=fraud_payload)

    body = client.get("/metrics-lite").json()
    assert body["min_latency_ms"] <= body["average_latency_ms"] <= body["max_latency_ms"]
    assert body["min_latency_ms"] >= 0.0


def test_metrics_are_not_incremented_by_validation_errors(
    client: TestClient, fresh_metrics: None, legit_payload: dict[str, float]
) -> None:
    client.post("/predict", json={**legit_payload, "Amount": -5.0})
    body = client.get("/metrics-lite").json()
    assert body["total_requests"] == 0


# --------------------------------------------------------------------------- #
# Documentation
# --------------------------------------------------------------------------- #


def test_openapi_documents_all_endpoints(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]

    assert "/" in paths
    assert "/health" in paths
    assert "/predict" in paths
    assert "/metrics-lite" in paths
    assert "post" in paths["/predict"]

    request_schema = paths["/predict"]["post"]["requestBody"]["content"]["application/json"]
    assert "schema" in request_schema


def test_openapi_request_schema_lists_all_30_features(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    components = schema["components"]["schemas"]
    transaction = components["TransactionFeatures"]

    assert set(transaction["properties"]) == set(FEATURE_NAMES)
    assert "Class" not in transaction["properties"]
    assert transaction.get("additionalProperties") is False


def test_docs_page_is_served(client: TestClient) -> None:
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
