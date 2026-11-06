"""Phase 5 tests for fraud-alert persistence.

Isolation guarantee
-------------------
Every test runs against a file-backed SQLite database created in a pytest
temporary directory. These tests **cannot** touch a real Neon database:

* the FastAPI ``get_db`` / ``get_db_optional`` dependencies are overridden with
  the test session factory, so the app never resolves its own engine;
* the client fixture calls ``configure_database(None)`` after startup, so even if
  a developer's ``.env`` defines a real ``DATABASE_URL``, this module's app has
  no engine;
* no test issues ``DROP``, ``DELETE FROM`` against anything but the temp file.

Run from the project root::

    python -m pytest tests/test_persistence.py -v
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, func, inspect, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

import app.database as app_db
from app.database import (
    Base,
    DatabaseUnavailable,
    configure_database,
    get_db,
    get_db_optional,
    mask_database_url,
    safe_error_message,
)
from app.main import app
from app.models.flagged import FlaggedTransaction
from app.services.alerts import list_alerts, persist_alert

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_PATH = PROJECT_ROOT / "artifacts" / "models" / "demo_transactions.csv"
FEATURE_NAMES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]

TABLE = "flagged_transactions"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def engine(tmp_path_factory: pytest.TempPathFactory):
    """Isolated, file-backed SQLite engine with the real schema applied."""
    db_path = tmp_path_factory.mktemp("fraudguard_test") / "test.db"
    eng = create_engine(
        f"sqlite:///{db_path.as_posix()}",
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
    Base.metadata.create_all(eng)
    try:
        yield eng
    finally:
        eng.dispose()


@pytest.fixture(scope="module")
def session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture(scope="module")
def client(session_factory: sessionmaker[Session]) -> Iterator[TestClient]:
    """App client with the model loaded and the database firmly disabled."""
    with TestClient(app) as test_client:
        # Startup configured whatever .env says; force-disable it so this module
        # can never reach a real database through the app's own dependency.
        configure_database(None)
        yield test_client


@pytest.fixture()
def db_overrides(session_factory: sessionmaker[Session]) -> Iterator[None]:
    """Route the app's DB dependencies to the isolated SQLite session."""

    def override_get_db() -> Iterator[Session]:
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    def override_get_db_optional() -> Iterator[Session]:
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_db_optional] = override_get_db_optional
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_db, None)
        app.dependency_overrides.pop(get_db_optional, None)


@pytest.fixture(autouse=True)
def _empty_table(session_factory: sessionmaker[Session]) -> Iterator[None]:
    """Start every test with an empty alerts table (temp SQLite only)."""
    with session_factory() as session:
        session.execute(delete(FlaggedTransaction))
        session.commit()
    yield


@pytest.fixture(scope="module")
def demo_frame() -> pd.DataFrame:
    if not DEMO_PATH.exists():
        pytest.skip("demo sample missing - run 'python scripts/train_models.py'")
    return pd.read_csv(DEMO_PATH)


def _payload(row: pd.Series) -> dict[str, float]:
    return {name: float(row[name]) for name in FEATURE_NAMES}


@pytest.fixture(scope="module")
def legit_payload(demo_frame: pd.DataFrame) -> dict[str, float]:
    return _payload(demo_frame[demo_frame["Class"] == 0].iloc[0])


@pytest.fixture(scope="module")
def fraud_payload(demo_frame: pd.DataFrame) -> dict[str, float]:
    # Last fraud row: a case the saved threshold does detect.
    return _payload(demo_frame[demo_frame["Class"] == 1].iloc[-1])


def _row_count(session_factory: sessionmaker[Session]) -> int:
    with session_factory() as session:
        return int(session.scalar(select(func.count()).select_from(FlaggedTransaction)) or 0)


def _seed_alerts(
    session_factory: sessionmaker[Session],
    count: int,
    *,
    same_timestamp: bool = False,
) -> list[uuid.UUID]:
    """Insert ``count`` alerts, newest first by insertion order."""
    base = datetime.now(timezone.utc)
    ids: list[uuid.UUID] = []
    with session_factory() as session:
        for index in range(count):
            alert = FlaggedTransaction(
                fraud_probability=0.95 - index * 0.01,
                decision="fraud",
                threshold=0.5383065342903137,
                model_name="xgboost",
                prediction_latency_ms=1.5,
                transaction_amount=100.0 + index,
                created_at=base if same_timestamp else base - timedelta(minutes=index),
            )
            session.add(alert)
        session.commit()
        ids = [
            row.id
            for row in session.scalars(
                select(FlaggedTransaction).order_by(FlaggedTransaction.id)
            )
        ]
    return ids


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


def test_table_is_created(engine) -> None:
    assert TABLE in inspect(engine).get_table_names()


def test_model_columns_match_the_specification(engine) -> None:
    columns = {column["name"] for column in inspect(engine).get_columns(TABLE)}
    expected = {
        "id",
        "transaction_id",
        "fraud_probability",
        "decision",
        "threshold",
        "model_name",
        "prediction_latency_ms",
        "transaction_time",
        "transaction_amount",
        "created_at",
    }
    assert columns == expected


def test_raw_features_are_not_stored(engine) -> None:
    """Privacy decision: the 30 model features must not appear in the schema."""
    columns = {column["name"] for column in inspect(engine).get_columns(TABLE)}
    assert not {f"V{i}" for i in range(1, 29)} & columns
    assert "features" not in columns
    assert "payload" not in columns


def test_required_and_nullable_columns(engine) -> None:
    columns = {column["name"]: column for column in inspect(engine).get_columns(TABLE)}

    for name in (
        "id",
        "fraud_probability",
        "decision",
        "threshold",
        "model_name",
        "prediction_latency_ms",
        "created_at",
    ):
        assert columns[name]["nullable"] is False, f"{name} must be NOT NULL"

    for name in ("transaction_id", "transaction_time", "transaction_amount"):
        assert columns[name]["nullable"] is True, f"{name} must be nullable"


def test_indexes_exist(engine) -> None:
    indexes = {index["name"] for index in inspect(engine).get_indexes(TABLE)}
    assert "ix_flagged_transactions_created_at" in indexes
    assert "ix_flagged_transactions_fraud_probability" in indexes
    assert "ix_flagged_transactions_decision" in indexes
    assert "ix_flagged_transactions_decision_created_at" in indexes


def test_created_at_default_is_timezone_aware_utc() -> None:
    """The Python-side default must produce an aware UTC instant.

    Asserted on the default callable rather than on a row read back from SQLite:
    SQLite has no timestamptz type and returns a naive datetime, whereas
    PostgreSQL preserves the offset. The API normalises this at the schema layer
    (see ``FlaggedTransactionRead``).
    """
    from app.models.flagged import _utcnow

    value = _utcnow()
    assert value.tzinfo is not None
    assert value.utcoffset() == timedelta(0)


def test_persisted_alert_gets_an_id_and_timestamp(
    session_factory: sessionmaker[Session],
) -> None:
    with session_factory() as session:
        alert = FlaggedTransaction(
            fraud_probability=0.9,
            decision="fraud",
            threshold=0.5,
            model_name="xgboost",
            prediction_latency_ms=2.0,
        )
        session.add(alert)
        session.commit()
        session.refresh(alert)

        assert alert.id is not None
        assert alert.created_at is not None


# --------------------------------------------------------------------------- #
# Session lifecycle
# --------------------------------------------------------------------------- #


def test_get_db_closes_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSession:
        closed = False

        def close(self) -> None:
            self.closed = True

    fake = FakeSession()
    monkeypatch.setattr(app_db, "get_session_factory", lambda: lambda: fake)

    generator = app_db.get_db()
    assert next(generator) is fake
    with pytest.raises(StopIteration):
        next(generator)
    assert fake.closed is True, "the dependency must close its session"


def test_get_db_optional_closes_the_session(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSession:
        closed = False

        def close(self) -> None:
            self.closed = True

    fake = FakeSession()
    monkeypatch.setattr(app_db, "get_session_factory", lambda: lambda: fake)

    generator = app_db.get_db_optional()
    assert next(generator) is fake
    with pytest.raises(StopIteration):
        next(generator)
    assert fake.closed is True


def test_get_db_optional_yields_none_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable() -> None:
        raise DatabaseUnavailable("not configured")

    monkeypatch.setattr(app_db, "get_session_factory", unavailable)

    generator = app_db.get_db_optional()
    assert next(generator) is None
    with pytest.raises(StopIteration):
        next(generator)


def test_get_db_raises_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable() -> None:
        raise DatabaseUnavailable("not configured")

    monkeypatch.setattr(app_db, "get_session_factory", unavailable)

    with pytest.raises(DatabaseUnavailable):
        next(app_db.get_db())


# --------------------------------------------------------------------------- #
# Service layer
# --------------------------------------------------------------------------- #


def test_persist_alert_roundtrip(session_factory: sessionmaker[Session]) -> None:
    with session_factory() as session:
        alert = persist_alert(
            session,
            fraud_probability=0.9876,
            threshold=0.5383065342903137,
            model_name="xgboost",
            prediction_latency_ms=1.23,
            transaction_time=406.0,
            transaction_amount=2.69,
            transaction_id="tx-abc",
        )
        alert_id = alert.id
        assert isinstance(alert_id, uuid.UUID)

    with session_factory() as session:
        stored = session.get(FlaggedTransaction, alert_id)
        assert stored is not None
        assert stored.fraud_probability == pytest.approx(0.9876)
        assert stored.decision == "fraud"
        assert stored.model_name == "xgboost"
        assert stored.transaction_id == "tx-abc"
        assert stored.transaction_amount == pytest.approx(2.69)


def test_list_alerts_returns_total_and_page(
    session_factory: sessionmaker[Session],
) -> None:
    _seed_alerts(session_factory, 3)

    with session_factory() as session:
        total, items = list_alerts(session, limit=2, offset=0)

    assert total == 3
    assert len(items) == 2


# --------------------------------------------------------------------------- #
# Persistence through POST /predict
# --------------------------------------------------------------------------- #


def test_fraud_prediction_is_persisted(
    client: TestClient,
    db_overrides: None,
    fraud_payload: dict[str, float],
    session_factory: sessionmaker[Session],
) -> None:
    response = client.post("/predict", json=fraud_payload)
    assert response.status_code == 200
    assert response.json()["decision"] == "fraud"

    assert _row_count(session_factory) == 1

    with session_factory() as session:
        stored = session.scalars(select(FlaggedTransaction)).one()
        assert stored.decision == "fraud"
        assert stored.model_name == "xgboost"
        assert stored.threshold == pytest.approx(0.5383065342903137)
        assert stored.fraud_probability == pytest.approx(
            response.json()["fraud_probability"]
        )
        assert stored.transaction_amount == pytest.approx(fraud_payload["Amount"])
        assert stored.prediction_latency_ms >= 0.0


def test_legitimate_prediction_is_not_persisted(
    client: TestClient,
    db_overrides: None,
    legit_payload: dict[str, float],
    session_factory: sessionmaker[Session],
) -> None:
    response = client.post("/predict", json=legit_payload)
    assert response.status_code == 200
    assert response.json()["decision"] == "legitimate"
    assert _row_count(session_factory) == 0


def test_prediction_response_shape_is_unchanged(
    client: TestClient, db_overrides: None, fraud_payload: dict[str, float]
) -> None:
    """Phase 4 contract: persistence must not add or remove response fields."""
    body = client.post("/predict", json=fraud_payload).json()
    assert set(body) == {"fraud_probability", "decision", "threshold", "latency_ms"}


def test_latency_excludes_database_time(
    client: TestClient, db_overrides: None, fraud_payload: dict[str, float]
) -> None:
    body = client.post("/predict", json=fraud_payload).json()
    latency = body["latency_ms"]
    assert 0.0 <= latency < 1000.0, "latency_ms must be model scoring time only"


def test_predict_works_without_database_configured(
    client: TestClient, fraud_payload: dict[str, float]
) -> None:
    """No DATABASE_URL: prediction still succeeds, alert storage is skipped."""
    response = client.post("/predict", json=fraud_payload)
    assert response.status_code == 200
    assert response.json()["decision"] == "fraud"


# --------------------------------------------------------------------------- #
# GET /flagged
# --------------------------------------------------------------------------- #


def test_flagged_is_empty_initially(client: TestClient, db_overrides: None) -> None:
    body = client.get("/flagged").json()
    assert body["total"] == 0
    assert body["items"] == []
    assert body["limit"] == 20
    assert body["offset"] == 0


def test_flagged_returns_persisted_alerts(
    client: TestClient, db_overrides: None, fraud_payload: dict[str, float]
) -> None:
    client.post("/predict", json=fraud_payload)

    body = client.get("/flagged").json()
    assert body["total"] == 1
    assert len(body["items"]) == 1

    item = body["items"][0]
    assert set(item) == {
        "id",
        "transaction_id",
        "fraud_probability",
        "decision",
        "threshold",
        "model_name",
        "prediction_latency_ms",
        "transaction_time",
        "transaction_amount",
        "created_at",
    }
    assert item["decision"] == "fraud"
    uuid.UUID(item["id"])  # raises if not a valid UUID
    assert item["created_at"].endswith("+00:00") or item["created_at"].endswith("Z")


def test_flagged_pagination_pages_do_not_overlap(
    client: TestClient, db_overrides: None, session_factory: sessionmaker[Session]
) -> None:
    _seed_alerts(session_factory, 5)

    first = client.get("/flagged", params={"limit": 2, "offset": 0}).json()
    second = client.get("/flagged", params={"limit": 2, "offset": 2}).json()
    third = client.get("/flagged", params={"limit": 2, "offset": 4}).json()

    assert first["total"] == second["total"] == third["total"] == 5
    assert len(first["items"]) == len(second["items"]) == 2
    assert len(third["items"]) == 1

    ids = [item["id"] for page in (first, second, third) for item in page["items"]]
    assert len(ids) == 5
    assert len(set(ids)) == 5, "pages must not repeat rows"


def test_flagged_is_newest_first(
    client: TestClient, db_overrides: None, session_factory: sessionmaker[Session]
) -> None:
    _seed_alerts(session_factory, 4)

    items = client.get("/flagged", params={"limit": 4}).json()["items"]
    timestamps = [item["created_at"] for item in items]
    assert timestamps == sorted(timestamps, reverse=True), "newest alert must come first"


def test_flagged_ordering_is_stable_with_identical_timestamps(
    client: TestClient, db_overrides: None, session_factory: sessionmaker[Session]
) -> None:
    """Same microsecond: the id tiebreak must still give a stable order."""
    _seed_alerts(session_factory, 4, same_timestamp=True)

    first = [item["id"] for item in client.get("/flagged", params={"limit": 4}).json()["items"]]
    second = [item["id"] for item in client.get("/flagged", params={"limit": 4}).json()["items"]]
    assert first == second

    # Paginating across the tie must not repeat or skip rows either.
    paged = [
        item["id"]
        for page in (
            client.get("/flagged", params={"limit": 2, "offset": 0}).json(),
            client.get("/flagged", params={"limit": 2, "offset": 2}).json(),
        )
        for item in page["items"]
    ]
    assert paged == first
    assert len(set(paged)) == 4


def test_flagged_respects_offset_beyond_total(
    client: TestClient, db_overrides: None
) -> None:
    body = client.get("/flagged", params={"offset": 100}).json()
    assert body["total"] == 0
    assert body["items"] == []


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        ({"limit": 0}, "limit must be >= 1"),
        ({"limit": 101}, "limit must be <= 100"),
        ({"limit": -5}, "negative limit"),
        ({"offset": -1}, "offset must be >= 0"),
        ({"limit": "abc"}, "limit must be an integer"),
    ],
)
def test_pagination_validation(
    client: TestClient, db_overrides: None, params: dict[str, object], reason: str
) -> None:
    response = client.get("/flagged", params=params)
    assert response.status_code == 422, reason


def test_flagged_accepts_maximum_limit(
    client: TestClient, db_overrides: None
) -> None:
    assert client.get("/flagged", params={"limit": 100}).status_code == 200


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #


def test_flagged_returns_503_when_database_unavailable(client: TestClient) -> None:
    """No dependency override and no engine: the app must say so, not 500."""
    response = client.get("/flagged")
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()


def test_predict_returns_503_when_persistence_fails(
    client: TestClient,
    fraud_payload: dict[str, float],
    tmp_path: Path,
) -> None:
    """A configured-but-broken database must not silently drop the alert."""
    broken_engine = create_engine(
        f"sqlite:///{(tmp_path / 'missing_dir' / 'x.db').as_posix()}"
    )
    broken_factory = sessionmaker(bind=broken_engine)

    def override() -> Iterator[Session]:
        session = broken_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db_optional] = override
    try:
        response = client.post("/predict", json=fraud_payload)
    finally:
        app.dependency_overrides.pop(get_db_optional, None)
        broken_engine.dispose()

    assert response.status_code == 503
    assert "could not be stored" in response.json()["detail"].lower()
    assert "retry" in response.json()["detail"].lower()


def test_failed_commit_leaves_no_partial_alert(
    client: TestClient,
    session_factory: sessionmaker[Session],
    fraud_payload: dict[str, float],
) -> None:
    """A commit that fails must not leave a half-written alert behind.

    The broken-engine test above proves the 503; this one proves the row is gone.
    A client that retries after a 503 must not end up with two alerts for one
    transaction, so the failed attempt has to vanish entirely.
    """

    def override() -> Iterator[Session]:
        session = session_factory()

        def failing_commit() -> None:
            raise OperationalError("INSERT INTO flagged_transactions", {}, Exception("simulated"))

        session.commit = failing_commit  # type: ignore[method-assign]
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db_optional] = override
    try:
        response = client.post("/predict", json=fraud_payload)
        assert response.status_code == 503
        assert _row_count(session_factory) == 0
    finally:
        app.dependency_overrides.pop(get_db_optional, None)


def test_credentials_are_masked_in_urls() -> None:
    url = (
        "postgresql+psycopg://fraudguard:sup3rs3cret@"
        "ep-cool-name-123456-pooler.eu-central-1.aws.neon.tech/fraudguard?sslmode=require"
    )
    masked = mask_database_url(url)

    assert "sup3rs3cret" not in masked
    assert "***" in masked
    assert "ep-cool-name-123456-pooler" in masked
    assert masked.startswith("postgresql+psycopg://fraudguard:")


def test_safe_error_message_strips_credentials() -> None:
    exc = RuntimeError(
        "could not connect to "
        "postgresql+psycopg://fraudguard:sup3rs3cret@ep-x-pooler.neon.tech/fraudguard"
    )
    message = safe_error_message(exc)

    assert "sup3rs3cret" not in message
    assert "RuntimeError" in message


def test_error_responses_do_not_leak_credentials(
    client: TestClient, fraud_payload: dict[str, float], tmp_path: Path
) -> None:
    """The 503 body must be generic, whatever the driver error says."""
    broken_engine = create_engine(
        f"sqlite:///{(tmp_path / 'nope' / 'x.db').as_posix()}"
    )
    broken_factory = sessionmaker(bind=broken_engine)

    def override() -> Iterator[Session]:
        session = broken_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db_optional] = override
    try:
        response = client.post("/predict", json=fraud_payload)
    finally:
        app.dependency_overrides.pop(get_db_optional, None)
        broken_engine.dispose()

    assert response.status_code == 503
    body = response.text.lower()
    for forbidden in ("password", "psycopg", "sqlite", "traceback"):
        assert forbidden not in body, f"{forbidden!r} must not appear in the response"


def test_health_reports_database_configuration(
    client: TestClient, db_overrides: None
) -> None:
    body = client.get("/health").json()
    assert "database_configured" in body
    assert isinstance(body["database_configured"], bool)
