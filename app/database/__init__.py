"""Database engine, session factory and FastAPI dependencies.

Design
------
* **One engine per process.** ``create_engine`` is expensive and holds a
  connection pool; building it per request would exhaust Neon's connection limit
  almost immediately. The engine is created lazily, once, behind a lock.
* **Sessions per request, always closed.** A SQLAlchemy ``Session`` is a unit of
  work, not a connection wrapper. The dependencies here yield a session inside a
  ``try/finally`` so it is returned to the pool even when a handler raises.
* **No credentials in logs or responses.** The URL is only ever logged through
  :func:`mask_database_url`, and error text passes through
  :func:`safe_error_message`, which strips anything that looks like a password.

Neon specifics
--------------
* Use the **pooled** connection string (the host contains ``-pooler``). Neon
  fronts PgBouncer in transaction mode, multiplexing many short-lived client
  connections onto a handful of real Postgres backends. Serverless functions
  scale out aggressively, and Postgres has a hard connection ceiling, so pooling
  is what stops a traffic spike from exhausting the database.
* The direct (non-pooled) string is for migrations and admin work, which need
  session-level state. Application traffic should use the pooled one.
* ``pool_pre_ping=True`` plus a short ``pool_recycle`` protects against a pooled
  backend being recycled underneath us.
* Transaction-mode pooling forbids session-level state (client-named prepared
  statements, locks held across statements). Ordinary ORM statements are fine.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from typing import Any, Final

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings

#: Matches the password portion of a URL so it can be redacted.
_PASSWORD_RE: Final[re.Pattern[str]] = re.compile(r"(://[^:/@]+:)([^@]*)@")
#: Splits a PostgreSQL URL into scheme and optional ``+driver`` token.
_DRIVER_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<scheme>postgres(?:ql)?)(?P<driver>\+[A-Za-z0-9_]+)?://"
)


class DatabaseUnavailable(RuntimeError):
    """Raised when no database is configured or the engine cannot be built."""


class Base(DeclarativeBase):
    """Declarative base shared by every ORM model."""


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
#: Once configured (even to "no database"), configuration is authoritative and
#: the settings are not re-read. This is what stops a test run from accidentally
#: connecting to a real database because ``.env`` happens to define one.
_explicit: bool = False
_lock: Final[threading.RLock] = threading.RLock()


# --------------------------------------------------------------------------- #
# URL handling and redaction
# --------------------------------------------------------------------------- #


def mask_database_url(url: str | None) -> str:
    """Return the URL with the password replaced by ``***``.

    Used for logging and for the init script's console output. Never log the raw
    URL: it embeds the database password.
    """
    if not url:
        return "(not configured)"
    return _PASSWORD_RE.sub(r"\1***@", url)


def safe_error_message(exc: BaseException) -> str:
    """Render an exception for logging without leaking credentials."""
    return _PASSWORD_RE.sub(r"\1***@", f"{type(exc).__name__}: {exc}")


def normalize_database_url(url: str | None) -> str | None:
    """Pin a driver-less PostgreSQL URL to the psycopg 3 dialect.

    Neon (and most managed providers) hand out ``postgresql://user:pass@host/db``.
    SQLAlchemy resolves that bare scheme to the psycopg2 dialect, which this
    project does not install -- only psycopg 3 is. Rewriting just the scheme
    token means the value pulled by ``neon env pull`` works verbatim, and a
    later pull cannot silently re-break the connection.

    Only the scheme is touched: an explicitly chosen driver is respected, and
    the credentials, host, database and query string are passed through as-is.
    """
    if not url:
        return url

    match = _DRIVER_RE.match(url)
    if match is None or match.group("driver"):
        return url

    return f"postgresql+psycopg://{url[match.end():]}"


# --------------------------------------------------------------------------- #
# Engine / session factory lifecycle
# --------------------------------------------------------------------------- #


def _engine_kwargs(url: str) -> dict[str, Any]:
    """Connection-pool settings appropriate for the target backend."""
    kwargs: dict[str, Any] = {"pool_pre_ping": True, "echo": settings.database_echo}

    if url.startswith("sqlite"):
        # Tests use a file-backed SQLite database across threads.
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        # Serverless-friendly: small pool, quick recycling, bounded wait.
        kwargs.update(
            pool_size=5,
            max_overflow=5,
            pool_timeout=10,
            pool_recycle=300,
            connect_args={
                "connect_timeout": settings.database_connect_timeout,
                # psycopg 3 auto-prepares a statement after `prepare_threshold`
                # executions (default 5). Behind PgBouncer in transaction mode
                # (Neon's pooled endpoint) the next execution can land on a
                # different backend where that prepared statement does not exist,
                # producing "prepared statement does not exist" errors under
                # load. Disabling client-side preparation keeps every statement
                # self-contained, which is what a transaction-mode pooler needs.
                "prepare_threshold": None,
            },
        )
    return kwargs


def _build_locked(url: str | None) -> None:
    """Create (or clear) the engine. Caller must hold ``_lock``."""
    global _engine, _session_factory, _explicit

    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
    _explicit = True

    if not url:
        return

    url = normalize_database_url(url) or url
    _engine = create_engine(url, **_engine_kwargs(url))
    _session_factory = sessionmaker(
        bind=_engine,
        autoflush=False,
        expire_on_commit=False,
    )


def configure_database(url: str | None) -> None:
    """Explicitly configure the database. ``None`` disables persistence."""
    with _lock:
        _build_locked(url)


def configure_from_settings() -> None:
    """Configure from ``settings.database_url`` (called at application startup)."""
    with _lock:
        _build_locked(settings.database_url or None)


def reset_database_state() -> None:
    """Forget any configuration (used by tests)."""
    global _explicit

    with _lock:
        if _engine is not None:
            _engine.dispose()
        _build_locked(None)
        _explicit = False


def is_database_configured() -> bool:
    """True when a database URL is configured (no network call is made)."""
    if _engine is not None:
        return True
    return bool(settings.database_url)


def get_engine() -> Engine:
    """Return the process-wide engine, building it on first use."""
    with _lock:
        if _engine is None and not _explicit:
            # Lazy path used outside the FastAPI lifespan (scripts, tests).
            _build_locked(settings.database_url or None)
        if _engine is None:
            raise DatabaseUnavailable(
                "Database is not configured: set DATABASE_URL in the environment."
            )
        return _engine


def get_session_factory() -> sessionmaker[Session]:
    """Return the session factory, building the engine on first use."""
    with _lock:
        if _session_factory is None and not _explicit:
            _build_locked(settings.database_url or None)
        if _session_factory is None:
            raise DatabaseUnavailable(
                "Database is not configured: set DATABASE_URL in the environment."
            )
        return _session_factory


def dispose_engine() -> None:
    """Release pooled connections (used on shutdown and by tests)."""
    with _lock:
        if _engine is not None:
            _engine.dispose()


# --------------------------------------------------------------------------- #
# Schema creation
# --------------------------------------------------------------------------- #


def init_db() -> list[str]:
    """Create missing tables. Never drops or alters existing ones.

    ``create_all`` reflects the schema first (``checkfirst=True``) and only issues
    ``CREATE TABLE`` / ``CREATE INDEX`` for tables that are absent, leaving
    existing tables and rows untouched. It is enough for this phase; a production
    system should use Alembic migrations (see ``scripts/init_db.py``).
    """
    from app import models  # noqa: F401  - import registers models on Base.metadata

    engine = get_engine()
    Base.metadata.create_all(engine)
    return sorted(Base.metadata.tables)


# --------------------------------------------------------------------------- #
# FastAPI dependencies
# --------------------------------------------------------------------------- #


def get_db() -> Iterator[Session]:
    """Yield a session for a request that requires the database.

    Raises :class:`DatabaseUnavailable` (mapped to HTTP 503) when unconfigured.
    The session is always closed, and an uncommitted transaction is rolled back
    implicitly by ``close()``.
    """
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def get_db_optional() -> Iterator[Session | None]:
    """Yield a session, or ``None`` when persistence is disabled.

    ``/predict`` must keep working without a database (prediction is the primary
    function; alert storage is secondary), so this variant degrades instead of
    raising.
    """
    try:
        factory = get_session_factory()
    except DatabaseUnavailable:
        yield None
        return

    session = factory()
    try:
        yield session
    finally:
        session.close()


__all__ = [
    "Base",
    "DatabaseUnavailable",
    "configure_database",
    "configure_from_settings",
    "dispose_engine",
    "get_db",
    "get_db_optional",
    "get_engine",
    "get_session_factory",
    "init_db",
    "is_database_configured",
    "mask_database_url",
    "normalize_database_url",
    "reset_database_state",
    "safe_error_message",
]
