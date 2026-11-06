"""Persistence operations for fraud alerts.

Kept out of the route handlers so the SQL is testable without HTTP, and so the
API layer only decides *whether* to persist, not *how*.
"""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.flagged import FlaggedTransaction


def persist_alert(
    session: Session,
    *,
    fraud_probability: float,
    threshold: float,
    model_name: str,
    prediction_latency_ms: float,
    transaction_time: float | None = None,
    transaction_amount: float | None = None,
    transaction_id: str | None = None,
    decision: str = "fraud",
) -> FlaggedTransaction:
    """Insert one fraud alert and return the stored row.

    Commits immediately: an alert that is only in an uncommitted transaction is
    an alert that can be lost when the process is recycled, which on serverless
    platforms happens at any moment.

    Raises:
        sqlalchemy.exc.SQLAlchemyError: on any database failure, so the caller
            can decide the HTTP behaviour instead of silently dropping the alert.
    """
    alert = FlaggedTransaction(
        transaction_id=transaction_id,
        fraud_probability=fraud_probability,
        decision=decision,
        threshold=threshold,
        model_name=model_name,
        prediction_latency_ms=prediction_latency_ms,
        transaction_time=transaction_time,
        transaction_amount=transaction_amount,
    )

    session.add(alert)
    session.commit()
    session.refresh(alert)
    return alert


def list_alerts(
    session: Session,
    *,
    limit: int,
    offset: int,
) -> tuple[int, Sequence[FlaggedTransaction]]:
    """Return ``(total_count, page)`` newest first.

    Ordering is ``created_at DESC, id DESC``. The second key matters: alerts
    written in the same microsecond (common in a burst) would otherwise come back
    in an arbitrary, potentially unstable order between requests. The UUID
    tiebreak is arbitrary in value but fully deterministic, so pages never
    overlap or skip rows.
    """
    total = session.scalar(select(func.count()).select_from(FlaggedTransaction)) or 0

    statement = (
        select(FlaggedTransaction)
        .order_by(
            FlaggedTransaction.created_at.desc(),
            FlaggedTransaction.id.desc(),
        )
        .limit(limit)
        .offset(offset)
    )
    items = session.scalars(statement).all()
    return int(total), items


__all__ = ["list_alerts", "persist_alert"]
