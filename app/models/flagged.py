"""SQLAlchemy model for persisted fraud alerts.

What is stored, and what deliberately is not
--------------------------------------------
Only the minimum an alert dashboard needs to triage a case:

* the model's verdict and its probability,
* the threshold in force at decision time (so an old alert stays explainable
  after the model is retrained),
* the model name and scoring latency,
* the transaction amount and its elapsed-time coordinate,
* when the alert was created.

**The 30 model features are not stored** (``V1``-``V28``, ``Time``, ``Amount`` as
a vector). Justification, weighed both ways:

* Privacy: ``V1``-``V28`` are PCA components derived from real cardholder
  behaviour. They are pseudonymised, not anonymous — enough of them can
  re-identify a cardholder or be correlated with another breached dataset.
  Retaining them turns every alert row into cardholder data, widening both the
  breach blast radius and the regulatory scope (GDPR "personal data", PCI-DSS).
* Storage: 30 floats per alert plus the JSON payload is pure overhead for a
  dashboard that shows a handful of numbers.
* Reproducibility: if you ever need to re-score an alert, re-score it from the
  original request in the ingestion layer, not from a table that has become a
  shadow copy of the transaction stream.

The trade-off accepted: you cannot replay a decision exactly from this table
alone. For dispute handling that is a real gap, and the mitigation is to keep the
mutable, access-controlled transaction store as the system of record and this
table as an index of *alerts*.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Index, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp.

    Set in Python rather than by the database so the value is timezone-aware on
    every backend (SQLite's ``CURRENT_TIMESTAMP`` is naive) and has microsecond
    precision, which keeps ordering meaningful for alerts created in the same
    second.
    """
    return datetime.now(timezone.utc)


class FlaggedTransaction(Base):
    """One transaction that the API flagged as fraudulent."""

    __tablename__ = "flagged_transactions"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    #: Optional caller-supplied reference. The API does not require one; it exists
    #: so an upstream payment system can correlate an alert with its own record.
    transaction_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    fraud_probability: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(16), nullable=False, default="fraud")
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    prediction_latency_ms: Mapped[float] = mapped_column(Float, nullable=False)

    #: Elapsed seconds from the training window's origin. Semantically odd for
    #: live traffic (see README); nullable so a caller may omit it.
    transaction_time: Mapped[float | None] = mapped_column(Float, nullable=True)
    transaction_amount: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (
        # Newest-first listing for the dashboard.
        Index("ix_flagged_transactions_created_at", "created_at"),
        # Sorting/analysing by confidence.
        Index("ix_flagged_transactions_fraud_probability", "fraud_probability"),
        # Filtering by verdict (useful once a review workflow adds other states).
        Index("ix_flagged_transactions_decision", "decision"),
        # The dashboard query: WHERE decision = ? ORDER BY created_at DESC.
        Index(
            "ix_flagged_transactions_decision_created_at",
            "decision",
            "created_at",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"FlaggedTransaction(id={self.id!s}, probability={self.fraud_probability:.4f}, "
            f"decision={self.decision!r})"
        )


__all__ = ["FlaggedTransaction"]
