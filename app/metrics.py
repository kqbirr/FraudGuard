"""Thread-safe, in-memory prediction metrics for the FraudGuard API.

Scope and honesty
-----------------
These counters live in the process that increments them. They are lost on
restart, are not shared between workers, and are not aggregated across
serverless instances. They exist so ``/metrics-lite`` can answer "is this
instance actually serving predictions?" during development and smoke tests.

They are NOT a monitoring solution. For real observability you need a metrics
backend (counters exported to Prometheus/CloudWatch/etc.) and persistent logs.

Mutating and reading the counters both take a lock, so concurrent requests in
the threadpool cannot produce torn reads or lost updates.
"""

from __future__ import annotations

import threading
from typing import Any, Final


class MetricsRegistry:
    """Counters plus latency extremes for served predictions."""

    __slots__ = ("_lock", "_total", "_fraud", "_legitimate", "_latency_sum",
                 "_latency_min", "_latency_max")

    def __init__(self) -> None:
        self._lock: Final[threading.Lock] = threading.Lock()
        self._total = 0
        self._fraud = 0
        self._legitimate = 0
        self._latency_sum = 0.0
        self._latency_min: float | None = None
        self._latency_max: float | None = None

    def record(self, *, decision_is_fraud: bool, latency_ms: float) -> None:
        """Record one served prediction.

        Args:
            decision_is_fraud: whether the response decision was ``"fraud"``.
            latency_ms: non-negative scoring duration in milliseconds.
        """
        if latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")

        with self._lock:
            self._total += 1
            if decision_is_fraud:
                self._fraud += 1
            else:
                self._legitimate += 1

            self._latency_sum += latency_ms
            if self._latency_min is None or latency_ms < self._latency_min:
                self._latency_min = latency_ms
            if self._latency_max is None or latency_ms > self._latency_max:
                self._latency_max = latency_ms

    def snapshot(self) -> dict[str, Any]:
        """Return a consistent copy of the current counters."""
        with self._lock:
            average = self._latency_sum / self._total if self._total else None
            return {
                "total_requests": self._total,
                "fraud_predictions": self._fraud,
                "legitimate_predictions": self._legitimate,
                "average_latency_ms": round(average, 6) if average is not None else None,
                "min_latency_ms": (
                    round(self._latency_min, 6) if self._latency_min is not None else None
                ),
                "max_latency_ms": (
                    round(self._latency_max, 6) if self._latency_max is not None else None
                ),
            }

    def reset(self) -> None:
        """Clear all counters (used by tests)."""
        with self._lock:
            self._total = 0
            self._fraud = 0
            self._legitimate = 0
            self._latency_sum = 0.0
            self._latency_min = None
            self._latency_max = None


#: Process-wide registry used by the API routes.
metrics: Final[MetricsRegistry] = MetricsRegistry()

__all__ = ["MetricsRegistry", "metrics"]
