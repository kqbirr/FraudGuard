"""Locust load profile for the FraudGuard API.

Run against a scratch instance, never production::

    python -m locust -f tests/load/locustfile.py --headless \
        -u 5 -r 5 -t 30s --host http://127.0.0.1:8010 \
        --csv artifacts/performance/locust_5u --only-summary

The predict task uses the bundled sanitised demo rows, which contain the 30
features only — `Class` is stripped at export time and is never sent, so the
service cannot be "helped" by the label. Every response is also checked against
the threshold rule: if `decision` ever disagrees with `fraud_probability >=
threshold`, the request is marked failed rather than merely slow.

Mix: 6 predict : 2 health : 2 flagged. The demo sample is 7 legitimate / 5 fraud,
so roughly 40% of predict calls are alerts and exercise the insert path.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from locust import HttpUser, between, task

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEMO_JSON = PROJECT_ROOT / "frontend" / "src" / "data" / "demoTransactions.json"

with DEMO_JSON.open(encoding="utf-8") as handle:
    _DEMO = json.load(handle)

#: Feature mappings only: no label, no row identifiers.
ROWS: list[dict[str, float]] = [row["features"] for row in _DEMO["rows"]]

if not ROWS:  # pragma: no cover - guards a broken export
    raise RuntimeError(f"No demo rows found in {DEMO_JSON}")


class FraudGuardUser(HttpUser):
    """A dashboard-like client: mostly scoring, some polling."""

    wait_time = between(0.1, 0.3)

    @task(6)
    def predict(self) -> None:
        payload = random.choice(ROWS)
        with self.client.post(
            "/predict",
            json=payload,
            name="POST /predict",
            catch_response=True,
            timeout=30,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:120]}")
                return

            body: dict[str, Any] = response.json()
            expected = "fraud" if body["fraud_probability"] >= body["threshold"] else "legitimate"
            if body["decision"] != expected:
                response.failure(
                    f"decision {body['decision']!r} contradicts threshold rule "
                    f"(p={body['fraud_probability']:.4f}, t={body['threshold']:.4f})"
                )

    @task(2)
    def health(self) -> None:
        self.client.get("/health", name="GET /health", timeout=10)

    @task(2)
    def flagged(self) -> None:
        self.client.get("/flagged?limit=20&offset=0", name="GET /flagged", timeout=10)
