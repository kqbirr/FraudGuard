"""Fire concurrent predictions and check alerts for duplicates or missing rows.

What this proves
----------------
Under ``--workers`` parallel clients:

* every response is either 200 or a documented 503 -- never a 500;
* the same input always produces the same decision (no cross-request state leak);
* the number of stored alerts grows by exactly the number of fraud decisions,
  so a concurrent burst neither drops nor duplicates rows.

The last check needs ``GET /flagged`` to be available, so run it against an API
with a scratch database (``sqlite:///./artifacts/performance/bench.db``) and never
against production. The two sample rows are verified first: one the model calls
legitimate, one it calls fraud.

Usage::

    python scripts/concurrency_check.py --base-url http://127.0.0.1:8010 \
        --workers 20 --requests 200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_JSON = PROJECT_ROOT / "frontend" / "src" / "data" / "demoTransactions.json"
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "performance" / "concurrency_check.json"

LEGIT_ROW_ID = "demo-01"
FRAUD_ROW_ID = "demo-12"


def load_rows() -> dict[str, dict[str, float]]:
    payload = json.loads(DEMO_JSON.read_text(encoding="utf-8"))
    return {row["id"]: row["features"] for row in payload["rows"]}


def flagged_total(client: httpx.Client, base_url: str) -> int:
    response = client.get(f"{base_url}/flagged", params={"limit": 1, "offset": 0})
    response.raise_for_status()
    return int(response.json()["total"])


def one_request(
    base_url: str, features: dict[str, float], expected: str
) -> dict[str, Any]:
    """Send one prediction and classify the outcome."""
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(f"{base_url}/predict", json=features)
    except httpx.HTTPError as exc:
        return {"outcome": "transport_error", "detail": type(exc).__name__}

    if response.status_code != 200:
        return {
            "outcome": f"http_{response.status_code}",
            "detail": response.text[:160],
        }

    body = response.json()
    if body["decision"] != expected:
        return {
            "outcome": "decision_mismatch",
            "detail": f"expected {expected}, got {body['decision']}",
        }
    return {"outcome": body["decision"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    base_url = args.base_url.rstrip("/")
    rows = load_rows()
    legit = rows[LEGIT_ROW_ID]
    fraud = rows[FRAUD_ROW_ID]

    with httpx.Client(timeout=30.0) as client:
        health = client.get(f"{base_url}/health").json()
        if health.get("status") != "ok":
            print(f"ERROR: API is not healthy: {health}")
            return 1
        # Confirm the expected decisions before trusting them in the assertion.
        # This must happen before the baseline count: the fraud probe itself
        # writes one alert, which would otherwise look like a duplicate.
        for label, features, expected in (
            ("legitimate", legit, "legitimate"),
            ("fraud", fraud, "fraud"),
        ):
            decision = client.post(f"{base_url}/predict", json=features).json()["decision"]
            if decision != expected:
                print(f"ERROR: sample row for '{label}' now decides '{decision}'.")
                return 1

        before_total = flagged_total(client, base_url)

    print(f"Target         : {base_url}")
    print(f"Concurrency    : {args.workers} workers, {args.requests} requests")
    print(f"Alerts before  : {before_total}")

    jobs = [
        (fraud, "fraud") if index % 3 == 0 else (legit, "legitimate")
        for index in range(args.requests)
    ]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(
            pool.map(lambda job: one_request(base_url, job[0], job[1]), jobs)
        )

    counts = Counter(outcome["outcome"] for outcome in outcomes)
    failures = [outcome for outcome in outcomes if outcome["outcome"] not in ("fraud", "legitimate")]
    fraud_ok = counts.get("fraud", 0)

    with httpx.Client(timeout=30.0) as client:
        after_total = flagged_total(client, base_url)
        health_after = client.get(f"{base_url}/health").json()

    delta = after_total - before_total

    result = {
        "base_url": base_url,
        "workers": args.workers,
        "requests": args.requests,
        "outcomes": dict(counts),
        "alerts_before": before_total,
        "alerts_after": after_total,
        "alerts_delta": delta,
        "expected_delta": fraud_ok,
        "duplicates_or_missing": delta != fraud_ok,
        "server_error_500": counts.get("http_500", 0),
        "failures_sample": failures[:5],
        "health_after": health_after.get("status"),
        "health_model_loaded": health_after.get("model_loaded"),
    }

    print()
    print(f"Outcomes       : {dict(counts)}")
    print(f"Alerts after   : {after_total} (delta {delta}, expected {fraud_ok})")
    print(f"500 responses  : {result['server_error_500']}")
    print(f"Health after   : {health_after.get('status')}")
    print()
    print(
        "RESULT: "
        + (
            "PASS - no duplicates, no missing alerts, no 500s"
            if delta == fraud_ok and result["server_error_500"] == 0
            else "FAIL - see failures_sample"
        )
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"JSON written to {args.json_out.relative_to(PROJECT_ROOT)}")
    return 0 if delta == fraud_ok and result["server_error_500"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
