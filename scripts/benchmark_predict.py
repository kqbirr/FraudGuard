"""Benchmark ``POST /predict`` and report model-scoring and HTTP latency separately.

Two different numbers come out of every request, and confusing them is the usual
way a "model latency" claim turns out to be wrong:

* **model scoring ms** — ``latency_ms`` in the response body, measured by the
  server around the model call only. It excludes JSON parsing, dependency
  resolution and any database work.
* **HTTP total ms** — wall-clock time measured by this client around the whole
  request. It includes TCP/keep-alive reuse, request serialisation, the full
  server handler (scoring *and* the alert insert for fraud results), and response
  parsing.

Both are reported with min/mean/p50/p95/p99/max.

Usage
-----
Point this at a running API. For benchmarks, start one bound to a scratch SQLite
database so production data is untouched::

    $env:DATABASE_URL = "sqlite:///./artifacts/performance/bench.db"
    python scripts/init_db.py
    python -m uvicorn app.main:app --port 8010

    python scripts/benchmark_predict.py --base-url http://127.0.0.1:8010

The script sends the 30 features only (the ``Class`` label is not in the bundled
demo JSON), performs warm-up requests before measuring, and verifies that the
sample rows still produce the decisions it expects.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_JSON = PROJECT_ROOT / "frontend" / "src" / "data" / "demoTransactions.json"
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "performance" / "predict_benchmark.json"

#: Rows chosen for their observed model decisions, verified again at run time.
LEGIT_ROW_ID = "demo-01"
FRAUD_ROW_ID = "demo-12"


@dataclass(frozen=True)
class Summary:
    """Distribution of one measured quantity, in milliseconds."""

    count: int
    minimum: float
    mean: float
    p50: float
    p95: float
    p99: float
    maximum: float

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> Summary:
        ordered = sorted(samples)
        return cls(
            count=len(ordered),
            minimum=ordered[0],
            mean=statistics.fmean(ordered),
            p50=percentile(ordered, 50),
            p95=percentile(ordered, 95),
            p99=percentile(ordered, 99),
            maximum=ordered[-1],
        )


def percentile(ordered: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile of an already sorted sequence.

    Implemented locally to keep the script dependant on nothing but httpx: a
    benchmark that needs numpy installed is one more thing to get wrong on a
    fresh machine.
    """
    if not ordered:
        raise ValueError("cannot take a percentile of no samples")
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]

    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def load_demo_rows() -> dict[str, dict[str, float]]:
    payload = json.loads(DEMO_JSON.read_text(encoding="utf-8"))
    return {row["id"]: row["features"] for row in payload["rows"]}


def benchmark_group(
    client: httpx.Client,
    url: str,
    features: dict[str, float],
    *,
    label: str,
    expected_decision: str,
    requests: int,
    warmup: int,
) -> dict[str, Any]:
    """Warm up, then time ``requests`` sequential calls for one sample row."""

    for index in range(warmup):
        response = client.post(url, json=features)
        if response.status_code != 200:
            raise SystemExit(
                f"warm-up request {index + 1} failed with HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
        decision = response.json()["decision"]
        if decision != expected_decision:
            raise SystemExit(
                f"{label} row now decides '{decision}', expected '{expected_decision}'. "
                "The model artifact or threshold changed; update the benchmark rows."
            )

    http_samples: list[float] = []
    model_samples: list[float] = []
    status_counts: dict[str, int] = {}
    decisions: dict[str, int] = {}

    for _ in range(requests):
        started = time.perf_counter()
        response = client.post(url, json=features)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        status_counts[str(response.status_code)] = status_counts.get(str(response.status_code), 0) + 1
        if response.status_code != 200:
            continue

        body = response.json()
        decisions[body["decision"]] = decisions.get(body["decision"], 0) + 1
        http_samples.append(elapsed_ms)
        model_samples.append(float(body["latency_ms"]))

    if not http_samples:
        raise SystemExit(f"every {label} request failed: {status_counts}")

    return {
        "label": label,
        "row_id": None,
        "requests_sent": requests,
        "requests_ok": len(http_samples),
        "status_counts": status_counts,
        "decisions": decisions,
        "model_scoring_ms": asdict(Summary.from_samples(model_samples)),
        "http_total_ms": asdict(Summary.from_samples(http_samples)),
    }


def format_table(groups: Sequence[dict[str, Any]]) -> str:
    header = (
        f"{'group':<12}{'n':>5}{'min':>9}{'mean':>9}{'p50':>9}"
        f"{'p95':>9}{'p99':>9}{'max':>9}"
    )
    lines = [header, "-" * len(header)]

    for group in groups:
        for metric_key, title in (
            ("model_scoring_ms", "model"),
            ("http_total_ms", "http"),
        ):
            stats = group[metric_key]
            name = f"{group['label'][:6]}/{title}"
            lines.append(
                f"{name:<12}{stats['count']:>5}{stats['minimum']:>9.2f}{stats['mean']:>9.2f}"
                f"{stats['p50']:>9.2f}{stats['p95']:>9.2f}{stats['p99']:>9.2f}{stats['maximum']:>9.2f}"
            )

    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8010")
    parser.add_argument("--requests", type=int, default=300, help="measured requests per row")
    parser.add_argument("--warmup", type=int, default=30, help="discarded requests per row")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    rows = load_demo_rows()
    for row_id in (LEGIT_ROW_ID, FRAUD_ROW_ID):
        if row_id not in rows:
            print(f"ERROR: {row_id} is missing from {DEMO_JSON.relative_to(PROJECT_ROOT)}.")
            return 1

    url = f"{args.base_url.rstrip('/')}/predict"
    print(f"Target      : {url}")
    print(f"Requests    : {args.requests} measured + {args.warmup} warm-up per row")
    print("Timing split: model scoring = response latency_ms; http total = client wall clock")
    print()

    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=args.timeout) as client:
        health = client.get(f"{args.base_url.rstrip('/')}/health")
        health.raise_for_status()
        print(f"Health      : {health.json()['status']} ({health.json()['model_name']})")
        print()

        for label, row_id, expected in (
            ("legitimate", LEGIT_ROW_ID, "legitimate"),
            ("fraud", FRAUD_ROW_ID, "fraud"),
        ):
            group = benchmark_group(
                client,
                url,
                rows[row_id],
                label=label,
                expected_decision=expected,
                requests=args.requests,
                warmup=args.warmup,
            )
            group["row_id"] = row_id
            results.append(group)
            print(f"completed {label} group ({row_id})")

    print()
    print(format_table(results))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": url,
        "method": {
            "sequential_requests": True,
            "connection_reused": True,
            "warmup_per_group": args.warmup,
            "measured_per_group": args.requests,
            "model_ms_source": "latency_ms field in the POST /predict response",
            "http_ms_source": "client-side wall clock around the request",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count(),
        },
        "groups": results,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"JSON written to {args.json_out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
