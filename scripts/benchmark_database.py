"""Measure fraud-alert insertion and retrieval separately, outside the HTTP stack.

Why a separate benchmark
------------------------
The predict benchmark shows HTTP latency that already contains a database write,
which makes the database's own cost invisible. This script times the persistence
layer directly so the two can be compared:

* **insert** — one ``persist_alert`` call, which adds, commits and refreshes, in a
  fresh session per call. That mirrors the per-request session the API uses.
* **retrieve** — one ``list_alerts`` call (count + one page), newest first.

All measurements are against a throwaway SQLite file under ``artifacts/`` by
default. The optional ``--neon-read-only`` flag runs three ``SELECT count(*)``
statements against ``DATABASE_URL`` and nothing else: no inserts, no schema
changes, no writes of any kind.

Usage::

    python scripts/benchmark_database.py --inserts 300 --reads 300
    python scripts/benchmark_database.py --neon-read-only
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from benchmark_predict import Summary, percentile  # same directory, no duplicate stats code

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:  # pragma: no cover - import path bootstrap
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import Base, configure_database, get_session_factory, mask_database_url  # noqa: E402
from app.models.flagged import FlaggedTransaction  # noqa: E402
from app.services.alerts import list_alerts, persist_alert  # noqa: E402

DEFAULT_DB = PROJECT_ROOT / "artifacts" / "performance" / "db_bench.sqlite"
DEFAULT_OUT = PROJECT_ROOT / "artifacts" / "performance" / "db_benchmark.json"


def timed(callable_: Any) -> tuple[Any, float]:
    started = time.perf_counter()
    result = callable_()
    return result, (time.perf_counter() - started) * 1000.0


def benchmark_inserts(factory: sessionmaker[Session], count: int) -> list[float]:
    samples: list[float] = []
    for index in range(count):
        def insert(index: int = index) -> None:
            with factory() as session:
                persist_alert(
                    session,
                    fraud_probability=0.9 + (index % 100) / 1000.0,
                    threshold=0.5383,
                    model_name="xgboost",
                    prediction_latency_ms=2.0,
                    transaction_time=float(index),
                    transaction_amount=float(index % 500),
                )

        _, elapsed = timed(insert)
        samples.append(elapsed)
    return samples


def benchmark_reads(factory: sessionmaker[Session], count: int, offset: int) -> list[float]:
    samples: list[float] = []
    for _ in range(count):
        def read() -> None:
            with factory() as session:
                list_alerts(session, limit=20, offset=offset)

        _, elapsed = timed(read)
        samples.append(elapsed)
    return samples


def neon_read_only(samples: int) -> dict[str, Any] | None:
    """Three read-only count queries against the configured database."""
    from app.config import settings
    from app.database import is_database_configured

    if not settings.database_url:
        print("Neon check skipped: DATABASE_URL is not configured.")
        return None

    configure_database(settings.database_url)
    if not is_database_configured():
        print("Neon check skipped: the configured engine did not initialise.")
        return None

    factory = get_session_factory()
    timings: list[float] = []
    total = 0
    for _ in range(samples):
        with factory() as session:
            def count_rows() -> int:
                return int(
                    session.scalar(select(func.count()).select_from(FlaggedTransaction)) or 0
                )

            rows, elapsed = timed(count_rows)
        total = rows
        timings.append(elapsed)

    return {
        "url": mask_database_url(settings.database_url),
        "statement": "SELECT count(*) FROM flagged_transactions",
        "writes": 0,
        "samples": len(timings),
        "rows": total,
        "timings_ms": [round(value, 2) for value in timings],
        "p50": round(percentile(sorted(timings), 50), 2),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inserts", type=int, default=300)
    parser.add_argument("--reads", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--neon-read-only",
        action="store_true",
        help="also run three read-only count queries against DATABASE_URL",
    )
    args = parser.parse_args(argv)

    args.db.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-journal", "-wal"):
        stale = Path(f"{args.db}{suffix}")
        if stale.exists():
            stale.unlink()

    engine = create_engine(
        f"sqlite:///{args.db.as_posix()}",
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    print(f"Database    : {args.db.relative_to(PROJECT_ROOT)} (SQLite, throwaway)")
    print(f"Workload    : {args.inserts} inserts, {args.reads} page reads, {args.warmup} warm-up each")

    # Warm-up: file page cache, statement cache, first-connection cost.
    for _ in range(args.warmup):
        with factory() as session:
            persist_alert(
                session,
                fraud_probability=0.99,
                threshold=0.5383,
                model_name="xgboost",
                prediction_latency_ms=1.0,
            )
    for _ in range(args.warmup):
        with factory() as session:
            list_alerts(session, limit=20, offset=0)

    insert_samples = benchmark_inserts(factory, args.inserts)

    with factory() as session:
        total_rows = int(
            session.scalar(select(func.count()).select_from(FlaggedTransaction)) or 0
        )

    page1_samples = benchmark_reads(factory, args.reads, offset=0)
    deep_offset = max(0, total_rows - 20)
    deep_samples = benchmark_reads(factory, max(10, args.reads // 4), offset=deep_offset)

    engine.dispose()

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database": str(args.db.relative_to(PROJECT_ROOT)),
        "engine": "sqlite",
        "rows_after_inserts": total_rows,
        "file_bytes": args.db.stat().st_size if args.db.exists() else 0,
        "insert_ms": asdict(Summary.from_samples(insert_samples)),
        "read_page1_ms": asdict(Summary.from_samples(page1_samples)),
        "read_deep_page_ms": asdict(
            Summary.from_samples(deep_samples)
        ),
        "deep_offset": deep_offset,
        "method": {
            "session_per_operation": True,
            "insert_includes_commit_and_refresh": True,
            "read_includes_count_plus_page": True,
            "single_process_single_thread": True,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    print()
    print(f"{'operation':<18}{'n':>5}{'min':>8}{'mean':>9}{'p50':>8}{'p95':>8}{'p99':>8}{'max':>8}")
    for key, label in (
        ("insert_ms", "insert alert"),
        ("read_page1_ms", "read page 1"),
        ("read_deep_page_ms", "read deep page"),
    ):
        s = result[key]
        print(
            f"{label:<18}{s['count']:>5}{s['minimum']:>8.2f}{s['mean']:>9.2f}"
            f"{s['p50']:>8.2f}{s['p95']:>8.2f}{s['p99']:>8.2f}{s['maximum']:>8.2f}"
        )
    print()
    print(f"Rows after inserts: {total_rows:,} | db file: {result['file_bytes'] / 1024:.0f} KiB")

    if args.neon_read_only:
        print()
        print("Neon read-only check (no writes):")
        neon = neon_read_only(samples=3)
        result["neon_read_only"] = neon
        if neon:
            print(
                f"  {neon['statement']} -> {neon['rows']:,} rows, "
                f"{neon['timings_ms']} ms over {neon['samples']} samples"
            )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print()
    print(f"JSON written to {args.json_out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
