"""Export the sanitised demo sample as a label-free JSON bundle for the dashboard.

Why this exists
---------------
``artifacts/models/demo_transactions.csv`` keeps the Kaggle ``Class`` column,
because Phase 3 used it to report how many demo rows each model caught. The
browser must never receive that label:

* the prediction contract has no label - sending one would be a leak, and the API
  rejects it outright (``extra="forbid"``);
* a dashboard that knows the ground truth invites "the model missed this" claims
  from a 12-row sample, which is exactly the kind of invented conclusion the demo
  panel is supposed to avoid.

So replay payloads are built here: the 30 features, in the model's exact order,
and nothing else. The 150 MB raw Kaggle CSV is never read by this script and is
never bundled.

Run after retraining (the demo sample is regenerated then):

    python scripts/export_demo_transactions.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

import ml_common as mc

#: Written into the dashboard's source tree and bundled by Vite.
OUTPUT_PATH: Path = mc.PROJECT_ROOT / "frontend" / "src" / "data" / "demoTransactions.json"


def main() -> int:
    source = mc.MODELS_DIR / "demo_transactions.csv"

    if not source.exists():
        print(f"ERROR: demo sample not found at {source.relative_to(mc.PROJECT_ROOT)}.")
        print("       It is produced by scripts/train_models.py; run Phase 3 first.")
        return 1

    frame = pd.read_csv(source)

    expected = [*mc.FEATURE_NAMES, mc.TARGET]
    if list(frame.columns) != expected:
        print("ERROR: demo sample columns do not match the model feature contract.")
        print(f"  expected 30 features + label: {expected}")
        print(f"  found:                        {list(frame.columns)}")
        return 1

    rows: list[dict[str, object]] = []
    for position, record in enumerate(frame.to_dict(orient="records"), start=1):
        features: dict[str, float] = {}
        for name in mc.FEATURE_NAMES:
            value = float(record[name])
            if not math.isfinite(value):
                print(f"ERROR: non-finite value in row {position}, feature {name}.")
                return 1
            features[name] = value

        # The label is deliberately dropped here, not later.
        rows.append({"id": f"demo-{position:02d}", "features": features})

    payload = {
        "source": source.relative_to(mc.PROJECT_ROOT).as_posix(),
        "featureCount": len(mc.FEATURE_NAMES),
        "note": (
            "Sanitised rows drawn from the held-out test split. Only the 30 model "
            "features are included; the Class label was stripped at export time."
        ),
        "rows": rows,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"Demo sample    : {source.relative_to(mc.PROJECT_ROOT)}")
    print(f"Rows exported  : {len(rows)} (labels stripped)")
    print(f"Feature count  : {len(mc.FEATURE_NAMES)}")
    print(f"Output         : {OUTPUT_PATH.relative_to(mc.PROJECT_ROOT)} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
