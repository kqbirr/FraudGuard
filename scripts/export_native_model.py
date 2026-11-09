"""Export the deployed Joblib bundle to XGBoost's native UBJ format.

Why this exists
---------------
``artifacts/models/fraud_model.joblib`` holds a pickled ``XGBClassifier``. Reading
it requires ``joblib`` *and* the whole scikit-learn/scipy stack to be installed on
the function platform. The estimator's underlying ``Booster`` can be written in
XGBoost's own UBJ format and read back with ``xgboost.Booster`` alone, which keeps
the deployed runtime small without touching the trained model.

Guarantees
----------
* the original Joblib bundle is only read, never written;
* all 269 boosted rounds are kept -- nothing is pruned, quantised or retrained;
* the decision threshold, the 30 feature names and their order, the training
  metadata and the random seed travel with the model as Booster attributes, so the
  runtime never has to unpickle anything to reconstruct them;
* ``best_iteration`` is recorded because ``XGBClassifier.predict_proba`` scores
  only trees ``[0, best_iteration + 1)`` once early stopping has been used. The
  earlier rounds are kept in the file (269 are saved) but the exported artifact
  reproduces the wrapper's decision range exactly;
* the written file is re-loaded and compared against ``predict_proba`` on the demo
  sample. The script fails loudly if a single probability differs.

Run from the project root, after ``scripts/train_models.py``::

    python scripts/export_native_model.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
MODELS_DIR: Final[Path] = PROJECT_ROOT / "artifacts" / "models"
SOURCE_PATH: Final[Path] = MODELS_DIR / "fraud_model.joblib"
TARGET_PATH: Final[Path] = MODELS_DIR / "fraud_model.ubj"
DEMO_PATH: Final[Path] = MODELS_DIR / "demo_transactions.csv"

#: Expected deployment values; a drift here means the wrong artifact was loaded.
EXPECTED_THRESHOLD: Final[float] = 0.5383065342903137
EXPECTED_ROUNDS: Final[int] = 269
EXPECTED_FEATURES: Final[int] = 30


def _demo_matrix(feature_names: list[str]) -> np.ndarray:
    """Load the demo sample in the model's exact feature order."""
    with DEMO_PATH.open(newline="", encoding="utf-8") as handle:
        rows = [[float(row[name]) for name in feature_names] for row in csv.DictReader(handle)]
    return np.array(rows, dtype=np.float64)


def main() -> int:
    import xgboost as xgb

    if not SOURCE_PATH.exists():
        print(f"error: {SOURCE_PATH} not found; run scripts/train_models.py first", file=sys.stderr)
        return 1

    artifact: dict[str, Any] = joblib.load(SOURCE_PATH)
    model = artifact["model"]
    feature_names = [str(name) for name in artifact["feature_names"]]
    threshold = float(artifact["decision_threshold"])

    booster = model.get_booster()
    rounds = booster.num_boosted_rounds()
    best_iteration = int(getattr(model, "best_iteration", rounds - 1))

    print(f"source            : {SOURCE_PATH.name} ({SOURCE_PATH.stat().st_size:,} bytes)")
    print(f"estimator         : {type(model).__module__}.{type(model).__name__}")
    print(f"boosted rounds    : {rounds} (best_iteration={best_iteration})")
    print(f"features          : {len(feature_names)} -> {feature_names[0]} .. {feature_names[-1]}")
    print(f"decision threshold: {threshold!r}")

    if rounds != EXPECTED_ROUNDS:
        print(f"error: expected {EXPECTED_ROUNDS} rounds, found {rounds}", file=sys.stderr)
        return 1
    if len(feature_names) != EXPECTED_FEATURES:
        print(f"error: expected {EXPECTED_FEATURES} features, found {len(feature_names)}", file=sys.stderr)
        return 1
    if abs(threshold - EXPECTED_THRESHOLD) > 1e-12:
        print(f"error: threshold drift (expected {EXPECTED_THRESHOLD!r})", file=sys.stderr)
        return 1

    # Everything in the bundle except the pickled estimator, plus what the native
    # runtime needs to reproduce predict_proba() semantics.
    metadata: dict[str, Any] = {key: value for key, value in artifact.items() if key != "model"}
    metadata["best_iteration"] = best_iteration
    metadata["booster_rounds"] = rounds
    metadata["native_format"] = "ubj"
    metadata["native_export_source"] = SOURCE_PATH.name
    metadata["decision_rule"] = "fraud_probability >= decision_threshold"

    booster.set_attr(
        fraudguard_bundle=json.dumps(metadata, sort_keys=True, default=str),
    )
    booster.save_model(str(TARGET_PATH))
    print(f"written           : {TARGET_PATH.name} ({TARGET_PATH.stat().st_size:,} bytes)")

    # --- verification: reload the written file and compare predictions ------- #
    reloaded = xgb.Booster()
    reloaded.load_model(str(TARGET_PATH))
    restored = json.loads(reloaded.attr("fraudguard_bundle") or "{}")

    demo = _demo_matrix(feature_names)
    expected = np.asarray(model.predict_proba(demo))[:, 1]
    actual = np.asarray(
        reloaded.predict(xgb.DMatrix(demo), iteration_range=(0, best_iteration + 1))
    )

    difference = np.abs(actual - expected)
    decisions_match = bool(
        np.array_equal(
            (actual >= threshold).astype(int),
            (expected >= threshold).astype(int),
        )
    )

    print()
    print(f"reloaded rounds    : {reloaded.num_boosted_rounds()}")
    print(f"reloaded features  : {reloaded.num_features()}")
    print(f"restored threshold : {restored.get('decision_threshold')!r}")
    print(f"restored features  : {len(restored.get('feature_names', []))}")
    print(f"demo rows          : {len(demo)}")
    print(f"max  abs diff      : {difference.max():.3e}")
    print(f"mean abs diff      : {difference.mean():.3e}")
    print(f"decisions identical: {decisions_match}")

    problems = []
    if not np.array_equal(actual, expected):
        problems.append("probabilities are not bit-identical")
    if not decisions_match:
        problems.append("fraud/legitimate decisions differ")
    if restored.get("feature_names") != feature_names:
        problems.append("feature names/order were not preserved")
    if reloaded.num_boosted_rounds() != rounds:
        problems.append("boosted round count changed")
    if abs(float(restored.get("decision_threshold", 0.0)) - threshold) > 1e-12:
        problems.append("threshold changed")

    if problems:
        print("\nexport FAILED: " + "; ".join(problems), file=sys.stderr)
        return 1

    print("\nexport OK: native artifact reproduces predict_proba() exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
