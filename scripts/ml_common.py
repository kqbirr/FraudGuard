"""Shared helpers for the FraudGuard Phase 3 modelling pipeline.

Used by both ``scripts/train_models.py`` and ``scripts/evaluate_models.py`` so
that the two scripts cannot drift apart: one definition of the feature list, the
grouping rule, the split, the metrics and the threshold rule.

Design notes
------------
* Features are passed to estimators as plain ``float64`` NumPy arrays in a fixed
  order (``FEATURE_NAMES``). That order is persisted inside every saved artifact,
  which is what makes the artifact safe to reuse from FastAPI later.
* Higher score always means "more likely fraudulent", for every model. For the
  Isolation Forest that means the score is ``-score_samples``; for the supervised
  models it is ``predict_proba[:, 1]``. A single threshold rule can then be
  applied to all three.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
MODELS_DIR: Final[Path] = PROJECT_ROOT / "artifacts" / "models"
RESULTS_DIR: Final[Path] = PROJECT_ROOT / "artifacts" / "model_results"
THRESHOLD_DIR: Final[Path] = RESULTS_DIR / "validation_thresholds"

# --------------------------------------------------------------------------- #
# Schema constants (must match the Kaggle CSV exactly)
# --------------------------------------------------------------------------- #

TARGET: Final[str] = "Class"
LEGIT: Final[int] = 0
FRAUD: Final[int] = 1

PCA_FEATURES: Final[list[str]] = [f"V{i}" for i in range(1, 29)]

#: Feature order used for training. Persisted in every artifact and enforced by
#: ``app.model_loader`` so column order at inference cannot silently change.
FEATURE_NAMES: Final[list[str]] = ["Time", *PCA_FEATURES, "Amount"]

GROUP_COL: Final[str] = "_group_id"

# --------------------------------------------------------------------------- #
# Splitting constants
# --------------------------------------------------------------------------- #

RANDOM_SEED: Final[int] = 42
TRAIN_FRAC: Final[float] = 0.70
VAL_FRAC: Final[float] = 0.15
TEST_FRAC: Final[float] = 0.15

#: Pre-declared operating-point rule, fixed before any model was fitted so the
#: threshold choice cannot be cherry-picked after seeing results:
#: maximise recall among thresholds whose precision is at least this value.
#: If no threshold reaches it, fall back to the best F1 threshold.
MIN_PRECISION_TARGET: Final[float] = 0.80

SPLIT_NAMES: Final[tuple[str, str, str]] = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Dataset loading and duplicate handling
# --------------------------------------------------------------------------- #


def load_raw_dataframe(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV and verify it is the expected file.

    The raw file is opened read-only and is never written to by any script.
    """
    if not path.exists():
        raise SystemExit(
            "\nERROR: dataset not found.\n"
            f"  Expected at: {path}\n"
            "  Download 'creditcard.csv' from Kaggle (mlg-ulb/creditcardfraud)\n"
            "  and place it at data/raw/creditcard.csv, then re-run.\n"
        )

    frame = pd.read_csv(path)

    expected = {*FEATURE_NAMES, TARGET}
    missing = expected - set(frame.columns)
    if missing:
        raise SystemExit(f"ERROR: dataset is missing columns: {sorted(missing)}")
    if frame.empty:
        raise SystemExit("ERROR: dataset loaded but is empty.")

    return frame


def assign_group_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach a ``_group_id`` identifying exactly-identical transactions.

    Two rows get the same id only when all 30 features *and* the label are
    identical. Phase 2 found 1,081 such excess copies across 773 groups and
    verified that **no group contains both classes**, so grouping never mixes
    labels.

    Why group instead of dropping duplicates: identical rows carry no extra
    information, but a repeated record may still be a genuine second transaction.
    Grouping keeps every legitimate and every fraud row (nothing is deleted)
    while guaranteeing that copies cannot straddle the train/val/test boundary.
    """
    grouped = frame.groupby([*FEATURE_NAMES, TARGET], sort=False, dropna=False)
    return frame.assign(**{GROUP_COL: grouped.ngroup().astype(np.int64)})


def group_stratified_split(
    frame: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> tuple[dict[str, np.ndarray], pd.Series]:
    """Split **groups** (not rows) 70/15/15, stratified by the group's label.

    Splitting whole groups means every copy of a duplicated transaction lands in
    the same split, so the test set cannot be contaminated by rows the model has
    already memorised from training.

    Returns positional index arrays per split and the per-group split label.
    """
    if GROUP_COL not in frame.columns:
        raise ValueError("assign_group_ids() must be called before splitting")

    group_label = frame.groupby(GROUP_COL, sort=True)[TARGET].first()
    rng = np.random.default_rng(seed)
    assignment: dict[int, str] = {}

    for label in (LEGIT, FRAUD):
        # .to_numpy(copy=True): pandas 3 index arrays are read-only, and
        # rng.shuffle() permutes in place, so a writable copy is required.
        group_ids = group_label.index[group_label == label].to_numpy(copy=True)
        rng.shuffle(group_ids)

        total = len(group_ids)
        if total < 3:
            raise SystemExit(f"ERROR: class {label} has too few groups to split.")

        n_train = int(round(total * TRAIN_FRAC))
        n_val = int(round(total * VAL_FRAC))
        # Guarantee at least one group in each split for every class.
        n_train = max(min(n_train, total - 2), 1)
        n_val = max(min(n_val, total - n_train - 1), 1)

        for group_id in group_ids[:n_train]:
            assignment[int(group_id)] = "train"
        for group_id in group_ids[n_train : n_train + n_val]:
            assignment[int(group_id)] = "val"
        for group_id in group_ids[n_train + n_val :]:
            assignment[int(group_id)] = "test"

    row_split = frame[GROUP_COL].map(assignment)
    if row_split.isna().any():
        raise RuntimeError("Internal error: some rows were not assigned to a split.")

    indices = {
        name: np.flatnonzero(row_split.to_numpy() == name) for name in SPLIT_NAMES
    }
    if sum(len(idx) for idx in indices.values()) != len(frame):
        raise RuntimeError("Internal error: split sizes do not sum to the row count.")

    return indices, row_split


def assert_no_group_leakage(
    frame: pd.DataFrame,
    indices: dict[str, np.ndarray],
) -> None:
    """Raise if any duplicate group appears in more than one split."""
    seen: dict[int, str] = {}
    collisions: list[tuple[int, str, str]] = []
    group_values = frame[GROUP_COL].to_numpy()

    for name in SPLIT_NAMES:
        for group_id in np.unique(group_values[indices[name]]):
            previous = seen.get(int(group_id))
            if previous is None:
                seen[int(group_id)] = name
            elif previous != name:
                collisions.append((int(group_id), previous, name))

    if collisions:
        raise RuntimeError(
            f"Group leakage detected: {len(collisions)} duplicate groups span splits "
            f"(first: {collisions[0]})"
        )


def split_class_counts(frame: pd.DataFrame, index: np.ndarray) -> dict[str, int]:
    """Return row counts by class for a set of positional indices."""
    labels = frame[TARGET].to_numpy()[index]
    return {
        "rows": int(len(index)),
        "legitimate": int((labels == LEGIT).sum()),
        "fraudulent": int((labels == FRAUD).sum()),
    }


# --------------------------------------------------------------------------- #
# Matrices and scoring
# --------------------------------------------------------------------------- #


def to_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Return the model matrix in the canonical feature order.

    Column order matters: a pipeline fitted on one ordering silently produces
    garbage if fed another, so the order lives in one place and is saved with
    each artifact.
    """
    missing = [name for name in FEATURE_NAMES if name not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")
    return frame.loc[:, FEATURE_NAMES].to_numpy(dtype=np.float64, copy=False)


def fraud_scores(model: Any, matrix: np.ndarray) -> np.ndarray:
    """Return a score where **higher means more likely fraudulent**."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix))[:, 1]
    # Unsupervised path (IsolationForest): score_samples returns the (negative)
    # anomaly score, so negating turns it into "higher = more anomalous".
    return -np.asarray(model.score_samples(matrix))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    """Compute the metrics this phase reports. Zero-division safe."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[LEGIT, FRAUD]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def build_threshold_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    min_precision: float = MIN_PRECISION_TARGET,
) -> pd.DataFrame:
    """One row per candidate threshold taken from the validation PR curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # precision_recall_curve returns len(thresholds) == len(precision) - 1.
    table = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
        }
    )
    table = table[np.isfinite(table["threshold"])].copy()
    table["f1"] = (
        2 * table["precision"] * table["recall"]
        / (table["precision"] + table["recall"]).replace(0, np.nan)
    ).fillna(0.0)

    positives = int((y_true == FRAUD).sum())
    table["flagged"] = (y_score[None, :] >= table["threshold"].to_numpy()[:, None]).sum(axis=1)
    table["true_positives"] = (table["recall"] * positives).round().astype(int)
    table["false_positives"] = table["flagged"] - table["true_positives"]
    table["false_negatives"] = positives - table["true_positives"]
    table["meets_precision_target"] = table["precision"] >= min_precision
    return table.sort_values("threshold").reset_index(drop=True)


def select_threshold(
    table: pd.DataFrame,
    min_precision: float = MIN_PRECISION_TARGET,
) -> tuple[float, str]:
    """Apply the pre-declared operating-point rule.

    Rule (fixed before training, so it is not fitted to the results):
      1. Among thresholds with ``precision >= min_precision``, take the highest
         recall (tie-break: higher precision, then higher F1).
      2. If none qualifies, take the highest F1 (tie-break: higher recall).

    Rationale: a missed fraud is a direct financial loss, while a false positive
    is a blocked customer and an analyst's time. Prioritising recall *subject to*
    a precision floor states that trade-off explicitly instead of assuming 0.5.
    """
    qualifying = table[table["meets_precision_target"]]
    if not qualifying.empty:
        best = qualifying.sort_values(
            ["recall", "precision", "f1"], ascending=False
        ).iloc[0]
        reason = (
            f"highest recall ({best['recall']:.4f}) among thresholds with "
            f"precision >= {min_precision:.2f}"
        )
        return float(best["threshold"]), reason

    best = table.sort_values(["f1", "recall"], ascending=False).iloc[0]
    reason = (
        f"no threshold reached precision >= {min_precision:.2f}; fell back to the "
        f"highest F1 ({best['f1']:.4f})"
    )
    return float(best["threshold"]), reason


def metrics_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Metrics for one specific threshold (prediction = score >= threshold)."""
    return compute_metrics(y_true, (y_score >= threshold).astype(int), y_score)


# --------------------------------------------------------------------------- #
# Persistence helpers
# --------------------------------------------------------------------------- #


def utc_now() -> str:
    """ISO-8601 UTC timestamp for artifact metadata."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(payload: dict[str, Any], path: Path) -> Path:
    """Write indented JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def ensure_directories() -> None:
    """Create the artifact directories used by this phase."""
    for directory in (MODELS_DIR, RESULTS_DIR, THRESHOLD_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    """Print a delimited console header."""
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
