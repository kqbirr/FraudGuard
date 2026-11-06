"""Tests for the Phase 3 model artifacts and the inference path.

These tests are skipped automatically if the training script has not been run,
so a fresh clone can still run ``pytest`` without a 143 MB dataset present.

Run from the project root::

    python -m pytest tests/test_model_pipeline.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from app.model_loader import (
    DEFAULT_MODEL_PATH,
    LoadedModel,
    ModelArtifactError,
    clear_model_cache,
    load_model,
    load_model_artifact,
    resolve_model_path,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "artifacts" / "models"
RESULTS_DIR = PROJECT_ROOT / "artifacts" / "model_results"
DEMO_PATH = MODELS_DIR / "demo_transactions.csv"
SPLIT_PATH = MODELS_DIR / "split_indices.npz"

CANONICAL_FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
MODEL_FILES = (
    "logistic_regression.joblib",
    "isolation_forest.joblib",
    "xgboost.joblib",
    "fraud_model.joblib",
)

requires_artifacts = pytest.mark.skipif(
    not DEFAULT_MODEL_PATH.exists(),
    reason="model artifacts missing - run 'python scripts/train_models.py' first",
)
requires_demo = pytest.mark.skipif(
    not DEMO_PATH.exists(),
    reason="demo sample missing - run 'python scripts/train_models.py' first",
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    """The raw deployment artifact dictionary."""
    return load_model_artifact()


@pytest.fixture(scope="module")
def model() -> LoadedModel:
    """The deployment model wrapped for inference."""
    clear_model_cache()
    return load_model()


@pytest.fixture(scope="module")
def demo_frame() -> pd.DataFrame:
    """The small sanitised demo sample of unseen transactions."""
    return pd.read_csv(DEMO_PATH)


# --------------------------------------------------------------------------- #
# Artifact contents
# --------------------------------------------------------------------------- #


@requires_artifacts
def test_artifact_has_required_keys(artifact: dict) -> None:
    for key in (
        "artifact_version",
        "model_name",
        "display_name",
        "model",
        "decision_threshold",
        "feature_names",
        "random_seed",
        "training_metadata",
    ):
        assert key in artifact, f"artifact is missing {key!r}"


@requires_artifacts
def test_feature_order_is_canonical(artifact: dict) -> None:
    assert artifact["feature_names"] == CANONICAL_FEATURES
    assert len(artifact["feature_names"]) == 30


@requires_artifacts
def test_threshold_is_usable(artifact: dict) -> None:
    threshold = artifact["decision_threshold"]
    assert isinstance(threshold, float)
    assert 0.0 <= threshold <= 50.0, "threshold should be a plausible score cut"


@requires_artifacts
def test_training_metadata_is_recorded(artifact: dict) -> None:
    metadata = artifact["training_metadata"]
    assert metadata["duplicates"]["handling"].startswith("grouped")
    assert metadata["resampling_applied"] is False
    assert metadata["test_set_used"] is False, "artifact must not claim test-set fit"


@requires_artifacts
def test_random_seed_is_recorded(artifact: dict) -> None:
    assert artifact["random_seed"] == 42


@requires_artifacts
def test_score_direction_is_documented(artifact: dict) -> None:
    assert artifact["score_direction"] == "higher_is_more_fraudulent"


@pytest.mark.parametrize("filename", MODEL_FILES)
def test_each_saved_model_loads(filename: str) -> None:
    path = MODELS_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not produced yet")
    loaded = joblib.load(path)
    assert isinstance(loaded, dict)
    assert hasattr(loaded["model"], "predict") or hasattr(loaded["model"], "score_samples")


# --------------------------------------------------------------------------- #
# Inference path
# --------------------------------------------------------------------------- #


@requires_artifacts
def test_loader_returns_loaded_model(model: LoadedModel) -> None:
    assert isinstance(model, LoadedModel)
    assert model.feature_names == tuple(CANONICAL_FEATURES)
    assert model.model_name in {"logistic_regression", "isolation_forest", "xgboost"}


@requires_artifacts
@requires_demo
def test_demo_sample_can_be_scored(model: LoadedModel, demo_frame: pd.DataFrame) -> None:
    scores = model.score_frame(demo_frame)
    decisions = model.predict_frame(demo_frame)

    assert scores.shape == (len(demo_frame),)
    assert np.isfinite(scores).all(), "scores must be finite"
    assert set(np.unique(decisions)).issubset({0, 1})


@requires_artifacts
@requires_demo
def test_demo_sample_contains_both_classes(demo_frame: pd.DataFrame) -> None:
    assert "Class" in demo_frame.columns
    assert demo_frame["Class"].nunique() == 2, "demo should show both legitimate and fraud"
    assert len(demo_frame) <= 50, "demo sample must stay small"


@requires_artifacts
@requires_demo
def test_column_order_does_not_change_scores(
    model: LoadedModel, demo_frame: pd.DataFrame
) -> None:
    """Shuffled input columns must yield identical scores."""
    shuffled = demo_frame[list(reversed(list(demo_frame.columns)))]
    assert list(shuffled.columns) != list(demo_frame.columns)

    expected = model.score_frame(demo_frame)
    actual = model.score_frame(shuffled)
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


@requires_artifacts
@requires_demo
def test_extra_columns_are_ignored(model: LoadedModel, demo_frame: pd.DataFrame) -> None:
    """A caller may pass the label or extra metadata; scoring must still work."""
    frame = demo_frame.copy()
    frame["unexpected_column"] = "extra"
    scores = model.score_frame(frame)
    assert scores.shape == (len(frame),)


@requires_artifacts
@requires_demo
def test_missing_feature_raises(model: LoadedModel, demo_frame: pd.DataFrame) -> None:
    broken = demo_frame.drop(columns=["Amount"])
    with pytest.raises(ModelArtifactError, match="missing required feature"):
        model.score_frame(broken)


@requires_artifacts
@requires_demo
def test_single_row_helpers(model: LoadedModel, demo_frame: pd.DataFrame) -> None:
    row = demo_frame.iloc[0]
    features = {name: float(row[name]) for name in CANONICAL_FEATURES}

    decision, score = model.predict_row(features)
    assert decision in (0, 1)
    assert np.isfinite(score)
    assert model.score_row(features) == pytest.approx(score)


@requires_artifacts
def test_cache_can_be_cleared() -> None:
    clear_model_cache()
    first = load_model()
    clear_model_cache()
    second = load_model()
    assert first.model_name == second.model_name


def test_missing_artifact_raises_model_artifact_error() -> None:
    with pytest.raises(ModelArtifactError, match="not found"):
        load_model_artifact(Path("artifacts/models/does_not_exist.joblib"))


def test_resolve_model_path_defaults_to_models_dir() -> None:
    resolved = resolve_model_path()
    assert resolved.name == "fraud_model.joblib"
    assert resolved.parent.name == "models"


# --------------------------------------------------------------------------- #
# Split integrity
# --------------------------------------------------------------------------- #


@requires_artifacts
def test_split_indices_are_disjoint() -> None:
    if not SPLIT_PATH.exists():
        pytest.skip("split_indices.npz not produced yet")

    saved = np.load(SPLIT_PATH)
    train, val, test = saved["train"], saved["val"], saved["test"]

    assert len(np.intersect1d(train, val)) == 0
    assert len(np.intersect1d(train, test)) == 0
    assert len(np.intersect1d(val, test)) == 0
    assert len(np.unique(np.concatenate([train, val, test]))) == (
        len(train) + len(val) + len(test)
    )


@requires_artifacts
def test_split_fractions_are_approximately_seventy_fifteen_fifteen() -> None:
    if not SPLIT_PATH.exists():
        pytest.skip("split_indices.npz not produced yet")

    saved = np.load(SPLIT_PATH)
    total = len(saved["train"]) + len(saved["val"]) + len(saved["test"])
    assert len(saved["train"]) / total == pytest.approx(0.70, abs=0.01)
    assert len(saved["val"]) / total == pytest.approx(0.15, abs=0.01)
    assert len(saved["test"]) / total == pytest.approx(0.15, abs=0.01)


# --------------------------------------------------------------------------- #
# Results bookkeeping
# --------------------------------------------------------------------------- #


def test_test_results_cover_all_three_models() -> None:
    path = RESULTS_DIR / "test_comparison.csv"
    if not path.exists():
        pytest.skip("run scripts/evaluate_models.py first")

    comparison = pd.read_csv(path)
    assert set(comparison["model"]) == {
        "Logistic Regression",
        "Isolation Forest",
        "XGBoost",
    }
    for column in ("precision", "recall", "f1", "pr_auc", "false_positives", "false_negatives"):
        assert column in comparison.columns
    assert comparison[["precision", "recall", "f1", "pr_auc"]].to_numpy().max() <= 1.0


def test_training_metadata_matches_results() -> None:
    metadata_path = MODELS_DIR / "training_metadata.json"
    if not metadata_path.exists():
        pytest.skip("run scripts/train_models.py first")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["test_split_evaluated"] is False
    assert metadata["resampling_applied"] is False
    assert metadata["selected_model"] in {
        "Logistic Regression",
        "Isolation Forest",
        "XGBoost",
    }
