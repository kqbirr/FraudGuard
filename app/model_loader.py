"""Load the trained FraudGuard model artifact for inference.

The Phase 3 training scripts persist a single ``.joblib`` bundle containing the
fitted estimator (with any preprocessing inside it), the decision threshold, the
canonical feature order and the training metadata. This module is the only place
that knows how to read that bundle, so Phase 4 endpoints can stay thin and the
tests can exercise exactly what production will load.

Nothing here trains, tunes or reads the test set.

Typical use::

    from app.model_loader import load_model

    model = load_model()
    score = model.score_frame(frame)      # higher = more likely fraudulent
    decision = model.predict_frame(frame) # 1 = fraud, 0 = legitimate
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Mapping

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

#: Where ``scripts/train_models.py`` writes the deployment artifact.
DEFAULT_MODEL_PATH: Final[Path] = PROJECT_ROOT / "artifacts" / "models" / "fraud_model.joblib"

REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "artifact_version",
    "model_name",
    "display_name",
    "model",
    "decision_threshold",
    "feature_names",
    "random_seed",
    "training_metadata",
)


class ModelArtifactError(RuntimeError):
    """Raised when the model artifact is missing, malformed or misused."""


@dataclass(frozen=True)
class LoadedModel:
    """A trained model plus everything needed to turn scores into decisions."""

    model: Any
    model_name: str
    display_name: str
    decision_threshold: float
    feature_names: tuple[str, ...]
    metadata: Mapping[str, Any]

    # -- feature handling --------------------------------------------------- #

    def _feature_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """Select and order columns, then convert to a float64 matrix.

        Column order is enforced here rather than trusted from the caller: a
        model fed the same numbers in a different order produces confident
        nonsense, so ordering is a correctness requirement, not a nicety.
        """
        if frame.empty:
            raise ModelArtifactError("Input frame is empty.")

        missing = [name for name in self.feature_names if name not in frame.columns]
        if missing:
            raise ModelArtifactError(
                f"Input is missing required feature(s): {missing}. "
                f"Expected {len(self.feature_names)} features."
            )

        selected = frame.loc[:, list(self.feature_names)]
        try:
            return selected.to_numpy(dtype=np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise ModelArtifactError(
                f"Input contains non-numeric values: {exc}"
            ) from exc

    # -- scoring ------------------------------------------------------------ #

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Return fraud scores (higher = more likely fraudulent)."""
        matrix = self._feature_matrix(frame)
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(matrix))[:, 1]
        # Unsupervised path: score_samples is negative for anomalous rows.
        return -np.asarray(self.model.score_samples(matrix))

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Return hard decisions using the artifact's saved threshold."""
        return (self.score_frame(frame) >= self.decision_threshold).astype(int)

    def score_row(self, features: Mapping[str, float]) -> float:
        """Score a single transaction given as a feature mapping."""
        return float(self.score_frame(pd.DataFrame([features]))[0])

    def predict_row(self, features: Mapping[str, float]) -> tuple[int, float]:
        """Return ``(decision, score)`` for a single transaction."""
        score = self.score_row(features)
        return int(score >= self.decision_threshold), score


def resolve_model_path(path: str | Path | None = None) -> Path:
    """Resolve the artifact path from an argument, ``MODEL_PATH`` or the default."""
    candidate = path if path is not None else os.getenv("MODEL_PATH")
    if not candidate:
        return DEFAULT_MODEL_PATH

    resolved = Path(candidate)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved


def load_model_artifact(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the raw artifact dictionary."""
    resolved = resolve_model_path(path)

    if not resolved.exists():
        raise ModelArtifactError(
            f"Model artifact not found at {resolved}. "
            "Run 'python scripts/train_models.py' to create it."
        )

    try:
        artifact = joblib.load(resolved)
    except Exception as exc:  # noqa: BLE001 - surface any deserialisation failure
        raise ModelArtifactError(f"Could not load artifact {resolved}: {exc}") from exc

    if not isinstance(artifact, dict):
        raise ModelArtifactError(
            f"Artifact {resolved} is a {type(artifact).__name__}, expected a dict bundle."
        )

    missing = [key for key in REQUIRED_KEYS if key not in artifact]
    if missing:
        raise ModelArtifactError(f"Artifact {resolved} is missing keys: {missing}")

    threshold = artifact["decision_threshold"]
    if not isinstance(threshold, (int, float)) or not (0.0 <= float(threshold)):
        raise ModelArtifactError(f"Invalid decision_threshold: {threshold!r}")

    feature_names = artifact["feature_names"]
    if not isinstance(feature_names, (list, tuple)) or not feature_names:
        raise ModelArtifactError("Artifact feature_names must be a non-empty sequence.")

    return artifact


@lru_cache(maxsize=4)
def load_model(path: str | Path | None = None) -> LoadedModel:
    """Load the artifact once per path and wrap it in :class:`LoadedModel`."""
    artifact = load_model_artifact(path)
    return LoadedModel(
        model=artifact["model"],
        model_name=str(artifact["model_name"]),
        display_name=str(artifact["display_name"]),
        decision_threshold=float(artifact["decision_threshold"]),
        feature_names=tuple(str(name) for name in artifact["feature_names"]),
        metadata=artifact.get("training_metadata", {}),
    )


def clear_model_cache() -> None:
    """Drop the cached artifact (used by tests and after retraining)."""
    load_model.cache_clear()


__all__ = [
    "DEFAULT_MODEL_PATH",
    "LoadedModel",
    "ModelArtifactError",
    "clear_model_cache",
    "load_model",
    "load_model_artifact",
    "resolve_model_path",
]
