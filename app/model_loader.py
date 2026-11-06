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
    """A trained model plus everything needed to turn scores into decisions.

    Inputs are deliberately agnostic of ``pandas``: the production API already
    validates payloads as mappings (``TransactionFeatures``), and the scorer here
    accepts a single mapping, a sequence of mappings, or a float64 matrix. The
    original training artifact is a ``joblib`` bundle that still requires
    ``xgboost`` (for the pickled estimator) and ``numpy`` (for the array math).
    """

    model: Any
    model_name: str
    display_name: str
    decision_threshold: float
    feature_names: tuple[str, ...]
    metadata: Mapping[str, Any]

    # -- feature handling --------------------------------------------------- #

    def _feature_matrix(
        self, payload: Any
    ) -> np.ndarray:
        """Normalize and order input into a float64 matrix.

        Accepted shapes:

        * a single feature mapping (``dict[str, float]`` or anything with a
          ``keys``/``__getitem__`` interface) — used by the HTTP ``/predict``
          path;
        * a sequence of such mappings — used by batch cooks and tests;
        * a ``(n, 30)`` float matrix — used by model-pipeline tests that pass a
          NumPy array directly.

        Column order is enforced here rather than trusted from the caller: a
        model fed the same numbers in a different order produces confident
        nonsense, so ordering is a correctness requirement, not a nicety.
        """
        if isinstance(payload, np.ndarray):
            matrix = payload
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            if matrix.ndim != 2:
                raise ModelArtifactError(
                    "Numeric input must be 1-D or 2-D."
                )
            if matrix.shape[1] != len(self.feature_names):
                raise ModelArtifactError(
                    f"Input has {matrix.shape[1]} feature(s), expected "
                    f"{len(self.feature_names)}."
                )
            try:
                return matrix.astype(np.float64, copy=False)
            except (TypeError, ValueError) as exc:
                raise ModelArtifactError(
                    f"Input contains non-numeric values: {exc}"
                ) from exc

        # DataFrame duck-type (pandas, polars-like, or test doubles).
        # We deliberately do not import ``pandas`` at module level: the deployed
        # runtime no longer ships it. Anything exposing ``columns``, ``loc`` and
        # ``to_numpy`` is treated as a DataFrame and scored column-wise.
        if (
            hasattr(payload, "columns")
            and callable(getattr(payload, "loc", None))
            and callable(getattr(payload, "to_numpy", None))
        ):
            frame_columns = list(payload.columns)
            if not frame_columns:
                raise ModelArtifactError("Input frame has no columns.")
            missing = [name for name in self.feature_names if name not in frame_columns]
            if missing:
                raise ModelArtifactError(
                    f"Input is missing required feature(s): {missing}. "
                    f"Expected {len(self.feature_names)} features."
                )
            selected = payload.loc[:, list(self.feature_names)]
            try:
                return selected.to_numpy(dtype=np.float64, copy=False)
            except (TypeError, ValueError) as exc:
                raise ModelArtifactError(
                    f"Input contains non-numeric values: {exc}"
                ) from exc

        # Mapping or sequence of mappings.
        if not isinstance(payload, (list, tuple, set)):
            payload = [payload]

        if not payload:
            raise ModelArtifactError("Input is empty.")

        first = payload[0]
        if not hasattr(first, "keys") or not callable(getattr(first, "__getitem__", None)):
            raise ModelArtifactError(
                "Input rows must be feature mappings (e.g. dicts)."
            )

        missing = [name for name in self.feature_names if name not in first]
        if missing:
            raise ModelArtifactError(
                f"Input is missing required feature(s): {missing}. "
                f"Expected {len(self.feature_names)} features."
            )

        rows: list[list[float]] = []
        for row in payload:
            try:
                rows.append([float(row[name]) for name in self.feature_names])
            except (TypeError, ValueError) as exc:
                raise ModelArtifactError(
                    f"Input contains non-numeric values: {exc}"
                ) from exc

        return np.array(rows, dtype=np.float64)

    # -- scoring ------------------------------------------------------------ #

    def score_frame(self, payload: Any) -> np.ndarray:
        """Return fraud scores (higher = more likely fraudulent)."""
        matrix = self._feature_matrix(payload)
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(matrix))[:, 1]
        # Unsupervised path: score_samples is negative for anomalous rows.
        return -np.asarray(self.model.score_samples(matrix))

    def predict_frame(self, payload: Any) -> np.ndarray:
        """Return hard decisions using the artifact's saved threshold."""
        return (self.score_frame(payload) >= self.decision_threshold).astype(int)

    def score_row(self, features: Mapping[str, float]) -> float:
        """Score a single transaction given as a feature mapping."""
        return float(self.score_frame(features)[0])

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
