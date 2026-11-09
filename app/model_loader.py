"""Load the deployed FraudGuard model artifact for inference.

Two artifact formats are supported:

* ``artifacts/models/fraud_model.ubj`` - the deployed artifact. It is the trained
  ``XGBClassifier``'s own ``Booster`` written in XGBoost's native UBJ format, read
  with ``xgboost.Booster.load_model``. The decision threshold, the canonical
  feature order, the training metadata and the early-stopping ``best_iteration``
  travel inside the file as Booster attributes, so the deployed runtime needs
  neither ``joblib`` nor ``scikit-learn`` and never unpickles anything.
* ``artifacts/models/fraud_model.joblib`` - the original Phase 3 bundle (a pickled
  ``XGBClassifier``). Kept as a development fallback for a working tree where the
  native file has not been exported yet; ``joblib`` is imported lazily only when
  that file is actually selected, so it is not a deployed dependency.

The two formats score identically: the native path reproduces
``XGBClassifier.predict_proba(X)[:, 1]`` bit for bit, including the wrapper's
habit of scoring only the trees up to ``best_iteration`` after early stopping.
``scripts/export_native_model.py`` re-verifies that on every export.

Nothing here trains, tunes or reads the test set.

Typical use::

    from app.model_loader import load_model

    model = load_model()
    score = model.score_frame(frame)      # higher = more likely fraudulent
    decision = model.predict_frame(frame) # 1 = fraud, 0 = legitimate
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, Mapping

import numpy as np
import xgboost as xgb

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

#: The deployed artifact: native XGBoost Booster, written by
#: ``scripts/export_native_model.py``.
DEFAULT_MODEL_PATH: Final[Path] = PROJECT_ROOT / "artifacts" / "models" / "fraud_model.ubj"

#: The original Phase 3 bundle, kept as a development fallback.
LEGACY_MODEL_PATH: Final[Path] = PROJECT_ROOT / "artifacts" / "models" / "fraud_model.joblib"

#: Formats read with ``xgboost.Booster.load_model``.
NATIVE_SUFFIXES: Final[frozenset[str]] = frozenset({".ubj", ".json"})

#: Booster attribute holding the JSON bundle (everything except the estimator).
BUNDLE_ATTRIBUTE: Final[str] = "fraudguard_bundle"

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
    accepts a single mapping, a sequence of mappings, a DataFrame-like object or a
    float64 matrix.
    """

    model: Any
    model_name: str
    display_name: str
    decision_threshold: float
    feature_names: tuple[str, ...]
    metadata: Mapping[str, Any]
    #: Tree range the wrapper would have scored. ``predict_proba`` only uses the
    #: trees up to ``best_iteration`` once early stopping ran, so the native path
    #: has to select the same range; ``None`` means "all trees".
    iteration_range: tuple[int, int] | None = None
    #: ``"ubj"`` for the deployed native artifact, ``"joblib"`` for the fallback.
    backend: str = "ubj"

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
          NumPy array directly;
        * anything DataFrame-like (``columns`` + ``loc`` + ``to_numpy``) — used by
          the test suite, which is pandas-based. ``pandas`` is never imported
          here, so a pandas-free runtime still supports it.

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

        if isinstance(self.model, xgb.Booster):
            # Native path: byte-for-byte the same computation the sklearn wrapper
            # performed, just on the booster directly.
            booster_input = xgb.DMatrix(matrix)
            if self.iteration_range is not None:
                scores = self.model.predict(
                    booster_input, iteration_range=self.iteration_range
                )
            else:
                scores = self.model.predict(booster_input)
            return np.asarray(scores, dtype=np.float64)

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
    """Resolve the artifact path from an argument, ``MODEL_PATH`` or the default.

    The native artifact is preferred. The legacy Joblib bundle is used only when
    the native file has not been exported in this working tree, which keeps a
    fresh clone runnable after ``scripts/train_models.py`` alone.
    """
    candidate = path if path is not None else os.getenv("MODEL_PATH")
    if candidate:
        resolved = Path(candidate)
        if not resolved.is_absolute():
            resolved = PROJECT_ROOT / resolved
        return resolved

    if DEFAULT_MODEL_PATH.exists() or not LEGACY_MODEL_PATH.exists():
        return DEFAULT_MODEL_PATH
    return LEGACY_MODEL_PATH


def _check_artifact(artifact: dict[str, Any], resolved: Path) -> dict[str, Any]:
    """Validate the bundle keys and the values the decision rule depends on."""
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


def _load_native_artifact(resolved: Path) -> dict[str, Any]:
    """Read a UBJ/JSON Booster plus the bundle stored in its attributes."""
    booster = xgb.Booster()
    try:
        booster.load_model(str(resolved))
    except Exception as exc:  # noqa: BLE001 - surface any XGBoost load failure
        raise ModelArtifactError(f"Could not load artifact {resolved}: {exc}") from exc

    raw = booster.attr(BUNDLE_ATTRIBUTE)
    if not raw:
        raise ModelArtifactError(
            f"Artifact {resolved} has no {BUNDLE_ATTRIBUTE!r} attribute. "
            "Re-run 'python scripts/export_native_model.py'."
        )

    try:
        bundle = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ModelArtifactError(
            f"Artifact {resolved} has an unreadable {BUNDLE_ATTRIBUTE!r} attribute: {exc}"
        ) from exc

    if not isinstance(bundle, dict):
        raise ModelArtifactError(
            f"Artifact {resolved} stores a {type(bundle).__name__} bundle, expected a dict."
        )

    artifact: dict[str, Any] = dict(bundle)
    artifact["model"] = booster
    return _check_artifact(artifact, resolved)


def _load_joblib_artifact(resolved: Path) -> dict[str, Any]:
    """Read the original pickled bundle (development fallback only)."""
    try:
        import joblib  # noqa: PLC0415 - optional; not installed in production
    except ImportError as exc:
        raise ModelArtifactError(
            f"Artifact {resolved} is a Joblib bundle but 'joblib' is not installed. "
            "Either install joblib or export the native artifact with "
            "'python scripts/export_native_model.py'."
        ) from exc

    try:
        artifact = joblib.load(resolved)
    except Exception as exc:  # noqa: BLE001 - surface any deserialisation failure
        raise ModelArtifactError(f"Could not load artifact {resolved}: {exc}") from exc

    if not isinstance(artifact, dict):
        raise ModelArtifactError(
            f"Artifact {resolved} is a {type(artifact).__name__}, expected a dict bundle."
        )

    return _check_artifact(artifact, resolved)


def load_model_artifact(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the raw artifact dictionary.

    The returned mapping always carries the same keys, whichever format is read:
    the bundle metadata plus ``"model"``. For the native format ``"model"`` is an
    ``xgboost.Booster``; for the fallback it is the pickled estimator.
    """
    resolved = resolve_model_path(path)

    if not resolved.exists():
        raise ModelArtifactError(
            f"Model artifact not found at {resolved}. "
            "Run 'python scripts/train_models.py' and then "
            "'python scripts/export_native_model.py' to create it."
        )

    if resolved.suffix.lower() in NATIVE_SUFFIXES:
        return _load_native_artifact(resolved)
    return _load_joblib_artifact(resolved)


@lru_cache(maxsize=4)
def load_model(path: str | Path | None = None) -> LoadedModel:
    """Load the artifact once per path and wrap it in :class:`LoadedModel`."""
    artifact = load_model_artifact(path)
    model = artifact["model"]

    iteration_range: tuple[int, int] | None = None
    backend = "joblib"
    if isinstance(model, xgb.Booster):
        backend = "ubj"
        best_iteration = artifact.get("best_iteration")
        if isinstance(best_iteration, int) and best_iteration >= 0:
            # ``XGBClassifier.predict_proba`` scores only [0, best_iteration + 1)
            # after early stopping; matching that range is what makes the native
            # artifact reproduce the wrapper exactly.
            iteration_range = (0, best_iteration + 1)

    return LoadedModel(
        model=model,
        model_name=str(artifact["model_name"]),
        display_name=str(artifact["display_name"]),
        decision_threshold=float(artifact["decision_threshold"]),
        feature_names=tuple(str(name) for name in artifact["feature_names"]),
        metadata=artifact.get("training_metadata", {}),
        iteration_range=iteration_range,
        backend=backend,
    )


def clear_model_cache() -> None:
    """Drop the cached artifact (used by tests and after retraining)."""
    load_model.cache_clear()


__all__ = [
    "DEFAULT_MODEL_PATH",
    "LEGACY_MODEL_PATH",
    "LoadedModel",
    "ModelArtifactError",
    "clear_model_cache",
    "load_model",
    "load_model_artifact",
    "resolve_model_path",
]
