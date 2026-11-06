"""Phase 3: train and validate three fraud detection models.

Pipeline
--------
1. Load ``data/raw/creditcard.csv`` (read-only).
2. Group exactly-identical rows and split **by group**, stratified 70/15/15.
3. Train Logistic Regression, Isolation Forest and XGBoost on the training split.
4. Select a decision threshold per model on the **validation** split only.
5. Persist every model, its threshold, metadata and the split indices.

The test split is never touched here. ``scripts/evaluate_models.py`` loads the
saved artifacts and scores the test split exactly once.

Run from the project root::

    python scripts/train_models.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ml_common as mc  # noqa: E402  (path set above so the import works standalone)


# --------------------------------------------------------------------------- #
# Model definitions
# --------------------------------------------------------------------------- #


def build_logistic_regression(seed: int) -> Pipeline:
    """Baseline linear model in a pipeline.

    ``class_weight="balanced"`` re-weights the loss by inverse class frequency, so
    the 578:1 imbalance no longer dominates the objective. ``StandardScaler``
    lives inside the pipeline, which means it is fitted on training rows only —
    fitting it outside would leak test statistics into the model.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )


def build_isolation_forest(seed: int) -> IsolationForest:
    """Unsupervised anomaly detector.

    ``max_samples=4096`` is larger than sklearn's default (256) because the
    training split has ~199k legitimate rows; tiny subsamples make the trees see
    too little of the distribution and the scores become noisy.
    """
    return IsolationForest(
        n_estimators=200,
        max_samples=4096,
        contamination="auto",
        random_state=seed,
        n_jobs=-1,
    )


def build_xgboost(seed: int, scale_pos_weight: float) -> xgboost.XGBClassifier:
    """Gradient-boosted trees, kept deliberately lightweight.

    Deliberately modest so the artifact stays small enough to deploy in a
    serverless Python function later: depth 4, 400 round cap with early stopping,
    histogram tree method.
    """
    return xgboost.XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="aucpr",
        early_stopping_rounds=40,
        random_state=seed,
        n_jobs=-1,
    )


# --------------------------------------------------------------------------- #
# Artifact assembly
# --------------------------------------------------------------------------- #


def build_artifact(
    *,
    model: Any,
    model_name: str,
    display_name: str,
    threshold: float,
    threshold_reason: str,
    metadata: dict[str, Any],
    validation_metrics: dict[str, float],
) -> dict[str, Any]:
    """Bundle everything FastAPI needs to serve predictions without retraining."""
    return {
        "artifact_version": 1,
        "model_name": model_name,
        "display_name": display_name,
        "model": model,
        "decision_threshold": float(threshold),
        "score_direction": "higher_is_more_fraudulent",
        "feature_names": list(mc.FEATURE_NAMES),
        "target_name": mc.TARGET,
        "positive_class": mc.FRAUD,
        "random_seed": mc.RANDOM_SEED,
        "created_utc": mc.utc_now(),
        "threshold_rule": threshold_reason,
        "min_precision_target": mc.MIN_PRECISION_TARGET,
        "validation_metrics": validation_metrics,
        "training_metadata": metadata,
        "library_versions": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
        },
    }


def write_demo_sample(frame: pd.DataFrame, test_index: np.ndarray) -> Path:
    """Write a small, sanitised demo file of unseen legitimate + fraud rows.

    Only the 30 features and the label are included — a handful of rows, never
    the full dataset. Drawn from the test split so the sample reflects data the
    models never trained on.
    """
    test_rows = frame.iloc[test_index]
    fraud = test_rows[test_rows[mc.TARGET] == mc.FRAUD].head(5)
    legit = test_rows[test_rows[mc.TARGET] == mc.LEGIT].head(7)
    sample = pd.concat([legit, fraud], ignore_index=True)
    sample = sample[[*mc.FEATURE_NAMES, mc.TARGET]]

    path = mc.MODELS_DIR / "demo_transactions.csv"
    sample.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    mc.ensure_directories()

    mc.section("PHASE 3: TRAIN AND VALIDATE FRAUD DETECTION MODELS")
    print(f"Project root : {mc.PROJECT_ROOT}")
    print(f"Random seed  : {mc.RANDOM_SEED}")
    print(f"Split        : {mc.TRAIN_FRAC:.0%} / {mc.VAL_FRAC:.0%} / {mc.TEST_FRAC:.0%} "
          "(group-aware, stratified)")

    # --- 1. load ------------------------------------------------------------ #
    mc.section("STEP 1: LOAD DATA AND INVESTIGATE DUPLICATES")
    raw = mc.load_raw_dataframe()
    print(f"Loaded {len(raw):,} rows x {raw.shape[1]} columns from {mc.DATA_PATH}")

    frame = mc.assign_group_ids(raw)
    n_groups = int(frame[mc.GROUP_COL].nunique())
    n_excess = len(frame) - n_groups
    print(f"Unique transactions (groups): {n_groups:,}")
    print(f"Excess duplicate rows      : {n_excess:,} "
          f"({n_excess / len(frame) * 100:.3f}%)")
    print(
        "\nDuplicate strategy: rows are NOT deleted. Identical rows share a group id\n"
        "and the split is performed on groups, so all copies of a transaction land\n"
        "in the same split. This removes leakage without discarding fraud rows."
    )

    # --- 2. split ----------------------------------------------------------- #
    mc.section("STEP 2: GROUP-AWARE STRATIFIED SPLIT")
    indices, row_split = mc.group_stratified_split(frame)
    mc.assert_no_group_leakage(frame, indices)

    split_report = []
    for name in mc.SPLIT_NAMES:
        counts = mc.split_class_counts(frame, indices[name])
        split_report.append(
            {
                "split": name,
                "rows": counts["rows"],
                "legitimate": counts["legitimate"],
                "fraudulent": counts["fraudulent"],
                "fraud_rate_pct": round(counts["fraudulent"] / counts["rows"] * 100, 4),
                "groups": int(frame.iloc[indices[name]][mc.GROUP_COL].nunique()),
            }
        )
    split_table = pd.DataFrame(split_report)
    print(split_table.to_string(index=False))
    print("\nNo duplicate group spans two splits (asserted).")
    print(f"Rows accounted for: {split_table['rows'].sum():,} of {len(frame):,}")

    # A chronological alternative exists (Time is ordered) and is discussed in the
    # report; a stratified split is used because the brief requires it and because
    # the 48-hour window is too short to model seasonal drift meaningfully.
    X = mc.to_matrix(frame)
    y = frame[mc.TARGET].to_numpy()

    X_train, y_train = X[indices["train"]], y[indices["train"]]
    X_val, y_val = X[indices["val"]], y[indices["val"]]

    train_legit = int((y_train == mc.LEGIT).sum())
    train_fraud = int((y_train == mc.FRAUD).sum())
    scale_pos_weight = train_legit / train_fraud
    print(f"\nXGBoost scale_pos_weight (train only) = {train_legit:,} / {train_fraud:,} "
          f"= {scale_pos_weight:.3f}")

    common_metadata = {
        "dataset": str(mc.DATA_PATH.relative_to(mc.PROJECT_ROOT)),
        "total_rows": int(len(frame)),
        "unique_transactions": n_groups,
        "excess_duplicate_rows": n_excess,
        "duplicate_strategy": "group-aware split, no rows dropped",
        "duplicates": {
            "total_rows": int(len(frame)),
            "unique_transactions": n_groups,
            "excess_duplicate_rows": n_excess,
            "handling": "grouped by identical rows, not dropped; each group stays in one split",
        },
        "split_strategy": "group-aware stratified 70/15/15",
        "split_sizes": split_table.set_index("split").to_dict(orient="index"),
        "train_class_balance": {
            "legitimate": train_legit,
            "fraudulent": train_fraud,
            "scale_pos_weight": round(scale_pos_weight, 6),
        },
        "resampling_applied": False,
        "test_set_used": False,
    }

    validation_rows: list[dict[str, Any]] = []

    # --- 3. Logistic Regression --------------------------------------------- #
    mc.section("STEP 3: LOGISTIC REGRESSION (baseline)")
    lr = build_logistic_regression(mc.RANDOM_SEED)
    lr.fit(X_train, y_train)
    lr_scores_val = mc.fraud_scores(lr, X_val)

    lr_table = mc.build_threshold_table(y_val, lr_scores_val)
    lr_table.to_csv(mc.THRESHOLD_DIR / "logistic_regression.csv", index=False)
    lr_threshold, lr_reason = mc.select_threshold(lr_table)
    lr_default = mc.metrics_at_threshold(y_val, lr_scores_val, 0.5)
    lr_selected = mc.metrics_at_threshold(y_val, lr_scores_val, lr_threshold)

    print(f"Threshold selected: {lr_threshold:.6f}  ({lr_reason})")
    print(f"  at 0.5 (for reference): precision {lr_default['precision']:.4f}, "
          f"recall {lr_default['recall']:.4f}, F1 {lr_default['f1']:.4f}")
    print(f"  at selected threshold : precision {lr_selected['precision']:.4f}, "
          f"recall {lr_selected['recall']:.4f}, F1 {lr_selected['f1']:.4f}, "
          f"PR-AUC {lr_selected['pr_auc']:.4f}")

    lr_artifact = build_artifact(
        model=lr,
        model_name="logistic_regression",
        display_name="Logistic Regression",
        threshold=lr_threshold,
        threshold_reason=lr_reason,
        metadata={
            **common_metadata,
            "model_family": "linear (supervised)",
            "class_weight": "balanced",
            "preprocessing": "StandardScaler fitted on training split only",
            "validation_metrics_at_0_5": lr_default,
        },
        validation_metrics=lr_selected,
    )
    joblib.dump(lr_artifact, mc.MODELS_DIR / "logistic_regression.joblib")
    validation_rows.append({"model": "Logistic Regression", "threshold": lr_threshold, **lr_selected})
    print("  saved -> artifacts/models/logistic_regression.joblib")

    # --- 4. Isolation Forest ------------------------------------------------- #
    mc.section("STEP 4: ISOLATION FOREST (unsupervised anomaly detection)")
    X_train_legit = X_train[y_train == mc.LEGIT]
    print(f"Training on legitimate transactions only: {len(X_train_legit):,} rows")
    print("Fraud labels are NOT used to fit this model.")

    iso = build_isolation_forest(mc.RANDOM_SEED)
    iso.fit(X_train_legit)
    iso_scores_val = mc.fraud_scores(iso, X_val)

    iso_table = mc.build_threshold_table(y_val, iso_scores_val)
    iso_table.to_csv(mc.THRESHOLD_DIR / "isolation_forest.csv", index=False)
    iso_threshold, iso_reason = mc.select_threshold(iso_table)
    iso_selected = mc.metrics_at_threshold(y_val, iso_scores_val, iso_threshold)
    # IsolationForest's own built-in cut (contamination="auto") for reference.
    iso_builtin_pred = (iso.predict(X_val) == -1).astype(int)
    iso_builtin = mc.compute_metrics(y_val, iso_builtin_pred, iso_scores_val)

    print(f"Threshold selected: {iso_threshold:.6f}  ({iso_reason})")
    print(f"  at selected threshold : precision {iso_selected['precision']:.4f}, "
          f"recall {iso_selected['recall']:.4f}, F1 {iso_selected['f1']:.4f}, "
          f"PR-AUC {iso_selected['pr_auc']:.4f}")
    print(f"  built-in offset cut   : precision {iso_builtin['precision']:.4f}, "
          f"recall {iso_builtin['recall']:.4f}")

    iso_artifact = build_artifact(
        model=iso,
        model_name="isolation_forest",
        display_name="Isolation Forest",
        threshold=iso_threshold,
        threshold_reason=iso_reason,
        metadata={
            **common_metadata,
            "model_family": "anomaly detection (unsupervised)",
            "trained_on": "legitimate training rows only",
            "contamination": "auto",
            "n_estimators": 200,
            "max_samples": 4096,
            "validation_metrics_builtin_cut": iso_builtin,
        },
        validation_metrics=iso_selected,
    )
    joblib.dump(iso_artifact, mc.MODELS_DIR / "isolation_forest.joblib")
    validation_rows.append({"model": "Isolation Forest", "threshold": iso_threshold, **iso_selected})
    print("  saved -> artifacts/models/isolation_forest.joblib")

    # --- 5. XGBoost ---------------------------------------------------------- #
    mc.section("STEP 5: XGBOOST (gradient boosted trees)")
    xgb = build_xgboost(mc.RANDOM_SEED, scale_pos_weight)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_iteration = getattr(xgb, "best_iteration", None)
    print(f"Early stopping: best iteration = {best_iteration} of {xgb.n_estimators} allowed")

    xgb_scores_val = mc.fraud_scores(xgb, X_val)
    xgb_table = mc.build_threshold_table(y_val, xgb_scores_val)
    xgb_table.to_csv(mc.THRESHOLD_DIR / "xgboost.csv", index=False)
    xgb_threshold, xgb_reason = mc.select_threshold(xgb_table)
    xgb_default = mc.metrics_at_threshold(y_val, xgb_scores_val, 0.5)
    xgb_selected = mc.metrics_at_threshold(y_val, xgb_scores_val, xgb_threshold)

    print(f"Threshold selected: {xgb_threshold:.6f}  ({xgb_reason})")
    print(f"  at 0.5 (for reference): precision {xgb_default['precision']:.4f}, "
          f"recall {xgb_default['recall']:.4f}, F1 {xgb_default['f1']:.4f}")
    print(f"  at selected threshold : precision {xgb_selected['precision']:.4f}, "
          f"recall {xgb_selected['recall']:.4f}, F1 {xgb_selected['f1']:.4f}, "
          f"PR-AUC {xgb_selected['pr_auc']:.4f}")

    xgb_artifact = build_artifact(
        model=xgb,
        model_name="xgboost",
        display_name="XGBoost",
        threshold=xgb_threshold,
        threshold_reason=xgb_reason,
        metadata={
            **common_metadata,
            "model_family": "gradient boosted trees (supervised)",
            "scale_pos_weight": round(scale_pos_weight, 6),
            "early_stopping_rounds": 40,
            "best_iteration": best_iteration,
            "hyperparameters": {
                "n_estimators": 400,
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "tree_method": "hist",
                "eval_metric": "aucpr",
            },
            "validation_metrics_at_0_5": xgb_default,
        },
        validation_metrics=xgb_selected,
    )
    joblib.dump(xgb_artifact, mc.MODELS_DIR / "xgboost.joblib")
    validation_rows.append({"model": "XGBoost", "threshold": xgb_threshold, **xgb_selected})
    print("  saved -> artifacts/models/xgboost.joblib")

    # --- 6. validation comparison ------------------------------------------- #
    mc.section("STEP 6: VALIDATION COMPARISON (thresholds selected on validation only)")
    columns = [
        "model", "threshold", "precision", "recall", "f1", "pr_auc",
        "true_positives", "false_positives", "false_negatives", "true_negatives",
    ]
    comparison = pd.DataFrame(validation_rows)[columns].round(6)
    comparison.to_csv(mc.RESULTS_DIR / "validation_comparison.csv", index=False)
    print(comparison.to_string(index=False))

    # --- 7. select the operating model --------------------------------------- #
    mc.section("STEP 7: SELECT THE OPERATING MODEL")
    # Same pre-declared logic used for thresholds: recall first, but only among
    # models that clear the precision floor; ties broken by PR-AUC.
    qualifying = comparison[comparison["precision"] >= mc.MIN_PRECISION_TARGET]
    pool = qualifying if not qualifying.empty else comparison
    selection_basis = (
        f"precision >= {mc.MIN_PRECISION_TARGET:.2f}"
        if not qualifying.empty
        else f"no model reached precision >= {mc.MIN_PRECISION_TARGET:.2f}; best F1"
    )
    ranked = pool.sort_values(["recall", "pr_auc", "f1"], ascending=False)
    best_row = ranked.iloc[0]
    best_name = str(best_row["model"])

    mapping = {
        "Logistic Regression": ("logistic_regression.joblib", lr_artifact),
        "Isolation Forest": ("isolation_forest.joblib", iso_artifact),
        "XGBoost": ("xgboost.joblib", xgb_artifact),
    }
    filename, selected_artifact = mapping[best_name]

    print(f"Selection pool: models with {selection_basis}")
    print(f"Ranking rule : highest recall, tie-break PR-AUC then F1")
    print(f"Selected     : {best_name}  "
          f"(recall {best_row['recall']:.4f}, precision {best_row['precision']:.4f}, "
          f"PR-AUC {best_row['pr_auc']:.4f})")
    print(f"Note: this decision used the validation split only; the test split is "
          f"still untouched.")

    selected_artifact = dict(selected_artifact)
    selected_artifact["selected_for_deployment"] = True
    selected_artifact["selection_basis"] = selection_basis
    selected_artifact["selection_rule"] = (
        "highest validation recall among models meeting the precision floor, "
        "tie-break PR-AUC then F1"
    )
    joblib.dump(selected_artifact, mc.MODELS_DIR / "fraud_model.joblib")
    print(f"  deployment artifact -> artifacts/models/fraud_model.joblib "
          f"(from {filename})")

    # --- 8. persist split indices and metadata ------------------------------- #
    mc.section("STEP 8: PERSIST SPLITS AND METADATA")
    np.savez_compressed(
        mc.MODELS_DIR / "split_indices.npz",
        train=indices["train"],
        val=indices["val"],
        test=indices["test"],
        seed=np.array([mc.RANDOM_SEED]),
    )
    print("  saved -> artifacts/models/split_indices.npz")

    demo_path = write_demo_sample(frame, indices["test"])
    demo = pd.read_csv(demo_path)
    print(f"  saved -> {demo_path.relative_to(mc.PROJECT_ROOT)} "
          f"({len(demo)} rows: {int((demo[mc.TARGET] == mc.LEGIT).sum())} legitimate, "
          f"{int((demo[mc.TARGET] == mc.FRAUD).sum())} fraudulent)")

    metadata = {
        "phase": 3,
        "created_utc": mc.utc_now(),
        "random_seed": mc.RANDOM_SEED,
        "feature_names": list(mc.FEATURE_NAMES),
        "n_features": len(mc.FEATURE_NAMES),
        "split": {
            "strategy": "group-aware stratified",
            "fractions": {"train": mc.TRAIN_FRAC, "val": mc.VAL_FRAC, "test": mc.TEST_FRAC},
            "sizes": split_table.set_index("split").to_dict(orient="index"),
        },
        "duplicates": {
            "total_rows": int(len(frame)),
            "unique_transactions": n_groups,
            "excess_duplicate_rows": n_excess,
            "handling": "grouped, not dropped; groups kept within a single split",
        },
        "models_trained": comparison.to_dict(orient="records"),
        "selected_model": best_name,
        "selected_model_file": "artifacts/models/fraud_model.joblib",
        "selection_basis": selection_basis,
        "min_precision_target": mc.MIN_PRECISION_TARGET,
        "resampling_applied": False,
        "test_split_evaluated": False,
        "notes": [
            "Thresholds were chosen on the validation split only.",
            "The test split is reserved for scripts/evaluate_models.py (single pass).",
            "No SMOTE or other resampling was applied; class_weight/scale_pos_weight "
            "handle imbalance inside the models.",
        ],
    }
    mc.save_json(metadata, mc.MODELS_DIR / "training_metadata.json")
    print("  saved -> artifacts/models/training_metadata.json")

    mc.section("TRAINING COMPLETE")
    print(f"Models      : {mc.MODELS_DIR}")
    print(f"Validation  : {mc.RESULTS_DIR}")
    print("Next        : python scripts/evaluate_models.py  (single test-set pass)")


if __name__ == "__main__":
    main()
