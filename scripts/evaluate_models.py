"""Phase 3: single-pass evaluation of the trained models on the held-out test set.

Loads the artifacts written by ``scripts/train_models.py`` and scores the test
split **once**. No threshold is selected, tuned or revised here — every threshold
comes from the saved artifact, which was chosen on the validation split.

Run from the project root::

    python scripts/evaluate_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ml_common as mc  # noqa: E402  (path set above so the import works standalone)

MODEL_FILES: tuple[tuple[str, str], ...] = (
    ("logistic_regression", "logistic_regression.joblib"),
    ("isolation_forest", "isolation_forest.joblib"),
    ("xgboost", "xgboost.joblib"),
)

COLORS = {
    "logistic_regression": "#4C72B0",
    "isolation_forest": "#55A868",
    "xgboost": "#C44E52",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_artifacts() -> dict[str, dict[str, Any]]:
    """Load each saved artifact, failing loudly if training has not been run."""
    if not mc.MODELS_DIR.exists():
        raise SystemExit(
            "\nERROR: artifacts/models/ does not exist.\n"
            "  Run the training script first:  python scripts/train_models.py\n"
        )

    artifacts: dict[str, dict[str, Any]] = {}
    for key, filename in MODEL_FILES:
        path = mc.MODELS_DIR / filename
        if not path.exists():
            raise SystemExit(
                f"\nERROR: missing model artifact {path}\n"
                "  Run the training script first:  python scripts/train_models.py\n"
            )
        artifact = joblib.load(path)
        for required in ("model", "decision_threshold", "feature_names", "model_name"):
            if required not in artifact:
                raise SystemExit(f"ERROR: {path} is missing key {required!r}")
        artifacts[key] = artifact
        print(f"  loaded {filename:<32} threshold={artifact['decision_threshold']:.6f}")

    return artifacts


def reproduce_test_split() -> tuple[pd.DataFrame, np.ndarray]:
    """Rebuild the dataset and confirm the saved split indices reproduce it."""
    frame = mc.assign_group_ids(mc.load_raw_dataframe())
    saved = np.load(mc.MODELS_DIR / "split_indices.npz")

    recomputed, _ = mc.group_stratified_split(frame)
    for name in mc.SPLIT_NAMES:
        if not np.array_equal(saved[name], recomputed[name]):
            raise SystemExit(
                f"ERROR: saved {name} indices do not match this run. The split is not "
                "reproducible; delete artifacts/models/ and retrain."
            )
    mc.assert_no_group_leakage(frame, {name: saved[name] for name in mc.SPLIT_NAMES})
    return frame, saved["test"]


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def plot_confusion_matrices(
    results: dict[str, dict[str, float]],
    display_names: dict[str, str],
) -> Path:
    """One annotated confusion-matrix heatmap per model."""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.4))
    axes = np.atleast_1d(axes)

    for ax, (key, metrics) in zip(axes, results.items()):
        matrix = np.array(
            [
                [metrics["true_negatives"], metrics["false_positives"]],
                [metrics["false_negatives"], metrics["true_positives"]],
            ]
        )
        ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for col in range(2):
                value = matrix[row, col]
                ax.text(
                    col, row, f"{value:,}",
                    ha="center", va="center", fontsize=13,
                    color="white" if value > matrix.max() * 0.45 else "black",
                )
        ax.set_xticks([0, 1], ["Predicted 0", "Predicted 1"])
        ax.set_yticks([0, 1], ["Actual 0", "Actual 1"])
        ax.set_title(
            f"{display_names[key]}\nthreshold={results[key]['threshold']:.4f}\n"
            f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}",
            fontsize=10,
        )

    fig.suptitle("Test-set confusion matrices (single pass, saved thresholds)")
    fig.tight_layout()
    path = mc.RESULTS_DIR / "test_confusion_matrices.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pr_curves(
    y_test: np.ndarray,
    scores: dict[str, np.ndarray],
    display_names: dict[str, str],
    baseline_rate: float,
) -> Path:
    """Precision-recall curves, the informative view under heavy imbalance."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for key, score in scores.items():
        precision, recall, _ = precision_recall_curve(y_test, score)
        ap = mc.compute_metrics(y_test, (score >= 0.0).astype(int), score)["pr_auc"]
        ax.plot(recall, precision, color=COLORS[key], linewidth=2,
                label=f"{display_names[key]} (AP = {ap:.3f})")

    ax.axhline(baseline_rate, color="black", linestyle=":", linewidth=1,
               label=f"random = fraud rate ({baseline_rate:.4f})")
    ax.set_xlabel("Recall (share of frauds caught)")
    ax.set_ylabel("Precision (share of alerts that are fraud)")
    ax.set_title("Test-set precision-recall curves")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)

    path = mc.RESULTS_DIR / "test_precision_recall_curves.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_roc_curves(
    y_test: np.ndarray,
    scores: dict[str, np.ndarray],
    display_names: dict[str, str],
) -> Path:
    """ROC curves, kept as supplementary analysis only."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for key, score in scores.items():
        fpr, tpr, _ = roc_curve(y_test, score)
        auc = mc.compute_metrics(y_test, (score >= 0.0).astype(int), score)["roc_auc"]
        ax.plot(fpr, tpr, color=COLORS[key], linewidth=2,
                label=f"{display_names[key]} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("Test-set ROC curves (supplementary: optimistic under imbalance)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    path = mc.RESULTS_DIR / "test_roc_curves.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    mc.ensure_directories()

    mc.section("PHASE 3: FINAL TEST-SET EVALUATION (single pass)")
    print("Thresholds are loaded from the artifacts and are NOT re-tuned here.")

    print("\nLoading artifacts:")
    artifacts = load_artifacts()

    frame, test_index = reproduce_test_split()
    X_test = mc.to_matrix(frame.iloc[test_index])
    y_test = frame[mc.TARGET].to_numpy()[test_index]

    test_counts = mc.split_class_counts(frame, test_index)
    mc.section("TEST SPLIT")
    print(f"Rows       : {test_counts['rows']:,}")
    print(f"Legitimate : {test_counts['legitimate']:,}")
    print(f"Fraudulent : {test_counts['fraudulent']:,}")
    print(f"Fraud rate : {test_counts['fraudulent'] / test_counts['rows'] * 100:.4f}%")
    print(f"Saved split indices verified reproducible: yes")

    # --- one scoring pass per model ----------------------------------------- #
    mc.section("SCORING THE TEST SET ONCE PER MODEL")
    results: dict[str, dict[str, float]] = {}
    scores: dict[str, np.ndarray] = {}
    display_names: dict[str, str] = {}

    for key, artifact in artifacts.items():
        model = artifact["model"]
        threshold = float(artifact["decision_threshold"])
        display_names[key] = artifact["display_name"]

        y_score = mc.fraud_scores(model, X_test)
        scores[key] = y_score

        metrics = mc.metrics_at_threshold(y_test, y_score, threshold)
        metrics["threshold"] = threshold
        results[key] = metrics

        print(
            f"{artifact['display_name']:<20} P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
            f"PR-AUC={metrics['pr_auc']:.4f} "
            f"FP={metrics['false_positives']:,} FN={metrics['false_negatives']:,}"
        )

    # --- comparison table ---------------------------------------------------- #
    mc.section("MODEL COMPARISON (TEST SET)")
    columns = ["model", "threshold", "precision", "recall", "f1", "pr_auc",
               "roc_auc", "false_positives", "false_negatives"]
    export = pd.DataFrame(
        [
            {"model": display_names[key], "threshold": results[key]["threshold"],
             "precision": results[key]["precision"], "recall": results[key]["recall"],
             "f1": results[key]["f1"], "pr_auc": results[key]["pr_auc"],
             "roc_auc": results[key]["roc_auc"],
             "false_positives": results[key]["false_positives"],
             "false_negatives": results[key]["false_negatives"]}
            for key in artifacts
        ]
    )[columns].round(6)

    detailed = pd.DataFrame(
        [
            {"model": display_names[key], **{k: v for k, v in results[key].items()}}
            for key in artifacts
        ]
    ).round(6)

    comparison_path = mc.RESULTS_DIR / "test_comparison.csv"
    export.to_csv(comparison_path, index=False)
    detailed.to_csv(mc.RESULTS_DIR / "test_metrics_detailed.csv", index=False)

    print(export.to_string(index=False))
    print(f"\n  saved -> artifacts/model_results/{comparison_path.name}")

    # --- confusion matrices in text ----------------------------------------- #
    mc.section("CONFUSION MATRICES")
    for key, metrics in results.items():
        matrix = np.array(
            [
                [metrics["true_negatives"], metrics["false_positives"]],
                [metrics["false_negatives"], metrics["true_positives"]],
            ]
        )
        print(f"\n{display_names[key]} (threshold={metrics['threshold']:.4f})")
        print("                  Predicted legit   Predicted fraud")
        print(f"Actual legit      {matrix[0][0]:>15,}   {matrix[0][1]:>15,}")
        print(f"Actual fraud      {matrix[1][0]:>15,}   {matrix[1][1]:>15,}")
        print(f"  caught {metrics['true_positives']} of "
              f"{metrics['true_positives'] + metrics['false_negatives']} frauds, "
              f"raised {metrics['false_positives']:,} false alerts")

    # --- plots --------------------------------------------------------------- #
    mc.section("CHARTS")
    baseline_rate = float(y_test.mean())
    for path in (
        plot_confusion_matrices(results, display_names),
        plot_pr_curves(y_test, scores, display_names, baseline_rate),
        plot_roc_curves(y_test, scores, display_names),
    ):
        print(f"  saved -> {path.relative_to(mc.PROJECT_ROOT)}")

    # --- selected model check ------------------------------------------------ #
    mc.section("DEPLOYMENT ARTIFACT CHECK")
    deployment_path = mc.MODELS_DIR / "fraud_model.joblib"
    deployment = joblib.load(deployment_path)
    selected_name = deployment["display_name"]
    print(f"Selected for deployment : {selected_name}")
    print(f"Threshold               : {deployment['decision_threshold']:.6f}")
    print(f"Features                : {len(deployment['feature_names'])} in fixed order")
    print(f"Selection basis         : {deployment.get('selection_basis', 'n/a')}")

    matching = [key for key in artifacts if display_names[key] == selected_name]
    if matching:
        sel = results[matching[0]]
        print(
            f"Its test-set result    : P={sel['precision']:.4f} R={sel['recall']:.4f} "
            f"F1={sel['f1']:.4f} PR-AUC={sel['pr_auc']:.4f}"
        )

    demo_path = mc.MODELS_DIR / "demo_transactions.csv"
    if demo_path.exists():
        demo = pd.read_csv(demo_path)
        demo_scores = mc.fraud_scores(deployment["model"], mc.to_matrix(demo))
        demo_pred = (demo_scores >= deployment["decision_threshold"]).astype(int)
        print(
            f"\nDemo sample ({len(demo)} rows) scored with the deployment artifact: "
            f"{int(demo_pred.sum())} flagged as fraud, "
            f"{len(demo) - int(demo_pred.sum())} as legitimate."
        )

    # --- markdown report ----------------------------------------------------- #
    mc.section("WRITING REPORT")
    report = build_report(export, detailed, results, display_names, test_counts, deployment)
    report_path = mc.RESULTS_DIR / "PHASE3_TEST_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  saved -> {report_path.relative_to(mc.PROJECT_ROOT)}")

    mc.section("EVALUATION COMPLETE")
    print(f"Results: {mc.RESULTS_DIR}")
    print("The test set has now been used once. Do not re-tune against it.")


def build_report(
    export: pd.DataFrame,
    detailed: pd.DataFrame,
    results: dict[str, dict[str, float]],
    display_names: dict[str, str],
    test_counts: dict[str, int],
    deployment: dict[str, Any],
) -> str:
    """Render a markdown summary of the actual measured results."""
    lines: list[str] = []
    lines.append("# FraudGuard Phase 3 - test-set report\n")
    lines.append(
        f"Generated from an actual run. Test split: {test_counts['rows']:,} rows, "
        f"{test_counts['legitimate']:,} legitimate and "
        f"{test_counts['fraudulent']:,} fraudulent "
        f"({test_counts['fraudulent'] / test_counts['rows'] * 100:.4f}% fraud).\n"
    )
    lines.append(
        "Thresholds were selected on the validation split and loaded from the saved "
        "artifacts; the test set was scored exactly once.\n"
    )
    lines.append("## Comparison\n")
    lines.append(
        "| Model | Threshold | Precision | Recall | F1 | PR-AUC | FP | FN |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    for _, row in export.iterrows():
        lines.append(
            f"| {row['model']} | {row['threshold']:.4f} | {row['precision']:.4f} | "
            f"{row['recall']:.4f} | {row['f1']:.4f} | {row['pr_auc']:.4f} | "
            f"{int(row['false_positives']):,} | {int(row['false_negatives'])} |"
        )

    lines.append("\n## Confusion matrices\n")
    for key, metrics in results.items():
        lines.append(f"### {display_names[key]} (threshold {metrics['threshold']:.4f})\n")
        lines.append("| | Predicted legitimate | Predicted fraud |\n| --- | --- | --- |")
        lines.append(
            f"| **Actual legitimate** | {metrics['true_negatives']:,} | "
            f"{metrics['false_positives']:,} |"
        )
        lines.append(
            f"| **Actual fraud** | {metrics['false_negatives']} | "
            f"{metrics['true_positives']} |\n"
        )

    lines.append("## Selected for deployment\n")
    lines.append(f"- Model: **{deployment['display_name']}**")
    lines.append(f"- Threshold: **{deployment['decision_threshold']:.6f}**")
    lines.append(f"- Selection basis: {deployment.get('selection_basis', 'n/a')}")
    lines.append(f"- Artifact: `artifacts/models/fraud_model.joblib`")
    selected_rows = detailed[detailed["model"] == deployment["display_name"]]
    if not selected_rows.empty:
        row = selected_rows.iloc[0]
        lines.append(
            f"- Test metrics: precision={row['precision']:.4f}, "
            f"recall={row['recall']:.4f}, f1={row['f1']:.4f}, "
            f"pr_auc={row['pr_auc']:.4f}, "
            f"false_positives={int(row['false_positives']):,}, "
            f"false_negatives={int(row['false_negatives'])}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
