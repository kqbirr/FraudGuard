"""Phase 2 exploratory data analysis for the FraudGuard credit card dataset.

Reads ``data/raw/creditcard.csv`` and writes plots to ``artifacts/eda/`` and
summary tables to ``data/processed/``. The raw CSV is opened read-only and is
never modified.

Run from the project root::

    python scripts/eda.py

This module trains nothing. It only describes the data. ``/predict`` and model
training belong to later phases.

Notes on the data
-----------------
* ``V1``-``V28`` are the output of a PCA transformation applied by the dataset
  authors. Their original business meaning is deliberately not public, so this
  script never assigns them financial interpretations.
* ``Time`` is elapsed seconds since the first transaction in the file, not a
  calendar timestamp. There is no date information in this dataset.
* ``Class`` is the target: ``0`` = legitimate, ``1`` = fraudulent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")  # headless backend: write PNGs without a display

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# --------------------------------------------------------------------------- #
# Paths and constants (pathlib only: no hardcoded absolute Windows paths)
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

# The dataset path is documented in the README, but fall back to a CSV sitting
# in the project root so a forgotten move does not block the analysis.
_FALLBACK_DATA_PATH: Final[Path] = PROJECT_ROOT / "creditcard.csv"

PLOTS_DIR: Final[Path] = PROJECT_ROOT / "artifacts" / "eda"
TABLES_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"

TARGET: Final[str] = "Class"
LEGIT: Final[int] = 0
FRAUD: Final[int] = 1
PCA_FEATURES: Final[list[str]] = [f"V{i}" for i in range(1, 29)]
EXPECTED_COLUMNS: Final[list[str]] = ["Time", *PCA_FEATURES, "Amount", TARGET]

SECONDS_PER_HOUR: Final[int] = 3600

sns.set_theme(style="whitegrid", context="notebook")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def section(title: str) -> None:
    """Print a clearly delimited console section header."""
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save a figure into ``artifacts/eda/`` and close it to free memory."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot  -> artifacts/eda/{filename}")
    return path


def save_table(frame: pd.DataFrame, filename: str, *, index: bool = True) -> Path:
    """Save a summary table into ``data/processed/``."""
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / filename
    frame.to_csv(path, index=index)
    print(f"  saved table -> data/processed/{filename}")
    return path


def pct(part: float, total: float) -> float:
    """Return ``part`` as a percentage of ``total``, guarding division by zero."""
    return float(part) / float(total) * 100.0 if total else 0.0


# --------------------------------------------------------------------------- #
# Step 1-2: load and verify
# --------------------------------------------------------------------------- #


def resolve_dataset_path() -> Path:
    """Return the dataset path, preferring ``data/raw/creditcard.csv``."""
    if DATA_PATH.exists():
        return DATA_PATH
    if _FALLBACK_DATA_PATH.exists():
        print(
            f"WARNING: dataset found at {_FALLBACK_DATA_PATH} instead of\n"
            f"         {DATA_PATH}.\n"
            f"         Move it there so the documented layout stays true:\n"
            f"         Move-Item creditcard.csv data\\raw\\creditcard.csv"
        )
        return _FALLBACK_DATA_PATH
    raise SystemExit(
        "\nERROR: dataset not found.\n"
        f"  Expected at: {DATA_PATH}\n\n"
        "  Fix:\n"
        "    1. Download 'creditcard.csv' from the Kaggle 'Credit Card Fraud\n"
        "       Detection' dataset (mlg-ulb/creditcardfraud).\n"
        "    2. Place the file at the path shown above (do not rename it).\n"
        "    3. Re-run:  python scripts/eda.py\n"
    )


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the dataset with pandas and verify it is the expected file."""
    print(f"Loading dataset from: {path}")
    print(f"File size: {path.stat().st_size / (1024**2):.1f} MB")

    frame = pd.read_csv(path)

    if frame.empty:
        raise SystemExit(f"ERROR: {path} loaded successfully but has no rows.")

    missing_columns = set(EXPECTED_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise SystemExit(
            f"ERROR: dataset is missing expected columns: {sorted(missing_columns)}\n"
            f"Found columns: {list(frame.columns)}"
        )

    print(f"Loaded OK: {frame.shape[0]:,} rows x {frame.shape[1]} columns")
    return frame


def inspect_structure(frame: pd.DataFrame) -> None:
    """Print shape, sample rows, dtypes, descriptive stats and data quality."""
    section("STEP 1: DATASET SHAPE")
    print(f"Rows:    {frame.shape[0]:,}")
    print(f"Columns: {frame.shape[1]}")

    section("STEP 2: FIRST FIVE ROWS")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(frame.head())

    section("STEP 3: COLUMN NAMES AND DATA TYPES")
    dtypes = pd.DataFrame(
        {
            "column": frame.columns,
            "dtype": [str(dtype) for dtype in frame.dtypes],
            "non_null": [int(frame[column].notna().sum()) for column in frame.columns],
        }
    )
    frame_dtypes = dtypes
    print(frame_dtypes.to_string(index=False))

    section("STEP 4: DESCRIPTIVE STATISTICS (Time, Amount)")
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(frame[["Time", "Amount", TARGET]].describe().T)

    print(
        "\nPCA features V1-V28 are already standardized by the dataset authors,\n"
        "so their raw means are near 0 and standard deviations near 1."
    )

    section("STEP 5: MISSING VALUES")
    null_counts = frame.isna().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls == 0:
        print("No missing values in any of the 31 columns.")
    else:
        print(f"Total missing cells: {total_nulls:,}")
        print(null_counts[null_counts > 0].to_string())

    section("STEP 6: DUPLICATE ROWS")
    duplicate_count = int(frame.duplicated().sum())
    print(f"Fully duplicated rows: {duplicate_count:,}")
    print(f"Share of rows: {pct(duplicate_count, len(frame)):.3f}%")
    print(
        "\nNote: these are exact duplicate records across all 31 columns. They are\n"
        "counted and reported here but NOT dropped in Phase 2. Deciding whether to\n"
        "drop them is a modelling decision for the next phase, because a repeated\n"
        "row may be a genuine second transaction with coincidentally identical\n"
        "PCA components, or a true data-entry duplicate."
    )

    summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "columns",
                "file_size_mb",
                "missing_cells",
                "duplicate_rows",
                "time_min_seconds",
                "time_max_seconds",
                "amount_min",
                "amount_max",
            ],
            "value": [
                frame.shape[0],
                frame.shape[1],
                round(DATA_PATH.stat().st_size / (1024**2), 2)
                if DATA_PATH.exists()
                else np.nan,
                total_nulls,
                duplicate_count,
                float(frame["Time"].min()),
                float(frame["Time"].max()),
                float(frame["Amount"].min()),
                float(frame["Amount"].max()),
            ],
        }
    )
    save_table(summary, "dataset_summary.csv", index=False)


# --------------------------------------------------------------------------- #
# Step 3: class imbalance
# --------------------------------------------------------------------------- #


def analyse_class_imbalance(frame: pd.DataFrame) -> pd.DataFrame:
    """Quantify the class imbalance, plot it, and compute a naive baseline."""
    section("STEP 7: TARGET CLASS DISTRIBUTION")

    counts = frame[TARGET].value_counts().sort_index()
    total = len(frame)
    legit_count = int(counts.get(LEGIT, 0))
    fraud_count = int(counts.get(FRAUD, 0))

    distribution = pd.DataFrame(
        {
            "class_label": ["legitimate (0)", "fraudulent (1)"],
            "count": [legit_count, fraud_count],
            "percentage": [
                round(pct(legit_count, total), 4),
                round(pct(fraud_count, total), 4),
            ],
        }
    )
    print(distribution.to_string(index=False))
    print(f"\nTotal transactions: {total:,}")

    ratio = legit_count / fraud_count if fraud_count else float("inf")
    print(f"Fraud-to-legitimate ratio: 1 : {ratio:.0f}")
    print(f"  i.e. about 1 fraudulent transaction for every {ratio:.0f} legitimate ones.")

    baseline_accuracy = pct(legit_count, total)
    print(
        "\nHYPOTHETICAL BASELINE (not a trained model):\n"
        f"  A dummy classifier that predicts 'legitimate' for every transaction\n"
        f"  would be correct on all {legit_count:,} legitimate rows and wrong on\n"
        f"  all {fraud_count:,} fraudulent rows.\n"
        f"  Its accuracy would be {baseline_accuracy:.3f}%,\n"
        f"  while its fraud recall would be 0.000%.\n"
        "  This is the floor any real model must beat, and it is why accuracy is\n"
        "  not a usable metric for this problem."
    )

    save_table(distribution, "class_distribution.csv", index=False)

    # --- bar chart: linear and log scale side by side ---------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    labels = ["Legitimate (0)", "Fraudulent (1)"]
    values = [legit_count, fraud_count]
    colors = ["#4C72B0", "#C44E52"]

    bars = axes[0].bar(labels, values, color=colors)
    axes[0].set_title("Class distribution (linear scale)")
    axes[0].set_ylabel("Number of transactions")
    for bar, value in zip(bars, values):
        axes[0].annotate(
            f"{value:,}\n({pct(value, total):.3f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[0].set_ylim(0, max(values) * 1.18)

    bars = axes[1].bar(labels, values, color=colors)
    axes[1].set_yscale("log")
    axes[1].set_title("Class distribution (log scale)")
    axes[1].set_ylabel("Number of transactions (log)")
    for bar, value in zip(bars, values):
        axes[1].annotate(
            f"{value:,}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[1].set_ylim(0.7, max(values) * 3)

    fig.suptitle(
        f"FraudGuard: class imbalance - {fraud_count:,} fraud in {total:,} transactions "
        f"({pct(fraud_count, total):.3f}%)",
        fontsize=12,
    )
    save_figure(fig, "01_class_distribution.png")

    return distribution


# --------------------------------------------------------------------------- #
# Step 4: transaction Amount
# --------------------------------------------------------------------------- #


def _describe_amount(series: pd.Series) -> dict[str, float]:
    """Return the amount statistics used in the summary table."""
    return {
        "count": float(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "min": float(series.min()),
        "q25": float(series.quantile(0.25)),
        "q75": float(series.quantile(0.75)),
        "max": float(series.max()),
        "skewness": float(series.skew()),
    }


def analyse_amount(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe Amount overall and per class, then plot its distribution."""
    section("STEP 8: TRANSACTION AMOUNT")

    legit_amount = frame.loc[frame[TARGET] == LEGIT, "Amount"]
    fraud_amount = frame.loc[frame[TARGET] == FRAUD, "Amount"]

    statistics = pd.DataFrame(
        {
            "all_transactions": _describe_amount(frame["Amount"]),
            "legitimate": _describe_amount(legit_amount),
            "fraudulent": _describe_amount(fraud_amount),
        }
    ).round(4)
    print(statistics.to_string())

    print(
        f"\nThe mean amount ({frame['Amount'].mean():.2f}) is far above the median "
        f"({frame['Amount'].median():.2f}),\n"
        f"and skewness is {frame['Amount'].skew():.2f}: the distribution is strongly\n"
        "right-skewed with a long tail of large transactions. That is why the\n"
        "median is a more honest 'typical transaction' than the mean."
    )

    zero_amount = int((frame["Amount"] == 0).sum())
    print(f"\nZero-amount transactions: {zero_amount:,} ({pct(zero_amount, len(frame)):.3f}%)")

    # --- IQR outliers: reported, never removed ----------------------------- #
    q1, q3 = frame["Amount"].quantile([0.25, 0.75])
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    outlier_rows = []
    for label, series in (("legitimate", legit_amount), ("fraudulent", fraud_amount)):
        count = int((series > upper_fence).sum())
        outlier_rows.append(
            {
                "class_label": label,
                "row_count": int(series.count()),
                "above_upper_fence": count,
                "share_of_class_pct": round(pct(count, series.count()), 3),
            }
        )
    outliers = pd.DataFrame(outlier_rows)
    print(f"\nIQR upper fence (Q3 + 1.5*IQR): {upper_fence:.2f}")
    print(outliers.to_string(index=False))
    print(
        "  Outliers are reported, not removed. A large amount is a legitimate\n"
        "  fraud signal, so deleting those rows would throw away the signal."
    )

    save_table(statistics, "amount_statistics.csv")
    save_table(outliers, "amount_outliers_iqr.csv", index=False)

    # --- distribution: raw and log-scaled ---------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(frame["Amount"], bins=80, color="#4C72B0")
    axes[0].set_title("Amount distribution (linear scale)")
    axes[0].set_xlabel("Amount")
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(frame["Amount"].mean(), color="#C44E52", linestyle="--", label=f"mean {frame['Amount'].mean():.0f}")
    axes[0].axvline(frame["Amount"].median(), color="#55A868", linestyle="-.", label=f"median {frame['Amount'].median():.0f}")
    axes[0].legend()

    log_amount = np.log1p(frame["Amount"])
    axes[1].hist(log_amount, bins=80, color="#4C72B0")
    axes[1].set_title("Amount distribution (log1p scale)")
    axes[1].set_xlabel("log(1 + Amount)")
    axes[1].set_ylabel("Frequency")
    axes[1].annotate(
        "log1p compresses the extreme tail\nso the bulk of transactions is visible",
        xy=(0.03, 0.92),
        xycoords="axes fraction",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    fig.suptitle("FraudGuard: transaction amount distribution", fontsize=12)
    save_figure(fig, "02_amount_distribution.png")

    # --- legitimate vs fraudulent ------------------------------------------ #
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    sns.histplot(log_amount[frame[TARGET] == LEGIT], bins=60, ax=axes[0][0],
                 color="#4C72B0", label="legitimate", kde=True)
    sns.histplot(log_amount[frame[TARGET] == FRAUD], bins=60, ax=axes[0][0],
                 color="#C44E52", label="fraudulent", kde=True)
    axes[0][0].set_title("log1p(Amount) by class (overlapping counts)")
    axes[0][0].set_xlabel("log(1 + Amount)")
    axes[0][0].legend()

    sns.boxplot(data=frame, x=TARGET, y="Amount", ax=axes[0][1], showfliers=False)
    axes[0][1].set_title("Amount by class (box, outliers hidden for readability)")
    axes[0][1].set_xticks([0, 1], ["Legitimate (0)", "Fraudulent (1)"])
    axes[0][1].set_ylabel("Amount")

    sns.violinplot(data=frame, x=TARGET, y="Amount", ax=axes[1][0], cut=0)
    axes[1][0].set_yscale("log")
    axes[1][0].set_title("Amount by class (violin, log y-axis)")
    axes[1][0].set_xticks([0, 1], ["Legitimate (0)", "Fraudulent (1)"])
    axes[1][0].set_ylabel("Amount (log)")

    # Largest transactions first: how much of the total value sits in the top x%?
    descending = np.sort(frame["Amount"].to_numpy())[::-1]
    cumulative = np.cumsum(descending) / descending.sum() * 100
    share_of_transactions = np.arange(1, len(descending) + 1) / len(descending) * 100
    axes[1][1].plot(share_of_transactions, cumulative, color="#4C72B0")
    axes[1][1].set_xscale("log")
    axes[1][1].set_title("Concentration of value (largest transactions first)")
    axes[1][1].set_xlabel("Top share of transactions, log scale (%)")
    axes[1][1].set_ylabel("Cumulative share of total amount (%)")
    for share in (0.1, 1.0, 10.0):
        index = max(int(len(descending) * share / 100) - 1, 0)
        axes[1][1].axvline(share, color="#999999", linestyle=":", linewidth=0.9)
        axes[1][1].annotate(
            f"top {share:g}% = {cumulative[index]:.1f}% of value",
            xy=(share, cumulative[index]),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=8,
        )

    fig.suptitle("FraudGuard: amount - legitimate vs fraudulent", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "03_amount_by_class.png")

    print(
        f"\nMedian amount - legitimate: {legit_amount.median():.2f} | "
        f"fraudulent: {fraud_amount.median():.2f}"
    )
    print(
        f"Mean amount   - legitimate: {legit_amount.mean():.2f} | "
        f"fraudulent: {fraud_amount.mean():.2f}"
    )
    print(
        "Amount alone is a weak separator: the two classes overlap heavily, which\n"
        "is expected. Its value shows up as one feature among many."
    )

    return statistics


# --------------------------------------------------------------------------- #
# Step 5: transaction Time
# --------------------------------------------------------------------------- #


def analyse_time(frame: pd.DataFrame) -> pd.DataFrame:
    """Describe the elapsed-time axis, plot volume over time and fraud rates."""
    section("STEP 9: TRANSACTION TIME (elapsed seconds, not a calendar date)")

    time_min = float(frame["Time"].min())
    time_max = float(frame["Time"].max())
    span_seconds = time_max - time_min
    span_hours = span_seconds / SECONDS_PER_HOUR

    print(f"Minimum Time: {time_min:,.0f} s  (elapsed at first transaction)")
    print(f"Maximum Time: {time_max:,.0f} s")
    print(f"Span:         {span_hours:,.2f} hours (~{span_hours / 24:.2f} days of continuous recording)")
    print(
        "\nImportant: Time is the number of seconds since the first record. It is\n"
        "NOT a Unix timestamp and the dataset contains no calendar date. Any\n"
        "'hour of day' figure below is an offset within the recording window,\n"
        "not a real clock time or a day of week."
    )

    hourly = (
        frame.assign(elapsed_hour=(frame["Time"] // SECONDS_PER_HOUR).astype(int))
        .groupby("elapsed_hour")[TARGET]
        .agg(transactions="size", fraudulent="sum")
        .reset_index()
    )
    hourly["legitimate"] = hourly["transactions"] - hourly["fraudulent"]
    hourly["fraud_rate_pct"] = (
        hourly["fraudulent"] / hourly["transactions"] * 100
    ).round(4)

    statistics = pd.DataFrame(
        {
            "statistic": [
                "min_seconds",
                "max_seconds",
                "span_hours",
                "span_days",
                "peak_transactions_in_hour",
                "peak_fraud_count_in_hour",
                "highest_hourly_fraud_rate_pct",
                "hours_with_zero_fraud",
            ],
            "value": [
                round(time_min, 2),
                round(time_max, 2),
                round(span_hours, 4),
                round(span_hours / 24, 4),
                int(hourly["transactions"].max()),
                int(hourly["fraudulent"].max()),
                float(hourly["fraud_rate_pct"].max()),
                int((hourly["fraudulent"] == 0).sum()),
            ],
        }
    )
    print("\nTime summary:")
    print(statistics.to_string(index=False))

    save_table(statistics, "time_statistics.csv", index=False)
    save_table(hourly, "time_hourly_summary.csv", index=False)

    busiest = hourly.loc[hourly["transactions"].idxmax()]
    riskiest = hourly.loc[hourly["fraud_rate_pct"].idxmax()]
    print(
        f"\nBusiest elapsed hour: hour {int(busiest['elapsed_hour'])} with "
        f"{int(busiest['transactions']):,} transactions."
    )
    print(
        f"Highest fraud rate:   hour {int(riskiest['elapsed_hour'])} with "
        f"{riskiest['fraud_rate_pct']:.3f}% "
        f"({int(riskiest['fraudulent'])} of {int(riskiest['transactions'])})."
    )
    print(
        "A single hour holding the top fraud rate usually has very few\n"
        "transactions, so that rate is noisy. Compare it against the overall rate\n"
        "before treating it as a pattern."
    )

    # --- volume over time --------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(hourly["elapsed_hour"], hourly["transactions"], color="#4C72B0")
    ax.set_title("Transaction volume per elapsed hour")
    ax.set_xlabel("Elapsed hour since first transaction (seconds // 3600)")
    ax.set_ylabel("Transactions")
    save_figure(fig, "04_time_volume.png")

    # --- fraud pattern over time ------------------------------------------- #
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].bar(hourly["elapsed_hour"], hourly["legitimate"], color="#4C72B0", label="legitimate")
    axes[0].bar(hourly["elapsed_hour"], hourly["fraudulent"], bottom=hourly["legitimate"],
                color="#C44E52", label="fraudulent")
    axes[0].set_title("Transactions per elapsed hour, stacked by class")
    axes[0].set_ylabel("Transactions")
    axes[0].legend()

    axes[1].plot(hourly["elapsed_hour"], hourly["fraudulent"], color="#C44E52",
                 marker="o", markersize=3, label="fraud count")
    axes[1].set_ylabel("Fraudulent transactions", color="#C44E52")
    axes[1].tick_params(axis="y", labelcolor="#C44E52")
    axes[1].set_xlabel("Elapsed hour since first transaction (seconds // 3600)")
    axes[1].set_title("Fraud count and fraud rate per elapsed hour")

    twin = axes[1].twinx()
    twin.plot(hourly["elapsed_hour"], hourly["fraud_rate_pct"], color="#55A868",
              marker="s", markersize=3, linestyle="--", label="fraud rate %")
    twin.axhline(pct(frame[TARGET].sum(), len(frame)), color="black", linestyle=":",
                 label="overall fraud rate")
    twin.set_ylabel("Fraud rate (%)", color="#55A868")
    twin.tick_params(axis="y", labelcolor="#55A868")

    lines = axes[1].get_lines() + twin.get_lines()
    axes[1].legend(lines, [line.get_label() for line in lines], loc="upper left", fontsize=8)

    fig.suptitle("FraudGuard: fraud pattern over elapsed time", fontsize=12)
    fig.tight_layout()
    save_figure(fig, "05_fraud_over_time.png")

    return statistics


# --------------------------------------------------------------------------- #
# Step 6: correlation analysis
# --------------------------------------------------------------------------- #


def analyse_correlations(frame: pd.DataFrame) -> pd.Series:
    """Compute the correlation matrix and rank features by |corr| with Class."""
    section("STEP 10: CORRELATION ANALYSIS")

    correlation = frame.corr(numeric_only=True)

    with_class = (
        correlation[TARGET]
        .drop(labels=[TARGET])
        .sort_values(key=lambda series: series.abs(), ascending=False)
    )
    ranking = pd.DataFrame(
        {
            "feature": with_class.index,
            "correlation_with_class": with_class.to_numpy().round(6),
            "abs_correlation": with_class.abs().to_numpy().round(6),
        }
    )
    print("Features ranked by absolute correlation with Class (top 15):")
    print(ranking.head(15).to_string(index=False))

    save_table(ranking, "correlation_with_class.csv", index=False)

    strongest = ranking.iloc[0]
    weakest = ranking.iloc[-1]
    max_abs = float(ranking["abs_correlation"].max())
    print(
        f"\nStrongest single linear association: {strongest['feature']} "
        f"(r = {strongest['correlation_with_class']:+.4f})"
    )
    print(
        f"Weakest: {weakest['feature']} (r = {weakest['correlation_with_class']:+.4f})"
    )
    print(
        f"No single feature exceeds |r| = {max_abs:.4f}. There is no feature that\n"
        "separates fraud by itself: fraud here is a multivariate, weakly-signalled\n"
        "pattern. That is the main modelling lesson of this phase."
    )

    # strongest positive / negative pairs among features (excluding target)
    feature_corr = correlation.drop(index=TARGET, columns=TARGET)
    upper = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
    stacked = upper.stack().sort_values(key=lambda series: series.abs(), ascending=False)
    print("\nMost strongly correlated feature pairs (excluding Class):")
    for (left, right), value in stacked.head(5).items():
        print(f"  {left:>7} - {right:<7} r = {value:+.4f}")
    print(
        "  These are expected artifacts of PCA: components are constructed to be\n"
        "  orthogonal, so near-perfect pairs are rare. Any pair that does show-up\n"
        "  strong correlation hints at redundancy worth reviewing before modelling."
    )

    # --- heatmap of the full matrix ---------------------------------------- #
    fig, ax = plt.subplots(figsize=(16, 13))
    sns.heatmap(
        correlation,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.2,
        cbar_kws={"shrink": 0.7, "label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Correlation matrix: Time, V1-V28, Amount, Class", fontsize=13)
    save_figure(fig, "06_correlation_heatmap.png")

    # --- ranked bar chart --------------------------------------------------- #
    top = ranking.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#C44E52" if value > 0 else "#4C72B0" for value in top["correlation_with_class"]]
    ax.barh(top["feature"], top["correlation_with_class"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Top 15 features by correlation with Class")
    ax.set_xlabel("Pearson correlation with Class")
    for index, value in enumerate(top["correlation_with_class"]):
        ax.text(value, index, f" {value:+.3f}", va="center",
                ha="left" if value > 0 else "right", fontsize=9)
    save_figure(fig, "07_correlation_with_class.png")

    return with_class


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    """Run every Phase 2 analysis in order."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("FraudGuard - Phase 2: dataset loading and exploratory data analysis")
    print(f"Project root: {PROJECT_ROOT}")

    path = resolve_dataset_path()
    frame = load_dataset(path)

    inspect_structure(frame)
    analyse_class_imbalance(frame)
    analyse_amount(frame)
    analyse_time(frame)
    analyse_correlations(frame)

    section("PHASE 2 COMPLETE")
    print(f"Plots written to:  {PLOTS_DIR}")
    print(f"Tables written to: {TABLES_DIR}")
    print(f"Raw dataset left untouched: {path}")
    print("\nNext phase (not started): model training and /predict. No model was")
    print("trained and no transaction was scored by this script.")


if __name__ == "__main__":
    main()
