"""Create the figures used in the README (writes PNG files to assets/)."""
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RANDOM_STATE
from src.data_loader import load_transactions
from src.feature_engineering import prepare_features
from src.models import score_isolation_forest, score_kmeans, score_lof

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
TOP_SHARE = 0.05  # same top-5% group as in main.py
N_INJECTED_PER_TYPE = 10
MODELS = {
    "Isolation Forest": score_isolation_forest,
    "LOF (scaled)": score_lof,
    "K-Means distance": score_kmeans,
}
COLORS = {"Isolation Forest": "#4f8cff", "LOF (scaled)": "#7c5cff", "K-Means distance": "#8b98b8"}

logger = logging.getLogger(__name__)


def recall_at_top(scores: np.ndarray, injected_pos: np.ndarray, n: int) -> float:
    """Share of injected rows that appear among the n most anomalous rows."""
    top = np.argsort(scores)[:n]
    return len(np.intersect1d(top, injected_pos)) / len(injected_pos)


def z_profile(X_ref: pd.DataFrame, top_idx: np.ndarray) -> pd.Series:
    """Mean of the flagged rows in units of the overall standard deviation, per feature."""
    return (X_ref.iloc[top_idx].mean() - X_ref.mean()) / X_ref.std()


def injected_recall_table(X_train: pd.DataFrame, X_test: pd.DataFrame, n_top: int) -> pd.DataFrame:
    """Recall in the top-5% for two injected anomaly types (tested separately)."""
    rng = np.random.default_rng(RANDOM_STATE)
    positions = rng.choice(len(X_test), size=2 * N_INJECTED_PER_TYPE, replace=False)
    pos_a, pos_b = positions[:N_INJECTED_PER_TYPE], positions[N_INJECTED_PER_TYPE:]

    rows = []
    # Type A: one extreme amount, from inside to outside the training range
    for amount in [800, 1500, 1900, 5000]:
        X_inj = X_test.astype(float).copy()
        X_inj.iloc[pos_a, X_inj.columns.get_loc("TransactionAmount")] = amount
        rows.append({"case": f"A: amount\n= {amount}",
                     **{name: recall_at_top(f(X_train, X_inj), pos_a, n_top) for name, f in MODELS.items()}})

    # Type B: three features moderately high (90% quantile) at the same time
    X_inj = X_test.astype(float).copy()
    for col in ["TransactionAmount", "TransactionDuration", "Transaction_diff"]:
        X_inj.iloc[pos_b, X_inj.columns.get_loc(col)] = X_train[col].quantile(0.9)
    rows.append({"case": "B: 3 features\nat 90% quantile",
                 **{name: recall_at_top(f(X_train, X_inj), pos_b, n_top) for name, f in MODELS.items()}})
    return pd.DataFrame(rows).set_index("case")


def plot_grouped_bars(table: pd.DataFrame, title: str, ylabel: str, path: Path, chance: float | None = None) -> None:
    """Grouped bar chart with one bar per model in each group."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    width = 0.8 / len(table.columns)
    x = np.arange(len(table))
    for i, name in enumerate(table.columns):
        ax.bar(x + i * width - 0.4 + width / 2, table[name], width, label=name, color=COLORS[name])
    if chance is not None:
        ax.axhline(chance, color="#d29922", linestyle="--", linewidth=1, label=f"chance level ({chance:.2f})")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", path.name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ASSETS_DIR.mkdir(exist_ok=True)

    X_train, X_test = prepare_features(load_transactions())
    n_top = int(len(X_test) * TOP_SHARE)

    recall = injected_recall_table(X_train, X_test, n_top)
    plot_grouped_bars(recall, "Injected anomalies found in the top 5% (recall)", "recall",
                      ASSETS_DIR / "injected-anomalies-recall.png", chance=TOP_SHARE)

    profile = pd.DataFrame({
        name: z_profile(X_test, np.argsort(f(X_train, X_test))[:n_top]) for name, f in MODELS.items()
    })
    plot_grouped_bars(profile, "Features of the flagged transactions\n(mean vs. all test rows, in standard deviations)",
                      "standard deviations above average", ASSETS_DIR / "feature-profile.png")

    logger.info("Injected-anomaly recall:\n%s", recall.rename(index=lambda s: s.replace("\n", " ")).round(2).to_string())
    logger.info("Feature profile (z):\n%s", profile.round(2).to_string())


if __name__ == "__main__":
    main()
