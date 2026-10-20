import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering import prepare_features
from src.models import score_isolation_forest, score_lof

DATA_PATH = Path("data") / "bank_transactions_data_2.csv"
TOP_SHARE = 0.05  # share of test transactions treated as "most anomalous"

logger = logging.getLogger(__name__)


def top_positions(scores: np.ndarray, n: int) -> np.ndarray:
    """Positions of the n most anomalous rows (lowest scores)."""
    return np.argsort(scores)[:n]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    logger.info("Loading data from %s", DATA_PATH)
    X_train, X_test = prepare_features(pd.read_csv(DATA_PATH))
    logger.info("Train: %s | Test: %s", X_train.shape, X_test.shape)

    scores = {
        "Isolation Forest": score_isolation_forest(X_train, X_test),
        "LOF (scaled)": score_lof(X_train, X_test),
    }

    n_top = int(len(X_test) * TOP_SHARE)
    tops = {name: top_positions(s, n_top) for name, s in scores.items()}

    for name, top in tops.items():
        flagged = X_test.iloc[top]
        logger.info("%s flags %d transactions | mean amount %.0f | mean login attempts %.2f",
                    name, n_top, flagged["TransactionAmount"].mean(), flagged["LoginAttempts"].mean())

    common = len(np.intersect1d(*tops.values()))
    logger.info("Overlap between the two methods: %d of %d", common, n_top)


if __name__ == "__main__":
    main()