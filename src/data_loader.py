from pathlib import Path

import pandas as pd

# Resolved relative to this file, so the code runs from any working directory
DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "bank_transactions_data_2.csv"

# Columns the feature pipeline needs
REQUIRED_COLUMNS = [
    "AccountID",
    "TransactionDate",
    "PreviousTransactionDate",
    "TransactionAmount",
    "LoginAttempts",
    "TransactionDuration",
    "CustomerAge",
]


def load_transactions(path: Path | str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw transaction CSV and check that the required columns exist.

    Args:
        path: Path to the CSV file. Defaults to data/bank_transactions_data_2.csv in the project.

    Returns:
        Raw transactions as a DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}. See the README for the dataset source.")

    df = pd.read_csv(path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df
