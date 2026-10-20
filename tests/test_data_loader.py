import pandas as pd
import pytest

from src.data_loader import REQUIRED_COLUMNS, load_transactions


def test_loads_a_valid_file(tmp_path):
    path = tmp_path / "transactions.csv"
    pd.DataFrame({col: [1, 2] for col in REQUIRED_COLUMNS}).to_csv(path, index=False)

    df = load_transactions(path)

    assert len(df) == 2
    assert set(REQUIRED_COLUMNS) <= set(df.columns)


def test_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_transactions(tmp_path / "does_not_exist.csv")


def test_raises_for_missing_columns(tmp_path):
    path = tmp_path / "transactions.csv"
    pd.DataFrame({"AccountID": [1, 2]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Missing columns"):
        load_transactions(path)