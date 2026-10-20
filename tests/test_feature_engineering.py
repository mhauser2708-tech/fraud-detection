import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    FEATURES_FOR_MODEL,
    convert_date_columns,
    create_transaction_difference,
    handle_missing_transaction_diff,
    prepare_features,
)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Small synthetic dataset with the same columns the pipeline needs."""
    rng = np.random.default_rng(0)
    n = 200
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n), unit="min")
    return pd.DataFrame({
        "AccountID": rng.choice([f"AC{i:03d}" for i in range(20)], n),
        "TransactionDate": dates.strftime("%Y-%m-%d %H:%M:%S"),
        "PreviousTransactionDate": "2024-11-04 08:00:00",
        "TransactionAmount": rng.uniform(1, 500, n),
        "LoginAttempts": rng.integers(1, 5, n),
        "TransactionDuration": rng.integers(10, 300, n),
        "CustomerAge": rng.integers(18, 80, n),
    })


def test_transaction_diff_is_hours_since_previous_transaction_of_same_account():
    # Rows are deliberately unsorted and belong to two accounts
    df = pd.DataFrame({
        "AccountID": ["AC1", "AC2", "AC1"],
        "TransactionDate": ["2023-11-14 16:56:34", "2023-01-10 16:00:32", "2023-09-15 17:00:20"],
        "PreviousTransactionDate": ["2024-11-04 08:00:00"] * 3,
    })

    result = create_transaction_difference(convert_date_columns(df))
    ac1 = result[result["AccountID"] == "AC1"]

    assert ac1["Transaction_diff"].iloc[1] == pytest.approx(1439.94)  # about 60 days
    assert ac1["Transaction_diff"].iloc[0] != ac1["Transaction_diff"].iloc[0]  # NaN for the first transaction
    assert result[result["AccountID"] == "AC2"]["Transaction_diff"].isna().all()  # single transaction, no predecessor


def test_handle_missing_flags_first_transactions_and_fills_with_median():
    # Median (30) and mean (40) differ, so the test notices if one is swapped for the other
    df = pd.DataFrame({"Transaction_diff": [np.nan, 10.0, 30.0, 80.0, np.nan]})

    result = handle_missing_transaction_diff(df)

    assert result["is_first_transaction"].tolist() == [1, 0, 0, 0, 1]
    assert result["Transaction_diff"].tolist() == [30.0, 10.0, 30.0, 80.0, 30.0]
    assert df["Transaction_diff"].isna().sum() == 2  # input is not modified


def test_handle_missing_uses_given_fill_value():
    df = pd.DataFrame({"Transaction_diff": [np.nan, 10.0, 30.0]})

    result = handle_missing_transaction_diff(df, fill_value=99.0)

    assert result["Transaction_diff"].iloc[0] == 99.0  # not the median (20)


def test_prepare_features_split_columns_and_no_missing_values(raw_df):
    X_train, X_test = prepare_features(raw_df, test_size=0.3, random_state=1)

    assert len(X_train) + len(X_test) == len(raw_df)
    assert len(X_test) == 60
    assert list(X_train.columns) == FEATURES_FOR_MODEL
    assert list(X_test.columns) == FEATURES_FOR_MODEL
    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0


def test_prepare_features_is_reproducible(raw_df):
    first_train, first_test = prepare_features(raw_df, random_state=1)
    second_train, second_test = prepare_features(raw_df, random_state=1)

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)