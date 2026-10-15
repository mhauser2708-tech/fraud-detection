import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, TEST_SIZE

def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts date columns to datetime format.
    
    Args:
        df: DataFrame with date columns to convert.
    
    Returns:
        DataFrame with converted date columns to DateTime format.
        
    """
    df = df.copy()  
    df["TransactionDate_dt"] = pd.to_datetime(df["TransactionDate"])
    df["PreviousTransactionDate_dt"] = pd.to_datetime(df["PreviousTransactionDate"])
    return df

def create_transaction_difference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sorts and groups the DataFrame by 'AccountID' and 'TransactionDate_dt', then calculates an intermediate column 'PrevTransactionDate_calc' that contains the previous transaction date for each account.
    Creates a new column 'Transaction_diff' that calculates the difference between 
    'TransactionDate_dt' and 'PreviousTransactionDate_calc'.
    
    Args:
        df: DataFrame with date columns.

    Returns:
        DataFrame with the new 'Transaction_diff' column.
    """
    df = df.copy()
    df = df.sort_values(by=['AccountID', 'TransactionDate_dt'])
    df = df.reset_index(drop=True) 
    df["PrevTransactionDate_calc"] = df.groupby("AccountID")["TransactionDate_dt"].shift(1) 
    df["Transaction_diff"] = ((df["TransactionDate_dt"]-df["PrevTransactionDate_calc"])/pd.Timedelta(hours=1)).round(2)
    return df

def add_location_deviation_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a flag indicating whether a transaction occurred at an unusual location for each account.
    
    Args:
        df: DataFrame with location columns.
    
    Returns:
        DataFrame with the new 'is_unusal_Location' and 'HomeLocation' columns.
    """
    df = df.copy()
    df["HomeLocation"] = df.groupby("AccountID")["Location"].transform(lambda x: x.mode()[0]) 
    df["is_unusal_Location"] = (df["Location"] != df["HomeLocation"]).astype(int)
    return df

def check_account_balance_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks the consistency of account balances by calculating the previous account balance, the difference in balance, and a flag indicating whether the transaction direction matches the balance change.
    
    Args:
        df: DataFrame with account balance and transaction type columns.

    Returns:
        DataFrame with the new 'PrevAccountBalance', 'BalanceDiff', and 'direction_matches' columns.

    Note:
    Rows representing the first transaction of an account have no previous
    balance, so 'direction_matches' is technically False there, even though
    no real comparison was possible.        
    """
    df = df.copy()
    df["PrevAccountBalance"] = df.groupby("AccountID")["AccountBalance"].shift(1)
    df["BalanceDiff"] = df["AccountBalance"] - df["PrevAccountBalance"]
    
    df["direction_matches"] =((df["BalanceDiff"] > 0) & (df["TransactionType"] == "Credit")) | ((df["BalanceDiff"] < 0) & (df["TransactionType"] == "Debit"))
    df_temp = df.dropna(subset=["BalanceDiff"])
    mismatch_rate = 1 - df_temp["direction_matches"].mean()
    print(mismatch_rate)

    return df

def build_features_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds features for the machine learning model by applying a series of feature engineering steps.
    
    Orchestrates the feature engineering pipeline by calling the individual
    feature functions in the correct order. Only includes features that were
    validated as useful during EDA (see notebooks/01_eda.ipynb for the
    discarded Location/AccountBalance checks).
    
    Args:
        df: DataFrame containing the raw transaction data. 
    
    Returns:
        DataFrame with the engineered features.
    """
    df = convert_date_columns(df)   
    df = create_transaction_difference(df)  
    return df

def handle_missing_transaction_diff(
    df: pd.DataFrame, fill_value: float | None = None
) -> pd.DataFrame:
    """
    Handles missing values in the transaction difference column.
    Adds new column 'is_first_transaction' to indicate whether a transaction is the first for an account.
    
    Args:
        df: DataFrame containing the transaction data.
        fill_value: Value to use for filling missing values. If None, uses the median.
    
    Returns:
        DataFrame with missing values in 'Transaction_diff' filled.
    """
    df = df.copy()
    df["is_first_transaction"] = df["Transaction_diff"].isna().astype(int)   
    if fill_value is None:
        fill_value = df["Transaction_diff"].median()               
    df["Transaction_diff"] = df["Transaction_diff"].fillna(fill_value)
    return df

FEATURES_FOR_MODEL = [
    "TransactionAmount",
    "LoginAttempts",
    "TransactionDuration",
    "CustomerAge",
    "Transaction_diff",
]


def prepare_features(
    df_raw: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds features, splits into train/test and handles missing values.

    The median used to fill 'Transaction_diff' is computed on the training
    part only and applied to both parts (fit on train, transform on test),
    so no information from the test set leaks into preprocessing.

    Args:
        df_raw: Raw transaction data as loaded from the CSV file.
        test_size: Fraction of rows used for the test set.
        random_state: Seed for a reproducible split.

    Returns:
        Tuple (X_train, X_test) containing only the columns in FEATURES_FOR_MODEL.
    """
    df_features = build_features_for_model(df_raw)

    df_train, df_test = train_test_split(
        df_features, test_size=test_size, random_state=random_state
    )

    median_train = df_train["Transaction_diff"].median()
    df_train = handle_missing_transaction_diff(df_train, fill_value=median_train)
    df_test = handle_missing_transaction_diff(df_test, fill_value=median_train)

    return df_train[FEATURES_FOR_MODEL], df_test[FEATURES_FOR_MODEL]