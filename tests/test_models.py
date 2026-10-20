import numpy as np
import pandas as pd
import pytest

from src.models import score_isolation_forest, score_kmeans, score_lof

MODELS = [score_isolation_forest, score_lof, score_kmeans]


@pytest.fixture
def data():
    """Dense normal cloud for training; the test set contains one obvious outlier at position 0."""
    rng = np.random.default_rng(0)
    columns = ["a", "b", "c"]
    X_train = pd.DataFrame(rng.normal(0, 1, size=(300, 3)), columns=columns)
    X_test = pd.DataFrame(rng.normal(0, 1, size=(50, 3)), columns=columns)
    X_test.iloc[0] = [15.0, 15.0, 15.0]
    return X_train, X_test


@pytest.mark.parametrize("score_function", MODELS)
def test_returns_one_score_per_test_row(score_function, data):
    X_train, X_test = data

    scores = score_function(X_train, X_test)

    assert scores.shape == (len(X_test),)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("score_function", MODELS)
def test_obvious_outlier_gets_the_lowest_score(score_function, data):
    # All models follow the convention: lower score = more anomalous
    X_train, X_test = data

    scores = score_function(X_train, X_test)

    assert np.argmin(scores) == 0