from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans

from src.config import RANDOM_STATE


def score_isolation_forest(X_train, X_test, random_state=RANDOM_STATE):
    """Fit an Isolation Forest on X_train and return anomaly scores for X_test (lower = more anomalous)."""
    model = IsolationForest(random_state=random_state).fit(X_train)
    return model.score_samples(X_test)


def score_lof(X_train, X_test, n_neighbors=20):
    """Fit a scaled LOF on X_train and return anomaly scores for X_test (lower = more anomalous)."""
    scaler = RobustScaler().fit(X_train)
    model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True).fit(scaler.transform(X_train))
    return model.score_samples(scaler.transform(X_test))

def score_kmeans(X_train, X_test, n_clusters=4, random_state=RANDOM_STATE):
    """
    Fit a scaled K-Means on X_train and return anomaly scores for X_test.

    The score is the negative distance to the nearest cluster centre, so that
    lower = more anomalous, like for the other models.
    """
    scaler = RobustScaler().fit(X_train)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(scaler.transform(X_train))
    # model.transform returns the distance to every centre; the minimum is the nearest one
    return -model.transform(scaler.transform(X_test)).min(axis=1)