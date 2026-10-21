"""Outlier detection: two statistical rules and two machine-learning methods.

Every detector returns a boolean pandas Series (True = flagged as an outlier)
aligned to the input index, so results from different methods can be compared
row by row.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from .config import RANDOM_STATE

METHODS = ["IQR", "Z-score", "Isolation Forest", "LOF"]


def iqr_bounds(x, k=1.5):
    """Tukey fences: [Q1 - k*IQR, Q3 + k*IQR]."""
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    spread = q3 - q1
    return q1 - k * spread, q3 + k * spread


def iqr_outliers(x: pd.Series, k=1.5) -> pd.Series:
    """Flag values outside the 1.5 x IQR fences."""
    lo, hi = iqr_bounds(x, k)
    return (x < lo) | (x > hi)


def zscore_outliers(x: pd.Series, threshold=3.0) -> pd.Series:
    """Flag values whose z-score |(x - mean) / std| is above the threshold."""
    z = (x - x.mean()) / x.std()
    return z.abs() > threshold


def per_feature_flags(df, columns, method):
    """Apply a univariate rule to each column; a row is an outlier if ANY column is flagged."""
    rule = iqr_outliers if method == "IQR" else zscore_outliers
    flags = pd.concat([rule(df[c]) for c in columns], axis=1)
    return flags.any(axis=1)


def isolation_forest_outliers(df, columns, contamination=0.05, random_state=RANDOM_STATE):
    """Isolation Forest on standardised features.

    Points that can be isolated with only a few random splits are unusual.
    """
    X = StandardScaler().fit_transform(df[columns])
    model = IsolationForest(contamination=contamination, random_state=random_state, n_estimators=300)
    return pd.Series(model.fit_predict(X) == -1, index=df.index)


def lof_outliers(df, columns, contamination=0.05, n_neighbors=20):
    """Local Outlier Factor on standardised features.

    Compares how dense a point's neighbourhood is to the density around its
    neighbours; much lower local density => outlier.
    """
    X = StandardScaler().fit_transform(df[columns])
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    return pd.Series(model.fit_predict(X) == -1, index=df.index)


def detect_all(df, univariate_cols, multivariate_cols, contamination=0.05):
    """Run all four methods and return a DataFrame of boolean flags (one column per method)."""
    return pd.DataFrame({
        "IQR": per_feature_flags(df, univariate_cols, "IQR"),
        "Z-score": per_feature_flags(df, univariate_cols, "Z-score"),
        "Isolation Forest": isolation_forest_outliers(df, multivariate_cols, contamination),
        "LOF": lof_outliers(df, multivariate_cols, contamination),
    })


def comparison_table(flags: pd.DataFrame) -> pd.DataFrame:
    """Number and percentage of rows flagged by each method."""
    n = len(flags)
    return pd.DataFrame({
        "Outliers flagged": flags.sum(),
        "% of rows": (flags.mean() * 100).round(2),
    })


def overlap_matrix(flags: pd.DataFrame) -> pd.DataFrame:
    """Cell (A, B) = number of rows flagged by BOTH method A and method B."""
    m = flags.astype(int)
    return m.T @ m


def agreement_summary(flags: pd.DataFrame) -> pd.DataFrame:
    """How many rows were flagged by exactly 0, 1, 2, 3 or 4 methods."""
    votes = flags.sum(axis=1)
    out = votes.value_counts().reindex(range(flags.shape[1] + 1), fill_value=0)
    return out.rename_axis("Methods agreeing").to_frame("Rows")
