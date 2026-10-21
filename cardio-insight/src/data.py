"""Loading and basic cleaning of the heart disease dataset."""
import numpy as np
import pandas as pd

from .config import RAW_DATA, ZERO_AS_MISSING, CATEGORY_LABELS


def load_raw(path=RAW_DATA) -> pd.DataFrame:
    """Read the original Kaggle CSV exactly as it is."""
    return pd.read_csv(path)


def zeros_to_nan(df: pd.DataFrame, columns=ZERO_AS_MISSING) -> pd.DataFrame:
    """Return a copy where impossible zeros (Cholesterol = 0, RestingBP = 0) become NaN."""
    out = df.copy()
    for col in columns:
        out[col] = out[col].replace(0, np.nan)
    return out


def load_clean(path=RAW_DATA) -> pd.DataFrame:
    """Raw data with impossible zeros marked as missing (NaN)."""
    return zeros_to_nan(load_raw(path))


def readable(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of the data with category codes replaced by readable names (for display only)."""
    out = df.copy()
    for col, mapping in CATEGORY_LABELS.items():
        if col in out:
            out[col] = out[col].map(mapping).fillna(out[col])
    return out
