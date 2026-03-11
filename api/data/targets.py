"""
Supervised targets: log returns and direction labels.
"""
import numpy as np
import pandas as pd


def add_return_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    For each horizon h, add log_return_{h}d = log(close_{t+h} / close_t) per asset.
    Rows at the tail (no future close) get NaN and should be dropped by caller.
    """
    out = df.copy()
    grouped = out.groupby("asset_id", group_keys=False)
    for h in horizons:
        col = f"log_return_{h}d"
        out[col] = grouped["close"].transform(lambda s: np.log(s.shift(-h) / s))
        out[col] = out[col].astype(np.float32)
    return out


def make_direction_label(
    df: pd.DataFrame, horizon: int, threshold: float = 0.0
) -> pd.Series:
    """Return int8 Series: 1 if log_return_{horizon}d > threshold, else 0."""
    col = f"log_return_{horizon}d"
    if col not in df.columns:
        raise KeyError(f"Column {col} not found; run add_return_targets first")
    return (df[col] > threshold).astype(np.int8)
