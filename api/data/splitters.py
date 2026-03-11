"""
Date-based train/val/test splitting; no percentage-based splits.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def split_by_date(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
    test_end: str,
    date_col: str = 'timestamp',
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by date boundaries: train <= train_end, val in (train_end, val_end], test in (val_end, test_end].
    """
    if date_col not in df.columns:
        raise ValueError(f"date_col '{date_col}' not in DataFrame")
    ts = pd.to_datetime(df[date_col])
    train_df = df.loc[ts <= train_end].copy()
    val_df = df.loc[(ts > train_end) & (ts <= val_end)].copy()
    test_df = df.loc[(ts > val_end) & (ts <= test_end)].copy()
    if train_df.empty:
        raise ValueError('empty_train_split')
    if val_df.empty:
        raise ValueError('empty_val_split')
    if test_df.empty:
        raise ValueError('empty_test_split')
    logger.info(
        'split_by_date: train=%d val=%d test=%d',
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


def validate_split_no_overlap(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    date_col: str = 'timestamp',
) -> None:
    """Assert no date overlap between splits."""
    t_max = pd.to_datetime(train_df[date_col]).max()
    v_min = pd.to_datetime(val_df[date_col]).min()
    v_max = pd.to_datetime(val_df[date_col]).max()
    s_min = pd.to_datetime(test_df[date_col]).min()
    if t_max >= v_min:
        raise ValueError(f'Train/val overlap: train max {t_max} >= val min {v_min}')
    if v_max >= s_min:
        raise ValueError(f'Val/test overlap: val max {v_max} >= test min {s_min}')
