"""
Alpha feature schema and merge logic for LLM-generated alpha factors.

Defines the alpha column schema (alpha_1..alpha_8 + alpha_missing_flag)
and provides merge/fill functions for integrating alpha features into
the multimodal feature matrix.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ALPHA_COLUMNS = [
    'alpha_1',
    'alpha_2',
    'alpha_3',
    'alpha_4',
    'alpha_5',
    'alpha_6',
    'alpha_7',
    'alpha_8',
]

ALPHA_MISSING_FLAG = 'alpha_missing_flag'

ALL_ALPHA_COLS = ALPHA_COLUMNS + [ALPHA_MISSING_FLAG]


def merge_alpha_features(
    base_df: pd.DataFrame,
    alpha_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join alpha factors onto the base dataframe by (timestamp, asset_id).

    Args:
        base_df: Base multimodal dataframe with timestamp and asset_id columns.
        alpha_df: Alpha dataframe with timestamp, asset_id, and alpha_1..alpha_8
            columns.

    Returns:
        Merged dataframe with alpha columns added.

    Raises:
        ValueError: If alpha_df has duplicate (timestamp, asset_id) entries.
    """
    if alpha_df is not None and not alpha_df.empty:
        # Check for duplicates
        dup_mask = alpha_df.duplicated(subset=['timestamp', 'asset_id'], keep=False)
        if dup_mask.any():
            raise ValueError(
                f'Duplicate (timestamp, asset_id) entries in alpha_df: '
                f'{dup_mask.sum()} rows'
            )
        # Normalize timestamps for join
        base_df = base_df.copy()
        alpha_df = alpha_df.copy()
        base_df['timestamp'] = pd.to_datetime(base_df['timestamp']).dt.normalize()
        alpha_df['timestamp'] = pd.to_datetime(alpha_df['timestamp']).dt.normalize()

        alpha_cols_to_merge = ['timestamp', 'asset_id'] + [
            c for c in ALPHA_COLUMNS if c in alpha_df.columns
        ]
        merged = base_df.merge(
            alpha_df[alpha_cols_to_merge],
            on=['timestamp', 'asset_id'],
            how='left',
        )
    else:
        merged = base_df.copy()

    return fill_missing_alphas(merged)


def fill_missing_alphas(
    df: pd.DataFrame,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Fill NaN alpha columns with fill_value and set alpha_missing_flag.

    Args:
        df: DataFrame that may have NaN alpha columns.
        fill_value: Value to use for missing alpha factors.

    Returns:
        DataFrame with alpha NaNs replaced and alpha_missing_flag set.
    """
    out = df.copy()

    # Ensure all alpha columns exist
    for col in ALPHA_COLUMNS:
        if col not in out.columns:
            out[col] = fill_value

    # Determine which rows have missing alphas (any alpha col is NaN)
    alpha_missing = out[ALPHA_COLUMNS].isna().any(axis=1).astype(np.int8)

    # Fill NaN alpha values
    for col in ALPHA_COLUMNS:
        out[col] = out[col].fillna(fill_value).astype(np.float32)

    # Clip alpha values to [-3, 3]
    for col in ALPHA_COLUMNS:
        out[col] = out[col].clip(-3.0, 3.0)

    out[ALPHA_MISSING_FLAG] = alpha_missing
    return out
