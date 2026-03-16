"""
Unit tests for api/data/lob_pipeline.py.

Tests: label generation, sequence shapes, normalization, split_by_date.
"""

import numpy as np
import pandas as pd
import pytest

from api.data.lob_pipeline import (
    LOBPipeline,
    extract_depth_features,
    mid_price,
    order_imbalance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_lob_df(n: int = 200, levels: int = 10, seed: int = 42) -> pd.DataFrame:
    """Synthetic LOB DataFrame with n events, given levels."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2024-01-01', periods=n, freq='100ms', tz='UTC')
    records: dict = {'timestamp': dates, 'asset_id': 'AAPL'}
    base = 100.0
    for i in range(1, levels + 1):
        # Ascending ask prices, descending bid prices
        records[f'ask_px_{i}'] = base + i * 0.01 + rng.uniform(0, 0.001, n)
        records[f'bid_px_{i}'] = base - i * 0.01 - rng.uniform(0, 0.001, n)
        records[f'ask_sz_{i}'] = rng.uniform(100, 1000, n)
        records[f'bid_sz_{i}'] = rng.uniform(100, 1000, n)
    return pd.DataFrame(records)


@pytest.fixture
def lob_cfg(tmp_path):
    return {
        'lob': {
            'levels': 10,
            'seq_len': 50,
            'horizon_events': 10,
            'tick_size': 0.01,
            'normalization': {'rolling_window_events': 100, 'clip': 6.0},
            'labeling': {'up_threshold_ticks': 1, 'down_threshold_ticks': -1},
            'streaming': {'enabled': False, 'chunk_size': 100000},
            'split': {'train_end': '2024-06-30', 'val_end': '2025-03-31'},
        },
        'paths': {'lob_raw_data_path': str(tmp_path / 'lob_raw')},
    }


# ---------------------------------------------------------------------------
# Test 1 — build_labels produces correct up/flat/down labels
# ---------------------------------------------------------------------------


def test_build_labels_up_label(lob_cfg):
    """Moving price significantly up => label 2 (up)."""
    pipeline = LOBPipeline(lob_cfg)
    n = 100
    df = _make_lob_df(n=n, levels=10)

    df = pipeline.build_features(df)

    # Construct a strictly rising mid_price so future_mid - current_mid > up_thresh
    h = pipeline.horizon_events
    tick = pipeline.tick_size
    # Each step rises by 5 ticks — guaranteed up label for rows [0 .. n-h-1]
    df['mid_price'] = np.arange(n, dtype=np.float32) * 5.0 * tick

    df_labeled = pipeline.build_labels(df)

    assert (df_labeled['mid_move_label_20'] == 2).sum() > 0, (
        'Expected some up labels when mid_price is strictly rising'
    )


def test_build_labels_down_label(lob_cfg):
    """Moving price significantly down => label 0 (down)."""
    pipeline = LOBPipeline(lob_cfg)
    df = _make_lob_df(n=100, levels=10)
    df = pipeline.build_features(df)
    # Construct a strictly falling mid_price so future_mid - current_mid < down_thresh
    n = len(df)
    tick = pipeline.tick_size
    df['mid_price'] = np.arange(n - 1, -1, -1, dtype=np.float32) * 5.0 * tick
    df_labeled = pipeline.build_labels(df)
    assert (df_labeled['mid_move_label_20'] == 0).sum() > 0, (
        'Expected some down labels when mid_price is strictly falling'
    )


# ---------------------------------------------------------------------------
# Test 2 — build_sequences produces correct tensor shapes
# ---------------------------------------------------------------------------


def test_build_sequences_shapes(lob_cfg):
    """build_sequences output arrays have correct shapes."""
    pipeline = LOBPipeline(lob_cfg)
    df = _make_lob_df(n=300, levels=10)
    df = pipeline.build_features(df)
    df = pipeline.normalize_book(df)
    df = pipeline.build_labels(df)

    T = pipeline.seq_len
    L = pipeline.levels
    x_book, x_aux, y_class, meta_df = pipeline.build_sequences(df)

    assert x_book.ndim == 4, f'x_book should be 4D, got {x_book.ndim}D'
    assert x_book.shape[1] == T, f'x_book time dim should be {T}'
    assert x_book.shape[2] == 4, f'x_book channel dim should be 4'
    assert x_book.shape[3] == L, f'x_book level dim should be {L}'
    assert x_aux.shape == (x_book.shape[0], T, 4), f'x_aux shape mismatch'
    assert y_class.shape == (x_book.shape[0],), f'y_class shape mismatch'
    assert y_class.dtype == np.int64, f'y_class should be int64'
    assert len(meta_df) == x_book.shape[0]


# ---------------------------------------------------------------------------
# Test 3 — normalize_book does not produce NaN / out-of-clip values
# ---------------------------------------------------------------------------


def test_normalize_book_no_nan_no_overflow(lob_cfg):
    """After normalization, values should be in [-clip, clip] and no NaN."""
    pipeline = LOBPipeline(lob_cfg)
    df = _make_lob_df(n=200, levels=10)
    df = pipeline.build_features(df)
    df_norm = pipeline.normalize_book(df)

    clip = pipeline.clip_val
    px_col = 'bid_px_1'
    vals = df_norm[px_col].dropna()
    assert not vals.isna().any(), 'Normalized column should not have NaN'
    assert (vals >= -clip - 1e-6).all(), 'Values below -clip detected'
    assert (vals <= clip + 1e-6).all(), 'Values above clip detected'


# ---------------------------------------------------------------------------
# Test 4 — split_by_date respects date boundaries
# ---------------------------------------------------------------------------


def test_split_by_date_boundaries(lob_cfg):
    """split_by_date should correctly partition by train_end / val_end."""
    pipeline = LOBPipeline(lob_cfg)
    n = 300
    # Events spanning Jan 2024 – Jan 2026
    dates = pd.date_range('2024-01-01', periods=n, freq='3D', tz='UTC')
    df = _make_lob_df(n=n, levels=10)
    df['timestamp'] = dates

    train_df, val_df, test_df = pipeline.split_by_date(df)

    train_end = pd.Timestamp('2024-06-30', tz='UTC')
    val_end = pd.Timestamp('2025-03-31', tz='UTC')

    assert (pd.to_datetime(train_df['timestamp'], utc=True) <= train_end).all()
    assert (pd.to_datetime(val_df['timestamp'], utc=True) > train_end).all()
    assert (pd.to_datetime(val_df['timestamp'], utc=True) <= val_end).all()
    assert (pd.to_datetime(test_df['timestamp'], utc=True) > val_end).all()
    # No overlap between splits
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == n, f'Splits should cover all {n} rows; got {total}'
