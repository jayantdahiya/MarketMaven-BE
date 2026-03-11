"""Unit tests for technical features."""

import numpy as np
import pandas as pd

from api.data import features


def test_calculate_sma_expected_values():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = features.calculate_sma(close, 2)
    expected_last = (4.0 + 5.0) / 2
    assert np.allclose(result.dropna().iloc[-1], expected_last, atol=1e-6)


def test_calculate_rsi_range_0_100():
    close = pd.Series(100 + np.random.randn(50).cumsum())
    close = close.clip(lower=1.0)
    result = features.calculate_rsi(close, window=14)
    assert result.min() >= 0 and result.max() <= 100


def test_calculate_macd_signal_length_match():
    close = pd.Series(100 + np.random.randn(100).cumsum())
    macd, signal = features.calculate_macd(close)
    assert len(macd) == len(signal)


def test_add_technical_features_columns_present(sample_daily_df):
    expected_cols = [
        'close',
        'volume',
        'sma_15',
        'sma_45',
        'rsi_14',
        'macd',
        'macd_signal',
        'obv',
    ]
    df = features.add_technical_features(sample_daily_df)
    assert set(expected_cols).issubset(set(df.columns))
