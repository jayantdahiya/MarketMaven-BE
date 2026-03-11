"""
Technical feature engineering: SMA, RSI, MACD, OBV.
"""

import numpy as np
import pandas as pd


def calculate_sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average of close."""
    return close.rolling(window=window, min_periods=1).mean().astype(np.float32)


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI using average gain/loss, clamped to [0, 100]."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = np.where(avg_loss == 0, 100.0, avg_gain / avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return np.clip(rsi, 0.0, 100.0).astype(np.float32)


def calculate_macd(
    close: pd.Series,
    short_window: int = 12,
    long_window: int = 26,
    signal_window: int = 9,
) -> tuple[pd.Series, pd.Series]:
    """MACD line and signal line using EMA (adjust=False)."""
    ema_short = close.ewm(span=short_window, adjust=False).mean()
    ema_long = close.ewm(span=long_window, adjust=False).mean()
    macd_line = (ema_short - ema_long).astype(np.float32)
    signal_line = (
        macd_line.ewm(span=signal_window, adjust=False).mean().astype(np.float32)
    )
    return macd_line, signal_line


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-balance volume: cumulative sum of sign(close_t - close_{t-1}) * volume_t."""
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * volume).cumsum()
    return obv.astype(np.float32)


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SMA(15), SMA(45), RSI(14), MACD(12,26,9), OBV per asset.
    Expects columns: open, high, low, close, volume, asset_id.
    """
    out = df.copy()
    if 'volume' not in out.columns:
        raise KeyError("DataFrame must contain 'volume' column for feature engineering")
    if 'close' not in out.columns:
        raise KeyError("DataFrame must contain 'close' column")
    grouped = out.groupby('asset_id', group_keys=False)
    out['sma_15'] = grouped['close'].transform(lambda s: calculate_sma(s, 15))
    out['sma_45'] = grouped['close'].transform(lambda s: calculate_sma(s, 45))
    out['rsi_14'] = grouped['close'].transform(lambda s: calculate_rsi(s, 14))
    macd_signal = grouped['close'].transform(lambda s: calculate_macd(s, 12, 26, 9)[1])
    out['macd'] = grouped['close'].transform(lambda s: calculate_macd(s, 12, 26, 9)[0])
    out['macd_signal'] = macd_signal
    obv_parts = []
    for _aid, g in grouped:
        obv_ser = calculate_obv(g['close'], g['volume'])
        obv_ser.index = g.index
        obv_parts.append(obv_ser)
    out['obv'] = pd.concat(obv_parts)
    return out
