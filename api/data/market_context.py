"""
Market context feature ingestion: SPY, QQQ, VIX daily returns and levels.

Produces context columns (spy_return_1d, qqq_return_1d, vix_close) keyed by
timestamp for merging into the multimodal feature matrix.
"""

import logging

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Default context tickers and their output column mapping
_RETURN_SYMBOLS = {'SPY': 'spy_return_1d', 'QQQ': 'qqq_return_1d'}
_LEVEL_SYMBOLS = {'^VIX': 'vix_close'}


def fetch_context_series(
    start_date: str,
    end_date: str,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV for context symbols via yfinance.

    Args:
        start_date: ISO date string (e.g. '2010-01-01').
        end_date: ISO date string (e.g. '2025-12-31').
        symbols: List of ticker symbols. Defaults to ['SPY', 'QQQ', '^VIX'].

    Returns:
        DataFrame with columns: timestamp, symbol, close.
    """
    if symbols is None:
        symbols = list(_RETURN_SYMBOLS.keys()) + list(_LEVEL_SYMBOLS.keys())
    frames = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(
                start=start_date, end=end_date, interval='1d', auto_adjust=True
            )
            if hist is None or hist.empty:
                logger.warning('No data returned for context symbol %s', sym)
                continue
            df = hist.reset_index()
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            if 'date' in df.columns:
                df = df.rename(columns={'date': 'timestamp'})
            df['symbol'] = sym
            df = df[['timestamp', 'symbol', 'close']].copy()
            frames.append(df)
        except Exception:
            logger.warning('Failed to fetch context symbol %s', sym, exc_info=True)
    if not frames:
        raise RuntimeError('No context data retrieved for any symbol')
    return pd.concat(frames, ignore_index=True)


def build_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns for SPY/QQQ and retain VIX close.

    Args:
        df: DataFrame from fetch_context_series with columns:
            timestamp, symbol, close.

    Returns:
        DataFrame keyed by timestamp with columns:
        spy_return_1d, qqq_return_1d, vix_close.
        Forward-fills missing dates (market holidays).
    """
    pivoted = df.pivot_table(
        index='timestamp', columns='symbol', values='close', aggfunc='first'
    )
    result = pd.DataFrame(index=pivoted.index)

    # Log returns for equity indices
    for sym, col_name in _RETURN_SYMBOLS.items():
        if sym in pivoted.columns:
            result[col_name] = np.log(pivoted[sym] / pivoted[sym].shift(1)).astype(
                np.float32
            )
        else:
            logger.warning('Context symbol %s missing, filling with zeros', sym)
            result[col_name] = np.float32(0.0)

    # Level for VIX
    for sym, col_name in _LEVEL_SYMBOLS.items():
        if sym in pivoted.columns:
            result[col_name] = pivoted[sym].astype(np.float32)
        else:
            logger.warning('Context symbol %s missing, filling with zeros', sym)
            result[col_name] = np.float32(0.0)

    # Forward-fill missing values (market holidays, misaligned dates)
    result = result.ffill().bfill()
    result = result.reset_index()
    # Normalize timestamp to date-only for joining with asset data
    result['timestamp'] = pd.to_datetime(result['timestamp']).dt.normalize()
    return result
