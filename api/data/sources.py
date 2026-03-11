"""
Centralize raw data acquisition from yfinance with retry and parquet caching.
"""
import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0


def fetch_yfinance_data(
    assets: list[str],
    start_date: str,
    end_date: str,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Download OHLCV for all assets, stack into single DataFrame with asset_id column.
    Forward-fill then backward-fill NaNs within each asset. Retry up to 3 times with backoff.
    """
    if not assets:
        raise RuntimeError("assets list is empty")
    pieces: list[pd.DataFrame] = []
    for symbol in assets:
        for attempt in range(MAX_RETRIES):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval=interval, auto_adjust=True)
                if df is None or df.empty:
                    logger.warning("No data returned for %s (attempt %d), skipping", symbol, attempt + 1)
                    break
                df = df.reset_index()
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                if "date" in df.columns:
                    df = df.rename(columns={"date": "timestamp"})
                df["asset_id"] = symbol
                df = df.ffill().bfill()
                pieces.append(df)
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logger.warning("Failed to fetch %s after %d attempts: %s; skipping", symbol, MAX_RETRIES, e)
                    break
                time.sleep(RETRY_BACKOFF ** attempt)
    if not pieces:
        raise RuntimeError("Zero assets returned data")
    out = pd.concat(pieces, ignore_index=True)
    if "timestamp" not in out.columns and out.index.name is None:
        # ensure timestamp column
        if hasattr(out.index, "normalize"):
            out = out.reset_index().rename(columns={"index": "timestamp"})
    out = out.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    return out


def save_raw_data(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to parquet."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_raw_data(path: str) -> pd.DataFrame:
    """Load DataFrame from parquet."""
    return pd.read_parquet(path)
