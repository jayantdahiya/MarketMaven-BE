#!/usr/bin/env python3
"""Fetch OHLCV from yfinance and save to parquet."""
import argparse
from pathlib import Path

from api.data.pipeline import load_config
from api.data import sources


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/phase_0.yaml", help="Path to config YAML")
    return p.parse_args()


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    paths = cfg.get("paths", {})
    data_dir = Path(paths.get("data_dir", "artifacts/data/phase0"))
    data_dir.mkdir(parents=True, exist_ok=True)
    data_cfg = cfg.get("data", {})
    assets = data_cfg.get("assets", ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"])
    start = data_cfg.get("start_date", "2010-01-01")
    end = data_cfg.get("end_date", "2025-12-31")
    interval = data_cfg.get("interval", "1d")
    df = sources.fetch_yfinance_data(assets, start, end, interval)
    out_path = data_dir / "raw_daily.parquet"
    sources.save_raw_data(df, str(out_path))
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args.config)
