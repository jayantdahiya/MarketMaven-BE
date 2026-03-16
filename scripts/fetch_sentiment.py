#!/usr/bin/env python3
"""Build reproducible sentiment feature tables offline via Finnhub.

Reads config from phase_3.yaml, fetches company news for each asset,
aggregates daily sentiment features, and writes a parquet file at
artifacts/data/phase3/sentiment_daily.parquet.
"""

import argparse
import logging
import os
from pathlib import Path

from api.data.pipeline import load_config
from api.data.sentiment import (
    FinnhubSentimentClient,
    build_sentiment_table,
    fill_neutral_defaults,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Fetch and aggregate daily sentiment from Finnhub',
    )
    p.add_argument(
        '--config',
        default='config/phase_3.yaml',
        help='Path to config YAML',
    )
    p.add_argument(
        '--start',
        default=None,
        help='Override start date (YYYY-MM-DD)',
    )
    p.add_argument(
        '--end',
        default=None,
        help='Override end date (YYYY-MM-DD)',
    )
    return p.parse_args()


def main(
    config_path: str, start_override: str | None, end_override: str | None
) -> None:
    cfg = load_config(config_path)
    data_cfg = cfg.get('data', {})
    paths_cfg = cfg.get('paths', {})
    multimodal_cfg = cfg.get('multimodal', {})
    sentiment_cfg = multimodal_cfg.get('sentiment', {})

    assets = data_cfg.get('assets', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY'])
    start_date = start_override or data_cfg.get('start_date', '2010-01-01')
    end_date = end_override or data_cfg.get('end_date', '2025-12-31')

    api_key = os.environ.get('FINNHUB_API_KEY', '')
    if not api_key:
        logger.error('FINNHUB_API_KEY environment variable not set')
        raise SystemExit(1)

    out_dir = Path(paths_cfg.get('sentiment_dir', 'artifacts/data/phase3'))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'sentiment_daily.parquet'

    client = FinnhubSentimentClient(api_key=api_key)

    logger.info(
        'Fetching sentiment for %d assets from %s to %s',
        len(assets),
        start_date,
        end_date,
    )

    df = build_sentiment_table(assets, start_date, end_date, client)

    # Apply neutral defaults for any missing values
    neutral_defaults = sentiment_cfg.get(
        'neutral_defaults',
        {
            'sentiment_mean': 0.0,
            'sentiment_std': 0.0,
            'sentiment_pos_ratio': 0.5,
            'sentiment_count': 0,
        },
    )
    df = fill_neutral_defaults(df, neutral_defaults)

    # Ensure correct column ordering
    cols = [
        'timestamp',
        'asset_id',
        'sentiment_mean',
        'sentiment_std',
        'sentiment_pos_ratio',
        'sentiment_count',
    ]
    df = df[[c for c in cols if c in df.columns]]

    df.to_parquet(out_path, index=False)
    logger.info(
        'Saved %d rows (%d unique assets) to %s',
        len(df),
        df['asset_id'].nunique() if 'asset_id' in df.columns else 0,
        out_path,
    )


if __name__ == '__main__':
    args = parse_args()
    main(args.config, args.start, args.end)
