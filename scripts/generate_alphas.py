#!/usr/bin/env python3
"""Offline batch generation of LLM alpha features.

Iterates over (asset_id, date) combinations, checks JSONL cache,
generates missing entries via AlphaService, and writes a consolidated
parquet at artifacts/data/phase3/alpha_daily.parquet.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from api.data.pipeline import load_config
from api.services.alpha_service import AlphaService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Generate alpha features for training data',
    )
    p.add_argument(
        '--config',
        default='config/phase_3.yaml',
        help='Path to config YAML',
    )
    p.add_argument(
        '--start',
        default='2022-01-01',
        help='Start date (YYYY-MM-DD)',
    )
    p.add_argument(
        '--end',
        default='2025-12-31',
        help='End date (YYYY-MM-DD)',
    )
    p.add_argument(
        '--force',
        action='store_true',
        help='Regenerate even if cache entry exists',
    )
    return p.parse_args()


def main(
    config_path: str,
    start_date: str,
    end_date: str,
    force: bool = False,
) -> None:
    cfg = load_config(config_path)
    data_cfg = cfg.get('data', {})
    paths_cfg = cfg.get('paths', {})
    multimodal_cfg = cfg.get('multimodal', {})
    alpha_cfg = multimodal_cfg.get('alpha', {})

    assets = data_cfg.get('assets', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY'])
    cache_path = alpha_cfg.get('cache_path', 'artifacts/data/phase3/alpha_cache.jsonl')
    out_dir = Path(paths_cfg.get('alpha_dir', 'artifacts/data/phase3'))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'alpha_daily.parquet'

    service = AlphaService(
        model_name=alpha_cfg.get('llm_model', 'gpt-4.1-mini'),
        cache_path=cache_path,
        temperature=alpha_cfg.get('temperature', 0.1),
    )

    # Generate trading dates in range
    trading_dates = pd.bdate_range(start=start_date, end=end_date)
    total = len(assets) * len(trading_dates)
    logger.info(
        'Generating alphas for %d assets x %d dates = %d entries',
        len(assets),
        len(trading_dates),
        total,
    )

    records = []
    cached_count = 0
    generated_count = 0

    for asset_id in assets:
        for dt in trading_dates:
            date_str = dt.strftime('%Y-%m-%d')

            # Skip if cached and not forcing
            if not force:
                existing = service.get_cached_alphas(asset_id, date_str)
                if existing is not None:
                    cached_count += 1
                    record = {
                        'timestamp': pd.Timestamp(date_str),
                        'asset_id': asset_id,
                    }
                    for i in range(1, 9):
                        record[f'alpha_{i}'] = float(existing.get(f'alpha_{i}', 0.0))
                    records.append(record)
                    continue

            # Build a minimal feature snapshot for the prompt
            feature_snapshot = {
                'asset_id': asset_id,
                'date': date_str,
            }

            result = service.generate_alphas(
                asset_id=asset_id,
                as_of_date=date_str,
                feature_snapshot=feature_snapshot,
            )
            generated_count += 1

            record = {
                'timestamp': pd.Timestamp(date_str),
                'asset_id': asset_id,
            }
            for i in range(1, 9):
                record[f'alpha_{i}'] = float(result.get(f'alpha_{i}', 0.0))
            records.append(record)

            if (cached_count + generated_count) % 100 == 0:
                logger.info(
                    'Progress: %d/%d (cached=%d, generated=%d)',
                    cached_count + generated_count,
                    total,
                    cached_count,
                    generated_count,
                )

    logger.info(
        'Done: cached=%d, generated=%d, total=%d',
        cached_count,
        generated_count,
        len(records),
    )

    if records:
        df = pd.DataFrame(records)
        df.to_parquet(out_path, index=False)
        logger.info('Saved %d rows to %s', len(df), out_path)
    else:
        logger.warning('No records generated; skipping parquet write')


if __name__ == '__main__':
    args = parse_args()
    main(args.config, args.start, args.end, args.force)
