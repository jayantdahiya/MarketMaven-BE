#!/usr/bin/env python3
"""Build LOB dataset: load raw events, run LOBPipeline, save processed splits.

Usage:
    python scripts/build_lob_dataset.py --config config/phase_4.yaml
    python scripts/build_lob_dataset.py --config config/phase_4.yaml --output-dir artifacts/data/phase4
"""

import argparse
import logging
from pathlib import Path

import numpy as np

from api.data.lob_pipeline import LOBPipeline
from api.data.pipeline import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build LOB dataset from raw event files.')
    p.add_argument(
        '--config', default='config/phase_4.yaml', help='Path to config YAML'
    )
    p.add_argument(
        '--output-dir',
        default=None,
        help='Override output directory (default: paths.lob_processed_dir from config)',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    paths = cfg.get('paths', {})
    output_dir = Path(
        args.output_dir or paths.get('lob_processed_dir', 'artifacts/data/phase4')
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = LOBPipeline(cfg)

    # Load raw events
    logger.info('Loading raw LOB events from %s', pipeline.raw_data_path)
    df = pipeline.load_raw()
    logger.info('Loaded %d raw events', len(df))

    # Chronological split
    logger.info('Splitting by date...')
    train_df, val_df, test_df = pipeline.split_by_date(df)
    logger.info(
        'Split sizes: train=%d  val=%d  test=%d',
        len(train_df),
        len(val_df),
        len(test_df),
    )

    # Process each split separately to prevent data leakage
    for split_name, split_df in [
        ('train', train_df),
        ('val', val_df),
        ('test', test_df),
    ]:
        if len(split_df) == 0:
            logger.warning('Split %s is empty — skipping', split_name)
            continue

        logger.info('Processing %s split (%d events)...', split_name, len(split_df))
        x_book, x_aux, y_class, meta_df = pipeline.run(split_df)
        logger.info(
            '%s: x_book=%s  x_aux=%s  y_class=%s',
            split_name,
            x_book.shape,
            x_aux.shape,
            y_class.shape,
        )

        # Save as compressed numpy
        split_out = output_dir / split_name
        split_out.mkdir(parents=True, exist_ok=True)
        np.save(split_out / 'x_book.npy', x_book)
        np.save(split_out / 'x_aux.npy', x_aux)
        np.save(split_out / 'y_class.npy', y_class)
        meta_df.to_parquet(split_out / 'meta.parquet', index=False)
        logger.info('Saved %s split to %s', split_name, split_out)

    logger.info('LOB dataset build complete. Output: %s', output_dir)


if __name__ == '__main__':
    main()
