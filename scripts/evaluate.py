#!/usr/bin/env python3
"""Evaluate a checkpoint: metrics, regime, backtest, write reports."""

import argparse
from pathlib import Path

from api.data.pipeline import DataPipeline, load_config
from api.models.factory import create_model
from api.training import evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        '--config', default='config/phase_0.yaml', help='Path to config YAML'
    )
    p.add_argument('--checkpoint', required=True, help='Path to checkpoint .pt file')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.checkpoint).exists():
        print(f'Checkpoint not found: {args.checkpoint}')
        exit(2)
    cfg = load_config(args.config)
    paths = cfg.get('paths', {})
    report_dir = Path(paths.get('reports_dir', 'artifacts/reports/phase0'))
    run_id = Path(args.checkpoint).parent.name
    report_dir = report_dir / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    pipeline = DataPipeline(cfg)
    loaders = pipeline.build_dataloaders(shuffle_train=False)
    model_cfg = cfg.get('model', {})
    model = create_model(model_cfg.get('type', 'lstm_baseline'), model_cfg)
    scaler = pipeline._scaler
    if scaler is None:
        print('Scaler not fitted; cannot evaluate')
        exit(2)
    results = evaluate.evaluate_checkpoint(
        cfg, args.checkpoint, loaders['test'], scaler, model
    )
    evaluate.write_reports(
        str(report_dir),
        results['summary'],
        results['regime_df'],
        results['bt_df'],
        results['bt_summary'],
    )
    print(f'Reports written to {report_dir}')


if __name__ == '__main__':
    main()
