#!/usr/bin/env python3
"""Evaluate a checkpoint: metrics, regime, backtest, write reports.

Routes to LOB evaluation when config task.type == 'lob'.
"""

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

    task_type = cfg.get('task', {}).get('type', 'daily')

    if task_type == 'lob':
        _evaluate_lob(cfg, args.checkpoint, report_dir)
        return

    # --- Daily regression path (unchanged) ---
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


def _evaluate_lob(cfg: dict, checkpoint_path: str, report_dir: Path) -> None:
    """Run LOB evaluation and write reports."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from torch.utils.data import DataLoader, TensorDataset  # noqa: PLC0415

    from api.training import evaluate  # noqa: PLC0415

    paths = cfg.get('paths', {})
    processed_dir = Path(paths.get('lob_processed_dir', 'artifacts/data/phase4'))
    test_dir = processed_dir / 'test'

    x_book = np.load(str(test_dir / 'x_book.npy'))
    x_aux = np.load(str(test_dir / 'x_aux.npy'))
    y_class = np.load(str(test_dir / 'y_class.npy'))

    x_book_t = torch.from_numpy(x_book)
    x_aux_t = torch.from_numpy(x_aux)
    y_t = torch.from_numpy(y_class)

    ds = TensorDataset(x_book_t, x_aux_t, y_t)

    def _collate(batch):
        xb = torch.stack([b[0] for b in batch])
        xa = torch.stack([b[1] for b in batch])
        yb = torch.stack([b[2] for b in batch])
        return xb, xa, yb, {}

    test_loader = DataLoader(ds, batch_size=256, shuffle=False, collate_fn=_collate)

    model_cfg = cfg.get('model', {})
    model = create_model(model_cfg.get('type', 'tlob_forecaster'), model_cfg)

    results = evaluate.evaluate_lob_checkpoint(cfg, checkpoint_path, test_loader, model)
    evaluate.write_lob_reports(
        str(report_dir),
        results['summary'],
        results['regime_df'],
        results['trades_df'],
        results['bt_summary'],
    )
    print(f'LOB reports written to {report_dir}')


if __name__ == '__main__':
    main()
