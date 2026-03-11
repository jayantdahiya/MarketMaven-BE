#!/usr/bin/env python3
"""Train LSTM (or other model) from config; multi-seed or single --seed."""

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from api.data.pipeline import DataPipeline, load_config
from api.models.factory import create_model
from api.training import losses
from api.training.train_loop import Trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        '--config', default='config/phase_0.yaml', help='Path to config YAML'
    )
    p.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Single seed; if omitted use all from config',
    )
    return p.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg.get('paths', {})
    Path(paths.get('checkpoints_dir', 'artifacts/checkpoints/phase0')).mkdir(
        parents=True, exist_ok=True
    )
    seeds = (
        [args.seed]
        if args.seed is not None
        else cfg.get('training', {}).get('seeds', [42])
    )
    train_cfg = cfg.get('training', {})
    device = train_cfg.get('device', 'cpu')
    loss_cfg = train_cfg.get('loss', {})
    criterion = losses.CompositeForecastLoss(
        lambda_sharpe=loss_cfg.get('lambda_sharpe', 0.1),
        temperature=loss_cfg.get('temperature', 0.02),
        warmup_epochs=loss_cfg.get('warmup_epochs', 3),
    )
    for seed in seeds:
        set_global_seed(seed)
        run_id = f'seed_{seed}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        pipeline = DataPipeline(cfg)
        loaders = pipeline.build_dataloaders(shuffle_train=True)
        model_cfg = cfg.get('model', {})
        model = create_model(model_cfg.get('type', 'lstm_baseline'), model_cfg)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.get('lr', 0.001),
            weight_decay=train_cfg.get('weight_decay', 0.0001),
        )
        sched_cfg = train_cfg.get('scheduler', {})
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched_cfg.get('factor', 0.5),
            patience=sched_cfg.get('patience', 3),
            min_lr=sched_cfg.get('min_lr', 1e-6),
        )
        trainer = Trainer(cfg, model, optimizer, scheduler, criterion, device)
        feature_cols = cfg.get('data', {}).get('feature_cols', [])
        scaler_path = str(
            Path(paths.get('data_dir', 'artifacts/data/phase0')) / 'scaler.joblib'
        )
        if pipeline._scaler is not None:
            pipeline._scaler.save(scaler_path)
        scaler_state = (
            pipeline._scaler.get_state() if pipeline._scaler is not None else {}
        )
        trainer.fit(
            loaders['train'],
            loaders['val'],
            run_id=run_id,
            feature_cols=feature_cols,
            scaler_path=scaler_path,
            scaler_state=scaler_state,
        )
        print(f'Seed {seed} done. Best checkpoint: {trainer.checkpoints_dir / run_id}')


if __name__ == '__main__':
    main()
