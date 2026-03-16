#!/usr/bin/env python3
"""Train a model (LSTM, CNN-Transformer, etc.) from config; multi-seed or single --seed.

MLflow tracking is enabled when config['logging']['experiment_tracker'] == 'mlflow'.
Set MLFLOW_TRACKING_URI or config['logging']['mlflow_tracking_uri'] to control storage.

For LOB task (config task.type == 'lob'), routes to LOBPipeline + fit_lob().
"""

import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau

from api.data.pipeline import DataPipeline, load_config
from api.models.factory import create_model
from api.training import losses
from api.training.train_loop import Trainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train a MarketMaven forecasting model.')
    p.add_argument(
        '--config', default='config/phase_0.yaml', help='Path to config YAML'
    )
    p.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Single seed; if omitted runs all seeds defined in config',
    )
    return p.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_cfg: dict,
    steps_per_epoch: int = 0,
    epochs: int = 1,
) -> ReduceLROnPlateau | CosineAnnealingLR | OneCycleLR | None:
    """Instantiate an LR scheduler from config.

    Args:
        optimizer: The optimizer to wrap.
        sched_cfg: Dict from config['training']['scheduler'].
        steps_per_epoch: Number of batches per epoch (required for OneCycleLR).
        epochs: Total training epochs (required for OneCycleLR).

    Returns:
        Scheduler instance, or None if type is unrecognised.
    """
    sched_type = sched_cfg.get('type', 'reduce_lr_on_plateau')
    if sched_type == 'cosine_annealing':
        return CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.get('T_max', 50),
            eta_min=sched_cfg.get('eta_min', 1e-6),
        )
    if sched_type == 'reduce_lr_on_plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched_cfg.get('factor', 0.5),
            patience=sched_cfg.get('patience', 3),
            min_lr=sched_cfg.get('min_lr', 1e-6),
        )
    if sched_type == 'one_cycle_lr':
        if steps_per_epoch <= 0:
            logger.warning(
                'OneCycleLR requires steps_per_epoch > 0; got %d — skipping.',
                steps_per_epoch,
            )
            return None
        return OneCycleLR(
            optimizer,
            max_lr=sched_cfg.get('max_lr', 6e-4),
            total_steps=epochs * steps_per_epoch,
            pct_start=sched_cfg.get('pct_start', 0.3),
            anneal_strategy=sched_cfg.get('anneal_strategy', 'cos'),
            div_factor=sched_cfg.get('div_factor', 25.0),
            final_div_factor=sched_cfg.get('final_div_factor', 1000.0),
        )
    logger.warning('Unknown scheduler type %r — skipping scheduler.', sched_type)
    return None


def _try_import_mlflow():
    """Import mlflow lazily so the rest of the script works without it."""
    try:
        import mlflow  # noqa: PLC0415

        return mlflow
    except ImportError:
        logger.warning('mlflow not installed — experiment tracking disabled.')
        return None


def _train_lob(cfg: dict, seed: int, run_id: str) -> dict:
    """Run LOB classification training for a single seed."""
    from torch.utils.data import DataLoader, TensorDataset  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from api.data.lob_pipeline import LOBPipeline  # noqa: PLC0415

    train_cfg = cfg.get('training', {})
    device = train_cfg.get('device', 'cpu')
    paths = cfg.get('paths', {})
    processed_dir = Path(paths.get('lob_processed_dir', 'artifacts/data/phase4'))

    # Prefer pre-built splits (from build_lob_dataset.py); fall back to in-memory
    def _load_split(split: str):
        split_dir = processed_dir / split
        x_book_path = split_dir / 'x_book.npy'
        x_aux_path = split_dir / 'x_aux.npy'
        y_path = split_dir / 'y_class.npy'
        if x_book_path.exists() and x_aux_path.exists() and y_path.exists():
            x_book = np.load(str(x_book_path))
            x_aux = np.load(str(x_aux_path))
            y_class = np.load(str(y_path))
            return (
                torch.from_numpy(x_book),
                torch.from_numpy(x_aux),
                torch.from_numpy(y_class),
            )
        raise FileNotFoundError(
            f'Pre-built LOB split not found under {split_dir}. '
            'Run scripts/build_lob_dataset.py first.'
        )

    x_book_tr, x_aux_tr, y_tr = _load_split('train')
    x_book_val, x_aux_val, y_val = _load_split('val')

    batch_size = train_cfg.get('batch_size', 128)

    # Build DataLoaders yielding 4-tuples (x_book, x_aux, y_class, meta={})
    # TensorDataset gives 3 tensors; we wrap with a collate that adds empty meta
    def _make_loader(xb, xa, y, shuffle: bool) -> DataLoader:
        ds = TensorDataset(xb, xa, y)

        def _collate(batch):
            xb_b = torch.stack([b[0] for b in batch])
            xa_b = torch.stack([b[1] for b in batch])
            y_b = torch.stack([b[2] for b in batch])
            return xb_b, xa_b, y_b, {}

        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate
        )

    train_loader = _make_loader(x_book_tr, x_aux_tr, y_tr, shuffle=True)
    val_loader = _make_loader(x_book_val, x_aux_val, y_val, shuffle=False)

    model_cfg = cfg.get('model', {})
    model = create_model(model_cfg.get('type', 'tlob_forecaster'), model_cfg)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get('lr', 8e-4),
        weight_decay=train_cfg.get('weight_decay', 1e-4),
    )

    loss_cfg = train_cfg.get('loss', {})
    criterion = losses.WeightedCEFocalLoss(
        class_weights=loss_cfg.get('class_weights', [1.2, 0.8, 1.2]),
        lambda_focal=loss_cfg.get('lambda_focal', 0.25),
        focal_gamma=loss_cfg.get('focal_gamma', 1.5),
    )

    sched_cfg = train_cfg.get('scheduler', {})
    scheduler = build_scheduler(
        optimizer,
        sched_cfg,
        steps_per_epoch=len(train_loader),
        epochs=train_cfg.get('epochs', 45),
    )

    trainer = Trainer(cfg, model, optimizer, scheduler, criterion, device)
    result = trainer.fit_lob(train_loader, val_loader, run_id=run_id)
    return result


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

    task_type = cfg.get('task', {}).get('type', 'daily')

    # MLflow setup (optional)
    log_cfg = cfg.get('logging', {})
    use_mlflow = log_cfg.get('experiment_tracker') == 'mlflow'
    mlflow = _try_import_mlflow() if use_mlflow else None
    if mlflow and use_mlflow:
        tracking_uri = log_cfg.get('mlflow_tracking_uri', 'artifacts/mlruns')
        Path(tracking_uri).mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri)
        experiment_name = cfg.get('project', {}).get('name', 'market-maven')
        mlflow.set_experiment(experiment_name)

    for seed in seeds:
        set_global_seed(seed)
        run_id = f'seed_{seed}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        if task_type == 'lob':
            result = _train_lob(cfg, seed, run_id)
            print(
                f'Seed {seed} done. Best LOB checkpoint: {result["best_checkpoint_path"]}'
            )
            continue

        # --- Daily regression path (unchanged) ---
        loss_cfg = train_cfg.get('loss', {})
        criterion = losses.CompositeForecastLoss(
            lambda_sharpe=loss_cfg.get('lambda_sharpe', 0.1),
            temperature=loss_cfg.get('temperature', 0.02),
            warmup_epochs=loss_cfg.get('warmup_epochs', 3),
            sharpe_window=loss_cfg.get('sharpe_window', 0),
        )

        # Build data pipeline
        pipeline = DataPipeline(cfg)
        loaders = pipeline.build_dataloaders(shuffle_train=True)

        # Build model
        model_cfg = cfg.get('model', {})
        model_type = model_cfg.get('type', 'lstm_baseline')
        model = create_model(model_type, model_cfg)

        # Build optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg.get('lr', 0.001),
            weight_decay=train_cfg.get('weight_decay', 0.0001),
        )

        # Build scheduler
        sched_cfg = train_cfg.get('scheduler', {})
        scheduler = build_scheduler(
            optimizer,
            sched_cfg,
            steps_per_epoch=len(loaders['train']),
            epochs=train_cfg.get('epochs', 40),
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

        # Start MLflow run if enabled
        run_name_tpl = log_cfg.get('run_name_template', 'run_seed_{seed}')
        run_name = run_name_tpl.format(seed=seed)

        if mlflow and use_mlflow:
            with mlflow.start_run(run_name=run_name):
                # Log config hyperparams
                mlflow.log_params({
                    'seed': seed,
                    'model_type': model_type,
                    'lr': train_cfg.get('lr'),
                    'batch_size': train_cfg.get('batch_size'),
                    'epochs': train_cfg.get('epochs'),
                    'grad_clip_norm': train_cfg.get('grad_clip_norm'),
                    'lambda_sharpe': loss_cfg.get('lambda_sharpe'),
                    'scheduler_type': sched_cfg.get('type'),
                })
                result = trainer.fit(
                    loaders['train'],
                    loaders['val'],
                    run_id=run_id,
                    feature_cols=feature_cols,
                    scaler_path=scaler_path,
                    scaler_state=scaler_state,
                )
                # Log per-epoch metrics
                for entry in result['history']:
                    mlflow.log_metrics(
                        {
                            'train_loss': entry['train_loss'],
                            'val_loss': entry['val_loss'],
                            'val_mae': entry['val_mae'],
                            'val_rmse': entry['val_rmse'],
                        },
                        step=entry['epoch'],
                    )
                mlflow.log_artifact(result['best_checkpoint_path'])
        else:
            result = trainer.fit(
                loaders['train'],
                loaders['val'],
                run_id=run_id,
                feature_cols=feature_cols,
                scaler_path=scaler_path,
                scaler_state=scaler_state,
            )

        print(f'Seed {seed} done. Best checkpoint: {result["best_checkpoint_path"]}')


if __name__ == '__main__':
    main()
