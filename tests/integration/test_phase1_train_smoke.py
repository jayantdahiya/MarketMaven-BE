"""Integration: CNN-Transformer training smoke test (2 epochs via factory)."""

from pathlib import Path

import pytest

from api.data import sources
from api.data.pipeline import DataPipeline
from api.models.factory import create_model
from api.training import losses
from api.training.train_loop import Trainer


@pytest.fixture
def phase1_cfg(tmp_path):
    """Phase 1 config with all paths redirected to tmp_path."""
    import yaml

    with open('config/phase_1.yaml') as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault('paths', {})
    cfg['paths']['data_dir'] = str(tmp_path / 'data')
    cfg['paths']['checkpoints_dir'] = str(tmp_path / 'checkpoints')
    cfg['paths']['reports_dir'] = str(tmp_path / 'reports')

    (tmp_path / 'data').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'reports').mkdir(parents=True, exist_ok=True)
    return cfg


@pytest.mark.slow
def test_cnn_trans_two_epoch_training_produces_checkpoint(
    phase1_cfg, sample_daily_df, tmp_artifact_dir
):
    """Run 2 epochs of CNN-Transformer training; verify checkpoint is created."""
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR

    data_path = Path(phase1_cfg['paths']['data_dir']) / 'raw_daily.parquet'
    sources.save_raw_data(sample_daily_df, str(data_path))

    # Override to 2 epochs and 1 seed for speed
    phase1_cfg['training']['epochs'] = 2
    phase1_cfg['training']['seeds'] = [42]
    phase1_cfg['training']['early_stopping_patience'] = 10
    # Disable mlflow to avoid filesystem side-effects in tests
    phase1_cfg.setdefault('logging', {})['experiment_tracker'] = 'none'
    # sample_daily_df spans 2020-01-01 → ~2021-11-30; adjust split dates accordingly
    phase1_cfg['data']['split']['train_end'] = '2021-03-31'
    phase1_cfg['data']['split']['val_end'] = '2021-07-31'
    phase1_cfg['data']['split']['test_end'] = '2021-12-31'

    pipeline = DataPipeline(phase1_cfg)
    loaders = pipeline.build_dataloaders(data_path=str(data_path))

    model_cfg = phase1_cfg.get('model', {})
    model = create_model('cnn_transformer', model_cfg)
    assert model.__class__.__name__ == 'CnnTransForecaster'

    train_cfg = phase1_cfg.get('training', {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get('lr', 7e-4),
        weight_decay=train_cfg.get('weight_decay', 1e-4),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=2, eta_min=1e-6)

    loss_cfg = train_cfg.get('loss', {})
    criterion = losses.CompositeForecastLoss(
        lambda_sharpe=loss_cfg.get('lambda_sharpe', 0.1),
        warmup_epochs=loss_cfg.get('warmup_epochs', 0),
    )

    trainer = Trainer(phase1_cfg, model, optimizer, scheduler, criterion, 'cpu')

    feature_cols = phase1_cfg.get('data', {}).get('feature_cols', [])
    scaler_state = pipeline._scaler.get_state() if pipeline._scaler else {}

    result = trainer.fit(
        loaders['train'],
        loaders['val'],
        run_id='test_phase1_run',
        feature_cols=feature_cols,
        scaler_path='',
        scaler_state=scaler_state,
    )

    # Checkpoint file must exist
    ckpt_path = Path(result['best_checkpoint_path'])
    assert ckpt_path.exists(), f'Checkpoint not found at {ckpt_path}'

    # History must have 2 entries (2 epochs ran)
    assert len(result['history']) == 2

    # All losses must be finite
    for entry in result['history']:
        assert entry['train_loss'] == entry['train_loss'], 'NaN train_loss detected'
        assert entry['val_loss'] == entry['val_loss'], 'NaN val_loss detected'


@pytest.mark.slow
def test_cnn_trans_checkpoint_load_round_trip(phase1_cfg, sample_daily_df):
    """Save checkpoint after 1 epoch; load into fresh model; predictions must match."""
    import torch
    from torch.optim.lr_scheduler import CosineAnnealingLR

    from api.training import checkpointing

    data_path = Path(phase1_cfg['paths']['data_dir']) / 'raw_daily.parquet'
    sources.save_raw_data(sample_daily_df, str(data_path))

    phase1_cfg['training']['epochs'] = 1
    phase1_cfg['training']['seeds'] = [42]
    phase1_cfg['training']['early_stopping_patience'] = 10
    phase1_cfg.setdefault('logging', {})['experiment_tracker'] = 'none'
    # sample_daily_df spans 2020-01-01 → ~2021-11-30; adjust split dates accordingly
    phase1_cfg['data']['split']['train_end'] = '2021-03-31'
    phase1_cfg['data']['split']['val_end'] = '2021-07-31'
    phase1_cfg['data']['split']['test_end'] = '2021-12-31'

    pipeline = DataPipeline(phase1_cfg)
    loaders = pipeline.build_dataloaders(data_path=str(data_path))

    model_cfg = phase1_cfg.get('model', {})
    model = create_model('cnn_transformer', model_cfg)

    train_cfg = phase1_cfg.get('training', {})
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get('lr', 7e-4))
    scheduler = CosineAnnealingLR(optimizer, T_max=1, eta_min=1e-6)
    criterion = losses.CompositeForecastLoss(warmup_epochs=0)

    trainer = Trainer(phase1_cfg, model, optimizer, scheduler, criterion, 'cpu')

    feature_cols = phase1_cfg.get('data', {}).get('feature_cols', [])
    scaler_state = pipeline._scaler.get_state() if pipeline._scaler else {}

    result = trainer.fit(
        loaders['train'],
        loaders['val'],
        run_id='test_roundtrip_run',
        feature_cols=feature_cols,
        scaler_path='',
        scaler_state=scaler_state,
    )

    ckpt_path = result['best_checkpoint_path']
    assert Path(ckpt_path).exists()

    # Load checkpoint into a fresh model
    fresh_model = create_model('cnn_transformer', model_cfg)
    checkpoint = checkpointing.load_checkpoint(ckpt_path, fresh_model, device='cpu')
    assert checkpoint is not None

    # Compare predictions between original and reloaded model
    model.eval()
    fresh_model.eval()
    x = torch.randn(4, 60, 8)
    with torch.no_grad():
        pred_orig = model(x)
        pred_loaded = fresh_model(x)

    assert torch.allclose(pred_orig, pred_loaded, atol=1e-5), (
        'Predictions differ after checkpoint round-trip'
    )
