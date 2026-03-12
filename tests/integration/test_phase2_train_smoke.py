"""Integration: Mamba/SSM training smoke test (2 epochs, with and without graph)."""

from pathlib import Path

import pytest
import torch
from torch.optim.lr_scheduler import OneCycleLR

from api.data import sources
from api.data.pipeline import DataPipeline
from api.models.factory import create_model
from api.training import checkpointing, losses
from api.training.train_loop import Trainer


@pytest.fixture
def phase2_cfg(tmp_path):
    """Phase 2 config with all paths redirected to tmp_path."""
    import yaml

    with open('config/phase_2.yaml') as f:
        cfg = yaml.safe_load(f) or {}

    cfg.setdefault('paths', {})
    cfg['paths']['data_dir'] = str(tmp_path / 'data')
    cfg['paths']['checkpoints_dir'] = str(tmp_path / 'checkpoints')
    cfg['paths']['reports_dir'] = str(tmp_path / 'reports')

    (tmp_path / 'data').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (tmp_path / 'reports').mkdir(parents=True, exist_ok=True)

    # Reduce model size for fast test
    cfg['model']['d_model'] = 32
    cfg['model']['d_state'] = 8
    cfg['model']['num_layers'] = 1
    cfg['model']['graph_dim'] = 8
    cfg['model']['num_assets'] = 2  # only 2 assets in sample_daily_df

    # sample_daily_df has ~87 rows per split with default date boundaries,
    # but Phase 2 default seq_len=90 requires seq_len+horizon=91 rows.
    # Reduce seq_len for integration smoke tests (unit tests cover full size).
    cfg['data']['seq_len'] = 30

    return cfg


@pytest.mark.slow
def test_mamba_two_epoch_training_with_graph(
    phase2_cfg, sample_daily_df, tmp_artifact_dir
):
    """Run 2 epochs of Mamba training with graph context; verify checkpoint."""
    data_path = Path(phase2_cfg['paths']['data_dir']) / 'raw_daily.parquet'
    sources.save_raw_data(sample_daily_df, str(data_path))

    # Override to 2 epochs and 1 seed for speed
    phase2_cfg['training']['epochs'] = 2
    phase2_cfg['training']['seeds'] = [42]
    phase2_cfg['training']['early_stopping_patience'] = 10
    phase2_cfg.setdefault('logging', {})['experiment_tracker'] = 'none'
    # sample_daily_df spans 2020-01-01 → ~2021-11-30; adjust split dates
    phase2_cfg['data']['split']['train_end'] = '2021-03-31'
    phase2_cfg['data']['split']['val_end'] = '2021-07-31'
    phase2_cfg['data']['split']['test_end'] = '2021-12-31'
    phase2_cfg['graph']['enabled'] = True

    pipeline = DataPipeline(phase2_cfg)
    loaders = pipeline.build_dataloaders(data_path=str(data_path))

    model_cfg = phase2_cfg.get('model', {})
    model = create_model('mamba_ssm', model_cfg)
    assert model.__class__.__name__ == 'MambaForecaster'

    train_cfg = phase2_cfg.get('training', {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get('lr', 6e-4),
        weight_decay=train_cfg.get('weight_decay', 1.5e-4),
    )

    # OneCycleLR needs total_steps
    steps_per_epoch = len(loaders['train'])
    scheduler = OneCycleLR(
        optimizer,
        max_lr=6e-4,
        total_steps=2 * steps_per_epoch,
        pct_start=0.3,
    )

    loss_cfg = train_cfg.get('loss', {})
    criterion = losses.CompositeForecastLoss(
        lambda_sharpe=loss_cfg.get('lambda_sharpe', 0.12),
        warmup_epochs=loss_cfg.get('warmup_epochs', 0),
        sharpe_window=loss_cfg.get('sharpe_window', 20),
    )

    trainer = Trainer(phase2_cfg, model, optimizer, scheduler, criterion, 'cpu')

    feature_cols = phase2_cfg.get('data', {}).get('feature_cols', [])
    scaler_state = pipeline._scaler.get_state() if pipeline._scaler else {}

    result = trainer.fit(
        loaders['train'],
        loaders['val'],
        run_id='test_phase2_graph_run',
        feature_cols=feature_cols,
        scaler_path='',
        scaler_state=scaler_state,
    )

    # Checkpoint must exist
    ckpt_path = Path(result['best_checkpoint_path'])
    assert ckpt_path.exists(), f'Checkpoint not found at {ckpt_path}'

    # History must have 2 entries
    assert len(result['history']) == 2

    # All losses must be finite
    for entry in result['history']:
        assert entry['train_loss'] == entry['train_loss'], 'NaN train_loss'
        assert entry['val_loss'] == entry['val_loss'], 'NaN val_loss'


@pytest.mark.slow
def test_mamba_two_epoch_training_without_graph(
    phase2_cfg, sample_daily_df, tmp_artifact_dir
):
    """Run 2 epochs of Mamba training without graph context."""
    data_path = Path(phase2_cfg['paths']['data_dir']) / 'raw_daily.parquet'
    sources.save_raw_data(sample_daily_df, str(data_path))

    phase2_cfg['training']['epochs'] = 2
    phase2_cfg['training']['seeds'] = [42]
    phase2_cfg['training']['early_stopping_patience'] = 10
    phase2_cfg.setdefault('logging', {})['experiment_tracker'] = 'none'
    phase2_cfg['data']['split']['train_end'] = '2021-03-31'
    phase2_cfg['data']['split']['val_end'] = '2021-07-31'
    phase2_cfg['data']['split']['test_end'] = '2021-12-31'
    phase2_cfg['graph']['enabled'] = False
    phase2_cfg['model']['use_graph_context'] = False

    pipeline = DataPipeline(phase2_cfg)
    loaders = pipeline.build_dataloaders(data_path=str(data_path))

    model_cfg = phase2_cfg.get('model', {})
    model = create_model('mamba_ssm', model_cfg)

    train_cfg = phase2_cfg.get('training', {})
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.get('lr', 6e-4))

    steps_per_epoch = len(loaders['train'])
    scheduler = OneCycleLR(
        optimizer, max_lr=6e-4, total_steps=2 * steps_per_epoch, pct_start=0.3
    )

    criterion = losses.CompositeForecastLoss(warmup_epochs=0)

    trainer = Trainer(phase2_cfg, model, optimizer, scheduler, criterion, 'cpu')

    feature_cols = phase2_cfg.get('data', {}).get('feature_cols', [])
    scaler_state = pipeline._scaler.get_state() if pipeline._scaler else {}

    result = trainer.fit(
        loaders['train'],
        loaders['val'],
        run_id='test_phase2_no_graph_run',
        feature_cols=feature_cols,
        scaler_path='',
        scaler_state=scaler_state,
    )

    assert len(result['history']) == 2
    for entry in result['history']:
        assert entry['train_loss'] == entry['train_loss'], 'NaN train_loss'


@pytest.mark.slow
def test_mamba_checkpoint_load_round_trip(phase2_cfg, sample_daily_df):
    """Save checkpoint after 1 epoch; load into fresh model; predictions match."""
    data_path = Path(phase2_cfg['paths']['data_dir']) / 'raw_daily.parquet'
    sources.save_raw_data(sample_daily_df, str(data_path))

    phase2_cfg['training']['epochs'] = 1
    phase2_cfg['training']['seeds'] = [42]
    phase2_cfg['training']['early_stopping_patience'] = 10
    phase2_cfg.setdefault('logging', {})['experiment_tracker'] = 'none'
    phase2_cfg['data']['split']['train_end'] = '2021-03-31'
    phase2_cfg['data']['split']['val_end'] = '2021-07-31'
    phase2_cfg['data']['split']['test_end'] = '2021-12-31'
    phase2_cfg['graph']['enabled'] = False
    phase2_cfg['model']['use_graph_context'] = False

    pipeline = DataPipeline(phase2_cfg)
    loaders = pipeline.build_dataloaders(data_path=str(data_path))

    model_cfg = phase2_cfg.get('model', {})
    model = create_model('mamba_ssm', model_cfg)

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)
    steps_per_epoch = len(loaders['train'])
    scheduler = OneCycleLR(
        optimizer, max_lr=6e-4, total_steps=1 * steps_per_epoch, pct_start=0.3
    )
    criterion = losses.CompositeForecastLoss(warmup_epochs=0)

    trainer = Trainer(phase2_cfg, model, optimizer, scheduler, criterion, 'cpu')

    feature_cols = phase2_cfg.get('data', {}).get('feature_cols', [])
    scaler_state = pipeline._scaler.get_state() if pipeline._scaler else {}

    result = trainer.fit(
        loaders['train'],
        loaders['val'],
        run_id='test_mamba_roundtrip',
        feature_cols=feature_cols,
        scaler_path='',
        scaler_state=scaler_state,
    )

    ckpt_path = result['best_checkpoint_path']
    assert Path(ckpt_path).exists()

    # Load into fresh model
    fresh_model = create_model('mamba_ssm', model_cfg)
    checkpoint = checkpointing.load_checkpoint(ckpt_path, fresh_model, device='cpu')
    assert checkpoint is not None

    # Compare predictions
    model.eval()
    fresh_model.eval()
    seq_len = phase2_cfg['data']['seq_len']
    x = torch.randn(4, seq_len, 8)
    with torch.no_grad():
        pred_orig = model(x)
        pred_loaded = fresh_model(x)

    assert torch.allclose(pred_orig, pred_loaded, atol=1e-5), (
        'Predictions differ after checkpoint round-trip'
    )
