"""Integration: training loop runs and produces checkpoint."""

from pathlib import Path

import pytest

from api.data.pipeline import DataPipeline
from api.models.factory import create_model
from api.training import losses
from api.training.train_loop import Trainer


@pytest.mark.slow
def test_train_one_epoch_produces_checkpoint(
    phase0_cfg, sample_daily_df, tmp_artifact_dir
):
    """Run 1 epoch and assert checkpoint file exists."""
    from api.data import sources

    data_path = tmp_artifact_dir / 'data' / 'raw_daily.parquet'
    data_path.parent.mkdir(parents=True, exist_ok=True)
    sources.save_raw_data(sample_daily_df, str(data_path))
    phase0_cfg['paths']['data_dir'] = str(tmp_artifact_dir / 'data')
    phase0_cfg['paths']['checkpoints_dir'] = str(tmp_artifact_dir / 'checkpoints')
    phase0_cfg['training']['epochs'] = 1
    phase0_cfg['training']['seeds'] = [42]
    phase0_cfg['training']['early_stopping_patience'] = 10
    phase0_cfg.setdefault('logging', {})['experiment_tracker'] = 'none'
    # sample_daily_df spans 2020-01-01 → ~2021-11-30; adjust split dates
    phase0_cfg['data']['split']['train_end'] = '2021-03-31'
    phase0_cfg['data']['split']['val_end'] = '2021-07-31'
    phase0_cfg['data']['split']['test_end'] = '2021-12-31'
    pipeline = DataPipeline(phase0_cfg)
    loaders = pipeline.build_dataloaders(data_path=str(data_path))
    model = create_model('lstm_baseline', phase0_cfg.get('model', {}))
    optimizer = __import__('torch').optim.AdamW(model.parameters(), lr=0.001)
    scheduler = __import__('torch').optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3
    )
    criterion = losses.CompositeForecastLoss(warmup_epochs=0)
    trainer = Trainer(phase0_cfg, model, optimizer, scheduler, criterion, 'cpu')
    run_id = 'test_run_1'
    scaler_state = pipeline._scaler.get_state() if pipeline._scaler else {}
    trainer.fit(
        loaders['train'],
        loaders['val'],
        run_id=run_id,
        feature_cols=phase0_cfg['data']['feature_cols'],
        scaler_path='',
        scaler_state=scaler_state,
    )
    ckpt_dir = Path(phase0_cfg['paths']['checkpoints_dir']) / run_id
    assert ckpt_dir.exists()
    assert any(ckpt_dir.glob('*.pt'))
