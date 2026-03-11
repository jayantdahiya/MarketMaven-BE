"""
ForecastService: load model from checkpoint, run inference, format response.
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from api.data.normalization import FeatureScaler
from api.data.pipeline import DataPipeline, load_config
from api.models.factory import create_model
from api.models.prophet_forecaster import DataUnavailableError, forecast_prophet
from api.schemas.forecast import (
    DailyForecastPoint,
    DailyForecastRequest,
    DailyForecastResponse,
)
from api.training import checkpointing


class ForecastService:
    def __init__(self, config: dict | None = None, config_path: str | None = None):
        if config is None and config_path:
            config = load_config(config_path)
        self.config = config or {}
        self._model_cache: dict[str, torch.nn.Module] = {}
        self._scaler_cache: dict[str, FeatureScaler] = {}
        self._pipeline: DataPipeline | None = None
        paths = self.config.get('paths', {})
        self.checkpoints_dir = Path(
            paths.get('checkpoints_dir', 'artifacts/checkpoints/phase0')
        )
        self.eval_cfg = self.config.get('evaluation', {})
        self.signal_threshold = self.eval_cfg.get('signal_threshold', 0.0)

    def _get_pipeline(self) -> DataPipeline:
        if self._pipeline is None:
            self._pipeline = DataPipeline(self.config)
        return self._pipeline

    def _load_model(self, model_name: str) -> torch.nn.Module | None:
        if model_name in self._model_cache:
            return self._model_cache.get(model_name)
        if model_name == 'prophet':
            return None
        model_cfg = self.config.get('model', {})
        model = create_model(model_name, model_cfg)
        ckpt_dir = self.checkpoints_dir
        if not ckpt_dir.exists():
            raise FileNotFoundError(f'Checkpoint dir not found: {ckpt_dir}')
        best_pt = None
        for run_dir in sorted(ckpt_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            for f in run_dir.glob('best_epoch_*.pt'):
                best_pt = f
                break
            if best_pt:
                break
        if best_pt is None:
            raise FileNotFoundError(f'No checkpoint found under {ckpt_dir}')
        state = checkpointing.load_checkpoint(str(best_pt), model, device='cpu')
        model.eval()
        self._model_cache[model_name] = model
        if 'scaler_state' in state and state['scaler_state']:
            self._scaler_cache[model_name] = FeatureScaler.from_state(
                state['scaler_state']
            )
        return model

    def _build_signal(
        self, predicted_return: float, threshold: float | None = None
    ) -> tuple[str, float]:
        th = threshold if threshold is not None else self.signal_threshold
        signal = 'long' if predicted_return > th else 'flat'
        confidence = 1.0 / (1.0 + np.exp(-abs(predicted_return) / 0.01))
        return signal, float(confidence)

    def predict_daily(self, req: DailyForecastRequest) -> DailyForecastResponse:
        if req.model == 'prophet':
            return self._predict_prophet(req)
        model = self._load_model(req.model)
        pipeline = self._get_pipeline()
        scaler = self._scaler_cache.get(req.model) or pipeline._scaler
        if scaler is None:
            raise RuntimeError('No scaler available; train and save a checkpoint first')
        try:
            window = pipeline.build_inference_window(
                req.asset_id,
                as_of_date=req.as_of_date,
                scaler=scaler,
            )
        except Exception as e:
            raise RuntimeError(f'Could not build inference window: {e}') from e
        x = torch.from_numpy(window).unsqueeze(0).float()  # [1, T, F]
        with torch.no_grad():
            y_hat = model(x)
        pred_return = y_hat.item()
        signal, confidence = self._build_signal(pred_return)
        as_of = req.as_of_date or date.today()
        next_day = as_of + timedelta(days=1)
        point = DailyForecastPoint(
            timestamp=datetime.combine(next_day, datetime.min.time()),
            predicted_return=pred_return,
            predicted_price=None,
            signal=signal,
            confidence=confidence,
        )
        return DailyForecastResponse(
            asset_id=req.asset_id,
            model=req.model,
            horizon_days=req.horizon_days,
            generated_at=datetime.utcnow(),
            predictions=[point],
        )

    def _predict_prophet(self, req: DailyForecastRequest) -> DailyForecastResponse:
        try:
            out = forecast_prophet(
                req.asset_id, horizon_days=req.horizon_days, end_date=req.as_of_date
            )
        except DataUnavailableError as e:
            raise RuntimeError(str(e)) from e
        points = [
            DailyForecastPoint(
                timestamp=datetime.fromisoformat(ts + 'T00:00:00'),
                predicted_return=0.0,
                predicted_price=trend,
                signal='flat',
                confidence=0.5,
            )
            for ts, trend in zip(out['timestamps'], out['trend'], strict=True)
        ]
        return DailyForecastResponse(
            asset_id=req.asset_id,
            model='prophet',
            horizon_days=req.horizon_days,
            generated_at=datetime.utcnow(),
            predictions=points[: req.horizon_days],
        )
