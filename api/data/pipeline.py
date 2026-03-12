"""
End-to-end daily data pipeline: load, features, targets, split, scale, DataLoaders, inference window.
"""

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf
from torch.utils.data import DataLoader

from api.data import datasets, features, normalization, sources, splitters, targets
from api.data.asset_graph import compute_graph_context

logger = logging.getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    out = dict(base)
    for k, v in override.items():
        if k == '_base_':
            continue
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict:
    """Load YAML config, resolving _base_ if present."""
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop('_base_', None)
    if base_ref:
        base_path = Path(path).parent / Path(base_ref).name
        with open(base_path) as b:
            base = yaml.safe_load(b) or {}
        cfg = _deep_merge(base, cfg)
    return cfg


class DataPipeline:
    """Orchestrates load → features → targets → split → scale → datasets → DataLoaders."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.data_cfg = cfg.get('data', {})
        self.train_cfg = cfg.get('training', {})
        self._scaler: normalization.FeatureScaler | None = None
        self._train_df: pd.DataFrame | None = None
        self._val_df: pd.DataFrame | None = None
        self._test_df: pd.DataFrame | None = None

    def load_raw_data(self, data_path: str | None = None) -> pd.DataFrame:
        """Load from parquet if path exists and has data, else fetch from yfinance."""
        path = data_path or str(
            Path(self.cfg.get('paths', {}).get('data_dir', 'artifacts/data/phase0'))
            / 'raw_daily.parquet'
        )
        if Path(path).exists():
            df = sources.load_raw_data(path)
            logger.info('Loaded raw data from %s: %d rows', path, len(df))
            return df
        assets = self.data_cfg.get('assets', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'SPY'])
        start = self.data_cfg.get('start_date', '2010-01-01')
        end = self.data_cfg.get('end_date', '2025-12-31')
        interval = self.data_cfg.get('interval', '1d')
        df = sources.fetch_yfinance_data(assets, start, end, interval)
        if data_path is None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            sources.save_raw_data(df, path)
        return df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical features and return targets."""
        df = features.add_technical_features(df)
        horizons = self.data_cfg.get('target_horizons', [1])
        df = targets.add_return_targets(df, horizons)
        return df

    def split_by_date(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split by config date boundaries."""
        split_cfg = self.data_cfg.get('split', {})
        train_end = split_cfg.get('train_end', '2021-12-31')
        val_end = split_cfg.get('val_end', '2023-12-31')
        test_end = split_cfg.get('test_end', '2025-12-31')
        return splitters.split_by_date(
            df, train_end, val_end, test_end, date_col='timestamp'
        )

    def fit_scaler(self, train_df: pd.DataFrame) -> normalization.FeatureScaler:
        """Fit scaler on train only."""
        norm_cfg = self.data_cfg.get('normalization', {})
        mode = norm_cfg.get('mode', 'per_asset_zscore')
        clip = norm_cfg.get('clip', 5.0)
        feature_cols = self.data_cfg.get(
            'feature_cols',
            [
                'close',
                'volume',
                'sma_15',
                'sma_45',
                'rsi_14',
                'macd',
                'macd_signal',
                'obv',
            ],
        )
        scaler = normalization.FeatureScaler(mode=mode, clip=clip)
        scaler.fit(train_df, feature_cols)
        self._scaler = scaler
        return scaler

    def build_graph_context(
        self, train_df: pd.DataFrame
    ) -> tuple[torch.Tensor, dict[str, int]]:
        """Compute graph-context features and asset-to-index mapping.

        Uses the *training* split only (to avoid look-ahead) for building the
        correlation graph and deriving per-asset features.

        Returns:
            Tuple of ``(graph_context, asset_to_index)`` where
            ``graph_context`` is a ``[A, 3]`` float32 tensor and
            ``asset_to_index`` maps asset-ID strings to integer indices.
        """
        graph_cfg = self.cfg.get('graph', {})
        assets = sorted(train_df['asset_id'].unique().tolist())
        asset_to_index = {a: i for i, a in enumerate(assets)}
        gc = compute_graph_context(
            train_df,
            assets,
            window=graph_cfg.get('correlation_window', 60),
            threshold=graph_cfg.get('correlation_threshold', 0.5),
            return_window=20,
        )
        return gc, asset_to_index

    def build_dataloaders(
        self,
        shuffle_train: bool = True,
        data_path: str | None = None,
    ) -> dict[str, DataLoader]:
        """Full pipeline: load, features, split, scale, datasets, DataLoaders."""
        df = self.load_raw_data(data_path=data_path)
        df = self.build_features(df)
        feature_cols = self.data_cfg.get('feature_cols', [])
        target_col = self.data_cfg.get('target_col', 'log_return_1d')
        df = df.dropna(subset=feature_cols + [target_col])
        self._train_df, self._val_df, self._test_df = self.split_by_date(df)
        self.fit_scaler(self._train_df)
        seq_len = self.data_cfg.get('seq_len', 60)
        horizon = self.data_cfg.get('horizon_days', 1)
        batch_size = self.train_cfg.get('batch_size', 64)

        # Graph context (Phase 2+): computed from training data only
        graph_enabled = self.cfg.get('graph', {}).get('enabled', False)
        if graph_enabled:
            gc, asset_to_index = self.build_graph_context(self._train_df)
        else:
            gc = None
            asset_to_index = None

        train_ds = datasets.DailySequenceDataset(
            self._scaler.transform(self._train_df),
            feature_cols,
            target_col,
            seq_len,
            horizon,
            asset_to_index=asset_to_index,
            graph_context=gc,
        )
        val_ds = datasets.DailySequenceDataset(
            self._scaler.transform(self._val_df),
            feature_cols,
            target_col,
            seq_len,
            horizon,
            asset_to_index=asset_to_index,
            graph_context=gc,
        )
        test_ds = datasets.DailySequenceDataset(
            self._scaler.transform(self._test_df),
            feature_cols,
            target_col,
            seq_len,
            horizon,
            asset_to_index=asset_to_index,
            graph_context=gc,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )
        logger.info(
            'DataLoaders: train=%d val=%d test=%d batches',
            len(train_loader),
            len(val_loader),
            len(test_loader),
        )
        return {'train': train_loader, 'val': val_loader, 'test': test_loader}

    def build_inference_window(
        self,
        asset_id: str,
        as_of_date: date | None = None,
        scaler: normalization.FeatureScaler | None = None,
    ) -> np.ndarray:
        """Fetch recent seq_len days for one asset, engineer features, scale; return array [seq_len, F]."""
        scaler = scaler or self._scaler
        if scaler is None:
            raise RuntimeError(
                'Scaler not fitted; run build_dataloaders or pass scaler'
            )
        seq_len = self.data_cfg.get('seq_len', 60)
        end = as_of_date or date.today()
        start = end - timedelta(days=seq_len + 100)  # extra for warmup
        ticker = yf.Ticker(asset_id)
        df = ticker.history(
            start=start.isoformat(),
            end=end.isoformat(),
            interval='1d',
            auto_adjust=True,
        )
        if df is None or len(df) < seq_len + 50:
            raise RuntimeError(f'Insufficient data for {asset_id} as of {as_of_date}')
        df = df.reset_index()
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})
        df['asset_id'] = asset_id
        df = features.add_technical_features(df)
        df = df.dropna(subset=self.data_cfg.get('feature_cols', []))
        if len(df) < seq_len:
            raise RuntimeError(
                f'After features, fewer than {seq_len} rows for {asset_id}'
            )
        window = df.tail(seq_len)
        scaled = scaler.transform(window)
        return scaled[self.data_cfg.get('feature_cols', [])].values.astype(np.float32)
