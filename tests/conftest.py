"""Shared fixtures for Phase 0 tests."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from api.models.lstm_baseline import LSTMBaseline


@pytest.fixture
def phase0_cfg(tmp_path):
    """Phase 0 config with paths redirected to tmp_path."""
    import yaml
    with open("config/phase_0.yaml") as f:
        cfg = yaml.safe_load(f) or {}
    cfg.setdefault("paths", {})
    cfg["paths"]["data_dir"] = str(tmp_path / "data")
    cfg["paths"]["checkpoints_dir"] = str(tmp_path / "checkpoints")
    cfg["paths"]["reports_dir"] = str(tmp_path / "reports")
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    return cfg


@pytest.fixture
def sample_daily_df():
    """Synthetic OHLCV: 500 rows, 2 assets (AAPL, SPY), deterministic."""
    np.random.seed(42)
    n = 500
    dates = pd.bdate_range(start="2020-01-01", periods=n, freq="B")
    out = []
    for asset in ["AAPL", "SPY"]:
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 1.0)
        high = close * (1 + np.abs(np.random.randn(n) * 0.01))
        low = close * (1 - np.abs(np.random.randn(n) * 0.01))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        volume = np.random.randint(1_000_000, 10_000_000, size=n)
        df = pd.DataFrame({
            "timestamp": dates,
            "open": open_, "high": high, "low": low, "close": close, "volume": volume,
            "asset_id": asset,
        })
        out.append(df)
    return pd.concat(out, ignore_index=True)


@pytest.fixture
def tmp_artifact_dir(tmp_path):
    """Create checkpoints, reports, data under tmp_path."""
    (tmp_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    return tmp_path


@pytest.fixture
def sample_lstm_model():
    """Small LSTM for fast tests: input_dim=8, hidden_dim=32, num_layers=1."""
    return LSTMBaseline(input_dim=8, hidden_dim=32, num_layers=1, dropout=0.0, horizon=1)
