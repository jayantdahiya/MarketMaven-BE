"""Unit tests for pipeline: split, scaler, dataset shape."""
import numpy as np
import pandas as pd
import pytest

from api.data import features, targets, splitters, normalization, datasets
from api.data.pipeline import DataPipeline


def test_split_by_date_no_overlap(sample_daily_df, phase0_cfg):
    df = features.add_technical_features(sample_daily_df)
    df = targets.add_return_targets(df, [1])
    df = df.dropna()
    # Use split boundaries that fit fixture (data 2020-01-01 to 2021-11-30)
    train, val, test = splitters.split_by_date(
        df, train_end="2020-12-31", val_end="2021-06-30", test_end="2021-12-31"
    )
    assert pd.to_datetime(train["timestamp"]).max() < pd.to_datetime(val["timestamp"]).min()
    assert pd.to_datetime(val["timestamp"]).max() < pd.to_datetime(test["timestamp"]).min()


def test_dataset_shapes_match_config_seq_len(sample_daily_df, phase0_cfg):
    df = features.add_technical_features(sample_daily_df)
    df = targets.add_return_targets(df, [1])
    df = df.dropna()
    feature_cols = phase0_cfg.get("data", {}).get("feature_cols", ["close", "volume", "sma_15", "sma_45", "rsi_14", "macd", "macd_signal", "obv"])
    target_col = "log_return_1d"
    seq_len = 60
    horizon = 1
    ds = datasets.DailySequenceDataset(df, feature_cols, target_col, seq_len, horizon)
    x, y, meta = ds[0]
    assert x.shape == (60, 8)
    assert y.shape == (1,)


def test_scaler_clip_applied(sample_daily_df, phase0_cfg):
    df = features.add_technical_features(sample_daily_df)
    df = targets.add_return_targets(df, [1])
    df = df.dropna()
    feature_cols = phase0_cfg.get("data", {}).get("feature_cols", [])
    scaler = normalization.FeatureScaler(mode="per_asset_zscore", clip=5.0)
    scaler.fit(df, feature_cols)
    out = scaler.transform(df)
    assert out[feature_cols].abs().max().max() <= 5.0
