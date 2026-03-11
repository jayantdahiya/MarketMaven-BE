"""
Per-asset or global z-score normalization with clipping; sklearn StandardScaler.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureScaler:
    """Z-score scaling per asset (or global), then clip. Fit on train only."""

    def __init__(self, mode: str = 'per_asset_zscore', clip: float = 5.0):
        if mode not in ('per_asset_zscore', 'global_zscore'):
            raise ValueError(
                f'mode must be per_asset_zscore or global_zscore, got {mode}'
            )
        self.mode = mode
        self.clip = clip
        self._scalers: dict[str, StandardScaler] = {}
        self._global_scaler: StandardScaler | None = None
        self._feature_cols: list[str] | None = None
        self._fitted = False

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> None:
        self._feature_cols = list(feature_cols)
        if self.mode == 'per_asset_zscore':
            for aid, g in df.groupby('asset_id'):
                sc = StandardScaler()
                sc.fit(g[self._feature_cols])
                self._scalers[aid] = sc
        else:
            self._global_scaler = StandardScaler()
            self._global_scaler.fit(df[self._feature_cols])
        self._fitted = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError('scaler_not_fitted')
        out = df.copy()
        out[self._feature_cols] = out[self._feature_cols].astype(np.float64)
        if self.mode == 'per_asset_zscore':
            for aid, sc in self._scalers.items():
                mask = df['asset_id'] == aid
                if mask.any():
                    out.loc[mask, self._feature_cols] = sc.transform(
                        df.loc[mask, self._feature_cols]
                    )
            # assets not seen in fit: leave as-is or use global fallback; spec says per-asset
        else:
            out[self._feature_cols] = self._global_scaler.transform(
                df[self._feature_cols]
            )
        out[self._feature_cols] = out[self._feature_cols].clip(-self.clip, self.clip)
        return out

    def inverse_transform(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError('scaler_not_fitted')
        out = df.copy()
        if self.mode == 'per_asset_zscore':
            for aid, sc in self._scalers.items():
                mask = df['asset_id'] == aid
                if mask.any():
                    out.loc[mask, cols] = sc.inverse_transform(df.loc[mask, cols])
        else:
            out[cols] = self._global_scaler.inverse_transform(df[cols])
        return out

    def get_state(self) -> dict[str, Any]:
        """Return serializable state for checkpoint portability."""
        if not self._fitted:
            raise RuntimeError('scaler_not_fitted')
        state = {
            'mode': self.mode,
            'clip': self.clip,
            'feature_cols': self._feature_cols,
        }
        if self.mode == 'per_asset_zscore':
            state['scalers'] = {
                aid: {'mean_': sc.mean_.tolist(), 'scale_': sc.scale_.tolist()}
                for aid, sc in self._scalers.items()
            }
        else:
            state['mean_'] = self._global_scaler.mean_.tolist()
            state['scale_'] = self._global_scaler.scale_.tolist()
        return state

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> 'FeatureScaler':
        """Create a FeatureScaler from a saved state dict (e.g. from checkpoint)."""
        inst = cls(mode=state['mode'], clip=state['clip'])
        inst._feature_cols = state['feature_cols']
        if state['mode'] == 'per_asset_zscore':
            for aid, s in state['scalers'].items():
                sc = StandardScaler()
                sc.mean_ = np.array(s['mean_'])
                sc.scale_ = np.array(s['scale_'])
                sc.n_features_in_ = len(sc.mean_)
                inst._scalers[aid] = sc
        else:
            inst._global_scaler = StandardScaler()
            inst._global_scaler.mean_ = np.array(state['mean_'])
            inst._global_scaler.scale_ = np.array(state['scale_'])
            inst._global_scaler.n_features_in_ = len(inst._global_scaler.mean_)
        inst._fitted = True
        return inst

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> 'FeatureScaler':
        return joblib.load(path)
