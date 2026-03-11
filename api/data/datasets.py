"""
PyTorch Dataset for sliding-window sequences; no cross-asset windows.
"""
import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DailySequenceDataset(Dataset):
    """
    Yields (x: [seq_len, F], y: [1], meta) per valid window.
    Windows do not cross asset boundaries. Data must be pre-scaled.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        seq_len: int,
        horizon: int = 1,
    ):
        self.df = df.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.horizon = horizon
        self._indices: list[tuple[int, int]] = []  # (start_row, asset_id_key)
        self._asset_rows: dict[str, np.ndarray] = {}
        self._build_indices()

    def _build_indices(self) -> None:
        for asset_id, g in self.df.groupby("asset_id", sort=True):
            # Use integer positions (iloc) in the full df for this group
            start_iloc = self.df.index.get_indexer(g.index)[0]
            n = len(g)
            if n < self.seq_len + self.horizon:
                logger.warning(
                    "Asset %s has %d rows < seq_len + horizon (%d), skipping",
                    asset_id, n, self.seq_len + self.horizon,
                )
                continue
            for i in range(n - self.seq_len - self.horizon + 1):
                self._indices.append((start_iloc + i, asset_id))
        if not self._indices:
            raise ValueError("No valid windows: all assets too short or missing target")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
        start_i, asset_id = self._indices[idx]
        end_i = start_i + self.seq_len
        block = self.df.iloc[start_i:end_i]
        x = block[self.feature_cols].values.astype(np.float32)
        y_val = block.iloc[-1][self.target_col]
        if pd.isna(y_val):
            y_val = 0.0
        y = np.array([y_val], dtype=np.float32)
        meta: dict[str, Any] = {
            "asset_id": asset_id,
            "timestamp": str(block.iloc[-1]["timestamp"]),
        }
        return torch.from_numpy(x), torch.from_numpy(y), meta
