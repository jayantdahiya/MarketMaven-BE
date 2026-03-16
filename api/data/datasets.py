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

# Canonical modality order for mask construction — matches config feature_groups key order.
_MODALITY_ORDER: list[str] = ['price_tech', 'context', 'sentiment', 'alpha']

# Hidden columns written by DataPipeline._merge_multimodal() to track per-row availability.
_MODALITY_AVAILABLE_COLS: dict[str, str] = {
    'price_tech': '_price_tech_available',
    'context': '_context_available',
    'sentiment': '_sentiment_available',
    'alpha': '_alpha_available',
}


class DailySequenceDataset(Dataset):
    """
    Yields (x: [seq_len, F], y: [1], meta) per valid window.
    Windows do not cross asset boundaries. Data must be pre-scaled.

    When *asset_to_index* is provided the ``meta`` dict includes an
    ``asset_index`` key (integer) suitable for graph-context look-ups.
    When *graph_context* is provided (a ``[A, G]`` tensor) it is included
    in the ``meta`` dict so the training loop can pass it to the model.

    Phase 3+: when *feature_groups* and *feature_group_slices* are provided,
    ``meta`` will include:
    - ``modality_mask``: ``torch.Tensor`` of shape ``[M]`` (M=4) indicating
      which feature groups are available for the **last** timestep of the
      window (1.0 = available, 0.0 = missing).
    - ``feature_group_slices``: ``dict[str, tuple[int, int]]`` mapping each
      group name to ``(start_idx, end_idx)`` within the feature dimension.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        seq_len: int,
        horizon: int = 1,
        asset_to_index: dict[str, int] | None = None,
        graph_context: torch.Tensor | None = None,
        feature_groups: dict[str, list[str]] | None = None,
        feature_group_slices: dict[str, tuple[int, int]] | None = None,
    ):
        self.df = df.sort_values(['asset_id', 'timestamp']).reset_index(drop=True)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        self.horizon = horizon
        self.asset_to_index = asset_to_index or {}
        self.graph_context = graph_context
        self.feature_groups = feature_groups
        self.feature_group_slices = feature_group_slices
        self._multimodal = (
            feature_groups is not None and feature_group_slices is not None
        )
        self._indices: list[tuple[int, int]] = []  # (start_row, asset_id_key)
        self._asset_rows: dict[str, np.ndarray] = {}
        self._build_indices()

    def _build_indices(self) -> None:
        for asset_id, g in self.df.groupby('asset_id', sort=True):
            # Use integer positions (iloc) in the full df for this group
            start_iloc = self.df.index.get_indexer(g.index)[0]
            n = len(g)
            if n < self.seq_len + self.horizon:
                logger.warning(
                    'Asset %s has %d rows < seq_len + horizon (%d), skipping',
                    asset_id,
                    n,
                    self.seq_len + self.horizon,
                )
                continue
            for i in range(n - self.seq_len - self.horizon + 1):
                self._indices.append((start_iloc + i, asset_id))
        if not self._indices:
            raise ValueError('No valid windows: all assets too short or missing target')

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
            'asset_id': asset_id,
            'timestamp': str(block.iloc[-1]['timestamp']),
        }
        # Phase 2: include asset_index and graph_context when available
        if self.asset_to_index:
            meta['asset_index'] = self.asset_to_index.get(str(asset_id), 0)
        if self.graph_context is not None:
            meta['graph_context'] = self.graph_context
        # Phase 3+: include multimodal metadata
        if self._multimodal:
            meta['feature_group_slices'] = self.feature_group_slices
            meta['modality_mask'] = self._build_modality_mask(block)
        return torch.from_numpy(x), torch.from_numpy(y), meta

    # ------------------------------------------------------------------
    # Multimodal helpers (Phase 3+)
    # ------------------------------------------------------------------

    def _build_modality_mask(self, block: pd.DataFrame) -> torch.Tensor:
        """Build a binary modality mask ``[M]`` from hidden ``_*_available`` columns.

        Uses the **last** row of the window to determine availability — this
        matches the prediction target timestep.  If a hidden column is absent
        from the DataFrame, the modality defaults to **available** (1.0) so
        that pure price_tech pipelines degrade gracefully.

        Returns:
            Float32 tensor of shape ``[4]`` with 1.0 (available) or 0.0 (missing).
        """
        last_row = block.iloc[-1]
        mask = np.ones(len(_MODALITY_ORDER), dtype=np.float32)
        for i, modality in enumerate(_MODALITY_ORDER):
            avail_col = _MODALITY_AVAILABLE_COLS.get(modality)
            if avail_col is not None and avail_col in block.columns:
                mask[i] = float(last_row[avail_col])
        if mask.sum() == 0.0:
            logger.warning(
                'All modalities masked for asset=%s ts=%s — sample kept but may degrade predictions',
                last_row.get('asset_id', '?'),
                last_row.get('timestamp', '?'),
            )
        return torch.from_numpy(mask)
