"""
LOB data pipeline: raw ingestion, per-asset rolling normalization, feature engineering,
label generation, and sequence creation for TLOBForecaster training and inference.

Supports streaming via chunked reads for large HDF5/parquet/CSV files to avoid OOM.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Raw book column prefixes
_BID_PX = 'bid_px_'
_ASK_PX = 'ask_px_'
_BID_SZ = 'bid_sz_'
_ASK_SZ = 'ask_sz_'


def mid_price(df: pd.DataFrame) -> pd.Series:
    """Compute mid-price from level-1 bid/ask.

    Args:
        df: DataFrame with bid_px_1 and ask_px_1 columns.

    Returns:
        Mid-price series.
    """
    return (df['bid_px_1'] + df['ask_px_1']) / 2.0


def order_imbalance(df: pd.DataFrame, levels: int) -> pd.Series:
    """Volume imbalance across specified depth levels.

    Args:
        df: DataFrame with bid_sz_* and ask_sz_* columns.
        levels: Number of price levels to aggregate.

    Returns:
        Imbalance series in [-1, 1].
    """
    bid_vol = sum(df[f'{_BID_SZ}{i}'] for i in range(1, levels + 1))
    ask_vol = sum(df[f'{_ASK_SZ}{i}'] for i in range(1, levels + 1))
    return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8)


def extract_depth_features(df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    """Extract exactly levels * 4 raw depth feature columns.

    Columns ordered as: ask_px_1..L, ask_sz_1..L, bid_px_1..L, bid_sz_1..L.

    Args:
        df: DataFrame with raw LOB columns.
        levels: Number of price levels.

    Returns:
        DataFrame with exactly levels * 4 columns.
    """
    cols = (
        [f'{_ASK_PX}{i}' for i in range(1, levels + 1)]
        + [f'{_ASK_SZ}{i}' for i in range(1, levels + 1)]
        + [f'{_BID_PX}{i}' for i in range(1, levels + 1)]
        + [f'{_BID_SZ}{i}' for i in range(1, levels + 1)]
    )
    return df[cols].copy()


class LOBPipeline:
    """Full LOB data pipeline: load → normalize → features → labels → sequences.

    Args:
        cfg: Config dict (reads keys under lob.* and paths.*).
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        lob_cfg = cfg.get('lob', {})
        self.levels: int = lob_cfg.get('levels', 10)
        self.seq_len: int = lob_cfg.get('seq_len', 100)
        self.horizon_events: int = lob_cfg.get('horizon_events', 20)
        self.tick_size: float = lob_cfg.get('tick_size', 0.01)

        norm_cfg = lob_cfg.get('normalization', {})
        self.rolling_window: int = norm_cfg.get('rolling_window_events', 5000)
        self.clip_val: float = norm_cfg.get('clip', 6.0)

        label_cfg = lob_cfg.get('labeling', {})
        self.up_thresh: int = label_cfg.get('up_threshold_ticks', 1)
        self.down_thresh: int = label_cfg.get('down_threshold_ticks', -1)

        streaming_cfg = lob_cfg.get('streaming', {})
        self.streaming_enabled: bool = streaming_cfg.get('enabled', True)
        self.chunk_size: int = streaming_cfg.get('chunk_size', 100_000)

        self.raw_data_path: str = cfg.get('paths', {}).get(
            'lob_raw_data_path', 'artifacts/data/lob_raw'
        )

        # Normalization state: per-asset running stats (filled by normalize_book)
        self._norm_stats: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Step 1 — Load raw LOB events
    # ------------------------------------------------------------------

    def load_raw(self) -> pd.DataFrame:
        """Read raw LOB events from parquet/CSV/HDF5; chunked reads for large files.

        Returns:
            DataFrame with raw book columns and timestamp/asset_id.

        Raises:
            FileNotFoundError: If raw data path does not exist.
        """
        raw_path = Path(self.raw_data_path)
        if not raw_path.exists():
            raise FileNotFoundError(
                f'LOB raw data path not found: {raw_path.resolve()}'
            )

        chunks: list[pd.DataFrame] = []
        files = sorted(raw_path.glob('*.parquet')) + sorted(raw_path.glob('*.csv'))

        if not files:
            # Try reading the path itself as a single file
            files = [raw_path] if raw_path.is_file() else []

        if not files:
            raise FileNotFoundError(f'No LOB data files found under {raw_path}')

        for file_path in files:
            suffix = file_path.suffix.lower()
            if suffix == '.parquet':
                df_chunk = pd.read_parquet(file_path)
                chunks.append(df_chunk)
            elif suffix == '.csv':
                if self.streaming_enabled:
                    for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                        chunks.append(chunk)
                else:
                    chunks.append(pd.read_csv(file_path))
            elif suffix in ('.h5', '.hdf5'):
                try:
                    import h5py  # noqa: PLC0415

                    with h5py.File(file_path, 'r') as hf:
                        # Expect dataset named 'lob_events'
                        dataset_key = list(hf.keys())[0]
                        data = hf[dataset_key][:]
                        chunks.append(pd.DataFrame(data))
                except ImportError:
                    logger.warning(
                        'h5py not available; skipping HDF5 file %s', file_path
                    )

        if not chunks:
            raise FileNotFoundError(
                f'No readable LOB data files found under {raw_path}'
            )

        df = pd.concat(chunks, ignore_index=True)

        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        else:
            logger.warning('No timestamp column found in LOB data')

        # Drop rows with NaN in core book columns
        core_cols = (
            [f'{_BID_PX}{i}' for i in range(1, self.levels + 1)]
            + [f'{_ASK_PX}{i}' for i in range(1, self.levels + 1)]
            + [f'{_BID_SZ}{i}' for i in range(1, self.levels + 1)]
            + [f'{_ASK_SZ}{i}' for i in range(1, self.levels + 1)]
        )
        existing_core = [c for c in core_cols if c in df.columns]
        before = len(df)
        df = df.dropna(subset=existing_core)
        dropped_nan = before - len(df)
        if dropped_nan > 0:
            logger.info('Dropped %d rows with NaN in core book columns', dropped_nan)

        # Drop crossed-book events (best_bid >= best_ask)
        if 'bid_px_1' in df.columns and 'ask_px_1' in df.columns:
            before = len(df)
            df = df[df['bid_px_1'] < df['ask_px_1']]
            dropped_crossed = before - len(df)
            if dropped_crossed > 0:
                logger.info('Dropped %d crossed-book rows', dropped_crossed)

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 2 — Per-asset rolling z-score normalization
    # ------------------------------------------------------------------

    def normalize_book(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-asset rolling z-score normalization. Global z-score is forbidden.

        Args:
            df: Raw LOB DataFrame.

        Returns:
            DataFrame with normalized book channel columns.
        """
        df = df.copy()
        norm_cols = (
            [f'{_BID_PX}{i}' for i in range(1, self.levels + 1)]
            + [f'{_ASK_PX}{i}' for i in range(1, self.levels + 1)]
            + [f'{_BID_SZ}{i}' for i in range(1, self.levels + 1)]
            + [f'{_ASK_SZ}{i}' for i in range(1, self.levels + 1)]
            + ['mid_price']
        )
        existing_norm = [c for c in norm_cols if c in df.columns]

        asset_col = 'asset_id' if 'asset_id' in df.columns else None
        if asset_col is None:
            # No asset column — treat as single asset
            df = self._rolling_zscore_group(df, existing_norm)
        else:
            groups = []
            for asset_id, grp in df.groupby(asset_col, sort=False):
                grp = grp.copy()
                grp = self._rolling_zscore_group(grp, existing_norm)
                groups.append(grp)
            df = pd.concat(groups).sort_index()

        return df

    def _rolling_zscore_group(self, grp: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Apply rolling z-score + clip to specified columns of a single-asset group."""
        w = self.rolling_window
        for col in cols:
            if col not in grp.columns:
                continue
            roll = grp[col].rolling(window=w, min_periods=1)
            mu = roll.mean()
            sigma = roll.std().fillna(0.0)
            grp[col] = ((grp[col] - mu) / (sigma + 1e-8)).clip(
                -self.clip_val, self.clip_val
            )
        return grp

    # ------------------------------------------------------------------
    # Step 3 — Feature engineering
    # ------------------------------------------------------------------

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute engineered columns: mid_price, spread, imbalance_l1, imbalance_l10.

        Args:
            df: LOB DataFrame (raw or normalized).

        Returns:
            DataFrame with added feature columns.
        """
        df = df.copy()
        df['mid_price'] = mid_price(df).astype('float32')
        df['spread'] = (df['ask_px_1'] - df['bid_px_1']).astype('float32')
        # Spread normalization: divide by tick_size; positive-only sanity check
        if self.tick_size > 0:
            df['spread'] = df['spread'] / self.tick_size
        df = df[df['spread'] >= 0].copy()

        df['imbalance_l1'] = (
            (
                (df['bid_sz_1'] - df['ask_sz_1'])
                / (df['bid_sz_1'] + df['ask_sz_1'] + 1e-8)
            )
            .clip(-1.0, 1.0)
            .astype('float32')
        )

        bid_vol = sum(df[f'{_BID_SZ}{i}'] for i in range(1, self.levels + 1))
        ask_vol = sum(df[f'{_ASK_SZ}{i}'] for i in range(1, self.levels + 1))
        df['imbalance_l10'] = (
            ((bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-8))
            .clip(-1.0, 1.0)
            .astype('float32')
        )

        return df

    # ------------------------------------------------------------------
    # Step 4 — Label generation
    # ------------------------------------------------------------------

    def build_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate mid_move_label_20 classification target.

        Label: 0=down (move <= down_thresh ticks), 1=flat, 2=up (move >= up_thresh ticks).

        Args:
            df: DataFrame with mid_price column.

        Returns:
            DataFrame with mid_move_label_20 column added.
        """
        df = df.copy()
        h = self.horizon_events
        future_mid = df['mid_price'].shift(-h)
        move_ticks = (future_mid - df['mid_price']) / self.tick_size

        labels = np.ones(len(df), dtype=np.int8)  # default flat
        labels[move_ticks >= self.up_thresh] = 2  # up
        labels[move_ticks <= self.down_thresh] = 0  # down

        df['mid_move_label_20'] = labels
        # Drop last horizon_events rows (no valid future label)
        df = df.iloc[:-h].copy()
        return df

    # ------------------------------------------------------------------
    # Step 5 — Sequence building
    # ------------------------------------------------------------------

    def build_sequences(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Build (x_book, x_aux, y_class, meta) arrays.

        x_book: [N, T, 4, L] — structured book tensor
        x_aux:  [N, T, 4]    — auxiliary features (mid_price, spread, imbalance_l1, imbalance_l10)
        y_class:[N]           — int64 class labels {0, 1, 2}
        meta:   DataFrame with asset_id and timestamp per sequence

        Args:
            df: Processed DataFrame with all required columns.

        Returns:
            Tuple (x_book, x_aux, y_class, meta_df).

        Raises:
            ValueError: If sequences span multiple asset_id values.
        """
        T = self.seq_len
        L = self.levels

        # Book channel column names: ask_px, ask_sz, bid_px, bid_sz per level
        ask_px_cols = [f'{_ASK_PX}{i}' for i in range(1, L + 1)]
        ask_sz_cols = [f'{_ASK_SZ}{i}' for i in range(1, L + 1)]
        bid_px_cols = [f'{_BID_PX}{i}' for i in range(1, L + 1)]
        bid_sz_cols = [f'{_BID_SZ}{i}' for i in range(1, L + 1)]

        aux_cols = ['mid_price', 'spread', 'imbalance_l1', 'imbalance_l10']
        label_col = 'mid_move_label_20'

        # Validate required columns exist
        required = (
            ask_px_cols
            + ask_sz_cols
            + bid_px_cols
            + bid_sz_cols
            + aux_cols
            + [label_col]
        )
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f'Missing required columns: {missing}')

        asset_col = 'asset_id' if 'asset_id' in df.columns else None
        ts_col = 'timestamp' if 'timestamp' in df.columns else None

        all_x_book: list[np.ndarray] = []
        all_x_aux: list[np.ndarray] = []
        all_y: list[int] = []
        meta_records: list[dict] = []

        def _process_group(grp: pd.DataFrame) -> None:
            grp = grp.reset_index(drop=True)
            n = len(grp)
            if n < T + 1:
                return

            # Validate single asset
            if asset_col and grp[asset_col].nunique() > 1:
                raise ValueError('Sequences must not span multiple asset_id values')

            # Extract arrays once
            ask_px = grp[ask_px_cols].values.astype(np.float32)  # [n, L]
            ask_sz = grp[ask_sz_cols].values.astype(np.float32)
            bid_px = grp[bid_px_cols].values.astype(np.float32)
            bid_sz = grp[bid_sz_cols].values.astype(np.float32)
            aux_arr = grp[aux_cols].values.astype(np.float32)  # [n, 4]
            labels = grp[label_col].values  # [n]

            for start in range(0, n - T, 1):
                end = start + T
                # x_book: [T, 4, L]
                book = np.stack(
                    [
                        ask_px[start:end],  # [T, L]
                        ask_sz[start:end],
                        bid_px[start:end],
                        bid_sz[start:end],
                    ],
                    axis=1,
                )  # [T, 4, L]

                all_x_book.append(book)
                all_x_aux.append(aux_arr[start:end])  # [T, 4]
                all_y.append(int(labels[end]))

                meta_rec: dict = {}
                if asset_col:
                    meta_rec['asset_id'] = grp[asset_col].iloc[end]
                if ts_col:
                    meta_rec['timestamp'] = grp[ts_col].iloc[end]
                meta_records.append(meta_rec)

        if asset_col:
            for _, grp in df.groupby(asset_col, sort=False):
                _process_group(grp)
        else:
            _process_group(df)

        if not all_x_book:
            # Return empty arrays with correct shapes
            return (
                np.zeros((0, T, 4, L), dtype=np.float32),
                np.zeros((0, T, 4), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                pd.DataFrame(meta_records),
            )

        x_book = np.stack(all_x_book, axis=0).astype(np.float32)  # [N, T, 4, L]
        x_aux = np.stack(all_x_aux, axis=0).astype(np.float32)  # [N, T, 4]
        y_class = np.array(all_y, dtype=np.int64)

        return x_book, x_aux, y_class, pd.DataFrame(meta_records)

    # ------------------------------------------------------------------
    # Convenience: run full pipeline on a DataFrame
    # ------------------------------------------------------------------

    def run(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Execute the full pipeline: features → normalize → labels → sequences.

        Args:
            df: Raw LOB DataFrame (already loaded).

        Returns:
            (x_book, x_aux, y_class, meta_df)
        """
        df = self.build_features(df)
        df = self.normalize_book(df)
        df = self.build_labels(df)
        return self.build_sequences(df)

    # ------------------------------------------------------------------
    # Chronological split helpers
    # ------------------------------------------------------------------

    def split_by_date(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split DataFrame chronologically into train/val/test.

        Uses dates from config lob.split.{train_end, val_end, test_end}.

        Args:
            df: DataFrame with timestamp column (tz-aware).

        Returns:
            (train_df, val_df, test_df)
        """
        split_cfg = self.cfg.get('lob', {}).get('split', {})
        train_end = pd.Timestamp(split_cfg.get('train_end', '2024-06-30'), tz='UTC')
        val_end = pd.Timestamp(split_cfg.get('val_end', '2025-03-31'), tz='UTC')

        ts = pd.to_datetime(df['timestamp'], utc=True)
        train_df = df[ts <= train_end].copy()
        val_df = df[(ts > train_end) & (ts <= val_end)].copy()
        test_df = df[ts > val_end].copy()
        return train_df, val_df, test_df
