"""
History routes: GET /history/returns, GET /history/predictions.

Serves historical log-returns and model predictions from the Phase 0 parquet data
and evaluation backtest reports for use in the frontend chart components.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request

from api.schemas.history import (
    HistoricalPredictionsResponse,
    HistoricalReturnsResponse,
    PredictionPoint,
    ReturnPoint,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/history', tags=['history'])

# Rolling window for volatility series (trading days)
_VOL_WINDOW = 30


@router.get('/returns', response_model=HistoricalReturnsResponse)
async def get_historical_returns(
    asset_id: str,
    request: Request,
    rolling_vol: bool = False,
    limit: int = 504,  # ~2 trading years default
) -> HistoricalReturnsResponse:
    """Return historical daily log-returns for an asset from the parquet data store.

    Args:
        asset_id: Ticker symbol (e.g. AAPL).
        rolling_vol: When true, also return a 30-day rolling volatility series.
        limit: Max number of most-recent data points to return (default 504 ≈ 2yr).
    """
    cfg = getattr(request.app.state, 'config', {})
    data_dir = Path(cfg.get('paths', {}).get('data_dir', 'artifacts/data/phase0'))
    parquet_path = data_dir / 'raw_daily.parquet'

    if not parquet_path.exists():
        raise HTTPException(
            status_code=503,
            detail={
                'code': 'data_unavailable',
                'message': f'Historical data not found at {parquet_path}',
            },
        )

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={'code': 'data_read_error', 'message': str(exc)},
        ) from exc

    # Normalise asset_id lookup — parquet uses a 'ticker' or 'asset_id' column
    ticker_col = 'ticker' if 'ticker' in df.columns else 'asset_id'
    asset_df = df[df[ticker_col] == asset_id.upper()].copy()

    if asset_df.empty:
        available = sorted(df[ticker_col].unique().tolist())
        raise HTTPException(
            status_code=404,
            detail={
                'code': 'asset_not_found',
                'message': f'Asset {asset_id!r} not found. Available: {available}',
            },
        )

    # Ensure sorted by date and compute log-return if not already present
    date_col = 'date' if 'date' in asset_df.columns else asset_df.index.name or 'date'
    if date_col in asset_df.columns:
        asset_df = asset_df.sort_values(date_col)
    else:
        asset_df = asset_df.sort_index()
        asset_df[date_col] = asset_df.index

    # Use pre-computed log_return_1d if available; otherwise compute from close
    if 'log_return_1d' in asset_df.columns:
        ret_series = asset_df['log_return_1d']
    elif 'close' in asset_df.columns:
        ret_series = np.log(asset_df['close'] / asset_df['close'].shift(1))
    else:
        raise HTTPException(
            status_code=503,
            detail={
                'code': 'data_schema_error',
                'message': 'Data has neither log_return_1d nor close column',
            },
        )

    asset_df = asset_df.assign(_ret=ret_series).dropna(subset=['_ret'])

    # Apply limit (most recent rows)
    if limit > 0:
        asset_df = asset_df.tail(limit)

    def _to_date_str(val) -> str:  # type: ignore[no-untyped-def]
        if isinstance(val, pd.Timestamp):
            return val.strftime('%Y-%m-%d')
        return str(val)[:10]

    date_vals = asset_df[date_col].tolist()
    ret_vals = asset_df['_ret'].tolist()

    returns = [
        ReturnPoint(date=_to_date_str(d), log_return=float(r))
        for d, r in zip(date_vals, ret_vals)
    ]

    rolling_vol_series: list[ReturnPoint] | None = None
    if rolling_vol:
        vol = (
            pd
            .Series(ret_vals)
            .rolling(window=_VOL_WINDOW, min_periods=_VOL_WINDOW // 2)
            .std()
            .fillna(0.0)
            .tolist()
        )
        rolling_vol_series = [
            ReturnPoint(date=_to_date_str(d), log_return=float(v))
            for d, v in zip(date_vals, vol)
        ]

    return HistoricalReturnsResponse(
        asset_id=asset_id.upper(),
        returns=returns,
        rolling_volatility=rolling_vol_series,
    )


@router.get('/predictions', response_model=HistoricalPredictionsResponse)
async def get_historical_predictions(
    asset_id: str,
    request: Request,
    model: str = 'lstm_baseline',
    limit: int = 252,  # ~1 trading year default
) -> HistoricalPredictionsResponse:
    """Return historical model predictions vs actual returns from backtest reports.

    Reads the most-recent backtest_report.csv from the appropriate phase report
    directory and filters to the requested asset.

    Args:
        asset_id: Ticker symbol (e.g. AAPL).
        model: Model type — determines which phase report dir to read.
        limit: Max number of most-recent data points to return.
    """
    # Map model type → reports directory
    _MODEL_PHASE_MAP: dict[str, str] = {
        'lstm_baseline': 'artifacts/reports/phase0',
        'cnn_transformer': 'artifacts/reports/phase1',
        'mamba_ssm': 'artifacts/reports/phase2',
    }
    reports_base = _MODEL_PHASE_MAP.get(model, 'artifacts/reports/phase0')
    reports_dir = Path(reports_base)

    if not reports_dir.exists():
        raise HTTPException(
            status_code=503,
            detail={
                'code': 'reports_unavailable',
                'message': f'No reports found for model {model!r}',
            },
        )

    # Find the most-recently modified backtest_report.csv
    candidates = sorted(
        reports_dir.rglob('backtest_report.csv'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise HTTPException(
            status_code=503,
            detail={
                'code': 'reports_unavailable',
                'message': f'No backtest reports found for model {model!r}',
            },
        )

    csv_path = candidates[0]
    try:
        bt_df = pd.read_csv(csv_path)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail={'code': 'report_read_error', 'message': str(exc)},
        ) from exc

    # The backtest CSV has columns: timestamp, predicted_return, actual_return, signal, ...
    # Filter by asset if asset_id column present (multi-asset reports); else use all rows
    if 'asset_id' in bt_df.columns:
        bt_df = bt_df[bt_df['asset_id'] == asset_id.upper()]

    if bt_df.empty:
        raise HTTPException(
            status_code=404,
            detail={
                'code': 'asset_not_found',
                'message': f'No predictions for asset {asset_id!r} in model {model!r} report',
            },
        )

    # Sort by timestamp
    ts_col = 'timestamp' if 'timestamp' in bt_df.columns else bt_df.columns[0]
    bt_df = bt_df.sort_values(ts_col)

    if limit > 0:
        bt_df = bt_df.tail(limit)

    def _to_date_str(val) -> str:  # type: ignore[no-untyped-def]
        s = str(val)
        return s[:10]

    predictions: list[PredictionPoint] = []
    for _, row in bt_df.iterrows():
        predictions.append(
            PredictionPoint(
                date=_to_date_str(row[ts_col]),
                predicted_return=float(
                    row.get('y_pred', row.get('predicted_return', 0.0))
                ),
                actual_return=float(
                    row.get(
                        'y_true', row.get('actual_return', row.get('net_return', 0.0))
                    )
                ),
                signal=str(row.get('signal', 'flat')),
            )
        )

    return HistoricalPredictionsResponse(
        asset_id=asset_id.upper(),
        model=model,
        predictions=predictions,
    )
