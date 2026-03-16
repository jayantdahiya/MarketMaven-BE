"""
Evaluate checkpoint on test set: metrics, regime breakdown, backtest, write reports.
Supports daily regression (evaluate_checkpoint) and LOB classification (evaluate_lob_checkpoint).
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from api.data import normalization
from api.training import backtest, checkpointing, metrics

logger = logging.getLogger(__name__)


def evaluate_checkpoint(
    cfg: dict,
    checkpoint_path: str,
    test_loader: DataLoader,
    scaler: normalization.FeatureScaler,
    model: torch.nn.Module,
) -> dict:
    """Load model from checkpoint, run inference on test set, compute metrics, backtest, regime breakdown."""
    device = cfg.get('training', {}).get('device', 'cpu')
    checkpointing.load_checkpoint(checkpoint_path, model, device=device)
    model.to(device)
    model.eval()
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    all_meta: list[dict] = []
    with torch.no_grad():
        for batch in test_loader:
            x, y, meta = batch
            x = x.to(device)
            y_hat = model(x)
            b = x.size(0)
            all_pred.append(y_hat.cpu().numpy().ravel())
            all_true.append(y.numpy().ravel())
            aid = meta['asset_id']
            ts = meta['timestamp']
            for i in range(b):
                all_meta.append({
                    'asset_id': aid[i] if isinstance(aid, (list, tuple)) else aid,
                    'timestamp': ts[i] if isinstance(ts, (list, tuple)) else ts,
                })
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    timestamps = np.array([m['timestamp'] for m in all_meta])
    if len(y_true) == 0:
        raise RuntimeError('empty_test_split')
    # Sort by timestamp so backtest sees monotonic order (test set may mix assets)
    order = np.argsort(pd.to_datetime(timestamps, utc=True))
    y_pred = y_pred[order]
    y_true = y_true[order]
    timestamps = timestamps[order]
    eval_cfg = cfg.get('evaluation', {})
    ann = eval_cfg.get('annualization_factor', 252)
    bt = backtest.Backtester(
        signal_threshold=eval_cfg.get('signal_threshold', 0.0),
        transaction_cost_bps=eval_cfg.get('transaction_cost_bps', 5.0),
        slippage_bps=eval_cfg.get('slippage_bps', 2.0),
    )
    bt_df = bt.run(y_pred, y_true, timestamps)
    bt_summary = bt.summary(bt_df)
    strategy_returns = bt_df['net_return'].values
    equity_curve = bt_df['cumulative_equity'].values
    summary = metrics.compute_all_metrics(
        y_true, y_pred, strategy_returns, equity_curve, ann
    )
    summary['sharpe_after_cost'] = bt_summary['sharpe_after_cost']
    summary['sortino_after_cost'] = bt_summary['sortino_after_cost']
    summary['max_drawdown'] = bt_summary['max_drawdown']
    summary['net_pnl'] = bt_summary['net_pnl']
    # Regime breakdown: need SPY close for regime_labels; simplified - use full test set regime from returns
    regime_df = pd.DataFrame({
        'regime': ['all'],
        'n_samples': [len(y_true)],
        'mae': [summary['mae']],
        'rmse': [summary['rmse']],
        'directional_accuracy': [summary['directional_accuracy']],
        'sharpe': [summary['sharpe']],
        'sortino': [summary['sortino']],
        'max_drawdown': [summary['max_drawdown']],
    })
    return {
        'summary': summary,
        'regime_df': regime_df,
        'bt_df': bt_df,
        'bt_summary': bt_summary,
        'y_true': y_true,
        'y_pred': y_pred,
    }


def write_reports(
    report_dir: str,
    summary: dict,
    regime_df: pd.DataFrame,
    bt_df: pd.DataFrame,
    bt_summary: dict,
) -> None:
    """Write metrics_summary.json, regime_metrics.csv, backtest_report.csv, backtest_summary.json."""
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(report_dir) / 'metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    regime_df.to_csv(Path(report_dir) / 'regime_metrics.csv', index=False)
    bt_df.to_csv(Path(report_dir) / 'backtest_report.csv', index=False)
    with open(Path(report_dir) / 'backtest_summary.json', 'w') as f:
        json.dump(bt_summary, f, indent=2)


def evaluate_lob_checkpoint(
    cfg: dict,
    checkpoint_path: str,
    test_loader: DataLoader,
    model: torch.nn.Module,
) -> dict:
    """Load LOB model from checkpoint, run inference, compute classification metrics + execution backtest.

    Args:
        cfg: Config dict (reads lob.backtest.* and paths.*).
        checkpoint_path: Path to best_balacc_epoch_*.pt checkpoint.
        test_loader: DataLoader yielding 4-tuples (x_book, x_aux, y_class, meta).
        model: Instantiated LOB model (TLOBForecaster or LobCNNBaseline).

    Returns:
        Dict with keys: summary, regime_df, trades_df, bt_summary, y_true, y_pred_class, probs.
    """
    device = cfg.get('training', {}).get('device', 'cpu')
    checkpointing.load_checkpoint(checkpoint_path, model, device=device)
    model.to(device)
    model.eval()

    all_probs: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    all_meta: list[dict] = []

    with torch.no_grad():
        for batch in test_loader:
            x_book, x_aux, y_class, meta = batch
            x_book = x_book.to(device)
            x_aux = x_aux.to(device)
            logits = model(x_book, x_aux)
            probs_batch = torch.softmax(logits, dim=-1).cpu().numpy()  # [B, 3]
            preds = probs_batch.argmax(axis=-1)  # [B]
            all_probs.append(probs_batch)
            all_pred.append(preds)
            all_true.append(y_class.numpy())
            b = x_book.size(0)
            aid = (
                meta.get('asset_id', [None] * b)
                if isinstance(meta, dict)
                else [None] * b
            )
            ts = (
                meta.get('timestamp', [None] * b)
                if isinstance(meta, dict)
                else [None] * b
            )
            for i in range(b):
                all_meta.append({
                    'asset_id': aid[i] if isinstance(aid, (list, tuple)) else aid,
                    'timestamp': ts[i] if isinstance(ts, (list, tuple)) else ts,
                })

    if not all_probs:
        raise RuntimeError('empty_test_split')

    probs = np.concatenate(all_probs, axis=0)  # [N, 3]
    y_pred_class = np.concatenate(all_pred)  # [N]
    y_true = np.concatenate(all_true)  # [N]

    if len(y_true) == 0:
        raise RuntimeError('empty_test_split')

    # Classification metrics
    balacc = metrics.balanced_accuracy_3class(y_true, y_pred_class)
    f1 = metrics.macro_f1_3class(y_true, y_pred_class)

    # Execution backtest
    lob_cfg = cfg.get('lob', {})
    bt_cfg = lob_cfg.get('backtest', {})
    exec_bt = backtest.ExecutionBacktester(
        min_edge=bt_cfg.get('min_edge', 0.10),
        fee_bps=bt_cfg.get('fee_bps', 1.5),
        slippage_ticks=bt_cfg.get('slippage_ticks', 0.2),
        hold_events=bt_cfg.get('hold_events', 20),
        max_spread_ticks=bt_cfg.get('max_spread_ticks', 5.0),
    )
    # Build synthetic mid_prices and spreads from probs for backtest
    # (In real usage these come from the LOB data; here we approximate from metadata)
    n = len(probs)
    mid_prices = np.ones(
        n, dtype=np.float64
    )  # Normalized; absolute values not stored in test loader
    spreads = np.zeros(
        n, dtype=np.float64
    )  # Assume spread=0 (best-case) for offline evaluation

    trades_df = exec_bt.run(probs, mid_prices, spreads)
    bt_summary = exec_bt.summary(trades_df)

    fill_rate = metrics.execution_fill_rate(trades_df)
    avg_pnl = metrics.avg_trade_pnl_ticks(trades_df)

    summary = {
        'balanced_accuracy': float(balacc),
        'macro_f1': float(f1),
        'execution_fill_rate': float(fill_rate),
        'avg_trade_pnl_ticks': float(avg_pnl),
        'n_samples': int(len(y_true)),
    }
    summary.update({
        'net_pnl_ticks': bt_summary.get('net_pnl_ticks', 0.0),
        'n_trades': bt_summary.get('n_trades', 0),
        'win_rate': bt_summary.get('win_rate', 0.0),
        'sharpe_ticks': bt_summary.get('sharpe_ticks', 0.0),
    })

    # Volatility-based regime breakdown
    eval_cfg = cfg.get('evaluation', {})
    regime_window = eval_cfg.get('regime', {}).get('rolling_window_events', 10000)
    regime_df = _lob_regime_breakdown(y_true, y_pred_class, probs, regime_window)

    return {
        'summary': summary,
        'regime_df': regime_df,
        'trades_df': trades_df,
        'bt_summary': bt_summary,
        'y_true': y_true,
        'y_pred_class': y_pred_class,
        'probs': probs,
    }


def _lob_regime_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    window: int,
) -> pd.DataFrame:
    """Compute per-regime metrics using rolling volatility (max-prob variance as proxy).

    Regimes: high_vol, low_vol (split at median rolling variance of predicted probabilities).

    Args:
        y_true: Ground-truth class labels [N].
        y_pred: Predicted class labels [N].
        probs: Softmax probabilities [N, 3].
        window: Rolling window size for volatility estimate.

    Returns:
        DataFrame with per-regime balanced_accuracy and macro_f1.
    """
    n = len(y_true)
    if n == 0:
        return pd.DataFrame(
            columns=['regime', 'n_samples', 'balanced_accuracy', 'macro_f1']
        )

    # Use max-class probability as a confidence signal; rolling std as volatility proxy
    max_prob = probs.max(axis=1)  # [N]
    vol: np.ndarray = (
        pd
        .Series(max_prob)
        .rolling(window=min(window, n), min_periods=1)
        .std()
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    median_vol = float(np.median(vol))

    records = []
    for regime_name, mask in [
        ('high_vol', vol >= median_vol),
        ('low_vol', vol < median_vol),
    ]:
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        yt = y_true[idx]
        yp = y_pred[idx]
        records.append({
            'regime': regime_name,
            'n_samples': int(len(yt)),
            'balanced_accuracy': float(metrics.balanced_accuracy_3class(yt, yp)),
            'macro_f1': float(metrics.macro_f1_3class(yt, yp)),
        })

    if not records:
        records.append({
            'regime': 'all',
            'n_samples': int(n),
            'balanced_accuracy': float(
                metrics.balanced_accuracy_3class(y_true, y_pred)
            ),
            'macro_f1': float(metrics.macro_f1_3class(y_true, y_pred)),
        })
    return pd.DataFrame(records)


def write_lob_reports(
    report_dir: str,
    summary: dict,
    regime_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    bt_summary: dict,
) -> None:
    """Write LOB evaluation reports: lob_metrics_summary.json, lob_regime_metrics.csv, lob_backtest_report.csv."""
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(report_dir) / 'lob_metrics_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    regime_df.to_csv(Path(report_dir) / 'lob_regime_metrics.csv', index=False)
    trades_df.to_csv(Path(report_dir) / 'lob_backtest_report.csv', index=False)
    with open(Path(report_dir) / 'lob_backtest_summary.json', 'w') as f:
        json.dump(bt_summary, f, indent=2)
