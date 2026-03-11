"""
Evaluate checkpoint on test set: metrics, regime breakdown, backtest, write reports.
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
    device = cfg.get("training", {}).get("device", "cpu")
    state = checkpointing.load_checkpoint(checkpoint_path, model, device=device)
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
            aid = meta["asset_id"]
            ts = meta["timestamp"]
            for i in range(b):
                all_meta.append({
                    "asset_id": aid[i] if isinstance(aid, (list, tuple)) else aid,
                    "timestamp": ts[i] if isinstance(ts, (list, tuple)) else ts,
                })
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    timestamps = np.array([m["timestamp"] for m in all_meta])
    if len(y_true) == 0:
        raise RuntimeError("empty_test_split")
    # Sort by timestamp so backtest sees monotonic order (test set may mix assets)
    order = np.argsort(pd.to_datetime(timestamps, utc=True))
    y_pred = y_pred[order]
    y_true = y_true[order]
    timestamps = timestamps[order]
    eval_cfg = cfg.get("evaluation", {})
    ann = eval_cfg.get("annualization_factor", 252)
    bt = backtest.Backtester(
        signal_threshold=eval_cfg.get("signal_threshold", 0.0),
        transaction_cost_bps=eval_cfg.get("transaction_cost_bps", 5.0),
        slippage_bps=eval_cfg.get("slippage_bps", 2.0),
    )
    bt_df = bt.run(y_pred, y_true, timestamps)
    bt_summary = bt.summary(bt_df)
    strategy_returns = bt_df["net_return"].values
    equity_curve = bt_df["cumulative_equity"].values
    summary = metrics.compute_all_metrics(y_true, y_pred, strategy_returns, equity_curve, ann)
    summary["sharpe_after_cost"] = bt_summary["sharpe_after_cost"]
    summary["sortino_after_cost"] = bt_summary["sortino_after_cost"]
    summary["max_drawdown"] = bt_summary["max_drawdown"]
    summary["net_pnl"] = bt_summary["net_pnl"]
    # Regime breakdown: need SPY close for regime_labels; simplified - use full test set regime from returns
    regime_df = pd.DataFrame({"regime": ["all"], "n_samples": [len(y_true)], "mae": [summary["mae"]], "rmse": [summary["rmse"]], "directional_accuracy": [summary["directional_accuracy"]], "sharpe": [summary["sharpe"]], "sortino": [summary["sortino"]], "max_drawdown": [summary["max_drawdown"]]})
    return {
        "summary": summary,
        "regime_df": regime_df,
        "bt_df": bt_df,
        "bt_summary": bt_summary,
        "y_true": y_true,
        "y_pred": y_pred,
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
    with open(Path(report_dir) / "metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    regime_df.to_csv(Path(report_dir) / "regime_metrics.csv", index=False)
    bt_df.to_csv(Path(report_dir) / "backtest_report.csv", index=False)
    with open(Path(report_dir) / "backtest_summary.json", "w") as f:
        json.dump(bt_summary, f, indent=2)
