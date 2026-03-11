"""
Statistical and financial metrics: MAE, RMSE, directional accuracy, Sharpe, Sortino, max drawdown, regime labels.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
EPS = 1e-8


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    if y_true.size == 0 or y_pred.size == 0:
        logger.warning('Empty input in mae')
        return float('nan')
    return float(np.abs(y_true - y_pred).mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    if y_true.size == 0 or y_pred.size == 0:
        logger.warning('Empty input in rmse')
        return float('nan')
    return float(np.sqrt(((y_true - y_pred) ** 2).mean()))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction where sign(y_true) == sign(y_pred). Tie at 0 counts as incorrect."""
    if y_true.size == 0 or y_pred.size == 0:
        logger.warning('Empty input in directional_accuracy')
        return float('nan')
    s_true = np.sign(y_true)
    s_pred = np.sign(y_pred)
    return float((s_true == s_pred).mean())


def sharpe(returns: np.ndarray, annualization_factor: int = 252) -> float:
    """Annualized Sharpe: sqrt(ann) * mean(r) / (std(r) + eps). Zero variance -> 0.0."""
    if returns.size == 0:
        logger.warning('Empty returns in sharpe')
        return float('nan')
    std = returns.std()
    if std < EPS:
        return 0.0
    return float(np.sqrt(annualization_factor) * returns.mean() / (std + EPS))


def sortino(returns: np.ndarray, annualization_factor: int = 252) -> float:
    """Sortino: downside std only (min(returns, 0))."""
    if returns.size == 0:
        logger.warning('Empty returns in sortino')
        return float('nan')
    downside = np.minimum(returns, 0.0)
    std_down = np.std(downside)
    if std_down < EPS:
        return 0.0 if returns.mean() <= 0 else float('inf')
    return float(np.sqrt(annualization_factor) * returns.mean() / (std_down + EPS))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Min of (equity / running_peak) - 1. Input is cumulative equity (e.g. cumprod of 1+r)."""
    if equity_curve.size == 0:
        logger.warning('Empty equity_curve in max_drawdown')
        return float('nan')
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve / peak) - 1.0
    return float(dd.min())


def regime_labels(
    market_close: pd.Series,
    window: int = 60,
    bull_threshold: float = 0.05,
    bear_threshold: float = -0.05,
) -> pd.Series:
    """Rolling window-day log return of market close; label bull / bear / sideways."""

    def rolling_log_ret(s):
        if len(s) < window:
            return np.nan
        a, b = s.iloc[0], s.iloc[-1]
        if a <= 0:
            return np.nan
        return np.log(b / a)

    rolling_ret = market_close.rolling(window=window, min_periods=window).apply(
        rolling_log_ret, raw=False
    )

    def label(r):
        if pd.isna(r):
            return 'sideways'
        if r >= bull_threshold:
            return 'bull'
        if r <= bear_threshold:
            return 'bear'
        return 'sideways'

    return rolling_ret.apply(label)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strategy_returns: np.ndarray,
    equity_curve: np.ndarray,
    annualization_factor: int = 252,
) -> dict[str, float]:
    """Convenience: return dict of mae, rmse, directional_accuracy, sharpe, sortino, max_drawdown."""
    return {
        'mae': mae(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'directional_accuracy': directional_accuracy(y_true, y_pred),
        'sharpe': sharpe(strategy_returns, annualization_factor),
        'sortino': sortino(strategy_returns, annualization_factor),
        'max_drawdown': max_drawdown(equity_curve),
    }
