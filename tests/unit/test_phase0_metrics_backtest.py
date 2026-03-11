"""Unit tests for metrics and backtest."""

import numpy as np

from api.training import backtest, metrics


def test_sharpe_positive_for_positive_returns():
    # Non-zero variance so Sharpe is defined and positive (zero variance -> 0.0 per spec)
    np.random.seed(42)
    returns = np.ones(100) * 0.001 + np.random.randn(100) * 0.0005
    assert metrics.sharpe(returns) > 0


def test_max_drawdown_known_series():
    equity = np.array([1.0, 2.0, 1.5, 3.0])
    mdd = metrics.max_drawdown(equity)
    assert abs(mdd - (-0.25)) < 1e-8


def test_backtest_cost_applied_on_position_change():
    pred = np.array([0.01, -0.01, 0.01])
    real = np.array([0.01, 0.0, 0.01])
    ts = np.arange(3)
    bt = backtest.Backtester(transaction_cost_bps=5.0, slippage_bps=2.0)
    df = bt.run(pred, real, ts)
    assert (df['cost'] > 0).any()


def test_zero_variance_returns_sharpe_zero():
    returns = np.ones(10) * 0.0
    assert metrics.sharpe(returns) == 0.0
