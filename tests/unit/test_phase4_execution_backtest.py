"""
Unit tests for ExecutionBacktester in api/training/backtest.py.

Tests: long signal produces positive gross PnL, spread filter skips trade, empty output.
"""

import numpy as np
import pytest

from api.training.backtest import ExecutionBacktester


# ---------------------------------------------------------------------------
# Test 1 — Long signal with rising price produces positive gross PnL
# ---------------------------------------------------------------------------


def test_long_signal_positive_gross_pnl():
    """A clear up-signal with a rising mid-price should yield positive gross PnL."""
    bt = ExecutionBacktester(
        min_edge=0.10,
        fee_bps=1.5,
        slippage_ticks=0.2,
        hold_events=5,
        max_spread_ticks=5.0,
    )

    n = 30
    # Probabilities strongly favour up (class 2): p_down=0.05, p_flat=0.10, p_up=0.85
    probs = np.tile([0.05, 0.10, 0.85], (n, 1)).astype(np.float32)

    # Strictly rising mid_price: +1 unit per event
    mid_prices = np.arange(100.0, 100.0 + n, dtype=np.float64)
    spreads = np.ones(n, dtype=np.float64) * 0.5  # tight spread

    trades = bt.run(probs, mid_prices, spreads)

    assert len(trades) > 0, 'Expected at least one trade'
    executed = trades[trades['executed']]
    assert len(executed) > 0, 'Expected at least one executed trade'
    # With price rising by 5 per hold window, gross pnl should be positive
    assert executed['gross_pnl_ticks'].sum() > 0, (
        'Expected positive gross PnL on rising price'
    )


# ---------------------------------------------------------------------------
# Test 2 — Spread filter: wide spread causes trade to be skipped (executed=False)
# ---------------------------------------------------------------------------


def test_spread_filter_skips_trade():
    """When spread > max_spread_ticks, trades are recorded with executed=False."""
    bt = ExecutionBacktester(
        min_edge=0.10,
        fee_bps=1.5,
        slippage_ticks=0.2,
        hold_events=5,
        max_spread_ticks=1.0,  # tight threshold
    )

    n = 10
    # Clear up signal
    probs = np.tile([0.05, 0.10, 0.85], (n, 1)).astype(np.float32)
    mid_prices = np.ones(n) * 100.0
    # All spreads are wide (10 >> max_spread_ticks=1.0)
    spreads = np.ones(n) * 10.0

    trades = bt.run(probs, mid_prices, spreads)

    assert len(trades) > 0, 'Expected spread-filtered signal records'
    assert (trades['executed'] == False).all(), (  # noqa: E712
        'All trades should be unexecuted when spread is too wide'
    )

    summary = bt.summary(trades)
    assert summary['n_trades'] == 0, 'fill_rate should be 0 with all filtered trades'
    assert summary['fill_rate'] == 0.0


# ---------------------------------------------------------------------------
# Test 3 — No signals => empty DataFrame with correct columns
# ---------------------------------------------------------------------------


def test_no_signals_returns_empty_dataframe():
    """When no signal exceeds min_edge, run() returns empty DataFrame."""
    bt = ExecutionBacktester(min_edge=0.50)  # very high threshold

    n = 20
    # Flat probabilities — no edge
    probs = np.tile([0.33, 0.34, 0.33], (n, 1)).astype(np.float32)
    mid_prices = np.ones(n) * 100.0
    spreads = np.ones(n) * 0.5

    trades = bt.run(probs, mid_prices, spreads)

    assert len(trades) == 0, 'Expected empty DataFrame when no signal fires'
    assert 'executed' in trades.columns, (
        'Empty DataFrame must still have expected columns'
    )

    summary = bt.summary(trades)
    assert summary['n_trades'] == 0
    assert summary['net_pnl_ticks'] == 0.0
