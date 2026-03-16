"""
Long-only backtester with transaction costs and slippage (daily regression).
ExecutionBacktester for LOB classification signals with edge threshold and spread filter.
"""

import logging

import numpy as np
import pandas as pd

from api.training import metrics

logger = logging.getLogger(__name__)


class Backtester:
    """
    signal_t = 1 if predicted_return_t > signal_threshold else 0.
    Position 1.0 when long, 0.0 when flat. Cost on position change.
    """

    def __init__(
        self,
        signal_threshold: float = 0.0,
        transaction_cost_bps: float = 5.0,
        fixed_cost_per_trade: float = 0.0,
        slippage_bps: float = 2.0,
    ):
        self.signal_threshold = signal_threshold
        self.transaction_cost_bps = transaction_cost_bps
        self.fixed_cost_per_trade = fixed_cost_per_trade
        self.slippage_bps = slippage_bps
        self.cost_per_trade = (transaction_cost_bps + slippage_bps) / 10000.0

    def run(
        self,
        predicted_returns: np.ndarray,
        realized_returns: np.ndarray,
        timestamps: np.ndarray,
    ) -> pd.DataFrame:
        """Return DataFrame with timestamp, signal, position, gross_return, cost, net_return, cumulative_equity."""
        n = len(predicted_returns)
        if len(realized_returns) != n:
            raise ValueError(
                'predicted_returns and realized_returns must have same length'
            )
        if len(timestamps) != n:
            raise ValueError('timestamps must have same length as returns')
        ts = pd.to_datetime(timestamps, utc=True)
        if not ts.is_monotonic_increasing:
            raise ValueError('timestamps must be monotonically increasing')
        signals = (predicted_returns > self.signal_threshold).astype(np.float64)
        positions = np.zeros(n)
        positions[0] = signals[0]
        for i in range(1, n):
            positions[i] = signals[i]
        gross_returns = positions * realized_returns
        cost = np.zeros(n)
        for i in range(1, n):
            if positions[i] != positions[i - 1]:
                cost[i] = self.cost_per_trade + self.fixed_cost_per_trade
        net_returns = gross_returns - cost
        equity = np.cumprod(1.0 + net_returns)
        return pd.DataFrame({
            'timestamp': timestamps,
            'predicted_return': predicted_returns,
            'realized_return': realized_returns,
            'signal': signals,
            'position': positions,
            'gross_return': gross_returns,
            'cost': cost,
            'net_return': net_returns,
            'cumulative_equity': equity,
        })

    def summary(self, bt_df: pd.DataFrame) -> dict:
        """net_pnl, gross_pnl, turnover, sharpe_after_cost, sortino_after_cost, max_drawdown, breakeven_cost_bps, n_trades."""
        net_returns = bt_df['net_return'].values
        gross_returns = bt_df['gross_return'].values
        equity = bt_df['cumulative_equity'].values
        net_pnl = equity[-1] - 1.0 if len(equity) else 0.0
        gross_equity = np.cumprod(1.0 + gross_returns)
        gross_pnl = gross_equity[-1] - 1.0 if len(gross_equity) else 0.0
        turnover = (bt_df['position'].diff().abs() > 0).sum()
        n_trades = turnover  # same for long-only
        sharpe_after = metrics.sharpe(net_returns)
        sortino_after = metrics.sortino(net_returns)
        mdd = metrics.max_drawdown(equity)
        breakeven_bps = (
            (self.transaction_cost_bps + self.slippage_bps) if net_pnl > 0 else 0.0
        )
        return {
            'net_pnl': net_pnl,
            'gross_pnl': gross_pnl,
            'turnover': int(turnover),
            'n_trades': int(n_trades),
            'sharpe_after_cost': sharpe_after,
            'sortino_after_cost': sortino_after,
            'max_drawdown': mdd,
            'breakeven_cost_bps': breakeven_bps,
        }


class ExecutionBacktester:
    """Execution-aware LOB trade simulation.

    Entry rules:
        Long  if (p_up - p_down) >= min_edge
        Short if (p_down - p_up) >= min_edge
        Flat  otherwise

    Trade filtering: skip if spread > max_spread_ticks.
    Transaction costs: fee_bps per side + slippage_ticks per fill.

    Args:
        min_edge: Minimum probability edge to enter a trade.
        fee_bps: One-way fee in basis points (round-trip = 2x).
        slippage_ticks: Slippage in ticks per fill side.
        hold_events: Number of events to hold position before evaluating exit.
        max_spread_ticks: Maximum spread to allow trade entry.
    """

    def __init__(
        self,
        min_edge: float = 0.10,
        fee_bps: float = 1.5,
        slippage_ticks: float = 0.2,
        hold_events: int = 20,
        max_spread_ticks: float = 5.0,
    ) -> None:
        self.min_edge = min_edge
        self.fee_bps = fee_bps
        self.slippage_ticks = slippage_ticks
        self.hold_events = hold_events
        self.max_spread_ticks = max_spread_ticks

    def run(
        self,
        probs: np.ndarray,
        mid_prices: np.ndarray,
        spreads: np.ndarray,
    ) -> pd.DataFrame:
        """Simulate trades and return per-trade log.

        Args:
            probs: [N, 3] softmax probabilities for (down, flat, up).
            mid_prices: [N] mid-price series.
            spreads: [N] spread series in ticks.

        Returns:
            DataFrame with one row per executed trade:
                entry_idx, exit_idx, direction, entry_price, exit_price,
                gross_pnl_ticks, fee_ticks, slippage_ticks_total,
                net_pnl_ticks, signal_event, executed.
        """
        n = len(probs)
        if n == 0 or len(mid_prices) != n or len(spreads) != n:
            raise ValueError('probs, mid_prices, spreads must have the same length')

        p_down = probs[:, 0]
        p_up = probs[:, 2]

        records: list[dict] = []
        i = 0
        while i < n:
            # Determine signal
            edge_long = float(p_up[i] - p_down[i])
            edge_short = float(p_down[i] - p_up[i])

            if edge_long >= self.min_edge:
                direction = 1
            elif edge_short >= self.min_edge:
                direction = -1
            else:
                i += 1
                continue

            # Spread filter
            if spreads[i] > self.max_spread_ticks:
                # Skip — spread too wide
                records.append({
                    'entry_idx': i,
                    'exit_idx': i,
                    'direction': 0,
                    'entry_price': float(mid_prices[i]),
                    'exit_price': float(mid_prices[i]),
                    'gross_pnl_ticks': 0.0,
                    'fee_ticks': 0.0,
                    'slippage_ticks_total': 0.0,
                    'net_pnl_ticks': 0.0,
                    'signal_event': True,
                    'executed': False,
                })
                i += 1
                continue

            # Execute trade
            entry_idx = i
            exit_idx = min(i + self.hold_events, n - 1)

            entry_price = float(mid_prices[entry_idx])
            exit_price = float(mid_prices[exit_idx])

            gross_pnl_ticks = direction * (exit_price - entry_price)

            # Fee: fee_bps per side × 2 (round-trip) converted to ticks
            # We express fee as bps of mid-price / tick_size; here we keep everything
            # in abstract ticks so we scale fee_bps into tick units via mid_price ratio.
            # For simplicity: fee_ticks = fee_bps/10000 * entry_price (in ticks via price)
            fee_ticks = (self.fee_bps / 10000.0) * entry_price * 2.0
            slippage_total = self.slippage_ticks * 2.0  # entry + exit side

            net_pnl_ticks = gross_pnl_ticks - fee_ticks - slippage_total

            records.append({
                'entry_idx': entry_idx,
                'exit_idx': exit_idx,
                'direction': direction,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'gross_pnl_ticks': gross_pnl_ticks,
                'fee_ticks': fee_ticks,
                'slippage_ticks_total': slippage_total,
                'net_pnl_ticks': net_pnl_ticks,
                'signal_event': True,
                'executed': True,
            })

            # Advance past hold window to avoid overlapping trades
            i = exit_idx + 1

        if not records:
            logger.warning('ExecutionBacktester: no trades generated')
            return pd.DataFrame(
                columns=[
                    'entry_idx',
                    'exit_idx',
                    'direction',
                    'entry_price',
                    'exit_price',
                    'gross_pnl_ticks',
                    'fee_ticks',
                    'slippage_ticks_total',
                    'net_pnl_ticks',
                    'signal_event',
                    'executed',
                ]
            )

        return pd.DataFrame(records)

    def summary(self, trades_df: pd.DataFrame) -> dict:
        """Compute aggregate execution metrics from trade log.

        Args:
            trades_df: Output of run().

        Returns:
            Dict with net_pnl_ticks, gross_pnl_ticks, n_trades, fill_rate,
            win_rate, avg_net_pnl_ticks, sharpe_ticks.
        """
        if trades_df is None or len(trades_df) == 0:
            return {
                'net_pnl_ticks': 0.0,
                'gross_pnl_ticks': 0.0,
                'n_trades': 0,
                'fill_rate': 0.0,
                'win_rate': 0.0,
                'avg_net_pnl_ticks': 0.0,
                'sharpe_ticks': 0.0,
            }

        executed = trades_df[trades_df['executed']]
        n_signals = len(trades_df)
        n_executed = len(executed)
        fill_rate = n_executed / n_signals if n_signals > 0 else 0.0

        if len(executed) == 0:
            return {
                'net_pnl_ticks': 0.0,
                'gross_pnl_ticks': 0.0,
                'n_trades': 0,
                'fill_rate': fill_rate,
                'win_rate': 0.0,
                'avg_net_pnl_ticks': 0.0,
                'sharpe_ticks': 0.0,
            }

        net_pnl_series = executed['net_pnl_ticks'].values.astype(float)
        gross_pnl_series = executed['gross_pnl_ticks'].values.astype(float)

        win_rate = float((net_pnl_series > 0).mean())
        avg_net = float(net_pnl_series.mean())
        sharpe_t = metrics.sharpe(net_pnl_series, annualization_factor=1)

        return {
            'net_pnl_ticks': float(net_pnl_series.sum()),
            'gross_pnl_ticks': float(gross_pnl_series.sum()),
            'n_trades': int(n_executed),
            'fill_rate': fill_rate,
            'win_rate': win_rate,
            'avg_net_pnl_ticks': avg_net,
            'sharpe_ticks': sharpe_t,
        }
