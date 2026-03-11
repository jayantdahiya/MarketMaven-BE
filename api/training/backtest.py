"""
Long-only backtester with transaction costs and slippage.
"""
import numpy as np
import pandas as pd

from api.training import metrics


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
            raise ValueError("predicted_returns and realized_returns must have same length")
        if len(timestamps) != n:
            raise ValueError("timestamps must have same length as returns")
        ts = pd.to_datetime(timestamps, utc=True)
        if not ts.is_monotonic_increasing:
            raise ValueError("timestamps must be monotonically increasing")
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
            "timestamp": timestamps,
            "predicted_return": predicted_returns,
            "realized_return": realized_returns,
            "signal": signals,
            "position": positions,
            "gross_return": gross_returns,
            "cost": cost,
            "net_return": net_returns,
            "cumulative_equity": equity,
        })

    def summary(self, bt_df: pd.DataFrame) -> dict:
        """net_pnl, gross_pnl, turnover, sharpe_after_cost, sortino_after_cost, max_drawdown, breakeven_cost_bps, n_trades."""
        net_returns = bt_df["net_return"].values
        gross_returns = bt_df["gross_return"].values
        equity = bt_df["cumulative_equity"].values
        net_pnl = equity[-1] - 1.0 if len(equity) else 0.0
        gross_equity = np.cumprod(1.0 + gross_returns)
        gross_pnl = gross_equity[-1] - 1.0 if len(gross_equity) else 0.0
        turnover = (bt_df["position"].diff().abs() > 0).sum()
        n_trades = turnover  # same for long-only
        sharpe_after = metrics.sharpe(net_returns)
        sortino_after = metrics.sortino(net_returns)
        mdd = metrics.max_drawdown(equity)
        breakeven_bps = (self.transaction_cost_bps + self.slippage_bps) if net_pnl > 0 else 0.0
        return {
            "net_pnl": net_pnl,
            "gross_pnl": gross_pnl,
            "turnover": int(turnover),
            "n_trades": int(n_trades),
            "sharpe_after_cost": sharpe_after,
            "sortino_after_cost": sortino_after,
            "max_drawdown": mdd,
            "breakeven_cost_bps": breakeven_bps,
        }
