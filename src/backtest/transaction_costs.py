"""Transaction cost model for portfolio rebalancing.

Models two cost components:
1. Brokerage: fixed per-trade commission (0.05%)
2. Slippage: market impact / bid-ask spread (0.03%)

Applied on both the buy leg and sell leg of each rebalance.
Total round-trip cost per unit of turnover = (brokerage + slippage) × 2

Typical usage:
    from src.backtest.transaction_costs import CostModel
    cost = CostModel()
    daily_cost = cost.compute(old_weights, new_weights, portfolio_value)
"""

import logging

import pandas as pd

from src.config import BROKERAGE, SLIPPAGE

logger = logging.getLogger(__name__)


class CostModel:
    """Two-sided transaction cost model.

    Args:
        brokerage: One-way brokerage as fraction of trade value (default 0.05%).
        slippage: One-way market impact as fraction (default 0.03%).
    """

    def __init__(
        self,
        brokerage: float = BROKERAGE,
        slippage: float = SLIPPAGE,
    ) -> None:
        self.brokerage = brokerage
        self.slippage = slippage
        self.one_way = brokerage + slippage
        self.round_trip = 2 * self.one_way

    def compute(
        self,
        old_weights: pd.Series,
        new_weights: pd.Series,
        portfolio_value: float = 1.0,
    ) -> float:
        """Compute total transaction cost for a rebalance.

        Args:
            old_weights: Portfolio weights before rebalance.
            new_weights: Target weights after rebalance.
            portfolio_value: Current portfolio value (default 1 = fractional cost).

        Returns:
            Total transaction cost as a fraction of portfolio value.
        """
        all_tickers = old_weights.index.union(new_weights.index)
        old = old_weights.reindex(all_tickers).fillna(0.0)
        new = new_weights.reindex(all_tickers).fillna(0.0)

        # One-way turnover = sum(|Δw|) / 2
        turnover = (new - old).abs().sum() / 2
        cost = turnover * self.round_trip * portfolio_value
        logger.debug(
            "Transaction cost — turnover=%.3f, cost=%.4f%%",
            turnover,
            cost * 100,
        )
        return cost

    def annual_cost_estimate(self, avg_turnover: float, rebalances_per_year: int) -> float:
        """Estimate annual drag from transaction costs.

        Args:
            avg_turnover: Average one-way turnover per rebalance.
            rebalances_per_year: Expected number of rebalances per year.

        Returns:
            Estimated annual cost as a fraction of portfolio value.
        """
        return avg_turnover * self.round_trip * rebalances_per_year
