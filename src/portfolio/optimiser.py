"""Portfolio optimisation: max-Sharpe, min-variance, and risk parity.

Three optimisers mapped to market regimes:
1. max_sharpe_beta_constrained — Bull markets: maximise Sharpe with beta bounds
2. min_variance — Bear/high-VIX: minimise portfolio volatility (capital protection)
3. risk_parity — Transition: equal risk contribution from each asset

All use Ledoit-Wolf shrinkage for covariance estimation (more stable than
sample covariance with 49 assets and short windows).

Typical usage:
    from src.portfolio.optimiser import PortfolioOptimiser
    opt = PortfolioOptimiser()
    weights = opt.optimise(prices_window, current_betas, 'max_sharpe_beta_constrained')
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from pypfopt import EfficientFrontier, risk_models, expected_returns
    PYPFOPT_AVAILABLE = True
except ImportError:
    PYPFOPT_AVAILABLE = False
    logger.warning("PyPortfolioOpt not installed — portfolio optimisation unavailable")

try:
    import riskfolio as rp
    RISKFOLIO_AVAILABLE = True
except ImportError:
    RISKFOLIO_AVAILABLE = False
    logger.warning("riskfolio-lib not installed — risk parity unavailable")

from src.config import MAX_WEIGHT, MIN_WEIGHT, DEFAULT_BETA_TARGET


class PortfolioOptimiser:
    """Portfolio weight optimiser with regime-aware mode selection.

    Args:
        max_weight: Maximum weight for any single stock (default 15%).
        min_weight: Minimum non-zero weight (default 1%).
        risk_free_rate: Annual risk-free rate used in Sharpe computation.
    """

    def __init__(
        self,
        max_weight: float = MAX_WEIGHT,
        min_weight: float = MIN_WEIGHT,
        risk_free_rate: float = 0.065,
    ) -> None:
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.risk_free_rate = risk_free_rate

    def optimise(
        self,
        prices_window: pd.DataFrame,
        current_betas: pd.Series,
        mode: str,
        beta_target: float = DEFAULT_BETA_TARGET,
    ) -> pd.Series:
        """Compute optimal portfolio weights.

        Args:
            prices_window: Historical close prices for covariance estimation.
                Typically 252 rows (1 year of trading days).
            current_betas: Latest beta estimate for each stock (index=tickers).
            mode: One of 'max_sharpe_beta_constrained' | 'min_variance' |
                'risk_parity' | 'hold'.
            beta_target: Target portfolio beta for beta-constrained MVO.

        Returns:
            Series of portfolio weights summing to 1.0 (index=tickers).
        """
        if mode == "hold":
            return pd.Series(dtype=float)  # signal: no change

        tickers = prices_window.columns.tolist()
        n = len(tickers)

        if not PYPFOPT_AVAILABLE:
            logger.warning("PyPortfolioOpt unavailable — returning equal weights")
            return self._equal_weight(tickers)

        try:
            if mode == "max_sharpe_beta_constrained":
                return self._max_sharpe_beta_constrained(
                    prices_window, current_betas, tickers, beta_target
                )
            elif mode == "min_variance":
                return self._min_variance(prices_window, tickers)
            elif mode == "risk_parity":
                return self._risk_parity(prices_window, tickers)
            else:
                logger.warning("Unknown mode '%s' — returning equal weights", mode)
                return self._equal_weight(tickers)
        except Exception as exc:
            logger.warning(
                "Optimisation failed (mode=%s): %s — falling back to min_variance",
                mode,
                exc,
            )
            try:
                return self._min_variance(prices_window, tickers)
            except Exception as exc2:
                logger.warning("Min-variance fallback also failed: %s — equal weights", exc2)
                return self._equal_weight(tickers)

    def _cov_and_mu(
        self,
        prices_window: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Estimate shrunk covariance matrix and mean historical returns.

        Args:
            prices_window: Price panel.

        Returns:
            Tuple of (S_shrunk, mu).
        """
        S = risk_models.CovarianceShrinkage(prices_window).ledoit_wolf()
        mu = expected_returns.mean_historical_return(prices_window, frequency=252)
        return S, mu

    def _max_sharpe_beta_constrained(
        self,
        prices: pd.DataFrame,
        betas: pd.Series,
        tickers: list[str],
        beta_target: float,
    ) -> pd.Series:
        """Maximum Sharpe optimisation with beta band constraint.

        Portfolio beta is constrained to [beta_target - 0.2, beta_target].

        Args:
            prices: Historical prices.
            betas: Current per-stock betas.
            tickers: Ordered list of tickers.
            beta_target: Target portfolio beta.

        Returns:
            Cleaned weight Series.
        """
        S, mu = self._cov_and_mu(prices)
        betas_arr = betas.reindex(tickers).fillna(1.0).values

        ef = EfficientFrontier(mu, S)
        ef.add_constraint(lambda w: w >= 0)
        ef.add_constraint(lambda w: w <= self.max_weight)

        # Beta band constraint
        ef.add_constraint(
            lambda w: sum(w[i] * betas_arr[i] for i in range(len(tickers))) <= beta_target
        )
        ef.add_constraint(
            lambda w: sum(w[i] * betas_arr[i] for i in range(len(tickers)))
            >= beta_target - 0.2
        )

        ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        raw_weights = ef.clean_weights()
        return pd.Series(raw_weights)

    def _min_variance(
        self,
        prices: pd.DataFrame,
        tickers: list[str],
    ) -> pd.Series:
        """Minimum variance (min-vol) optimisation.

        Args:
            prices: Historical prices.
            tickers: Ordered list of tickers.

        Returns:
            Cleaned weight Series.
        """
        S, _ = self._cov_and_mu(prices)
        ef = EfficientFrontier(None, S)
        ef.add_constraint(lambda w: w >= 0)
        ef.add_constraint(lambda w: w <= self.max_weight)
        ef.min_volatility()
        return pd.Series(ef.clean_weights())

    def _risk_parity(
        self,
        prices: pd.DataFrame,
        tickers: list[str],
    ) -> pd.Series:
        """Risk parity: equal risk contribution from each asset.

        Falls back to min-variance if riskfolio-lib is unavailable.

        Args:
            prices: Historical prices.
            tickers: Ordered list of tickers.

        Returns:
            Weight Series.
        """
        if not RISKFOLIO_AVAILABLE:
            logger.warning("riskfolio-lib unavailable — using min-variance for risk parity")
            return self._min_variance(prices, tickers)

        returns = np.log(prices / prices.shift(1)).dropna()
        port = rp.Portfolio(returns=returns)
        port.assets_stats(method_mu="hist", method_cov="ledoit")
        w = port.optimization(
            model="Classic",
            rm="MV",
            obj="Sharpe",
            rf=self.risk_free_rate,
            l=0,
            hist=True,
        )
        if w is not None:
            weights = w["weights"]
            weights.index = tickers[: len(weights)]
            return weights
        return self._min_variance(prices, tickers)

    @staticmethod
    def _equal_weight(tickers: list[str]) -> pd.Series:
        """Equal weight fallback.

        Args:
            tickers: List of tickers.

        Returns:
            Equal weight Series.
        """
        n = len(tickers)
        return pd.Series({t: 1.0 / n for t in tickers})
