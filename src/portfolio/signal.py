"""Threshold trigger and rebalance signal generation.

This is the key innovation of AdaptiveBeta:
Instead of rebalancing on a fixed calendar schedule, we trigger
rebalancing only when the LSTM-predicted future beta volatility
exceeds a statistically calibrated threshold.

Signal priority:
1. VIX override: if VIX > 25 → MIN_VARIANCE (market stress, defensive)
2. Beta threshold: if predicted_betavol > threshold → REBALANCE
3. Otherwise: HOLD (save transaction costs)

Portfolio mode routing:
- MIN_VARIANCE: min-vol optimisation regardless of regime
- bull regime + REBALANCE → max_sharpe_beta_constrained
- bear regime + REBALANCE → min_variance (protect capital)
- transition + REBALANCE → risk_parity

Typical usage:
    from src.portfolio.signal import SignalGenerator
    sig_gen = SignalGenerator.from_training_data(portfolio_betavol_train)
    signal = sig_gen.get_signal(predicted_betavol=0.12, vix=22.5)
    mode = sig_gen.get_portfolio_mode(signal, regime='bull')
"""

import logging
from dataclasses import dataclass
from typing import Literal

import pandas as pd
import numpy as np

from src.config import THRESHOLD_QUANTILE, VIX_STRESS_LEVEL

logger = logging.getLogger(__name__)

Signal = Literal["HOLD", "REBALANCE", "MIN_VARIANCE"]
PortfolioMode = Literal["hold", "max_sharpe_beta_constrained", "min_variance", "risk_parity"]


@dataclass
class SignalState:
    """Represents a rebalance decision at a point in time."""

    signal: Signal
    mode: PortfolioMode
    predicted_betavol: float
    vix_level: float
    regime: str
    threshold: float


class SignalGenerator:
    """Generates rebalance signals from LSTM predictions and VIX.

    Args:
        threshold: Beta volatility trigger level. Predictions above this
            trigger a rebalance event.
        vix_threshold: VIX level above which we force min-variance.
    """

    def __init__(
        self,
        threshold: float,
        vix_threshold: float = VIX_STRESS_LEVEL,
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"Threshold must be positive, got {threshold}")
        self.threshold = threshold
        self.vix_threshold = vix_threshold
        logger.info(
            "SignalGenerator initialised — betavol_threshold=%.4f, vix_threshold=%.1f",
            threshold,
            vix_threshold,
        )

    @classmethod
    def from_training_data(
        cls,
        portfolio_betavol_train: pd.Series,
        quantile: float = THRESHOLD_QUANTILE,
        vix_threshold: float = VIX_STRESS_LEVEL,
    ) -> "SignalGenerator":
        """Calibrate the threshold from in-sample data.

        The threshold is the `quantile`-th percentile of the portfolio-level
        beta volatility over the training period. Using the 75th percentile
        means only the top 25% most volatile beta environments trigger rebalancing.

        Args:
            portfolio_betavol_train: Time series of portfolio betavol in the
                training period (equal-weighted average across all stocks).
            quantile: Quantile for threshold (default 0.75).
            vix_threshold: VIX override level.

        Returns:
            Calibrated SignalGenerator.
        """
        threshold = portfolio_betavol_train.dropna().quantile(quantile)
        logger.info(
            "Threshold calibrated at %.0f%% quantile: %.4f (N=%d training days)",
            quantile * 100,
            threshold,
            portfolio_betavol_train.notna().sum(),
        )
        return cls(threshold=float(threshold), vix_threshold=vix_threshold)

    def get_signal(
        self,
        predicted_betavol: float,
        vix_level: float,
    ) -> Signal:
        """Determine the rebalancing signal.

        Args:
            predicted_betavol: LSTM prediction of forward beta volatility.
            vix_level: Current India VIX level.

        Returns:
            'MIN_VARIANCE' | 'REBALANCE' | 'HOLD'
        """
        if vix_level > self.vix_threshold:
            return "MIN_VARIANCE"
        if predicted_betavol > self.threshold:
            return "REBALANCE"
        return "HOLD"

    def get_portfolio_mode(self, signal: Signal, regime: str) -> PortfolioMode:
        """Route signal + market regime to a specific portfolio optimiser.

        Args:
            signal: Output of get_signal().
            regime: Current HMM regime label ('bull' | 'bear' | 'transition').

        Returns:
            Portfolio mode string consumed by the Optimiser.
        """
        if signal == "MIN_VARIANCE":
            return "min_variance"
        if signal == "REBALANCE":
            if regime == "bull":
                return "max_sharpe_beta_constrained"
            if regime == "bear":
                return "min_variance"
            return "risk_parity"
        return "hold"

    def evaluate(
        self,
        predicted_betavol: float,
        vix_level: float,
        regime: str,
    ) -> SignalState:
        """Full evaluation returning a SignalState dataclass.

        Args:
            predicted_betavol: LSTM predicted beta volatility.
            vix_level: India VIX level.
            regime: Current market regime.

        Returns:
            SignalState with all decision fields populated.
        """
        signal = self.get_signal(predicted_betavol, vix_level)
        mode = self.get_portfolio_mode(signal, regime)
        return SignalState(
            signal=signal,
            mode=mode,
            predicted_betavol=predicted_betavol,
            vix_level=vix_level,
            regime=regime,
            threshold=self.threshold,
        )
