"""Rolling OLS beta and beta volatility computation.

Beta (β) is the systematic risk coefficient in CAPM:
    r_stock = α + β * r_market + ε

This module computes:
1. Rolling OLS beta for multiple windows (30d, 60d, 120d)
2. Rolling beta volatility (std-dev of beta over the same window)
3. Forward beta volatility (the prediction TARGET — betavol shifted back
   TARGET_HORIZON trading days so each row aligns with "what betavol
   will be TARGET_HORIZON days from now")

Typical usage:
    from src.features.beta import rolling_beta, beta_volatility, forward_target
    betas = rolling_beta(stock_ret, market_ret, windows=[30, 60, 120])
    betavols = beta_volatility(betas)
    target = forward_target(betavols[60], horizon=20)
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd

from src.config import BETA_WINDOWS, TARGET_HORIZON

logger = logging.getLogger(__name__)


def rolling_beta(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    window: int,
) -> pd.Series:
    """Compute rolling OLS beta for a single stock.

    Uses the standard formula: β = Cov(r_s, r_m) / Var(r_m)

    Args:
        stock_ret: Log returns of the stock (aligned with market_ret).
        market_ret: Log returns of the market index.
        window: Rolling window in trading days.

    Returns:
        Series of rolling beta values, same index as inputs.
    """
    cov = stock_ret.rolling(window).cov(market_ret)
    var = market_ret.rolling(window).var()
    beta = cov / var
    beta.name = stock_ret.name
    return beta


def rolling_beta_panel(
    stock_returns: pd.DataFrame,
    market_ret: pd.Series,
    windows: list[int] | None = None,
) -> Dict[int, pd.DataFrame]:
    """Compute rolling beta for all stocks across multiple windows.

    Args:
        stock_returns: Log returns panel — rows=dates, cols=tickers.
        market_ret: Log returns of the market index.
        windows: List of rolling window sizes. Defaults to BETA_WINDOWS.

    Returns:
        Dict mapping window size → DataFrame of betas (same shape as stock_returns).
    """
    if windows is None:
        windows = BETA_WINDOWS

    betas: Dict[int, pd.DataFrame] = {}
    for w in windows:
        logger.info("Computing rolling beta — window=%dd", w)
        betas[w] = stock_returns.apply(
            lambda s: rolling_beta(s, market_ret, w)
        )
    return betas


def beta_volatility(
    beta_panel: Dict[int, pd.DataFrame],
    windows: list[int] | None = None,
) -> Dict[int, pd.DataFrame]:
    """Compute rolling standard deviation of beta (beta volatility).

    Beta volatility captures how unstable a stock's systematic risk
    exposure has been — the primary signal in our model.

    Args:
        beta_panel: Dict mapping window size → beta DataFrame.
        windows: Windows to compute betavol for. Defaults to keys of beta_panel.

    Returns:
        Dict mapping window size → DataFrame of beta volatilities.
    """
    if windows is None:
        windows = list(beta_panel.keys())

    betavols: Dict[int, pd.DataFrame] = {}
    for w in windows:
        if w not in beta_panel:
            raise KeyError(f"Beta window {w} not found in beta_panel")
        betavols[w] = beta_panel[w].rolling(w).std()
        logger.info("Computed betavol — window=%dd, non-NaN rows: %d", w, betavols[w].notna().any(axis=1).sum())
    return betavols


def forward_target(
    betavol_series: pd.Series | pd.DataFrame,
    horizon: int = TARGET_HORIZON,
) -> pd.Series | pd.DataFrame:
    """Shift betavol back by `horizon` days to create the forward prediction target.

    After this shift, row t contains the betavol that will be observed
    at time t + horizon. This is what the LSTM predicts given features
    at time t.

    Note: The last `horizon` rows will be NaN and must be dropped before
    training.

    Args:
        betavol_series: Rolling betavol series or panel.
        horizon: Number of trading days ahead to predict.

    Returns:
        Forward-shifted betavol target, same type as input.
    """
    return betavol_series.shift(-horizon)


def portfolio_betavol(
    betavol_panel: pd.DataFrame,
    weights: pd.Series | None = None,
) -> pd.Series:
    """Compute portfolio-level beta volatility as a weighted average.

    Args:
        betavol_panel: DataFrame of per-stock betavols.
        weights: Optional per-stock weights. If None, uses equal weights.

    Returns:
        Series of portfolio-level betavol.
    """
    if weights is None:
        return betavol_panel.mean(axis=1)
    w = weights.reindex(betavol_panel.columns).fillna(0)
    w = w / w.sum()
    return betavol_panel.mul(w).sum(axis=1)
