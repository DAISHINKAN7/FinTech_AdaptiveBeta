"""Kalman Filter for time-varying beta estimation.

The classical Kalman filter models beta as a random walk (state-space):
    State:   [alpha_t, beta_t]  (intercept + slope)
    Obs:     r_stock_t = alpha_t + beta_t * r_market_t + eps_t

This is the state-space alternative to rolling OLS beta — it's smoother
and responds faster to regime changes, but is less interpretable.

Typical usage:
    from src.models.kalman import kalman_beta_panel
    kalman_betas = kalman_beta_panel(stock_returns, market_returns)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from pykalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    logger.warning("pykalman not installed — Kalman model unavailable")


def kalman_beta(
    stock_ret: pd.Series,
    market_ret: pd.Series,
    observation_noise: float = 0.1,
    transition_noise: float = 0.01,
) -> pd.Series:
    """Estimate time-varying beta for a single stock via Kalman filter.

    State space formulation:
        x_t = [alpha_t, beta_t]
        x_t = F * x_{t-1} + w_t,  w_t ~ N(0, Q)
        y_t = H_t * x_t + v_t,    v_t ~ N(0, R)

    where H_t = [1, market_ret_t].

    Args:
        stock_ret: Log returns of the stock.
        market_ret: Log returns of the market, aligned with stock_ret.
        observation_noise: Observation covariance R.
        transition_noise: State transition noise (Q = noise * I).

    Returns:
        Series of Kalman-smoothed beta values (index = stock_ret.index).

    Raises:
        ImportError: If pykalman is not installed.
    """
    if not KALMAN_AVAILABLE:
        raise ImportError("pykalman is not installed. Run: pip install pykalman")

    common = stock_ret.dropna().index.intersection(market_ret.dropna().index)
    s = stock_ret.loc[common].values
    m = market_ret.loc[common].values
    n = len(common)

    if n < 60:
        logger.warning("Only %d observations for Kalman filter on %s — skipping", n, stock_ret.name)
        return pd.Series(np.full(n, np.nan), index=common, name=stock_ret.name)

    # Build observation matrices H_t = [1, market_ret_t] — shape (n, 1, 2)
    obs_matrices = np.column_stack([np.ones(n), m]).reshape(n, 1, 2)

    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=obs_matrices,
        initial_state_mean=np.array([0.0, 1.0]),
        initial_state_covariance=np.eye(2),
        observation_covariance=np.array([[observation_noise]]),
        transition_covariance=transition_noise * np.eye(2),
        em_vars=["transition_covariance", "observation_covariance"],
    )

    try:
        kf = kf.em(s.reshape(-1, 1), n_iter=10)
        state_means, _ = kf.filter(s.reshape(-1, 1))
        beta_series = pd.Series(state_means[:, 1], index=common, name=stock_ret.name)
    except Exception as exc:
        logger.warning("Kalman filter failed for %s: %s", stock_ret.name, exc)
        beta_series = pd.Series(np.full(n, np.nan), index=common, name=stock_ret.name)

    return beta_series


def kalman_beta_panel(
    stock_returns: pd.DataFrame,
    market_ret: pd.Series,
    observation_noise: float = 0.1,
    transition_noise: float = 0.01,
) -> pd.DataFrame:
    """Compute Kalman filter betas for all stocks.

    Args:
        stock_returns: Log return panel (dates × tickers).
        market_ret: Log returns of NIFTY50.
        observation_noise: Observation noise for KF.
        transition_noise: Transition noise for KF.

    Returns:
        DataFrame of Kalman betas (same shape as stock_returns).
    """
    result = {}
    tickers = stock_returns.columns.tolist()
    for i, ticker in enumerate(tickers, 1):
        logger.info("Kalman filter — %d/%d %s", i, len(tickers), ticker)
        result[ticker] = kalman_beta(
            stock_returns[ticker],
            market_ret,
            observation_noise,
            transition_noise,
        )
    panel = pd.DataFrame(result)
    logger.info("Kalman beta panel built — shape %s", panel.shape)
    return panel
