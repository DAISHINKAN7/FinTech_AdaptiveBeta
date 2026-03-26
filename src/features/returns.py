"""Log return computation utilities.

All returns in this project are natural-log returns: ln(P_t / P_{t-1}).
Log returns are additive across time, better-behaved statistically, and
standard in academic finance literature.

Typical usage:
    from src.features.returns import compute_log_returns, align_returns
    stock_ret, market_ret = align_returns(raw_stock_returns, raw_market_returns)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Compute daily log returns from price levels.

    Args:
        prices: Price levels with DatetimeIndex. Can be a DataFrame
            (multiple tickers) or a Series (single series).

    Returns:
        Log returns of the same shape, with the first row as NaN.
    """
    if isinstance(prices, pd.Series):
        ret = np.log(prices / prices.shift(1))
        ret.name = prices.name
        return ret
    return np.log(prices / prices.shift(1))


def align_returns(
    stock_ret: pd.DataFrame,
    market_ret: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Align stock and market returns to a common DatetimeIndex.

    The first row of both series will be NaN (from the log shift).
    We drop those rows so both series are NaN-free at the start.

    Args:
        stock_ret: Log returns of all stocks.
        market_ret: Log returns of the market index (NIFTY50).

    Returns:
        Tuple of (aligned_stock_returns, aligned_market_returns).
    """
    common_idx = stock_ret.index.intersection(market_ret.index)
    stock_aligned = stock_ret.loc[common_idx].dropna(how="all")
    market_aligned = market_ret.loc[stock_aligned.index]

    # Re-intersect after dropping rows where market is NaN
    valid_idx = market_aligned.dropna().index
    stock_aligned = stock_aligned.loc[valid_idx]
    market_aligned = market_aligned.loc[valid_idx]

    logger.info(
        "Returns aligned — %d stocks, %d dates (%s → %s)",
        stock_aligned.shape[1],
        len(valid_idx),
        valid_idx[0].date(),
        valid_idx[-1].date(),
    )
    return stock_aligned, market_aligned


def rolling_realized_vol(
    returns: pd.Series | pd.DataFrame,
    window: int,
    annualise: bool = True,
) -> pd.Series | pd.DataFrame:
    """Rolling annualised realised volatility.

    Args:
        returns: Log return series or DataFrame.
        window: Rolling window size in trading days.
        annualise: If True, multiply by sqrt(252).

    Returns:
        Rolling volatility of the same shape.
    """
    vol = returns.rolling(window).std()
    if annualise:
        vol = vol * np.sqrt(252)
    return vol
