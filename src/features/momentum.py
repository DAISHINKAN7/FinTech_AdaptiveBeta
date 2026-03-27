# src/features/momentum.py
"""Price momentum and quality features for the AdaptiveBeta pipeline.

These are added to the per-stock feature matrix in script 01.
Also used standalone in script 04 for the momentum_quality strategy.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def price_momentum(prices: pd.DataFrame, window: int, skip: int = 21) -> pd.DataFrame:
    """Return (t-skip) / (t-skip-window) - 1 for each stock.

    Args:
        prices: (T x N) close price panel, date-indexed.
        window: lookback in trading days (e.g. 126 = 6 months).
        skip:   days to skip at the end to avoid short-term reversal.

    Returns:
        DataFrame same shape as prices with momentum scores.
    """
    lagged = prices.shift(skip)
    base   = prices.shift(skip + window)
    return (lagged / base - 1).replace([np.inf, -np.inf], np.nan)


def rsi(prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Wilder RSI for each stock in the panel."""
    delta = prices.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(columns=lambda c: c)


def ma_distance(prices: pd.DataFrame, window: int = 200) -> pd.DataFrame:
    """Price distance from rolling mean: (price / MA) - 1."""
    ma = prices.rolling(window, min_periods=window // 2).mean()
    return (prices / ma - 1).replace([np.inf, -np.inf], np.nan)


def realized_vol(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Annualised rolling realised volatility."""
    return returns.rolling(window, min_periods=window // 2).std() * np.sqrt(252)


def append_momentum_features(
    stock_features: pd.DataFrame,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    ticker: str,
    momentum_windows: list[int] = (21, 63, 126, 252),
    skip: int = 21,
) -> pd.DataFrame:
    """Add momentum/quality columns to a single stock's feature DataFrame.

    Columns added (for each window w):
        mom_{w}d   — price momentum
    And additional:
        rsi_14     — RSI
        ma200_dist — distance from 200-day MA
        rvol_20d   — 20-day realised vol
    """
    idx = stock_features.index

    if ticker in prices.columns:
        px = prices[ticker]
        for w in momentum_windows:
            col = f"mom_{w}d"
            mom = (px.shift(skip) / px.shift(skip + w) - 1).replace(
                [np.inf, -np.inf], np.nan
            )
            stock_features[col] = mom.reindex(idx)

        # RSI
        delta = px.diff()
        gain  = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss  = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs    = gain / loss.replace(0, np.nan)
        stock_features["rsi_14"] = (100 - 100 / (1 + rs)).reindex(idx)

        # Distance from 200-day MA
        ma200 = px.rolling(200, min_periods=100).mean()
        stock_features["ma200_dist"] = (px / ma200 - 1).replace(
            [np.inf, -np.inf], np.nan
        ).reindex(idx)

    if ticker in returns.columns:
        rv = returns[ticker].rolling(20, min_periods=10).std() * np.sqrt(252)
        stock_features["rvol_20d"] = rv.reindex(idx)

    return stock_features


def momentum_portfolio_weights(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    top_n: int = 15,
    lookback: int = 126,
    skip: int = 21,
    max_w: float = 0.15,
) -> dict[str, float]:
    """Pick top_n stocks by 6-month momentum (skip last month).

    Returns equal-weight dict clipped at max_w.
    """
    end_idx = prices.index.get_indexer([date], method="ffill")[0]
    if end_idx < 0:
        return {}
    lag_idx  = max(0, end_idx - skip)
    base_idx = max(0, end_idx - skip - lookback)
    if lag_idx == base_idx:
        return {}

    lag_px  = prices.iloc[lag_idx]
    base_px = prices.iloc[base_idx]
    mom     = (lag_px / base_px - 1).dropna().sort_values(ascending=False)

    top     = mom.head(top_n).index.tolist()
    n       = len(top)
    if n == 0:
        return {}
    raw_w   = 1.0 / n
    w       = min(raw_w, max_w)
    weights = {t: w for t in top}
    # re-normalise
    total   = sum(weights.values())
    return {t: v / total for t, v in weights.items()}
