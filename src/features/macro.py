"""Macro, VIX, flow, and sector feature construction.

Builds the market-level feature matrix that is shared across all stocks.
Per-stock beta features are appended in the notebook / pipeline.

All features are:
- Stationary (returns / changes / z-scores, not price levels)
- Forward-looking leak-free (only use data available at time t)
- Aligned to the NIFTY50 trading calendar

Typical usage:
    from src.features.macro import build_market_features
    features = build_market_features(loader, market_ret)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_market_features(loader: object, market_ret: pd.Series) -> pd.DataFrame:
    """Build the full market-level feature matrix.

    Args:
        loader: DataLoader instance (provides all raw series).
        market_ret: Daily log returns of NIFTY50.

    Returns:
        DataFrame with DatetimeIndex and one column per feature.
        Rows: NIFTY50 trading calendar.
        Initial rows with NaN (due to rolling windows) are NOT dropped here —
        the caller should dropna() after merging with per-stock features.
    """
    idx = loader.trading_days
    f = pd.DataFrame(index=idx)

    vix = loader.vix.reindex(idx).ffill()
    usdinr = loader.usdinr.reindex(idx).ffill()
    crude = loader.crude.reindex(idx).ffill()
    repo = loader.repo_rate.reindex(idx).ffill()
    rfr = loader.risk_free_rate.reindex(idx).ffill()
    fii = loader.fii_flows.reindex(idx).ffill()
    dii = loader.dii_flows.reindex(idx).ffill()
    bank = loader.nifty_bank.reindex(idx).ffill()
    nit = loader.nifty_it.reindex(idx).ffill()
    fmcg = loader.nifty_fmcg.reindex(idx).ffill()
    mret = market_ret.reindex(idx)

    # ------------------------------------------------------------------
    # VIX features
    # ------------------------------------------------------------------
    f["vix_level"] = vix
    f["vix_5d_chg"] = vix.pct_change(5)
    f["vix_20d_chg"] = vix.pct_change(20)
    f["vix_zscore"] = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
    f["vix_above_25"] = (vix > 25).astype(int)

    # ------------------------------------------------------------------
    # Macro features
    # ------------------------------------------------------------------
    log_usdinr = np.log(usdinr)
    f["usdinr_5d_chg"] = log_usdinr.diff(5)
    f["usdinr_20d_vol"] = log_usdinr.diff(1).rolling(20).std()

    log_crude = np.log(crude.clip(lower=1e-6))  # guard zero
    f["crude_5d_chg"] = log_crude.diff(5)
    f["crude_20d_vol"] = log_crude.diff(1).rolling(20).std()

    f["repo_rate"] = repo
    f["rfr"] = rfr
    f["rate_spread"] = repo - rfr

    # ------------------------------------------------------------------
    # Market (NIFTY50) features
    # ------------------------------------------------------------------
    f["market_ret_5d"] = mret.rolling(5).sum()
    f["market_ret_20d"] = mret.rolling(20).sum()
    f["market_vol_20d"] = mret.rolling(20).std()
    f["market_vol_60d"] = mret.rolling(60).std()

    # ------------------------------------------------------------------
    # Sector momentum features
    # ------------------------------------------------------------------
    log_bank = np.log(bank)
    log_it = np.log(nit)
    log_fmcg = np.log(fmcg)

    f["bank_mom_10d"] = log_bank.diff(10)
    f["it_mom_10d"] = log_it.diff(10)
    f["fmcg_mom_10d"] = log_fmcg.diff(10)
    f["bank_vs_nifty"] = f["bank_mom_10d"] - f["market_ret_20d"]

    # ------------------------------------------------------------------
    # FII / DII flow features
    # ------------------------------------------------------------------
    f["fii_net"] = fii
    f["dii_net"] = dii
    f["combined_flow"] = fii + dii
    f["fii_dii_ratio"] = fii / (dii.abs() + 1)  # +1 to avoid zero-division

    logger.info(
        "Market features built — %d features, %d rows (%s → %s)",
        f.shape[1],
        f.shape[0],
        f.index[0].date(),
        f.index[-1].date(),
    )
    return f


def append_stock_features(
    market_features: pd.DataFrame,
    stock_returns: pd.DataFrame,
    beta_panel: dict,
    betavol_panel: dict,
    target: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    """Append per-stock beta/return features to the market feature matrix.

    Args:
        market_features: Output of build_market_features().
        stock_returns: Log return panel for all stocks.
        beta_panel: Dict[window → DataFrame] of rolling betas.
        betavol_panel: Dict[window → DataFrame] of rolling betavols.
        target: Forward betavol target DataFrame (all stocks).
        ticker: Stock ticker to build features for.

    Returns:
        DataFrame with market + stock features and target column.
        Rows with ANY NaN are dropped.
    """
    df = market_features.copy()

    ret = stock_returns[ticker]

    # Per-stock beta features
    df["beta_30"] = beta_panel[30][ticker]
    df["beta_60"] = beta_panel[60][ticker]
    df["beta_120"] = beta_panel[120][ticker]
    df["betavol_30"] = betavol_panel[30][ticker]
    df["betavol_60"] = betavol_panel[60][ticker]
    df["betavol_120"] = betavol_panel[120][ticker]
    df["beta_60_30_spread"] = beta_panel[60][ticker] - beta_panel[30][ticker]
    df["betavol_trend"] = betavol_panel[60][ticker].diff(10)

    # Per-stock return features
    df["stock_ret_5d"] = ret.rolling(5).sum()
    df["stock_vol_20d"] = ret.rolling(20).std()
    df["stock_vs_mkt"] = df["stock_ret_5d"] - market_features["market_ret_5d"]

    # Target
    df["target"] = target[ticker]

    # Drop any row with NaN in features or target
    df = df.dropna()
    return df
