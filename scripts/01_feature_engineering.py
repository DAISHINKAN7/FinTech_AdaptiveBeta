"""
Script 01 — Feature Engineering
================================
AdaptiveBeta: AI-Powered Portfolio Optimisation
Author: Kunal | M.Tech AI & ML, SIT Pune
 
What this script does:
  1. Loads all raw CSV files from data/
  2. Aligns every dataset to the NIFTY50 trading calendar
  3. Computes log returns for all 49 stocks + NIFTY50
  4. Computes rolling OLS beta (30d, 60d, 120d) and beta volatility
  5. Builds the shared market feature matrix (VIX, macro, flows, sectors)
  6. Builds per-stock feature matrices and stacks into one CSV
  7. Adds momentum + quality features (RSI, MA distance, price momentum)
  8. Saves everything to features/
 
Run:
    python scripts/01_feature_engineering.py
 
Outputs (all in features/):
    stacked_features.csv        — full cross-sectional feature matrix
    beta30d.csv, beta60d.csv, beta120d.csv
    betavol_30d.csv, betavol_60d.csv, betavol_120d.csv
    target_betavol_20d_ahead.csv
    market_features.csv
"""
 
import sys
import time
import logging
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
 
from src.config import (
    DATA_DIR, FEATURES_DIR, RESULTS_DIR,
    TICKERS, BETA_WINDOWS, TARGET_HORIZON,
    LATE_IPO_TICKERS,
)
from src.utils.logging import setup_logging
 
logger = setup_logging(level="INFO", log_file=str(RESULTS_DIR / "01_feature_engineering.log"))
 
 
# ===========================================================================
# HELPERS
# ===========================================================================
 
def read_csv(rel_path: str, date_col: str = "date") -> pd.DataFrame:
    full = DATA_DIR / rel_path
    if not full.exists():
        raise FileNotFoundError(
            f"\n\n  Data file not found: {full}\n"
            f"  Put your CSV at: {full}\n"
        )
    df = pd.read_csv(full, parse_dates=[date_col]).set_index(date_col).sort_index()
    logger.info("Loaded %-45s  shape=%s", str(rel_path), df.shape)
    return df
 
 
def close_col(df: pd.DataFrame) -> pd.Series:
    if "Close" in df.columns:
        return df["Close"]
    candidates = [c for c in df.columns if c.startswith("Close")]
    if candidates:
        return df[candidates[0]]
    raise KeyError(f"No Close column found. Available: {list(df.columns)}")
 
 
def align(df_or_series, trading_days: pd.DatetimeIndex):
    return df_or_series.reindex(trading_days).ffill()
 
 
def rolling_beta(s_ret: pd.Series, m_ret: pd.Series, window: int) -> pd.Series:
    cov  = s_ret.rolling(window).cov(m_ret)
    var  = m_ret.rolling(window).var()
    beta = cov / var
    beta.name = s_ret.name
    return beta
 
 
# ===========================================================================
# MOMENTUM & QUALITY FEATURES  (self-contained, no external dependency)
# ===========================================================================
 
def add_momentum_features(
    df: pd.DataFrame,
    prices: pd.DataFrame,
    stock_ret: pd.DataFrame,
    ticker: str,
    windows: list = (21, 63, 126, 252),
    skip: int = 21,
) -> pd.DataFrame:
    """Add price momentum, RSI-14, MA-200 distance and realised vol to df.
 
    All series are aligned to df.index — NaNs from warmup periods are handled
    by the final dropna() in the caller loop.
 
    Args:
        df        : per-stock feature DataFrame already containing beta cols.
        prices    : (T x N) close price panel, date-indexed.
        stock_ret : (T x N) log return panel, date-indexed.
        ticker    : stock symbol string.
        windows   : momentum lookback windows in trading days.
        skip      : days to skip at end to avoid short-term reversal.
    """
    idx = df.index
 
    if ticker not in prices.columns:
        return df
 
    px = prices[ticker]
 
    # Price momentum: (price[t-skip] / price[t-skip-w]) - 1
    for w in windows:
        lagged = px.shift(skip)
        base   = px.shift(skip + w)
        mom    = (lagged / base - 1).replace([np.inf, -np.inf], np.nan)
        df[f"mom_{w}d"] = mom.reindex(idx)
 
    # RSI-14 (Wilder EMA smoothing)
    delta = px.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = (100 - 100 / (1 + rs)).reindex(idx)
 
    # Distance from 200-day MA: (price / MA200) - 1
    ma200 = px.rolling(200, min_periods=100).mean()
    df["ma200_dist"] = (px / ma200 - 1).replace([np.inf, -np.inf], np.nan).reindex(idx)
 
    # 20-day annualised realised volatility
    if ticker in stock_ret.columns:
        rv = stock_ret[ticker].rolling(20, min_periods=10).std() * np.sqrt(252)
        df["rvol_20d"] = rv.reindex(idx)
 
    return df
 
 
# ===========================================================================
# MAIN
# ===========================================================================
 
def main() -> None:
    t0 = time.time()
    logger.info("=" * 65)
    logger.info("STEP 1 — Feature Engineering")
    logger.info("Data dir  : %s", DATA_DIR)
    logger.info("Output dir: %s", FEATURES_DIR)
    logger.info("=" * 65)
 
    # -----------------------------------------------------------------------
    # 1. Master trading calendar
    # -----------------------------------------------------------------------
    logger.info("\n[1/7] Loading master trading calendar (NIFTY50)...")
    nifty_raw    = read_csv("market/nifty50.csv")
    TRADING_DAYS = nifty_raw.index
    logger.info("Trading calendar: %s -> %s  (%d days)",
                TRADING_DAYS[0].date(), TRADING_DAYS[-1].date(), len(TRADING_DAYS))
 
    # -----------------------------------------------------------------------
    # 2. Load & align all datasets
    # -----------------------------------------------------------------------
    logger.info("\n[2/7] Loading & aligning all data sources...")
 
    prices_raw = read_csv("stocks/all_stocks_prices.csv")
    available  = [t for t in TICKERS if t in prices_raw.columns]
    missing    = set(TICKERS) - set(available)
    if missing:
        logger.warning("Tickers not found in prices file (skipped): %s", missing)
    prices = align(prices_raw[available], TRADING_DAYS)
 
    nifty_close = align(close_col(nifty_raw), TRADING_DAYS).rename("NIFTY50")
    vix_raw     = read_csv("market/india_vix.csv")
    vix         = align(close_col(vix_raw), TRADING_DAYS).rename("india_vix")
 
    usdinr = align(close_col(read_csv("macro/usdinr.csv")), TRADING_DAYS).rename("usdinr")
    crude  = align(close_col(read_csv("macro/crude_oil.csv")), TRADING_DAYS).clip(lower=0).rename("crude")
 
    rfr_df  = read_csv("macro/risk_free_rate_91d_daily.csv")
    rfr_col = "risk_free_rate" if "risk_free_rate" in rfr_df.columns else rfr_df.columns[0]
    rfr     = align(rfr_df[[rfr_col]], TRADING_DAYS)[rfr_col].rename("risk_free_rate")
 
    repo_df  = read_csv("macro/repo_rate_daily.csv")
    repo_col = "repo_rate" if "repo_rate" in repo_df.columns else repo_df.columns[0]
    repo     = align(repo_df[[repo_col]], TRADING_DAYS)[repo_col].rename("repo_rate")
 
    fii_df  = read_csv("flows/fii_flows.csv")
    fii_col = "fii_net_investment" if "fii_net_investment" in fii_df.columns else fii_df.columns[0]
    fii     = align(fii_df[[fii_col]], TRADING_DAYS)[fii_col].rename("fii_net_investment")
 
    dii_df  = read_csv("flows/dii_flows.csv")
    dii_col = "dii_net_investment" if "dii_net_investment" in dii_df.columns else dii_df.columns[0]
    dii     = align(dii_df[[dii_col]], TRADING_DAYS)[dii_col].rename("dii_net_investment")
 
    bank = align(close_col(read_csv("sector/nifty_bank.csv")), TRADING_DAYS).rename("nifty_bank")
    nit  = align(close_col(read_csv("sector/nifty_it.csv")),   TRADING_DAYS).rename("nifty_it")
    fmcg = align(close_col(read_csv("sector/nifty_fmcg.csv")), TRADING_DAYS).rename("nifty_fmcg")
 
    logger.info("All data loaded and aligned.")
 
    # -----------------------------------------------------------------------
    # 3. Log returns
    # -----------------------------------------------------------------------
    logger.info("\n[3/7] Computing log returns...")
    stock_ret  = np.log(prices / prices.shift(1))
    market_ret = np.log(nifty_close / nifty_close.shift(1)).rename("market_ret")
 
    common_idx = stock_ret.dropna(how="all").index.intersection(market_ret.dropna().index)
    stock_ret  = stock_ret.loc[common_idx]
    market_ret = market_ret.loc[common_idx]
 
    for t in LATE_IPO_TICKERS:
        if t in stock_ret.columns:
            n_pre = stock_ret[t].isna().sum()
            stock_ret[t] = stock_ret[t].fillna(0)
            logger.info("  %s: filled %d pre-IPO NaN returns with 0", t, n_pre)
 
    logger.info("Returns — stocks: %s, market: %d rows (%s -> %s)",
                stock_ret.shape, len(market_ret),
                common_idx[0].date(), common_idx[-1].date())
 
    # -----------------------------------------------------------------------
    # 4. Rolling beta & beta volatility
    # -----------------------------------------------------------------------
    logger.info("\n[4/7] Computing rolling betas (%d windows x %d stocks)...",
                len(BETA_WINDOWS), len(available))
 
    betas:    dict = {}
    betavols: dict = {}
 
    for w in BETA_WINDOWS:
        logger.info("  Beta window = %dd", w)
        betas[w]    = stock_ret.apply(lambda s: rolling_beta(s, market_ret, w))
        betavols[w] = betas[w].rolling(w).std()
 
    target = betavols[60].shift(-TARGET_HORIZON)
    logger.info("Target (20d fwd betavol_60d) — non-NaN rows: %d",
                target.notna().any(axis=1).sum())
 
    # -----------------------------------------------------------------------
    # 5. Market feature matrix
    # -----------------------------------------------------------------------
    logger.info("\n[5/7] Building market feature matrix...")
    f = pd.DataFrame(index=common_idx)
 
    vix_s = vix.reindex(common_idx).ffill()
    f["vix_level"]    = vix_s
    f["vix_5d_chg"]   = vix_s.pct_change(5)
    f["vix_20d_chg"]  = vix_s.pct_change(20)
    f["vix_zscore"]   = (vix_s - vix_s.rolling(252).mean()) / vix_s.rolling(252).std()
    f["vix_above_25"] = (vix_s > 25).astype(int)
 
    log_usd = np.log(usdinr.reindex(common_idx).ffill())
    f["usdinr_5d_chg"]  = log_usd.diff(5)
    f["usdinr_20d_vol"] = log_usd.diff(1).rolling(20).std()
 
    log_crd = np.log(crude.reindex(common_idx).ffill().clip(lower=1e-6))
    f["crude_5d_chg"]  = log_crd.diff(5)
    f["crude_20d_vol"] = log_crd.diff(1).rolling(20).std()
 
    f["repo_rate"]   = repo.reindex(common_idx).ffill()
    f["rfr"]         = rfr.reindex(common_idx).ffill()
    f["rate_spread"] = f["repo_rate"] - f["rfr"]
 
    f["market_ret_5d"]  = market_ret.rolling(5).sum()
    f["market_ret_20d"] = market_ret.rolling(20).sum()
    f["market_vol_20d"] = market_ret.rolling(20).std()
    f["market_vol_60d"] = market_ret.rolling(60).std()
 
    f["bank_mom_10d"]  = np.log(bank.reindex(common_idx).ffill()).diff(10)
    f["it_mom_10d"]    = np.log(nit.reindex(common_idx).ffill()).diff(10)
    f["fmcg_mom_10d"]  = np.log(fmcg.reindex(common_idx).ffill()).diff(10)
    f["bank_vs_nifty"] = f["bank_mom_10d"] - f["market_ret_20d"]
 
    fii_s = fii.reindex(common_idx).ffill()
    dii_s = dii.reindex(common_idx).ffill()
    f["fii_net"]       = fii_s
    f["dii_net"]       = dii_s
    f["combined_flow"] = fii_s + dii_s
    f["fii_dii_ratio"] = fii_s / (dii_s.abs() + 1)
 
    logger.info("Market features: %d columns", f.shape[1])
 
    # -----------------------------------------------------------------------
    # 6. Per-stock feature matrices → stack
    # -----------------------------------------------------------------------
    logger.info("\n[6/7] Building per-stock feature matrices...")
    all_stocks: list = []
    skipped:    list = []
 
    for i, ticker in enumerate(available, 1):
        df  = f.copy()
        ret = stock_ret[ticker]
 
        # Beta features
        df["beta_30"]           = betas[30][ticker]
        df["beta_60"]           = betas[60][ticker]
        df["beta_120"]          = betas[120][ticker]
        df["betavol_30"]        = betavols[30][ticker]
        df["betavol_60"]        = betavols[60][ticker]
        df["betavol_120"]       = betavols[120][ticker]
        df["beta_60_30_spread"] = betas[60][ticker] - betas[30][ticker]
        df["betavol_trend"]     = betavols[60][ticker].diff(10)
 
        # Stock return features
        df["stock_ret_5d"]  = ret.rolling(5).sum()
        df["stock_vol_20d"] = ret.rolling(20).std()
        df["stock_vs_mkt"]  = df["stock_ret_5d"] - f["market_ret_5d"]
 
        # Momentum & quality features
        df = add_momentum_features(
            df, prices, stock_ret, ticker,
            windows=[21, 63, 126, 252], skip=21,
        )
 
        # Target
        df["target"] = target[ticker]
 
        before = len(df)
        df     = df.dropna()
        after  = len(df)
 
        if after < 200:
            skipped.append(ticker)
            logger.warning("  SKIP %-20s — only %d valid rows after dropna", ticker, after)
            continue
 
        df["ticker"] = ticker
        all_stocks.append(df)
 
        if i % 10 == 0 or i == len(available):
            logger.info("  [%2d/%2d] %-20s  valid rows: %d  (dropped %d NaN rows)",
                        i, len(available), ticker, after, before - after)
 
    logger.info("Feature matrices built for %d / %d stocks", len(all_stocks), len(available))
    if skipped:
        logger.warning("Skipped (< 200 rows): %s", skipped)
 
    stacked = pd.concat(all_stocks).sort_index()
    logger.info("Stacked matrix: %s", stacked.shape)
 
    # Report new momentum columns
    new_cols = [c for c in stacked.columns
                if c.startswith("mom_") or c in ("rsi_14", "ma200_dist", "rvol_20d")]
    if new_cols:
        logger.info("Momentum/quality features added: %s", new_cols)
 
    # -----------------------------------------------------------------------
    # 7. Save everything
    # -----------------------------------------------------------------------
    logger.info("\n[7/7] Saving features to %s ...", FEATURES_DIR)
 
    stacked.to_csv(FEATURES_DIR / "stacked_features.csv")
    logger.info("  stacked_features.csv  (%d rows x %d cols)", *stacked.shape)
 
    for w in BETA_WINDOWS:
        betas[w].to_csv(FEATURES_DIR / f"beta{w}d.csv")
        betavols[w].to_csv(FEATURES_DIR / f"betavol_{w}d.csv")
    logger.info("  beta{30,60,120}d.csv and betavol_{30,60,120}d.csv")
 
    target.to_csv(FEATURES_DIR / "target_betavol_20d_ahead.csv")
    logger.info("  target_betavol_20d_ahead.csv")
 
    f.to_csv(FEATURES_DIR / "market_features.csv")
    logger.info("  market_features.csv")
 
    _save_overview_plot(betas, betavols, vix_s, market_ret, common_idx)
 
    elapsed = time.time() - t0
    logger.info("\nFeature engineering complete in %.1f seconds.", elapsed)
    logger.info("    Next step: python scripts/02_train_models.py")
 
 
def _save_overview_plot(betas, betavols, vix_s, market_ret, idx) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    colors = ["#1D9E75", "#7F77DD", "#EF9F27"]
 
    ax = axes[0, 0]
    for w, c in zip([30, 60, 120], colors):
        betas[w]["RELIANCE.NS"].plot(ax=ax, color=c, lw=1, label=f"Beta {w}d")
    ax.axhline(1.0, color="white", ls="--", lw=0.7, alpha=0.5)
    ax.set_title("RELIANCE.NS — Rolling Beta")
    ax.legend(fontsize=8)
 
    ax = axes[0, 1]
    for w, c in zip([30, 60, 120], colors):
        betavols[w]["RELIANCE.NS"].plot(ax=ax, color=c, lw=1, label=f"BetaVol {w}d")
    ax.set_title("RELIANCE.NS — Beta Volatility")
    ax.legend(fontsize=8)
 
    ax = axes[1, 0]
    vix_s.plot(ax=ax, color="#D85A30", lw=1)
    ax.axhline(25, color="yellow", ls="--", lw=0.8, label="VIX=25")
    ax.set_title("India VIX")
    ax.legend(fontsize=8)
 
    ax = axes[1, 1]
    market_ret.plot(ax=ax, color="#888780", lw=0.5, alpha=0.7)
    ax.set_title("NIFTY50 Log Returns")
 
    plt.suptitle("Feature Engineering Overview", fontsize=13)
    plt.tight_layout()
    out = RESULTS_DIR / "feature_overview.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  Diagnostic chart saved: %s", out)
 
 
if __name__ == "__main__":
    main()