"""
Script 05 — Export Real Demo Data for Next.js Interactive Demo
==============================================================
Reads the real backtest output (from script 04) and packages it into
website/public/demo-data.json which the browser demo loads at startup.

When this JSON exists, the interactive demo shows REAL equity curves,
REAL India VIX levels, and REAL betavol predictions instead of synthetic data.
Parameter sliders then replay the signal logic on real data.

Requires:
    results/strategy_returns.csv     (from script 04)
    data/market/india_vix.csv        (your raw VIX file)
    features/betavol_60d.csv         (from script 01, optional — falls back to proxy)
    features/hmm_regimes.csv         (from script 02, optional)

Run:
    python scripts/05_export_demo_data.py
    python scripts/05_export_demo_data.py --results-csv path/to/strategy_returns.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR,
    FEATURES_DIR,
    RESULTS_DIR,
    RAW_FILES,
    THRESHOLD_QUANTILE,
    VIX_STRESS_LEVEL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

WEBSITE_PUBLIC = PROJECT_ROOT / "website" / "public"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path: Path, date_col: str = "date") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=[date_col]).set_index(date_col)
    logger.info("  Loaded %s  shape=%s", path.name, df.shape)
    return df


def safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    return df[col].dropna()


def to_list(series: pd.Series, index: pd.DatetimeIndex) -> list:
    """Reindex to common index, fill NaN with forward-fill then 0."""
    aligned = series.reindex(index).ffill().fillna(0)
    return [round(float(v), 6) for v in aligned]


def compute_metrics(log_returns: pd.Series, rf_annual: float = 0.065) -> dict:
    r = log_returns.dropna()
    if len(r) < 5:
        return {}
    n_years = len(r) / 252
    total_ret = float(np.exp(r.sum()) - 1)
    cagr = float((1 + total_ret) ** (1 / max(n_years, 0.1)) - 1) * 100
    annual_vol = float(r.std() * np.sqrt(252)) * 100
    rf_daily = rf_annual / 252
    excess = r - rf_daily
    sharpe = float(excess.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    neg = r[r < 0]
    sortino = float(excess.mean() / neg.std() * np.sqrt(252)) if len(neg) > 1 else 0.0
    equity = np.exp(r.cumsum())
    running_max = equity.cummax()
    dd = (equity / running_max - 1) * 100
    max_dd = float(dd.min())
    calmar = abs(cagr / max_dd) if max_dd < 0 else 0.0
    return {
        "cagr": round(cagr, 3),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "maxDrawdown": round(max_dd, 3),
        "calmar": round(calmar, 4),
        "annualVol": round(annual_vol, 3),
        "totalReturn": round(total_ret * 100, 2),
        "tradingDays": len(r),
    }


def compute_betavol_thresholds(betavol: pd.Series) -> dict:
    """Compute what betavol value corresponds to each percentile 40–90."""
    clean = betavol.dropna()
    if clean.empty:
        return {}
    result = {}
    for pct in range(40, 95, 5):
        result[str(pct)] = round(float(np.percentile(clean, pct)), 6)
    return result


def compute_crisis_stats(
    returns: pd.Series,
    nifty_returns: pd.Series,
    vix: pd.Series,
    betavol: pd.Series,
    betavol_threshold: float,
    vix_threshold: float,
) -> dict:
    CRISES = {
        "COVID Crash":    ("2020-02-01", "2020-05-01"),
        "ADANI Crisis":   ("2023-01-25", "2023-03-15"),
        "IL&FS Crisis":   ("2018-08-01", "2018-10-31"),
        "2018 Rate Hike": ("2018-09-01", "2018-11-30"),
    }
    result = {}
    for name, (start, end) in CRISES.items():
        try:
            r  = returns.loc[start:end].dropna()
            nr = nifty_returns.loc[start:end].dropna()
            v  = vix.loc[start:end].dropna()
            bv = betavol.loc[start:end].dropna()
            if r.empty or nr.empty:
                continue
            p_ret = float((np.exp(r.sum()) - 1) * 100)
            n_ret = float((np.exp(nr.sum()) - 1) * 100)
            equity = np.exp(r.cumsum())
            running_max = equity.cummax()
            dd = float((equity / running_max - 1).min() * 100)
            ne = np.exp(nr.cumsum())
            nm = ne.cummax()
            ndd = float((ne / nm - 1).min() * 100)
            min_var_days = int((v > vix_threshold).sum())
            rebalances   = int((bv > betavol_threshold).sum())
            result[name] = {
                "portfolioReturn": round(p_ret, 2),
                "niftyReturn":     round(n_ret, 2),
                "portfolioMaxDD":  round(dd, 2),
                "niftyMaxDD":      round(ndd, 2),
                "protection":      round(p_ret - n_ret, 2),
                "rebalances":      rebalances,
                "minVarDays":      min_var_days,
            }
        except Exception as e:
            logger.warning("Crisis %s failed: %s", name, e)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(returns_csv: Path) -> None:
    logger.info("=" * 60)
    logger.info("Script 05 — Export Real Demo Data")
    logger.info("=" * 60)

    WEBSITE_PUBLIC.mkdir(parents=True, exist_ok=True)
    out_path = WEBSITE_PUBLIC / "demo-data.json"

    # ------------------------------------------------------------------
    # 1. Load strategy returns (required)
    # ------------------------------------------------------------------
    logger.info("Loading strategy returns...")
    returns_df = load_csv(returns_csv)
    if returns_df.empty:
        logger.error(
            "strategy_returns.csv not found at %s. Run script 04 first.", returns_csv
        )
        sys.exit(1)

    # Ensure datetime index
    returns_df.index = pd.to_datetime(returns_df.index)
    idx: pd.DatetimeIndex = returns_df.index

    logger.info("  Trading days: %d  (%s → %s)",
                len(idx), idx[0].date(), idx[-1].date())

    # ------------------------------------------------------------------
    # 2. Load real India VIX
    # ------------------------------------------------------------------
    logger.info("Loading India VIX...")
    vix_path = DATA_DIR / RAW_FILES["india_vix"]
    vix_df   = load_csv(vix_path)

    # Handle the Close_^INDIAVIX column name
    vix_col = None
    if not vix_df.empty:
        close_cols = [c for c in vix_df.columns if c.startswith("Close")]
        if close_cols:
            vix_col = close_cols[0]
            vix_series = vix_df[vix_col].reindex(idx).ffill().bfill().fillna(18.0)
        else:
            vix_series = pd.Series(18.0, index=idx)
    else:
        logger.warning("VIX file not found — using default 18.0")
        vix_series = pd.Series(18.0, index=idx)

    # ------------------------------------------------------------------
    # 3. Load betavol (60d rolling — proxy for LSTM prediction target)
    # ------------------------------------------------------------------
    logger.info("Loading betavol...")
    bv_path = FEATURES_DIR / "betavol_60d.csv"
    bv_df   = load_csv(bv_path)

    if not bv_df.empty:
        # Average across all stocks to get portfolio-level betavol
        bv_series = bv_df.mean(axis=1).reindex(idx).ffill().bfill()
    else:
        logger.warning("betavol_60d.csv not found — computing proxy from NIFTY50 returns")
        if "buy_hold_nifty" in returns_df.columns:
            nifty_r = returns_df["buy_hold_nifty"].dropna()
            bv_series = nifty_r.rolling(60).std() * np.sqrt(252)
            bv_series = bv_series.reindex(idx).ffill().bfill().fillna(0.15)
        else:
            bv_series = pd.Series(0.12, index=idx)

    bv_series = bv_series.fillna(bv_series.median())

    # ------------------------------------------------------------------
    # 4. Load HMM regimes
    # ------------------------------------------------------------------
    logger.info("Loading regime labels...")
    regime_path = FEATURES_DIR / "hmm_regimes.csv"
    regime_df   = load_csv(regime_path)

    if not regime_df.empty:
        col = "regime_label" if "regime_label" in regime_df.columns else regime_df.columns[0]
        regime_series = regime_df[col].reindex(idx).ffill().fillna("transition")
    else:
        logger.warning("hmm_regimes.csv not found — defaulting to 'transition'")
        regime_series = pd.Series("transition", index=idx)

    # ------------------------------------------------------------------
    # 5. Compute betavol percentile thresholds
    # ------------------------------------------------------------------
    logger.info("Computing betavol thresholds...")
    bv_thresholds = compute_betavol_thresholds(bv_series)
    default_threshold = bv_thresholds.get(str(int(THRESHOLD_QUANTILE * 100)), float(bv_series.quantile(THRESHOLD_QUANTILE)))
    logger.info("  Default threshold (%.0f%%): %.5f", THRESHOLD_QUANTILE * 100, default_threshold)

    # ------------------------------------------------------------------
    # 6. Build strategy return arrays
    # ------------------------------------------------------------------
    STRATEGY_MAP = {
        "adaptive_beta":   "adaptive_beta",
        "equal_weight":    "equal_weight",
        "buy_hold_nifty":  "buy_hold_nifty",
        "static_capm_mvo": "static_capm_mvo",
        "kalman_mvo":      "kalman_mvo",
        "momentum_quality":"momentum_quality",
    }

    returns_out: dict[str, list] = {}
    available_strategies = []
    for key, col in STRATEGY_MAP.items():
        if col in returns_df.columns:
            s = returns_df[col].reindex(idx).fillna(0)
            returns_out[key] = [round(float(v), 8) for v in s]
            available_strategies.append(key)
            logger.info("  Strategy %-22s  %d non-zero days", key,
                        (returns_df[col].fillna(0) != 0).sum())
        else:
            logger.warning("  Strategy '%s' not found in returns CSV — skipping", col)

    # ------------------------------------------------------------------
    # 7. Compute per-strategy metrics
    # ------------------------------------------------------------------
    logger.info("Computing metrics...")
    metrics_out = {}
    for key in available_strategies:
        r = returns_df.get(STRATEGY_MAP[key], pd.Series(dtype=float))
        if not r.empty:
            metrics_out[key] = compute_metrics(r.dropna())

    # ------------------------------------------------------------------
    # 8. Build the original signal sequence
    # ------------------------------------------------------------------
    logger.info("Computing original signal sequence...")
    signals_out = []
    for date in idx:
        v  = float(vix_series.get(date, 18.0))
        bv = float(bv_series.get(date, default_threshold))
        if v > VIX_STRESS_LEVEL:
            sig = "min_variance"
        elif bv > default_threshold:
            sig = "rebalance"
        else:
            sig = "hold"
        signals_out.append(sig)

    # ------------------------------------------------------------------
    # 9. Compute crisis stats
    # ------------------------------------------------------------------
    logger.info("Computing crisis statistics...")
    nifty_key = "buy_hold_nifty" if "buy_hold_nifty" in returns_df.columns else None
    ab_key    = "adaptive_beta"  if "adaptive_beta"  in returns_df.columns else None

    crisis_stats = {}
    if nifty_key and ab_key:
        crisis_stats = compute_crisis_stats(
            returns=returns_df[ab_key],
            nifty_returns=returns_df[nifty_key],
            vix=vix_series,
            betavol=bv_series,
            betavol_threshold=default_threshold,
            vix_threshold=float(VIX_STRESS_LEVEL),
        )

    # ------------------------------------------------------------------
    # 10. Build equity curves (indexed to 100 at start)
    # ------------------------------------------------------------------
    logger.info("Building equity curves...")
    equity_curves: dict[str, list] = {}
    for key in available_strategies:
        r = returns_df.get(STRATEGY_MAP[key], pd.Series(dtype=float))
        r = r.reindex(idx).fillna(0)
        equity = 100 * np.exp(r.cumsum())
        equity_curves[key] = [round(float(v), 4) for v in equity]

    # ------------------------------------------------------------------
    # 11. Write JSON
    # ------------------------------------------------------------------
    logger.info("Writing demo-data.json...")

    output = {
        "source": "real",
        "meta": {
            "generatedAt": pd.Timestamp.now().isoformat(),
            "tradingDays": len(idx),
            "startDate": str(idx[0].date()),
            "endDate":   str(idx[-1].date()),
            "vixThreshold": float(VIX_STRESS_LEVEL),
            "betavolQuantile": float(THRESHOLD_QUANTILE),
            "betavolThreshold": round(default_threshold, 6),
            "strategies": available_strategies,
        },
        "dates":   [str(d.date()) for d in idx],
        "vix":     to_list(vix_series, idx),
        "betavol": to_list(bv_series, idx),
        "regime":  [str(regime_series.get(d, "transition")) for d in idx],
        "signals": signals_out,
        "returns": returns_out,
        "equityCurves": equity_curves,
        "metrics": metrics_out,
        "betavolThresholds": bv_thresholds,
        "crisisStats": crisis_stats,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))  # compact (no pretty-print for size)

    size_kb = out_path.stat().st_size / 1024
    logger.info("  Wrote %s  (%.0f KB)", out_path, size_kb)

    # ------------------------------------------------------------------
    # 12. Print summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Demo data export complete!")
    logger.info("  Output: %s", out_path)
    logger.info("  Size:   %.0f KB", size_kb)
    logger.info("")
    logger.info("Performance summary:")
    for key, m in metrics_out.items():
        logger.info(
            "  %-22s  CAGR=%5.1f%%  Sharpe=%.3f  MaxDD=%.1f%%",
            key, m.get("cagr", 0), m.get("sharpe", 0), m.get("maxDrawdown", 0)
        )
    logger.info("=" * 60)
    logger.info("Now run:  cd website && npm run dev")
    logger.info("The demo at /demo will automatically use real data.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export real backtest data for the web demo")
    p.add_argument(
        "--results-csv",
        type=Path,
        default=RESULTS_DIR / "strategy_returns.csv",
        help="Path to strategy_returns.csv (default: results/strategy_returns.csv)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.results_csv)
