"""Script 03 — Portfolio Optimisation Validation.

Loads the signal configuration calibrated by script 02 and validates all three
portfolio optimisers on recent in-sample price data. Produces:

    features/optimiser_weights_maxsharpe.csv
    features/optimiser_weights_minvar.csv
    features/optimiser_weights_riskparity.csv
    results/optimiser_comparison.csv
    results/optimiser_weights_chart.png
    results/signal_threshold_demo.png

Run:
    python scripts/03_portfolio_optimisation.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src` is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FEATURES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    TICKERS,
    VIX_STRESS_LEVEL,
    THRESHOLD_QUANTILE,
    TRAIN_END,
)
from src.data.loader import DataLoader
from src.portfolio.signal import SignalGenerator
from src.portfolio.optimiser import PortfolioOptimiser

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_signal_config(models_dir: Path) -> dict:
    """Load signal threshold config saved by script 02."""
    path = models_dir / "signal_config.json"
    if not path.exists():
        logger.warning("signal_config.json not found — using default threshold 0.05")
        return {"betavol_threshold": 0.05, "vix_threshold": VIX_STRESS_LEVEL}
    with open(path) as f:
        cfg = json.load(f)
    logger.info("Loaded signal_config.json: threshold=%.4f", cfg.get("betavol_threshold", 0.05))
    return cfg


def load_betas(features_dir: Path) -> pd.DataFrame:
    """Load 60-day rolling beta panel."""
    path = features_dir / "beta60d.csv"
    if not path.exists():
        logger.warning("beta60d.csv not found — betas will be set to 1.0")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    logger.info("Loaded beta60d.csv — shape %s", df.shape)
    return df


def load_target_betavol(features_dir: Path) -> pd.Series:
    """Load target betavol series for threshold visualisation."""
    path = features_dir / "target_betavol_20d_ahead.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    # Average across stocks to get portfolio-level betavol
    return df.mean(axis=1).rename("portfolio_betavol")


def get_price_window(prices: pd.DataFrame, end_date: str, window: int = 252) -> pd.DataFrame:
    """Get a trailing window of prices up to end_date."""
    end = pd.Timestamp(end_date)
    subset = prices.loc[:end].tail(window)
    # Drop stocks with > 10% NaN
    thresh = int(len(subset) * 0.9)
    subset = subset.dropna(axis=1, thresh=thresh).ffill().bfill()
    return subset


def get_current_betas(betas_df: pd.DataFrame, as_of: str) -> pd.Series:
    """Get the most recent beta row on or before as_of date."""
    if betas_df.empty:
        return pd.Series({t: 1.0 for t in TICKERS})
    as_of_ts = pd.Timestamp(as_of)
    valid = betas_df.loc[betas_df.index <= as_of_ts]
    if valid.empty:
        return pd.Series({t: 1.0 for t in TICKERS})
    row = valid.iloc[-1].fillna(1.0)
    return row


# ---------------------------------------------------------------------------
# Optimiser validation
# ---------------------------------------------------------------------------

def run_optimiser_validation(
    prices: pd.DataFrame,
    betas: pd.Series,
    optimiser: PortfolioOptimiser,
    as_of: str,
) -> dict[str, pd.Series]:
    """Run all three optimisers and return a dict of mode → weights."""
    modes = [
        "max_sharpe_beta_constrained",
        "min_variance",
        "risk_parity",
    ]
    results = {}
    for mode in modes:
        logger.info("Running optimiser: %s", mode)
        try:
            w = optimiser.optimise(prices, betas, mode)
            if w.empty:
                logger.warning("  → returned empty weights for %s", mode)
                n = len(prices.columns)
                w = pd.Series({t: 1.0 / n for t in prices.columns})
            # Normalise
            w = w[w > 0]
            w = w / w.sum()
            results[mode] = w
            logger.info(
                "  → %d stocks, max=%.2f%%, min=%.2f%%, portfolio_beta=%.3f",
                len(w),
                w.max() * 100,
                w.min() * 100,
                float((w * betas.reindex(w.index).fillna(1.0)).sum()),
            )
        except Exception as exc:
            logger.error("Optimiser %s failed: %s", mode, exc)
            n = len(prices.columns)
            results[mode] = pd.Series({t: 1.0 / n for t in prices.columns})
    return results


def compute_optimiser_stats(
    weights_dict: dict[str, pd.Series],
    prices_window: pd.DataFrame,
    betas: pd.Series,
    rfr: float = 0.065,
) -> pd.DataFrame:
    """Compute expected annual return, volatility, Sharpe, and beta for each mode."""
    import numpy as np
    rows = {}
    daily_ret = np.log(prices_window / prices_window.shift(1)).dropna()
    cov = daily_ret.cov() * 252  # annualised

    for mode, w in weights_dict.items():
        tickers = w.index.tolist()
        w_arr = w.values

        # Expected annual return (mean historical)
        mu = daily_ret[tickers].mean() * 252
        port_ret = float(w_arr @ mu.reindex(tickers).fillna(0).values)

        # Portfolio variance
        cov_sub = cov.reindex(index=tickers, columns=tickers).fillna(0)
        port_var = float(w_arr @ cov_sub.values @ w_arr)
        port_vol = np.sqrt(max(port_var, 0))

        sharpe = (port_ret - rfr) / port_vol if port_vol > 0 else 0.0
        port_beta = float((w * betas.reindex(w.index).fillna(1.0)).sum())

        rows[mode] = {
            "Expected Annual Return (%)": round(port_ret * 100, 2),
            "Annual Volatility (%)": round(port_vol * 100, 2),
            "Sharpe Ratio": round(sharpe, 3),
            "Portfolio Beta": round(port_beta, 3),
            "N Stocks": len(w),
            "Max Weight (%)": round(w.max() * 100, 2),
        }
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------------------
# Signal threshold demo
# ---------------------------------------------------------------------------

def plot_signal_threshold_demo(
    betavol_series: pd.Series,
    threshold: float,
    vix_series: pd.Series,
    save_path: Path,
) -> None:
    """Plot betavol time series with threshold overlay and VIX stress zones."""
    if betavol_series.empty:
        logger.info("No betavol series — skipping threshold demo plot")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("AdaptiveBeta: Signal Threshold Trigger", fontsize=14, fontweight="bold")

    # Top: portfolio betavol vs threshold
    ax = axes[0]
    ax.plot(betavol_series.index, betavol_series.values, color="#4A90D9", linewidth=1, label="Portfolio BetaVol")
    ax.axhline(threshold, color="#E74C3C", linewidth=2, linestyle="--",
               label=f"Threshold ({THRESHOLD_QUANTILE*100:.0f}th pct = {threshold:.4f})")

    # Shade REBALANCE zones
    rebalance_mask = betavol_series > threshold
    prev = False
    seg_start = None
    for date, val in rebalance_mask.items():
        if val and not prev:
            seg_start = date
        elif not val and prev and seg_start is not None:
            ax.axvspan(seg_start, date, alpha=0.15, color="#E74C3C")
        prev = val

    ax.set_ylabel("Beta Volatility")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_title("REBALANCE signal: predicted betavol > threshold (red zones)")

    # Bottom: VIX with stress level
    ax2 = axes[1]
    if not vix_series.empty:
        aligned_vix = vix_series.reindex(betavol_series.index).ffill()
        ax2.plot(aligned_vix.index, aligned_vix.values, color="#F39C12", linewidth=1, label="India VIX")
        ax2.axhline(VIX_STRESS_LEVEL, color="#8E44AD", linewidth=2, linestyle="--",
                    label=f"VIX Stress Level ({VIX_STRESS_LEVEL})")
        stress_mask = aligned_vix > VIX_STRESS_LEVEL
        prev = False
        seg_start = None
        for date, val in stress_mask.items():
            if val and not prev:
                seg_start = date
            elif not val and prev and seg_start is not None:
                ax2.axvspan(seg_start, date, alpha=0.2, color="#8E44AD")
            prev = val
        ax2.set_ylabel("VIX Level")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("MIN_VARIANCE override: VIX > 25 (purple zones)")

    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved threshold demo → %s", save_path)


# ---------------------------------------------------------------------------
# Weights bar chart
# ---------------------------------------------------------------------------

def plot_weights_chart(
    weights_dict: dict[str, pd.Series],
    save_path: Path,
) -> None:
    """Plot top-20 holdings for each optimiser mode side by side."""
    modes = list(weights_dict.keys())
    fig, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 7))
    if len(modes) == 1:
        axes = [axes]

    colors = ["#4A90D9", "#27AE60", "#E67E22"]

    for ax, mode, color in zip(axes, modes, colors):
        w = weights_dict[mode].sort_values(ascending=False).head(20)
        labels = [t.replace(".NS", "") for t in w.index]
        ax.barh(range(len(w)), w.values * 100, color=color, alpha=0.85)
        ax.set_yticks(range(len(w)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Weight (%)")
        ax.set_title(mode.replace("_", "\n"), fontsize=10, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        ax.invert_yaxis()

    plt.suptitle("Portfolio Weights by Optimiser Mode (Top 20 Holdings)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved weights chart → %s", save_path)


# ---------------------------------------------------------------------------
# Signal frequency analysis
# ---------------------------------------------------------------------------

def analyse_signal_frequency(
    betavol_series: pd.Series,
    vix_series: pd.Series,
    sig_gen: SignalGenerator,
) -> pd.DataFrame:
    """Compute signal statistics over the full history."""
    if betavol_series.empty:
        logger.info("No betavol series — skipping signal frequency analysis")
        return pd.DataFrame()

    dates = betavol_series.index
    signals = []
    for date in dates:
        bv = float(betavol_series.loc[date]) if date in betavol_series.index else sig_gen.threshold
        vix = float(vix_series.loc[date]) if date in vix_series.index else 20.0
        sig = sig_gen.get_signal(bv, vix)
        signals.append(sig)

    s = pd.Series(signals, index=dates, name="signal")
    counts = s.value_counts()
    freq = (counts / len(s) * 100).round(1)
    result = pd.DataFrame({
        "Count": counts,
        "Frequency (%)": freq,
    })
    logger.info("Signal distribution:\n%s", result.to_string())
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("=" * 60)
    logger.info("Script 03 — Portfolio Optimisation Validation")
    logger.info("=" * 60)

    # 1. Load config and data
    cfg = load_signal_config(MODELS_DIR)
    threshold = cfg.get("betavol_threshold", 0.05)
    vix_threshold = cfg.get("vix_threshold", VIX_STRESS_LEVEL)

    loader = DataLoader()
    prices = loader.prices
    vix = loader.vix
    betas_df = load_betas(FEATURES_DIR)
    betavol_series = load_target_betavol(FEATURES_DIR)

    logger.info("Prices shape: %s | date range: %s → %s",
                prices.shape, prices.index.min().date(), prices.index.max().date())

    # 2. Build signal generator
    sig_gen = SignalGenerator(threshold=threshold, vix_threshold=vix_threshold)
    logger.info("SignalGenerator: threshold=%.4f, vix_threshold=%.1f", threshold, vix_threshold)

    # 3. Get price window and betas for validation (use TRAIN_END as 'as of' date)
    as_of = TRAIN_END
    price_window = get_price_window(prices, end_date=as_of, window=252)
    current_betas = get_current_betas(betas_df, as_of=as_of)

    logger.info("Validation date: %s | Price window: %d days x %d stocks",
                as_of, len(price_window), len(price_window.columns))

    # 4. Run all optimisers
    optimiser = PortfolioOptimiser()
    weights_dict = run_optimiser_validation(price_window, current_betas, optimiser, as_of)

    # 5. Compute optimiser stats
    stats_df = compute_optimiser_stats(weights_dict, price_window, current_betas)
    logger.info("\nOptimiser Comparison:\n%s", stats_df.to_string())

    # 6. Signal frequency analysis
    signal_freq = analyse_signal_frequency(
        betavol_series.loc[:as_of] if not betavol_series.empty else betavol_series,
        vix,
        sig_gen,
    )

    # 7. Save outputs
    for mode, w in weights_dict.items():
        fname = f"optimiser_weights_{mode}.csv"
        w.reset_index().rename(columns={"index": "ticker", 0: "weight"}).to_csv(
            FEATURES_DIR / fname, index=False
        )
        logger.info("Saved %s", fname)

    stats_df.to_csv(RESULTS_DIR / "optimiser_comparison.csv")
    logger.info("Saved optimiser_comparison.csv")

    if not signal_freq.empty:
        signal_freq.to_csv(RESULTS_DIR / "signal_frequency.csv")
        logger.info("Saved signal_frequency.csv")

    # 8. Plots
    plot_signal_threshold_demo(
        betavol_series.loc[:as_of] if not betavol_series.empty else betavol_series,
        threshold,
        vix,
        RESULTS_DIR / "signal_threshold_demo.png",
    )
    plot_weights_chart(weights_dict, RESULTS_DIR / "optimiser_weights_chart.png")

    # 9. Summary
    logger.info("=" * 60)
    logger.info("Script 03 complete. Files written:")
    logger.info("  features/optimiser_weights_*.csv")
    logger.info("  results/optimiser_comparison.csv")
    logger.info("  results/signal_frequency.csv")
    logger.info("  results/signal_threshold_demo.png")
    logger.info("  results/optimiser_weights_chart.png")
    logger.info("=" * 60)

    # Print stats table
    print("\n" + "=" * 60)
    print("OPTIMISER COMPARISON (as of", as_of, ")")
    print("=" * 60)
    print(stats_df.to_string())
    print()

    if not signal_freq.empty:
        print("SIGNAL FREQUENCY (full history up to train end)")
        print(signal_freq.to_string())
        print()

    print("Next step: python scripts/04_backtest.py")


if __name__ == "__main__":
    main()
