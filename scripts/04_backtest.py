"""Script 04 — Walk-Forward Backtest.

Runs the full 6-year walk-forward backtest (2018–2024) across 5 strategies:
1. adaptive_beta       — LSTM threshold trigger + regime routing
2. static_capm_mvo     — Monthly MVO with static OLS betas
3. kalman_mvo          — Monthly MVO with Kalman-filtered betas
4. equal_weight        — Monthly equal-weight rebalance
5. buy_hold_nifty      — Buy & hold NIFTY50 benchmark

Outputs:
    results/strategy_returns.csv          ← daily return series, all strategies
    results/performance_metrics.csv       ← annual return, Sharpe, Sortino, etc.
    results/stress_analysis.csv           ← COVID, ADANI, IL&FS, 2018 hike
    results/equity_curves.png
    results/drawdown_curves.png
    results/rolling_sharpe.png
    results/stress_heatmap.png

Run:
    python scripts/04_backtest.py [--fast] [--device auto|cpu|cuda|mps]

Flags:
    --fast     Skip Kalman betas (use static OLS betas for kalman_mvo strategy)
    --device   PyTorch device for LSTM retraining (default: auto)
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BACKTEST_START_YEAR,
    BACKTEST_END_YEAR,
    FEATURES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    STRATEGIES,
    STRESS_EVENTS,
    TRAIN_YEARS,
    TEST_YEARS,
)
from src.data.loader import DataLoader
from src.portfolio.signal import SignalGenerator
from src.portfolio.optimiser import PortfolioOptimiser
from src.backtest.engine import WalkForwardBacktester
from src.backtest.metrics import (
    build_metrics_table,
    stress_analysis,
    rolling_sharpe,
    drawdown_series,
)

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load supporting artefacts
# ---------------------------------------------------------------------------

def load_signal_config(models_dir: Path) -> dict:
    import json
    path = models_dir / "signal_config.json"
    if not path.exists():
        logger.warning("signal_config.json not found — using default threshold 0.05")
        return {"betavol_threshold": 0.05, "vix_threshold": 25.0}
    with open(path) as f:
        cfg = json.load(f)
    logger.info("Loaded signal_config.json: threshold=%.4f", cfg.get("betavol_threshold"))
    return cfg


def load_optional_csv(path: Path, index_col: str = "date") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=[index_col]).set_index(index_col)
    logger.info("Loaded %s — shape %s", path.name, df.shape)
    return df


def load_hmm_regimes(features_dir: Path) -> pd.Series:
    path = features_dir / "hmm_regimes.csv"
    if not path.exists():
        logger.warning("hmm_regimes.csv not found — defaulting to 'transition' everywhere")
        return pd.Series(dtype=str)
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    col = "regime_label" if "regime_label" in df.columns else df.columns[0]
    regime = df[col].rename("regime")
    logger.info("Loaded hmm_regimes.csv — %d dates", len(regime))
    return regime


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

STRATEGY_COLORS = {
    "adaptive_beta":     "#E74C3C",
    "kalman_mvo":        "#3498DB",
    "static_capm_mvo":   "#2ECC71",
    "equal_weight":      "#F39C12",
    "buy_hold_nifty":    "#95A5A6",
}

STRATEGY_LABELS = {
    "adaptive_beta":     "AdaptiveBeta (ours)",
    "kalman_mvo":        "Kalman-Beta MVO",
    "static_capm_mvo":   "Static CAPM MVO",
    "equal_weight":      "Equal Weight",
    "buy_hold_nifty":    "Buy & Hold NIFTY50",
}


def plot_equity_curves(
    returns_dict: dict[str, pd.Series],
    save_path: Path,
) -> None:
    """Plot cumulative wealth (starting ₹100) for all strategies."""
    fig, ax = plt.subplots(figsize=(14, 7))

    for strategy in STRATEGIES:
        if strategy not in returns_dict:
            continue
        r = returns_dict[strategy].dropna()
        if r.empty:
            continue
        equity = 100 * np.exp(r.cumsum())
        lw = 2.5 if strategy == "adaptive_beta" else 1.5
        ax.plot(
            equity.index, equity.values,
            color=STRATEGY_COLORS.get(strategy, "gray"),
            linewidth=lw,
            label=STRATEGY_LABELS.get(strategy, strategy),
            zorder=3 if strategy == "adaptive_beta" else 2,
        )

    # Shade stress events
    for event, (start, end) in STRESS_EVENTS.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.08, color="red")
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ax.text(mid, ax.get_ylim()[0] * 1.02, event[:8], fontsize=7,
                ha="center", color="darkred", rotation=90, va="bottom")

    ax.set_title("Equity Curves — Walk-Forward Backtest 2018–2024\n(₹100 invested, log returns, 0.08% round-trip costs)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Portfolio Value (₹)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.0f}"))

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved equity curves → %s", save_path)


def plot_drawdown_curves(
    returns_dict: dict[str, pd.Series],
    save_path: Path,
) -> None:
    """Plot drawdown time series for all strategies."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for strategy in STRATEGIES:
        if strategy not in returns_dict:
            continue
        r = returns_dict[strategy].dropna()
        if r.empty:
            continue
        dd = drawdown_series(r)
        lw = 2.5 if strategy == "adaptive_beta" else 1.2
        ax.plot(
            dd.index, dd.values,
            color=STRATEGY_COLORS.get(strategy, "gray"),
            linewidth=lw,
            label=STRATEGY_LABELS.get(strategy, strategy),
        )

    ax.fill_between(ax.get_lines()[0].get_xdata(), 0, ax.get_ylim()[0], alpha=0.03, color="red")
    ax.set_title("Drawdown Curves — Walk-Forward Backtest 2018–2024", fontsize=13, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved drawdown curves → %s", save_path)


def plot_rolling_sharpe(
    returns_dict: dict[str, pd.Series],
    save_path: Path,
    window: int = 252,
) -> None:
    """Plot 1-year rolling Sharpe ratio for all strategies."""
    fig, ax = plt.subplots(figsize=(14, 5))

    for strategy in STRATEGIES:
        if strategy not in returns_dict:
            continue
        r = returns_dict[strategy].dropna()
        if len(r) < window + 10:
            continue
        rs = rolling_sharpe(r, window=window)
        lw = 2.5 if strategy == "adaptive_beta" else 1.5
        ax.plot(
            rs.index, rs.values,
            color=STRATEGY_COLORS.get(strategy, "gray"),
            linewidth=lw,
            label=STRATEGY_LABELS.get(strategy, strategy),
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_title(f"Rolling {window}d Sharpe Ratio", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved rolling Sharpe → %s", save_path)


def plot_stress_heatmap(
    stress_dict: dict[str, pd.DataFrame],
    save_path: Path,
) -> None:
    """Plot heatmap of cumulative returns during stress events."""
    if not stress_dict:
        return

    # Build a combined dataframe: rows = strategies, cols = events
    events = list(stress_dict.keys())
    strategies_ordered = [s for s in STRATEGIES if any(s in df.index for df in stress_dict.values())]

    cum_matrix = pd.DataFrame(index=strategies_ordered, columns=events, dtype=float)
    dd_matrix = pd.DataFrame(index=strategies_ordered, columns=events, dtype=float)

    for event, df in stress_dict.items():
        for strategy in strategies_ordered:
            if strategy in df.index:
                cum_matrix.loc[strategy, event] = df.loc[strategy, "Cumulative Return (%)"]
                dd_matrix.loc[strategy, event] = df.loc[strategy, "Max Drawdown (%)"]

    cum_matrix = cum_matrix.astype(float)
    dd_matrix = dd_matrix.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Cumulative return heatmap
    ax = axes[0]
    vmax = max(abs(cum_matrix.values.flatten()))
    im = ax.imshow(cum_matrix.values, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(events)))
    ax.set_xticklabels([e[:12] for e in events], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(strategies_ordered)))
    ax.set_yticklabels([STRATEGY_LABELS.get(s, s) for s in strategies_ordered], fontsize=9)
    ax.set_title("Cumulative Return (%) During Stress Events", fontsize=11, fontweight="bold")
    for i in range(len(strategies_ordered)):
        for j in range(len(events)):
            val = cum_matrix.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > vmax * 0.6 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Max drawdown heatmap
    ax2 = axes[1]
    dd_vals = dd_matrix.values
    im2 = ax2.imshow(dd_vals, cmap="RdYlGn_r", aspect="auto",
                     vmin=min(dd_vals.flatten()), vmax=0)
    ax2.set_xticks(range(len(events)))
    ax2.set_xticklabels([e[:12] for e in events], rotation=30, ha="right", fontsize=9)
    ax2.set_yticks(range(len(strategies_ordered)))
    ax2.set_yticklabels([STRATEGY_LABELS.get(s, s) for s in strategies_ordered], fontsize=9)
    ax2.set_title("Max Drawdown (%) During Stress Events", fontsize=11, fontweight="bold")
    for i in range(len(strategies_ordered)):
        for j in range(len(events)):
            val = dd_matrix.iloc[i, j]
            if not np.isnan(val):
                ax2.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8,
                         color="white" if abs(val) > 15 else "black")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    plt.suptitle("Stress Event Analysis — AdaptiveBeta vs Benchmarks", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved stress heatmap → %s", save_path)


def plot_monthly_returns_heatmap(
    returns: pd.Series,
    strategy_name: str,
    save_path: Path,
) -> None:
    """Calendar heatmap of monthly returns for a single strategy."""
    if returns.empty:
        return

    monthly = returns.resample("ME").sum() * 100  # percentage
    df = monthly.to_frame(name="return")
    df["year"] = df.index.year
    df["month"] = df.index.month

    pivot = df.pivot(index="year", columns="month", values="return")

    fig, ax = plt.subplots(figsize=(13, 5))
    vmax = max(abs(pivot.values.flatten()))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names, fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str), fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(12):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > vmax * 0.6 else "black")

    plt.colorbar(im, ax=ax, label="Monthly Return (%)", shrink=0.8)
    ax.set_title(
        f"Monthly Returns — {STRATEGY_LABELS.get(strategy_name, strategy_name)}",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved monthly returns heatmap → %s", save_path)


# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------

def print_results_table(metrics_df: pd.DataFrame) -> None:
    """Pretty-print the performance comparison table."""
    cols = [
        "CAGR (%)", "Sharpe Ratio", "Sortino Ratio",
        "Max Drawdown (%)", "Calmar Ratio", "Annual Volatility (%)"
    ]
    available = [c for c in cols if c in metrics_df.columns]
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY — Walk-Forward Backtest 2018–2024")
    print("=" * 80)

    # Rename index for display
    display = metrics_df[available].copy()
    display.index = [STRATEGY_LABELS.get(i, i) for i in display.index]
    print(display.to_string())
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AdaptiveBeta Walk-Forward Backtest")
    p.add_argument("--fast", action="store_true",
                   help="Skip Kalman betas (faster, uses static OLS for kalman_mvo)")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="PyTorch device for LSTM fold retraining (default: auto)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Script 04 — Walk-Forward Backtest")
    logger.info("  fast=%s  device=%s", args.fast, args.device)
    logger.info("  Period: %d–%d  |  Folds: train %dy / test %dy",
                BACKTEST_START_YEAR, BACKTEST_END_YEAR, TRAIN_YEARS, TEST_YEARS)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    loader = DataLoader()
    betas_60 = load_optional_csv(FEATURES_DIR / "beta60d.csv")
    kalman_betas = (
        pd.DataFrame()
        if args.fast
        else load_optional_csv(FEATURES_DIR / "kalman_betas.csv")
    )
    regimes = load_hmm_regimes(FEATURES_DIR)

    stacked_path = FEATURES_DIR / "stacked_features.csv"
    if not stacked_path.exists():
        logger.warning("stacked_features.csv not found — LSTM predictions will be disabled")
        stacked_path = None

    # ------------------------------------------------------------------
    # 2. Build signal generator and optimiser
    # ------------------------------------------------------------------
    cfg = load_signal_config(MODELS_DIR)
    sig_gen = SignalGenerator(
        threshold=cfg.get("betavol_threshold", 0.05),
        vix_threshold=cfg.get("vix_threshold", 25.0),
    )
    optimiser = PortfolioOptimiser()

    # ------------------------------------------------------------------
    # 3. Initialise backtester
    # ------------------------------------------------------------------
    backtester = WalkForwardBacktester(
        loader=loader,
        signal_gen=sig_gen,
        optimiser=optimiser,
        kalman_betas=kalman_betas if not kalman_betas.empty else None,
        feature_matrix_path=str(stacked_path) if stacked_path else None,
    )
    if not regimes.empty:
        backtester.set_regimes(regimes)

    # ------------------------------------------------------------------
    # 4. Run backtest
    # ------------------------------------------------------------------
    logger.info("Starting walk-forward backtest...")
    bt_results = backtester.run(
        betas_60=betas_60 if not betas_60.empty else None,
    )

    returns_dict = bt_results.returns
    logger.info("Backtest complete. Strategies with returns:")
    for strat, r in returns_dict.items():
        logger.info("  %-25s %d days  mean=%.4f%%  total=%.1f%%",
                    strat, len(r.dropna()),
                    r.mean() * 100 if not r.empty else 0,
                    (np.exp(r.dropna().sum()) - 1) * 100 if not r.empty else 0)

    # ------------------------------------------------------------------
    # 5. Performance metrics
    # ------------------------------------------------------------------
    metrics_df = build_metrics_table(returns_dict)
    print_results_table(metrics_df)

    # ------------------------------------------------------------------
    # 6. Stress analysis
    # ------------------------------------------------------------------
    logger.info("Running stress event analysis...")
    stress_dict = stress_analysis(returns_dict, STRESS_EVENTS)
    for event, df in stress_dict.items():
        logger.info("Stress event '%s':\n%s", event, df.to_string())

    # Flatten stress results for CSV
    stress_rows = []
    for event, df in stress_dict.items():
        for strategy in df.index:
            row = df.loc[strategy].to_dict()
            row["strategy"] = strategy
            row["event"] = event
            stress_rows.append(row)
    stress_flat = pd.DataFrame(stress_rows).set_index(["event", "strategy"])

    # ------------------------------------------------------------------
    # 7. Save CSVs
    # ------------------------------------------------------------------
    # Daily returns
    returns_out = pd.DataFrame(returns_dict)
    returns_out.to_csv(RESULTS_DIR / "strategy_returns.csv")
    logger.info("Saved strategy_returns.csv")

    # Metrics
    metrics_df.to_csv(RESULTS_DIR / "performance_metrics.csv")
    logger.info("Saved performance_metrics.csv")

    # Stress
    stress_flat.to_csv(RESULTS_DIR / "stress_analysis.csv")
    logger.info("Saved stress_analysis.csv")

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    logger.info("Generating plots...")
    plot_equity_curves(returns_dict, RESULTS_DIR / "equity_curves.png")
    plot_drawdown_curves(returns_dict, RESULTS_DIR / "drawdown_curves.png")
    plot_rolling_sharpe(returns_dict, RESULTS_DIR / "rolling_sharpe.png")
    plot_stress_heatmap(stress_dict, RESULTS_DIR / "stress_heatmap.png")

    # Monthly returns heatmap for AdaptiveBeta
    if "adaptive_beta" in returns_dict and not returns_dict["adaptive_beta"].empty:
        plot_monthly_returns_heatmap(
            returns_dict["adaptive_beta"],
            "adaptive_beta",
            RESULTS_DIR / "monthly_returns_heatmap.png",
        )

    # ------------------------------------------------------------------
    # 9. Final summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Script 04 complete. Files written:")
    logger.info("  results/strategy_returns.csv")
    logger.info("  results/performance_metrics.csv")
    logger.info("  results/stress_analysis.csv")
    logger.info("  results/equity_curves.png")
    logger.info("  results/drawdown_curves.png")
    logger.info("  results/rolling_sharpe.png")
    logger.info("  results/stress_heatmap.png")
    logger.info("  results/monthly_returns_heatmap.png  (AdaptiveBeta)")
    logger.info("=" * 60)

    # Print key result
    if "adaptive_beta" in metrics_df.index:
        ab = metrics_df.loc["adaptive_beta"]
        nifty = metrics_df.loc["buy_hold_nifty"] if "buy_hold_nifty" in metrics_df.index else None
        print("\n=== KEY RESULTS ===")
        print(f"AdaptiveBeta CAGR:    {ab.get('CAGR (%)', 'N/A')}%")
        print(f"AdaptiveBeta Sharpe:  {ab.get('Sharpe Ratio', 'N/A')}")
        print(f"AdaptiveBeta MaxDD:   {ab.get('Max Drawdown (%)', 'N/A')}%")
        if nifty is not None:
            print(f"NIFTY50 CAGR:         {nifty.get('CAGR (%)', 'N/A')}%")
            print(f"NIFTY50 Sharpe:       {nifty.get('Sharpe Ratio', 'N/A')}")


if __name__ == "__main__":
    main()
