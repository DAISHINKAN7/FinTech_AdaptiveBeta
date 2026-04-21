"""Script 04 — Walk-Forward Backtest.
 
Runs the full walk-forward backtest (2018-2025) across 6 strategies:
1. adaptive_beta       — LSTM/XGB threshold trigger + regime routing
2. static_capm_mvo     — Monthly MVO with static OLS betas
3. kalman_mvo          — Monthly MVO with Kalman-filtered betas
4. equal_weight        — Monthly equal-weight rebalance
5. buy_hold_nifty      — Buy & hold NIFTY50 benchmark
6. momentum_quality    — 6-month price momentum, monthly rebalance, top 15 stocks
 
Outputs:
    results/strategy_returns.csv          <- daily return series, all strategies
    results/{strategy}_returns.csv        <- individual CSVs for dashboard
    results/performance_metrics.csv       <- annual return, Sharpe, Sortino, etc.
    results/stress_analysis.csv           <- COVID, ADANI, IL&FS, 2018 hike
    results/equity_curves.png
    results/drawdown_curves.png
    results/rolling_sharpe.png
    results/stress_heatmap.png
    results/monthly_returns_heatmap.png
 
Run:
    python scripts/04_backtest.py
    python scripts/04_backtest.py --fast          # skip Kalman (much faster)
    python scripts/04_backtest.py --device cuda   # GPU for LSTM fold retraining
"""
 
import argparse
import logging
import sys
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    STRESS_EVENTS,
    TRAIN_YEARS,
    TEST_YEARS,
    MOMENTUM_TOP_N,
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
# Strategy display config
# ---------------------------------------------------------------------------
 
STRATEGY_COLORS = {
    "adaptive_beta":    "#00D4AA",
    "kalman_mvo":       "#818CF8",
    "static_capm_mvo":  "#FBBF24",
    "equal_weight":     "#64748B",
    "buy_hold_nifty":   "#F87171",
    "momentum_quality": "#34D399",
}
 
STRATEGY_LABELS = {
    "adaptive_beta":    "AdaptiveBeta (ours)",
    "kalman_mvo":       "Kalman-Beta MVO",
    "static_capm_mvo":  "Static CAPM MVO",
    "equal_weight":     "Equal Weight",
    "buy_hold_nifty":   "Buy & Hold NIFTY50",
    "momentum_quality": "Momentum-Quality",
}
 
ALL_STRATEGIES = list(STRATEGY_COLORS.keys())
 
 
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
    logger.info("Loaded signal_config.json: threshold=%.4f  model=%s",
                cfg.get("betavol_threshold"), cfg.get("best_model", "unknown"))
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
    df  = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    col = "regime_label" if "regime_label" in df.columns else df.columns[0]
    regime = df[col].rename("regime")
    logger.info("Loaded hmm_regimes.csv — %d dates | distribution:\n%s",
                len(regime), regime.value_counts().to_string())
    return regime
 
 
# ---------------------------------------------------------------------------
# MOMENTUM-QUALITY STRATEGY
# Pure price-momentum: buy top N stocks by 6-month momentum, rebalance monthly.
# No dependency on LSTM or HMM — completely standalone.
# ---------------------------------------------------------------------------
 
def run_momentum_strategy(
    prices: pd.DataFrame,
    backtest_start: str = "2018-01-01",
    backtest_end:   str = "2025-12-31",
    top_n:      int   = 15,
    lookback:   int   = 126,   # 6-month momentum window (trading days)
    skip:       int   = 21,    # skip last 1 month to avoid short-term reversal
    rebal_freq: int   = 21,    # rebalance every ~1 month (trading days)
    max_w:      float = 0.15,  # max weight per stock
    tc:         float = 0.0008,  # 0.08% round-trip transaction cost
) -> pd.Series:
    """
    At each monthly rebalance date:
      1. Look back 6 months (excluding most recent month)
      2. Rank all 49 stocks by price momentum
      3. Buy equal-weight top-N (capped at max_w per stock)
      4. Deduct transaction costs on turnover
 
    Returns a daily log-return Series.
    """
    prices = prices.loc[backtest_start:backtest_end].copy()
    prices = prices.dropna(axis=1, how="all")  # drop columns that are all NaN
 
    # Pre-compute log returns for the whole panel
    log_ret = np.log(prices / prices.shift(1))
 
    daily_rets:   list[tuple] = []
    current_w:    dict[str, float] = {}
    last_rebal_i: int = -rebal_freq  # force rebalance on first day
 
    idx = prices.index.tolist()
 
    for i in range(1, len(idx)):
        date = idx[i]
 
        # Monthly rebalance
        if (i - last_rebal_i) >= rebal_freq:
            lag_i  = max(0, i - skip)
            base_i = max(0, i - skip - lookback)
 
            if lag_i > base_i:
                lag_px  = prices.iloc[lag_i]
                base_px = prices.iloc[base_i]
 
                # Compute momentum, drop NaN (pre-IPO stocks)
                mom = (lag_px / base_px - 1).replace([np.inf, -np.inf], np.nan).dropna()
                mom = mom.sort_values(ascending=False)
 
                top_tickers = mom.head(top_n).index.tolist()
                n = len(top_tickers)
                if n > 0:
                    raw_w   = 1.0 / n
                    w_each  = min(raw_w, max_w)
                    new_w   = {t: w_each for t in top_tickers}
                    # Renormalise to sum to 1
                    total_w = sum(new_w.values())
                    new_w   = {t: v / total_w for t, v in new_w.items()}
 
                    # Transaction cost: half of two-way turnover
                    if current_w:
                        all_tickers = set(current_w) | set(new_w)
                        turnover = sum(
                            abs(new_w.get(t, 0.0) - current_w.get(t, 0.0))
                            for t in all_tickers
                        ) / 2.0
                        tc_drag = turnover * tc
                    else:
                        tc_drag = 0.0
 
                    current_w   = new_w
                    last_rebal_i = i
 
                    # Deduct cost on rebalance day
                    daily_rets.append((date, -tc_drag))
                    continue
 
        # Non-rebalance day: compute portfolio return
        day_ret = 0.0
        if current_w:
            for ticker, w in current_w.items():
                if ticker in log_ret.columns:
                    r = log_ret.loc[date, ticker]
                    if pd.notna(r):
                        day_ret += w * r
        daily_rets.append((date, day_ret))
 
    series = pd.Series(dict(daily_rets), name="momentum_quality")
    total  = (np.exp(series.sum()) - 1) * 100
    logger.info(
        "Momentum-Quality: %d days | mean=%.4f%% | total=%.1f%%",
        len(series), series.mean() * 100, total
    )
    return series
 
 
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
 
def plot_equity_curves(returns_dict: dict, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0A0F1E")
    ax.set_facecolor("#0F1629")
 
    for strategy in ALL_STRATEGIES:
        if strategy not in returns_dict:
            continue
        r = returns_dict[strategy].dropna()
        if r.empty:
            continue
        equity = 100 * np.exp(r.cumsum())
        lw = 2.5 if strategy == "adaptive_beta" else (2.0 if strategy == "momentum_quality" else 1.5)
        ax.plot(
            equity.index, equity.values,
            color=STRATEGY_COLORS.get(strategy, "gray"),
            linewidth=lw,
            label=STRATEGY_LABELS.get(strategy, strategy),
            zorder=4 if strategy in ("adaptive_beta", "momentum_quality") else 2,
        )
 
    for event, (start, end) in STRESS_EVENTS.items():
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.07, color="red")
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ymin = ax.get_ylim()[0]
        ax.text(mid, ymin * 1.02, event[:8], fontsize=7,
                ha="center", color="#FF6B6B", rotation=90, va="bottom")
 
    ax.set_title(
        f"Equity Curves — Walk-Forward Backtest {BACKTEST_START_YEAR}–{BACKTEST_END_YEAR}\n"
        "(₹100 invested, log returns, 0.08% round-trip costs)",
        fontsize=13, fontweight="bold", color="white"
    )
    ax.set_ylabel("Portfolio Value (₹)", color="white")
    ax.set_xlabel("Date", color="white")
    ax.legend(loc="upper left", fontsize=10, facecolor="#141D3A", labelcolor="white")
    ax.grid(True, alpha=0.2, color="#2A3A5A")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#2A3A5A")
    ax.spines["left"].set_color("#2A3A5A")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"₹{x:.0f}"))
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved equity curves -> %s", save_path)
 
 
def plot_drawdown_curves(returns_dict: dict, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0A0F1E")
    ax.set_facecolor("#0F1629")
 
    for strategy in ALL_STRATEGIES:
        if strategy not in returns_dict:
            continue
        r = returns_dict[strategy].dropna()
        if r.empty:
            continue
        dd = drawdown_series(r)
        lw = 2.5 if strategy == "adaptive_beta" else (2.0 if strategy == "momentum_quality" else 1.2)
        ax.plot(
            dd.index, dd.values,
            color=STRATEGY_COLORS.get(strategy, "gray"),
            linewidth=lw,
            label=STRATEGY_LABELS.get(strategy, strategy),
        )
 
    ax.set_title(
        f"Drawdown Curves — Walk-Forward Backtest {BACKTEST_START_YEAR}–{BACKTEST_END_YEAR}",
        fontsize=13, fontweight="bold", color="white"
    )
    ax.set_ylabel("Drawdown (%)", color="white")
    ax.set_xlabel("Date", color="white")
    ax.legend(loc="lower left", fontsize=10, facecolor="#141D3A", labelcolor="white")
    ax.grid(True, alpha=0.2, color="#2A3A5A")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#2A3A5A")
    ax.spines["left"].set_color("#2A3A5A")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved drawdown curves -> %s", save_path)
 
 
def plot_rolling_sharpe(returns_dict: dict, save_path: Path, window: int = 252) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0A0F1E")
    ax.set_facecolor("#0F1629")
 
    for strategy in ALL_STRATEGIES:
        if strategy not in returns_dict:
            continue
        r = returns_dict[strategy].dropna()
        if len(r) < window + 10:
            continue
        rs = rolling_sharpe(r, window=window)
        lw = 2.5 if strategy == "adaptive_beta" else (2.0 if strategy == "momentum_quality" else 1.5)
        ax.plot(
            rs.index, rs.values,
            color=STRATEGY_COLORS.get(strategy, "gray"),
            linewidth=lw,
            label=STRATEGY_LABELS.get(strategy, strategy),
        )
 
    ax.axhline(0, color="#444", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="#555", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.set_title(f"Rolling {window}d Sharpe Ratio", fontsize=13, fontweight="bold", color="white")
    ax.set_ylabel("Sharpe Ratio", color="white")
    ax.set_xlabel("Date", color="white")
    ax.legend(loc="upper left", fontsize=10, facecolor="#141D3A", labelcolor="white")
    ax.grid(True, alpha=0.2, color="#2A3A5A")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#2A3A5A")
    ax.spines["left"].set_color("#2A3A5A")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved rolling Sharpe -> %s", save_path)
 
 
def plot_stress_heatmap(stress_dict: dict, save_path: Path) -> None:
    if not stress_dict:
        return
 
    events = list(stress_dict.keys())
    strategies_ordered = [s for s in ALL_STRATEGIES
                          if any(s in df.index for df in stress_dict.values())]
 
    cum_matrix = pd.DataFrame(index=strategies_ordered, columns=events, dtype=float)
    dd_matrix  = pd.DataFrame(index=strategies_ordered, columns=events, dtype=float)
 
    for event, df in stress_dict.items():
        for strategy in strategies_ordered:
            if strategy in df.index:
                cum_matrix.loc[strategy, event] = df.loc[strategy, "Cumulative Return (%)"]
                dd_matrix.loc[strategy, event]  = df.loc[strategy, "Max Drawdown (%)"]
 
    cum_matrix = cum_matrix.astype(float)
    dd_matrix  = dd_matrix.astype(float)
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0A0F1E")
 
    for ax, matrix, title, cmap, center in [
        (axes[0], cum_matrix, "Cumulative Return (%)", "RdYlGn", True),
        (axes[1], dd_matrix,  "Max Drawdown (%)",      "RdYlGn_r", False),
    ]:
        ax.set_facecolor("#0F1629")
        if center:
            vmax = max(abs(matrix.values.flatten()[~np.isnan(matrix.values.flatten())]))
            im   = ax.imshow(matrix.values, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(matrix.values, cmap=cmap, aspect="auto",
                           vmin=matrix.values[~np.isnan(matrix.values)].min(), vmax=0)
 
        ax.set_xticks(range(len(events)))
        ax.set_xticklabels([e[:12] for e in events], rotation=30, ha="right",
                           fontsize=9, color="white")
        ax.set_yticks(range(len(strategies_ordered)))
        ax.set_yticklabels([STRATEGY_LABELS.get(s, s) for s in strategies_ordered],
                           fontsize=9, color="white")
        ax.set_title(title, fontsize=11, fontweight="bold", color="white")
 
        flat = matrix.values.flatten()
        vmax_for_text = max(abs(flat[~np.isnan(flat)])) if not np.all(np.isnan(flat)) else 1
        for i in range(len(strategies_ordered)):
            for j in range(len(events)):
                val = matrix.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=8,
                            color="white" if abs(val) > vmax_for_text * 0.5 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8)
 
    plt.suptitle(f"Stress Event Analysis — {BACKTEST_START_YEAR}–{BACKTEST_END_YEAR}",
                 fontsize=12, fontweight="bold", color="white")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved stress heatmap -> %s", save_path)
 
 
def plot_monthly_returns_heatmap(returns: pd.Series, strategy_name: str,
                                  save_path: Path) -> None:
    if returns.empty:
        return
 
    monthly = returns.resample("ME").sum() * 100
    df      = monthly.to_frame(name="return")
    df["year"]  = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot(index="year", columns="month", values="return")
 
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#0A0F1E")
    ax.set_facecolor("#0F1629")
 
    flat = pivot.values.flatten()
    vmax = max(abs(flat[~np.isnan(flat)])) if not np.all(np.isnan(flat)) else 1
    im   = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
 
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    ax.set_xticks(range(12))
    ax.set_xticklabels(month_names, fontsize=9, color="white")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(str), fontsize=9, color="white")
 
    for i in range(len(pivot.index)):
        for j in range(12):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > vmax * 0.6 else "black")
 
    plt.colorbar(im, ax=ax, label="Monthly Return (%)", shrink=0.8)
    ax.set_title(
        f"Monthly Returns — {STRATEGY_LABELS.get(strategy_name, strategy_name)}",
        fontsize=12, fontweight="bold", color="white"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved monthly returns heatmap -> %s", save_path)
 
 
# ---------------------------------------------------------------------------
# Print summary table
# ---------------------------------------------------------------------------
 
def print_results_table(metrics_df: pd.DataFrame) -> None:
    cols = [
        "CAGR (%)", "Sharpe Ratio", "Sortino Ratio",
        "Max Drawdown (%)", "Calmar Ratio", "Annual Volatility (%)"
    ]
    available = [c for c in cols if c in metrics_df.columns]
    print("\n" + "=" * 80)
    print(f"PERFORMANCE SUMMARY — Walk-Forward Backtest {BACKTEST_START_YEAR}–{BACKTEST_END_YEAR}")
    print("=" * 80)
    display = metrics_df[available].copy()
    display.index = [STRATEGY_LABELS.get(i, i) for i in display.index]
    print(display.to_string())
    print("=" * 80)
 
 
# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AdaptiveBeta Walk-Forward Backtest")
    p.add_argument("--fast", action="store_true",
                   help="Skip Kalman betas (faster, uses static OLS for kalman_mvo)")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="PyTorch device for LSTM fold retraining (default: auto)")
    return p.parse_args()
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def main() -> None:
    args = parse_args()
 
    logger.info("=" * 60)
    logger.info("Script 04 — Walk-Forward Backtest")
    logger.info("  fast=%s  device=%s", args.fast, args.device)
    logger.info("  Period: %d-%d  |  Folds: train %dy / test %dy",
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
    # 3. Initialise and run walk-forward backtester (5 existing strategies)
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
 
    logger.info("Starting walk-forward backtest (5 core strategies)...")
    bt_results   = backtester.run(betas_60=betas_60 if not betas_60.empty else None)
    returns_dict = bt_results.returns
 
    # ------------------------------------------------------------------
    # 4. Run Momentum-Quality strategy (standalone, no engine dependency)
    # ------------------------------------------------------------------
    logger.info("\n--- Strategy 6: Momentum-Quality (6m price momentum, top %d) ---",
                MOMENTUM_TOP_N)
    try:
        mom_returns = run_momentum_strategy(
            prices=loader.prices,
            backtest_start=f"{BACKTEST_START_YEAR}-01-01",
            backtest_end=f"{BACKTEST_END_YEAR}-12-31",
            top_n=MOMENTUM_TOP_N,
            lookback=126,
            skip=21,
            rebal_freq=21,
            max_w=0.15,
            tc=0.0008,
        )
        returns_dict["momentum_quality"] = mom_returns
    except Exception as e:
        logger.error("Momentum strategy failed: %s", e)
 
    # ------------------------------------------------------------------
    # 5. Log summary
    # ------------------------------------------------------------------
    logger.info("Backtest complete. Strategies with returns:")
    for strat, r in returns_dict.items():
        r_clean = r.dropna()
        logger.info("  %-25s %d days  mean=%.4f%%  total=%.1f%%",
                    strat, len(r_clean),
                    r_clean.mean() * 100 if not r_clean.empty else 0,
                    (np.exp(r_clean.sum()) - 1) * 100 if not r_clean.empty else 0)
 
    # ------------------------------------------------------------------
    # 6. Performance metrics
    # ------------------------------------------------------------------
    metrics_df = build_metrics_table(returns_dict)
    print_results_table(metrics_df)
 
    # ------------------------------------------------------------------
    # 7. Stress analysis
    # ------------------------------------------------------------------
    logger.info("Running stress event analysis...")
    stress_dict = stress_analysis(returns_dict, STRESS_EVENTS)
    for event, df in stress_dict.items():
        logger.info("Stress event '%s':\n%s", event, df.to_string())
 
    stress_rows = []
    for event, df in stress_dict.items():
        for strategy in df.index:
            row          = df.loc[strategy].to_dict()
            row["strategy"] = strategy
            row["event"]    = event
            stress_rows.append(row)
    stress_flat = pd.DataFrame(stress_rows).set_index(["event", "strategy"])
 
    # ------------------------------------------------------------------
    # 8. Save CSVs
    # ------------------------------------------------------------------
    # Combined returns file
    returns_out = pd.DataFrame(returns_dict)
    returns_out.index.name = "date"
    returns_out.to_csv(RESULTS_DIR / "strategy_returns.csv")
    logger.info("Saved strategy_returns.csv")
 
    # Individual returns CSVs (for Dash dashboard)
    for strat, ret in returns_dict.items():
        out = ret.to_frame(name=strat)
        out.index.name = "date"
        out.to_csv(RESULTS_DIR / f"{strat}_returns.csv")
    logger.info("Saved individual {strategy}_returns.csv files")
 
    # Metrics
    metrics_df.to_csv(RESULTS_DIR / "performance_metrics.csv")
    logger.info("Saved performance_metrics.csv")
 
    # Stress
    stress_flat.to_csv(RESULTS_DIR / "stress_analysis.csv")
    logger.info("Saved stress_analysis.csv")
 
    # ------------------------------------------------------------------
    # 9. Plots
    # ------------------------------------------------------------------
    logger.info("Generating plots...")
    plot_equity_curves(returns_dict,   RESULTS_DIR / "equity_curves.png")
    plot_drawdown_curves(returns_dict, RESULTS_DIR / "drawdown_curves.png")
    plot_rolling_sharpe(returns_dict,  RESULTS_DIR / "rolling_sharpe.png")
    plot_stress_heatmap(stress_dict,   RESULTS_DIR / "stress_heatmap.png")
 
    # Monthly heatmaps for AdaptiveBeta and Momentum
    for strat in ("adaptive_beta", "momentum_quality"):
        if strat in returns_dict and not returns_dict[strat].empty:
            plot_monthly_returns_heatmap(
                returns_dict[strat], strat,
                RESULTS_DIR / f"monthly_returns_heatmap_{strat}.png",
            )
 
    # ------------------------------------------------------------------
    # 10. Final summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Script 04 complete. Files written:")
    logger.info("  results/strategy_returns.csv")
    logger.info("  results/{strategy}_returns.csv  (one per strategy)")
    logger.info("  results/performance_metrics.csv")
    logger.info("  results/stress_analysis.csv")
    logger.info("  results/equity_curves.png")
    logger.info("  results/drawdown_curves.png")
    logger.info("  results/rolling_sharpe.png")
    logger.info("  results/stress_heatmap.png")
    logger.info("  results/monthly_returns_heatmap_*.png")
    logger.info("=" * 60)
 
    # Key results print
    print("\n=== KEY RESULTS ===")
    for strat in ("adaptive_beta", "momentum_quality", "equal_weight", "buy_hold_nifty"):
        if strat in metrics_df.index:
            row = metrics_df.loc[strat]
            print(f"{STRATEGY_LABELS.get(strat, strat):<28}  "
                  f"CAGR={row.get('CAGR (%)', 'N/A'):>6}%  "
                  f"Sharpe={row.get('Sharpe Ratio', 'N/A'):>5}  "
                  f"MaxDD={row.get('Max Drawdown (%)', 'N/A'):>7}%")
 
 
if __name__ == "__main__":
    main()