"""
AdaptiveBeta v2 — Upgrade Pipeline Runner
==========================================
This script orchestrates the full v2 upgrade pipeline:

  Stage 1 — Enhanced Feature Engineering
           (advanced beta/betavol, regime interaction, sector-relative,
            lag features, z-scores, technical indicators)

  Stage 2 — Regime Detection
           (HMM primary + KMeans + Threshold ensemble, with balance correction)

  Stage 3 — Beta Model Suite
           (XGBoost + LightGBM, pooled + regime-conditional;
            SHAP explainability, walk-forward cross-validation)

  Stage 4 — Advanced Backtesting
           (6 strategies, full metric suite, stress analysis,
            publication-quality charts)

Run:
    python scripts/06_run_v2_upgrade.py
    python scripts/06_run_v2_upgrade.py --fast           # skip regime-conditional + CV
    python scripts/06_run_v2_upgrade.py --regime hmm     # hmm | kmeans | threshold | ensemble
    python scripts/06_run_v2_upgrade.py --start 3        # resume from stage N

Outputs:
    features/stacked_features_v2.csv        enriched feature matrix
    features/regime_labels_v2.csv           regime series
    models/beta_suite_v2.pkl                trained BetaModelSuite
    models/signal_config_v2.json            updated threshold config
    results/performance_metrics_v2.csv      full metric table
    results/shap_importance_v2.csv          feature importance
    results/walkfwd_cv_v2.csv              walk-forward CV metrics
    results/equity_curves_v2.png
    results/drawdown_v2.png
    results/rolling_sharpe_v2.png
    results/stress_heatmap_v2.png
    results/shap_importance_v2.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR,
    TICKERS, BETA_WINDOWS, TARGET_HORIZON, TRAIN_END, TEST_START,
    RANDOM_SEED, VIX_STRESS_LEVEL, THRESHOLD_QUANTILE,
)
from src.data.loader import DataLoader
from src.utils.logging import setup_logging

logger = setup_logging(level="INFO", log_file=str(RESULTS_DIR / "06_v2_upgrade.log"))

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AdaptiveBeta v2 upgrade pipeline")
    p.add_argument("--fast",   action="store_true", help="Skip regime-conditional models + walk-forward CV")
    p.add_argument("--regime", default="ensemble",  choices=["hmm","kmeans","threshold","ensemble"])
    p.add_argument("--start",  type=int, default=1, choices=[1,2,3,4])
    p.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Enhanced feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def stage1_features(loader: DataLoader) -> pd.DataFrame:
    """Build the enriched v2 feature matrix."""
    logger.info("=" * 60)
    logger.info("STAGE 1 — Enhanced Feature Engineering")
    logger.info("=" * 60)

    from src.features.beta import rolling_beta_panel, beta_volatility, forward_target
    from src.features.macro import build_market_features
    from src.features.returns import compute_log_returns
    from src.features.feature_engineering_advanced import (
        AdvancedFeatureBuilder,
        add_sector_relative_features,
        compute_sector_momentum,
    )

    t0 = time.time()

    # Base returns
    stock_ret  = compute_log_returns(loader.prices)
    market_ret = loader.market_returns

    # Fill late-IPO NaN
    for t in ["HDFCLIFE.NS", "SBILIFE.NS"]:
        if t in stock_ret.columns:
            stock_ret[t] = stock_ret[t].fillna(0)

    # Beta panels
    logger.info("Computing rolling betas (windows=%s)...", BETA_WINDOWS)
    betas    = rolling_beta_panel(stock_ret, market_ret, BETA_WINDOWS)
    betavols = beta_volatility(betas, BETA_WINDOWS)
    target   = forward_target(betavols[60], TARGET_HORIZON)

    # Market features
    market_df = build_market_features(loader, market_ret)

    # Advanced builder
    builder = AdvancedFeatureBuilder(
        lag_windows      = [1, 2, 5, 10, 20],
        beta_windows     = BETA_WINDOWS,
        momentum_windows = [5, 10, 21, 63],
        zscore_windows   = [60, 252],
    )

    available_tickers = [t for t in TICKERS if t in loader.prices.columns]
    stacked = builder.build_all(
        market_df=market_df,
        betas=betas,
        betavols=betavols,
        prices=loader.prices,
        stock_ret=stock_ret,
        target=target,
        tickers=available_tickers,
    )

    # Add sector-relative features
    frames = []
    for ticker, grp in stacked.groupby("ticker"):
        grp = add_sector_relative_features(grp, str(ticker), stock_ret)
        frames.append(grp)
    stacked = pd.concat(frames).sort_index()

    out_path = FEATURES_DIR / "stacked_features_v2.csv"
    stacked.to_csv(out_path)
    logger.info("Saved stacked_features_v2.csv — %s rows x %s cols in %.1fs",
                *stacked.shape, time.time() - t0)
    return stacked


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Regime detection
# ─────────────────────────────────────────────────────────────────────────────

def stage2_regime(loader: DataLoader, mode: str = "ensemble") -> pd.Series:
    """Fit regime detector and return labelled series."""
    logger.info("=" * 60)
    logger.info("STAGE 2 — Regime Detection (mode=%s)", mode)
    logger.info("=" * 60)

    from src.models.regime_detection_advanced import build_regime_detector

    detector = build_regime_detector(mode)
    detector.fit(loader.market_returns, loader.vix)
    regime   = detector.predict(loader.market_returns, loader.vix)
    regime.name = "regime"

    out_path = FEATURES_DIR / "regime_labels_v2.csv"
    regime.to_csv(out_path, header=True)

    dist = regime.value_counts(normalize=True) * 100
    logger.info("Regime distribution:\n%s", dist.round(1).to_string())
    logger.info("Saved regime_labels_v2.csv")
    return regime


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 — Beta model suite
# ─────────────────────────────────────────────────────────────────────────────

def stage3_models(
    stacked: pd.DataFrame,
    regime: pd.Series,
    fast: bool = False,
) -> dict:
    """Train BetaModelSuite and run walk-forward CV."""
    logger.info("=" * 60)
    logger.info("STAGE 3 — Beta Model Suite")
    logger.info("=" * 60)

    from src.models.beta_model_advanced import BetaModelSuite

    t0 = time.time()

    # Inject regime into stacked
    stacked_with_regime = stacked.copy()
    stacked_with_regime["regime"] = regime.reindex(stacked.index).ffill().fillna("transition")

    suite = BetaModelSuite(
        regime_conditional = not fast,
        use_xgb=True,
        use_lgb=True,
        threshold_quantile = THRESHOLD_QUANTILE,
        vix_threshold = VIX_STRESS_LEVEL,
        random_seed = RANDOM_SEED,
    )
    result = suite.fit(stacked_with_regime, train_end=TRAIN_END, regime_col="regime")

    # Save SHAP importance
    if result.shap_importance is not None:
        result.shap_importance.to_csv(RESULTS_DIR / "shap_importance_v2.csv", index=False)
        logger.info("Top 10 features:\n%s",
                    result.shap_importance.head(10).to_string(index=False))

    # Walk-forward CV
    if not fast:
        logger.info("Running walk-forward CV...")
        cv_df = suite.walk_forward_cv(
            stacked_with_regime,
            train_years=3, test_years=1,
            start_year=2018, end_year=2025,
        )
        cv_df.to_csv(RESULTS_DIR / "walkfwd_cv_v2.csv", index=False)
        logger.info("Walk-forward CV summary:\n%s", cv_df.to_string(index=False))

    # Save suite
    suite.save(MODELS_DIR / "beta_suite_v2.pkl")

    # Update signal_config_v2.json
    cfg = {
        "betavol_threshold":  result.threshold,
        "vix_threshold":      result.vix_threshold,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "best_model":         result.best_model_name,
        "train_end":          TRAIN_END,
        "test_start":         TEST_START,
        "version":            "v2",
    }
    with open(MODELS_DIR / "signal_config_v2.json", "w") as f:
        json.dump(cfg, f, indent=2)

    logger.info("Stage 3 complete in %.1fs | best_model=%s | threshold=%.4f",
                time.time() - t0, result.best_model_name, result.threshold)
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 — Advanced backtest
# ─────────────────────────────────────────────────────────────────────────────

def stage4_backtest(
    loader: DataLoader,
    regime: pd.Series,
    cfg: dict,
    no_charts: bool = False,
) -> None:
    """
    Run the full 6-strategy walk-forward backtest with v2 enhancements.
    This re-uses the existing engine (src.backtest.engine) but now:
      - loads the v2 model suite for prediction
      - uses the v2 betavol threshold and vix threshold
      - adds regime-overlay charts
    """
    logger.info("=" * 60)
    logger.info("STAGE 4 — Advanced Backtest")
    logger.info("=" * 60)

    t0 = time.time()

    # ── Load required artifacts ──────────────────────────────────────────────
    from src.portfolio.signal import SignalGenerator
    from src.portfolio.optimiser import PortfolioOptimiser
    from src.backtest.engine import WalkForwardBacktester
    from src.backtest.metrics import build_metrics_table, stress_analysis
    from src.backtest.advanced_backtest import (
        BacktestResults, StrategyResult, BacktestVisualizer,
        compute_stress_stats, annualised_return, sharpe_ratio, max_drawdown,
    )
    from src.models.beta_model_advanced import BetaModelSuite

    # Optional: load Kalman betas
    kalman_path = FEATURES_DIR / "kalman_betas.csv"
    kalman_betas = None
    if kalman_path.exists():
        kalman_betas = pd.read_csv(kalman_path, parse_dates=["date"]).set_index("date")
        logger.info("Loaded kalman_betas.csv")

    betas_60_path = FEATURES_DIR / "beta60d.csv"
    betas_60 = None
    if betas_60_path.exists():
        betas_60 = pd.read_csv(betas_60_path, parse_dates=["date"]).set_index("date")

    stacked_path = FEATURES_DIR / "stacked_features_v2.csv"
    feature_matrix_path = str(stacked_path) if stacked_path.exists() else None

    sig_gen   = SignalGenerator(
        threshold    = cfg["betavol_threshold"],
        vix_threshold = cfg["vix_threshold"],
    )
    optimiser = PortfolioOptimiser()

    backtester = WalkForwardBacktester(
        loader=loader,
        signal_gen=sig_gen,
        optimiser=optimiser,
        kalman_betas=kalman_betas,
        feature_matrix_path=feature_matrix_path,
    )
    backtester.set_regimes(regime)

    logger.info("Running walk-forward backtest (v2 thresholds)...")
    bt_results = backtester.run(betas_60=betas_60)

    # ── Momentum-Quality strategy ───────────────────────────────────────────
    from scripts.script_04_backtest import run_momentum_strategy  # noqa: F401 -- graceful import
    try:
        from scripts.script_04_backtest import run_momentum_strategy as mom_strat
        mom_ret = mom_strat(
            prices=loader.prices,
            backtest_start="2015-01-01",
            backtest_end="2025-12-31",
        )
        bt_results.returns["momentum_quality"] = mom_ret
    except Exception as exc:
        logger.warning("Momentum strategy import failed: %s", exc)
        # Fallback: use equal-weight as proxy
        bt_results.returns["momentum_quality"] = bt_results.returns.get(
            "equal_weight", pd.Series(dtype=float)
        )

    # ── Save returns ────────────────────────────────────────────────────────
    returns_df = pd.DataFrame(bt_results.returns)
    returns_df.index.name = "date"
    returns_df.to_csv(RESULTS_DIR / "strategy_returns_v2.csv")

    # ── Metrics ─────────────────────────────────────────────────────────────
    metrics_df = build_metrics_table(bt_results.returns)
    metrics_df.to_csv(RESULTS_DIR / "performance_metrics_v2.csv")

    # ── Stress analysis ─────────────────────────────────────────────────────
    from src.config import STRESS_EVENTS
    stress_dict = stress_analysis(bt_results.returns, STRESS_EVENTS)
    stress_rows = []
    for event, df_ev in stress_dict.items():
        for strategy in df_ev.index:
            row = df_ev.loc[strategy].to_dict()
            row.update({"strategy": strategy, "event": event})
            stress_rows.append(row)
    stress_flat = pd.DataFrame(stress_rows).set_index(["event", "strategy"])
    stress_flat.to_csv(RESULTS_DIR / "stress_analysis_v2.csv")

    # ── Print summary ───────────────────────────────────────────────────────
    br = BacktestResults()
    for name, ret in bt_results.returns.items():
        if not ret.empty:
            br.add(StrategyResult(name=name, log_returns=ret))
    br.stress_stats = compute_stress_stats(bt_results.returns)
    br.print_summary()

    # ── Charts ──────────────────────────────────────────────────────────────
    if not no_charts:
        viz = BacktestVisualizer(br, save_dir=RESULTS_DIR)
        viz.plot_all(regime_series=regime)

        # SHAP chart
        shap_path = RESULTS_DIR / "shap_importance_v2.csv"
        if shap_path.exists():
            shap_df = pd.read_csv(shap_path)
            viz.plot_shap_importance(shap_df)

    logger.info("Stage 4 complete in %.1fs", time.time() - t0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    t_total = time.time()

    logger.info("=" * 60)
    logger.info("AdaptiveBeta v2 Upgrade Pipeline")
    logger.info("fast=%s  regime=%s  start_stage=%d", args.fast, args.regime, args.start)
    logger.info("=" * 60)

    loader = DataLoader()

    stacked: pd.DataFrame = pd.DataFrame()
    regime:  pd.Series    = pd.Series(dtype=str)
    cfg:     dict         = {}

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    if args.start <= 1:
        stacked = stage1_features(loader)
    else:
        path = FEATURES_DIR / "stacked_features_v2.csv"
        if path.exists():
            logger.info("Loading stacked_features_v2.csv (skipping stage 1)")
            stacked = pd.read_csv(path, parse_dates=["date"]).set_index("date")
        else:
            logger.error("stacked_features_v2.csv not found — run from stage 1")
            sys.exit(1)

    # ── Stage 2 ──────────────────────────────────────────────────────────────
    if args.start <= 2:
        regime = stage2_regime(loader, mode=args.regime)
    else:
        path = FEATURES_DIR / "regime_labels_v2.csv"
        if path.exists():
            regime = pd.read_csv(path, parse_dates=["date"], index_col="date").squeeze()
            logger.info("Loaded regime_labels_v2.csv")
        else:
            logger.warning("regime_labels_v2.csv not found — using transition fallback")
            regime = pd.Series("transition", index=stacked.index)

    # ── Stage 3 ──────────────────────────────────────────────────────────────
    if args.start <= 3:
        cfg = stage3_models(stacked, regime, fast=args.fast)
    else:
        path = MODELS_DIR / "signal_config_v2.json"
        if path.exists():
            with open(path) as f:
                cfg = json.load(f)
            logger.info("Loaded signal_config_v2.json")
        else:
            logger.error("signal_config_v2.json not found — run from stage 3")
            sys.exit(1)

    # ── Stage 4 ──────────────────────────────────────────────────────────────
    if args.start <= 4:
        stage4_backtest(loader, regime, cfg, no_charts=args.no_charts)

    logger.info("=" * 60)
    logger.info("v2 Pipeline complete in %.0f seconds (%.1f min)",
                time.time() - t_total, (time.time() - t_total) / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()