"""run_pipeline.py — Master runner for the AdaptiveBeta pipeline.

Executes all four scripts in sequence, logging each step's output to
results/pipeline_run.log in addition to the terminal.

Usage:
    python run_pipeline.py                   # full pipeline
    python run_pipeline.py --fast            # skip Kalman betas
    python run_pipeline.py --start 2         # start from script 02
    python run_pipeline.py --only 1          # run only script 01
    python run_pipeline.py --device cuda     # use GPU for LSTM
    python run_pipeline.py --fast --device cpu --start 3

Flags:
    --fast       Pass --fast to scripts 02 and 04 (skip Kalman, faster LSTM)
    --start N    Start from script N (1–4). Useful when resuming after a failure.
    --only N     Run only script N.
    --device     PyTorch device: auto | cpu | cuda | mps (default: auto)
    --epochs N   Override LSTM epochs (passed to script 02)
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / "pipeline_run.log"

# ---------------------------------------------------------------------------
# Logging to both terminal and file
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Script definitions
# ---------------------------------------------------------------------------
SCRIPTS = {
    1: {
        "path": "scripts/01_feature_engineering.py",
        "name": "Feature Engineering",
        "desc": "Aligns data, computes rolling betas, builds feature matrices",
        "extra_args": [],
    },
    2: {
        "path": "scripts/02_train_models.py",
        "name": "Model Training",
        "desc": "XGBoost, Kalman, LSTM, HMM — trains all models",
        "extra_args": [],   # device, epochs, fast injected at runtime
    },
    3: {
        "path": "scripts/03_portfolio_optimisation.py",
        "name": "Portfolio Optimisation",
        "desc": "Validates optimisers, plots signal threshold",
        "extra_args": [],
    },
    4: {
        "path": "scripts/04_backtest.py",
        "name": "Walk-Forward Backtest",
        "desc": "6-year OOS backtest, performance tables, stress analysis",
        "extra_args": [],   # fast, device injected at runtime
    },
}


# ---------------------------------------------------------------------------
def check_data_directory() -> bool:
    """Verify that the minimum required data files exist."""
    required = [
        "data/stocks/all_stocks_prices.csv",
        "data/market/nifty50.csv",
    ]
    missing = []
    for rel in required:
        if not (PROJECT_ROOT / rel).exists():
            missing.append(rel)

    if missing:
        logger.error("Missing required data files:")
        for m in missing:
            logger.error("  ✗  %s", m)
        logger.error("")
        logger.error("Put your CSV files under adaptivebeta/data/")
        logger.error("See README.md for the full directory structure.")
        return False

    logger.info("Data files found — OK")
    return True


def run_script(
    script_num: int,
    extra_args: list[str],
    timeout: int = 7200,
) -> bool:
    """Run a single script as a subprocess.

    Args:
        script_num: Script number (1–4).
        extra_args: Additional CLI arguments for the script.
        timeout: Maximum seconds to wait (default 2 hours).

    Returns:
        True if the script exited with code 0.
    """
    info = SCRIPTS[script_num]
    script_path = PROJECT_ROOT / info["path"]

    if not script_path.exists():
        logger.error("Script not found: %s", script_path)
        return False

    cmd = [sys.executable, str(script_path)] + info["extra_args"] + extra_args

    logger.info("")
    logger.info("─" * 60)
    logger.info("STEP %d / 4 — %s", script_num, info["name"])
    logger.info("  %s", info["desc"])
    logger.info("  Command: %s", " ".join(cmd))
    logger.info("─" * 60)

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.error("Script %d timed out after %d seconds", script_num, timeout)
        return False
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user at script %d", script_num)
        raise

    elapsed = time.time() - t0
    if result.returncode == 0:
        logger.info("✓  Script %d completed in %.1f seconds", script_num, elapsed)
        return True
    else:
        logger.error("✗  Script %d failed (exit code %d) after %.1f seconds",
                     script_num, result.returncode, elapsed)
        return False


# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AdaptiveBeta master pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--fast", action="store_true",
                   help="Skip Kalman betas (scripts 02 + 04 run faster)")
    p.add_argument("--start", type=int, default=1, choices=[1, 2, 3, 4],
                   help="Start from this script number (default: 1)")
    p.add_argument("--only", type=int, choices=[1, 2, 3, 4],
                   help="Run only this script")
    p.add_argument("--device", default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="PyTorch device (default: auto)")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override LSTM epoch count in script 02")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("AdaptiveBeta Pipeline — %s", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("  fast=%s  device=%s  start=%s  only=%s",
                args.fast, args.device,
                args.only if args.only else args.start,
                args.only)
    logger.info("  Log file: %s", LOG_FILE)
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Data check (always, unless skipping script 1)
    # ------------------------------------------------------------------
    start_at = args.only if args.only else args.start
    if start_at == 1:
        if not check_data_directory():
            sys.exit(1)

    # ------------------------------------------------------------------
    # Build script list
    # ------------------------------------------------------------------
    if args.only:
        run_order = [args.only]
    else:
        run_order = list(range(args.start, 5))

    # Inject per-script extra args
    # Script 02
    s2_args = [f"--device={args.device}"]
    if args.fast:
        s2_args.append("--fast")
    if args.epochs:
        s2_args.append(f"--epochs={args.epochs}")
    SCRIPTS[2]["extra_args"] = s2_args

    # Script 04
    s4_args = [f"--device={args.device}"]
    if args.fast:
        s4_args.append("--fast")
    SCRIPTS[4]["extra_args"] = s4_args

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
    failed_at = None
    for script_num in run_order:
        success = run_script(script_num, extra_args=[])
        if not success:
            failed_at = script_num
            break

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    elapsed_total = (datetime.now() - start_time).total_seconds()
    logger.info("")
    logger.info("=" * 60)
    if failed_at is None:
        logger.info("✓  PIPELINE COMPLETE in %.0f seconds (%.1f min)",
                    elapsed_total, elapsed_total / 60)
        logger.info("")
        logger.info("Output files:")
        logger.info("  features/  ← beta panels, stacked features, regimes")
        logger.info("  models/    ← lstm_best.pt, scaler.pkl, signal_config.json")
        logger.info("  results/   ← performance_metrics.csv, equity_curves.png, ...")
        logger.info("")
        logger.info("Next: streamlit run dashboard.py")
    else:
        logger.error("✗  PIPELINE FAILED at script %d", failed_at)
        logger.error("Fix the error above, then resume with:")
        logger.error("  python run_pipeline.py --start %d", failed_at)
        sys.exit(1)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
