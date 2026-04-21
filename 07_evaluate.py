"""
Quick evaluation script — loads existing backtest results and prints
the full evaluation report + answers to the two research questions.

Run after stage 4 completes:
    python scripts/07_evaluate.py
    python scripts/07_evaluate.py --results v2      # use v2 results
    python scripts/07_evaluate.py --results base    # use original results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RESULTS_DIR, FEATURES_DIR
from src.backtest.evaluation import EvaluationReport


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AdaptiveBeta evaluation report")
    p.add_argument("--results", default="v2", choices=["v2", "base"],
                   help="Which results to load (default: v2)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    suffix = "_v2" if args.results == "v2" else ""

    returns_path = RESULTS_DIR / f"strategy_returns{suffix}.csv"
    if not returns_path.exists():
        # Fall back to base
        returns_path = RESULTS_DIR / "strategy_returns.csv"
    if not returns_path.exists():
        print(f"[ERROR] Returns CSV not found at {returns_path}")
        print("Run the pipeline first:  python run_pipeline.py")
        sys.exit(1)

    df = pd.read_csv(returns_path, parse_dates=["date"]).set_index("date").sort_index()
    returns_dict = {col: df[col].dropna() for col in df.columns}

    # Load regime
    regime = None
    regime_path = FEATURES_DIR / "regime_labels_v2.csv"
    if regime_path.exists():
        regime = pd.read_csv(regime_path, parse_dates=["date"], index_col="date").squeeze()

    report = EvaluationReport(returns_dict, benchmark="buy_hold_nifty")
    report.print_full()
    report.answer_research_questions(regime=regime)
    report.export(str(RESULTS_DIR / f"evaluation_report{suffix}.csv"))


if __name__ == "__main__":
    main()