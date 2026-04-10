"""Recommended dashboard data service module placeholder.

This file is intentionally descriptive only. It documents the proposed
modular split for a future Dash-based Python demo without changing runtime
behaviour elsewhere in the project.
"""

RECOMMENDED_RESPONSIBILITIES = [
    "Load strategy returns from results/strategy_returns.csv or per-strategy CSVs",
    "Load performance metrics, stress analysis, rebalance logs, and model comparison",
    "Load India VIX raw data from data/ via src.config RAW_FILES mapping",
    "Load feature artifacts such as features/betavol_60d.csv and features/hmm_regimes.csv",
    "Expose normalized data payloads for Dash callbacks and dcc.Store objects",
    "Reuse transformation ideas from scripts/05_export_demo_data.py",
]

FRAMEWORK_FIT = "Dash + Plotly is the recommended Python-native dashboard stack for this project."