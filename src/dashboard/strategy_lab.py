"""Recommended strategy lab module placeholder.

Suggested future wrapper around project-native portfolio logic for an
interactive Dash demo.
"""

REUSABLE_CORE_OBJECTS = [
    "src.portfolio.signal.SignalGenerator",
    "src.portfolio.signal.SignalState",
    "src.portfolio.optimiser.PortfolioOptimiser",
    "src.backtest.metrics.compute_metrics",
    "src.backtest.metrics.stress_analysis",
]

DEMO_IDEAS = [
    "Threshold slider for betavol trigger quantile",
    "VIX stress override slider",
    "Regime-aware routing explainer",
    "Interactive optimisation mode comparison",
    "Date-based crisis replay with real project data",
]