"""Matplotlib / Plotly chart functions for the AdaptiveBeta pipeline.

All visualisation code lives here — notebooks import from this module
to keep notebook cells clean and charts consistent.

Typical usage:
    from src.utils.plotting import plot_equity_curves, plot_drawdowns, plot_beta_panel
    fig = plot_equity_curves(returns_dict)
    fig.savefig("results/equity_curves.png", dpi=150)
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Palette consistent with the website design system
PALETTE = {
    "adaptive_beta": "#1D9E75",
    "static_capm_mvo": "#7F77DD",
    "kalman_mvo": "#EF9F27",
    "equal_weight": "#888780",
    "buy_hold_nifty": "#D85A30",
    "bull": "#1D9E75",
    "bear": "#D85A30",
    "transition": "#888780",
    "high-vol": "#EF9F27",
}


def _mpl_available() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def _plotly_available() -> bool:
    try:
        import plotly  # noqa: F401
        return True
    except ImportError:
        return False


def plot_equity_curves(
    returns_dict: Dict[str, pd.Series],
    base_value: float = 100.0,
    title: str = "Cumulative Returns (Walk-Forward OOS)",
    figsize: tuple = (14, 7),
):
    """Plot cumulative equity curves for multiple strategies.

    Args:
        returns_dict: Dict of strategy_name → daily return Series.
        base_value: Starting portfolio value for normalisation.
        title: Chart title.
        figsize: Figure size (width, height) in inches.

    Returns:
        Matplotlib Figure, or Plotly Figure if matplotlib unavailable.
    """
    if not _mpl_available():
        logger.warning("matplotlib not available — skipping equity curve plot")
        return None

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0A0F1E")
    ax.set_facecolor("#0F1629")

    for name, returns in returns_dict.items():
        cum = np.exp(returns.cumsum()) * base_value
        color = PALETTE.get(name, "#AAAAAA")
        lw = 2.5 if name == "adaptive_beta" else 1.5
        ax.plot(cum.index, cum.values, label=name.replace("_", " ").title(), color=color, lw=lw)

    ax.axhline(base_value, color="#444", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.set_ylabel("Portfolio Value", color="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
    ax.legend(facecolor="#141D3A", labelcolor="white", framealpha=0.8)
    fig.tight_layout()
    return fig


def plot_drawdowns(
    returns_dict: Dict[str, pd.Series],
    title: str = "Drawdown Analysis",
    figsize: tuple = (14, 5),
):
    """Plot drawdown time series for multiple strategies.

    Args:
        returns_dict: Dict of strategy_name → daily return Series.
        title: Chart title.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    if not _mpl_available():
        return None

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0A0F1E")
    ax.set_facecolor("#0F1629")

    for name, returns in returns_dict.items():
        cum = np.exp(returns.cumsum())
        dd = (cum / cum.cummax() - 1) * 100
        color = PALETTE.get(name, "#AAAAAA")
        lw = 2.0 if name == "adaptive_beta" else 1.2
        ax.fill_between(dd.index, dd.values, 0, alpha=0.25, color=color)
        ax.plot(dd.index, dd.values, label=name.replace("_", " ").title(), color=color, lw=lw)

    ax.set_title(title, color="white", fontsize=14, pad=12)
    ax.set_ylabel("Drawdown (%)", color="white")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(facecolor="#141D3A", labelcolor="white", framealpha=0.8)
    fig.tight_layout()
    return fig


def plot_beta_panel(
    beta_panel: pd.DataFrame,
    ticker: str,
    vix: Optional[pd.Series] = None,
    figsize: tuple = (14, 6),
):
    """Plot rolling beta for a single stock across multiple windows.

    Args:
        beta_panel: Dict of window → DataFrame or single DataFrame with window columns.
        ticker: Stock to plot.
        vix: Optional VIX series for secondary axis overlay.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    if not _mpl_available():
        return None

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0A0F1E")
    ax1.set_facecolor("#0F1629")

    colors = ["#1D9E75", "#7F77DD", "#EF9F27"]
    windows = [30, 60, 120]

    if isinstance(beta_panel, dict):
        for w, color in zip(windows, colors):
            if w in beta_panel:
                series = beta_panel[w][ticker] if ticker in beta_panel[w].columns else pd.Series(dtype=float)
                ax1.plot(series.index, series.values, label=f"Beta {w}d", color=color, lw=1.5)
    else:
        ax1.plot(beta_panel.index, beta_panel.values, color="#1D9E75", label="Beta", lw=1.5)

    ax1.axhline(1.0, color="#555", lw=0.8, ls="--", alpha=0.6, label="Beta=1.0")
    ax1.set_ylabel("Rolling Beta", color="white")
    ax1.tick_params(colors="white")
    ax1.set_title(f"Rolling Beta — {ticker}", color="white", fontsize=13)

    if vix is not None:
        ax2 = ax1.twinx()
        ax2.plot(vix.index, vix.values, color="#D85A30", alpha=0.4, lw=1.0, label="VIX")
        ax2.set_ylabel("India VIX", color="#D85A30")
        ax2.tick_params(axis="y", colors="#D85A30")

    ax1.legend(facecolor="#141D3A", labelcolor="white", framealpha=0.8, loc="upper left")
    ax1.spines["bottom"].set_color("#333")
    ax1.spines["left"].set_color("#333")
    ax1.spines["top"].set_visible(False)
    fig.tight_layout()
    return fig


def plot_regime_timeline(
    regime_series: pd.Series,
    market_ret: Optional[pd.Series] = None,
    figsize: tuple = (14, 4),
):
    """Plot a colour-coded regime timeline.

    Args:
        regime_series: Series with 'bull'/'bear'/'transition' values.
        market_ret: Optional cumulative market returns for overlay.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    if not _mpl_available():
        return None

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0A0F1E")
    ax.set_facecolor("#0F1629")

    regime_colors = {"bull": "#1D9E75", "bear": "#D85A30", "transition": "#888780"}

    # Shade background by regime
    prev_date = regime_series.index[0]
    prev_regime = regime_series.iloc[0]
    for date, regime in regime_series.items():
        if regime != prev_regime:
            color = regime_colors.get(str(prev_regime), "#888780")
            ax.axvspan(prev_date, date, alpha=0.25, color=color)
            prev_date = date
            prev_regime = regime
    ax.axvspan(prev_date, regime_series.index[-1], alpha=0.25,
               color=regime_colors.get(str(prev_regime), "#888780"))

    if market_ret is not None:
        cum = np.exp(market_ret.cumsum()) * 100
        ax.plot(cum.index, cum.values, color="white", lw=1.5, label="NIFTY50")
        ax.set_ylabel("NIFTY50 (base=100)", color="white")

    legend_elements = [Patch(facecolor=c, alpha=0.6, label=r) for r, c in regime_colors.items()]
    ax.legend(handles=legend_elements, facecolor="#141D3A", labelcolor="white")
    ax.set_title("Market Regime Classification (HMM)", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig
