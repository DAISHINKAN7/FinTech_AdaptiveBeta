"""Performance metrics for backtesting results.

Computes a comprehensive set of risk-adjusted return metrics:
- Sharpe ratio (annualised)
- Sortino ratio (downside deviation)
- Maximum drawdown
- Calmar ratio (return / max_dd)
- CAGR (compound annual growth rate)

Also includes stress event analysis and rolling metric computation.

Typical usage:
    from src.backtest.metrics import compute_metrics, stress_analysis
    metrics = compute_metrics(returns_series)
    stress = stress_analysis(all_returns_dict, STRESS_EVENTS)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.config import STRESS_EVENTS

logger = logging.getLogger(__name__)


def compute_metrics(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    trading_days_per_year: int = 252,
) -> dict[str, float]:
    """Compute comprehensive risk-adjusted performance metrics.

    Args:
        returns: Daily log returns series.
        benchmark: Optional benchmark returns for alpha/beta computation.
        trading_days_per_year: Annualisation factor (252 for equity markets).

    Returns:
        Dict of named metrics (all floats, percentages already multiplied by 100).
    """
    r = returns.dropna()
    if len(r) < 10:
        logger.warning("Less than 10 return observations — metrics may be unreliable")

    ann_factor = trading_days_per_year

    # Annualised return and volatility
    ann_ret = r.mean() * ann_factor
    ann_vol = r.std() * np.sqrt(ann_factor)

    # Sharpe ratio
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    # Sortino ratio (penalises only downside deviation)
    downside = r[r < 0].std() * np.sqrt(ann_factor)
    sortino = ann_ret / downside if downside > 0 else 0.0

    # CAGR
    n_years = len(r) / ann_factor
    total_return = np.exp(r.sum()) - 1
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Maximum drawdown
    cum_ret = np.exp(r.cumsum())
    rolling_max = cum_ret.cummax()
    drawdowns = (cum_ret - rolling_max) / rolling_max
    max_dd = float(drawdowns.min())

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    metrics: dict[str, float] = {
        "CAGR (%)": round(cagr * 100, 2),
        "Annual Return (%)": round(ann_ret * 100, 2),
        "Annual Volatility (%)": round(ann_vol * 100, 2),
        "Sharpe Ratio": round(sharpe, 3),
        "Sortino Ratio": round(sortino, 3),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Calmar Ratio": round(calmar, 3),
        "Total Return (%)": round(total_return * 100, 2),
        "N Days": len(r),
    }

    # Benchmark-relative metrics
    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        common = r.reindex(b.index).dropna()
        if len(common) >= 20:
            cov = np.cov(common.values, b.loc[common.index].values)
            strat_beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else np.nan
            strat_alpha = (common.mean() - strat_beta * b.loc[common.index].mean()) * ann_factor
            metrics["Alpha (%)"] = round(strat_alpha * 100, 2)
            metrics["Market Beta"] = round(float(strat_beta), 3)

    return metrics


def build_metrics_table(
    returns_dict: dict[str, pd.Series],
    benchmark_name: str = "buy_hold_nifty",
) -> pd.DataFrame:
    """Build a comparison table for multiple strategies.

    Args:
        returns_dict: Dict of strategy_name → daily returns Series.
        benchmark_name: Key in returns_dict to use as benchmark.

    Returns:
        DataFrame with strategies as rows and metrics as columns.
    """
    benchmark = returns_dict.get(benchmark_name)
    rows = {}
    for name, returns in returns_dict.items():
        rows[name] = compute_metrics(returns, benchmark=benchmark)
    df = pd.DataFrame(rows).T
    return df


def stress_analysis(
    returns_dict: dict[str, pd.Series],
    events: dict[str, tuple[str, str]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Analyse strategy performance during market stress events.

    Args:
        returns_dict: Dict of strategy_name → daily returns.
        events: Dict of event_name → (start_date, end_date).
            Defaults to STRESS_EVENTS from config.

    Returns:
        Dict of event_name → DataFrame with strategies as rows
        and cumulative_return / max_drawdown as columns.
    """
    if events is None:
        events = STRESS_EVENTS

    results = {}
    for event, (start, end) in events.items():
        event_rows = {}
        for strategy, returns in returns_dict.items():
            r = returns.loc[start:end]
            if len(r) == 0:
                continue
            cum = float(np.exp(r.sum()) - 1) * 100
            dd_series = (np.exp(r.cumsum()) / np.exp(r.cumsum()).cummax() - 1)
            max_dd = float(dd_series.min()) * 100
            event_rows[strategy] = {
                "Cumulative Return (%)": round(cum, 2),
                "Max Drawdown (%)": round(max_dd, 2),
            }
        results[event] = pd.DataFrame(event_rows).T
        logger.info("Stress event '%s': %d strategies analysed", event, len(event_rows))
    return results


def rolling_sharpe(returns: pd.Series, window: int = 252) -> pd.Series:
    """Compute rolling Sharpe ratio.

    Args:
        returns: Daily return series.
        window: Rolling window in trading days.

    Returns:
        Rolling Sharpe ratio series.
    """
    roll_mean = returns.rolling(window).mean() * 252
    roll_std = returns.rolling(window).std() * np.sqrt(252)
    return roll_mean / roll_std.replace(0, np.nan)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute the full drawdown time series.

    Args:
        returns: Daily log return series.

    Returns:
        Drawdown series in percentage terms (always ≤ 0).
    """
    cum = np.exp(returns.cumsum())
    dd = (cum / cum.cummax() - 1) * 100
    return dd
