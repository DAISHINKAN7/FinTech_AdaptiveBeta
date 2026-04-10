"""
AdaptiveBeta — Premium Python-Native Quant Research Dashboard
Framework: Dash + Plotly + dash-bootstrap-components
Design:    Institutional dark research workstation | Real NIFTY50 data

Run:  python app.py
URL:  http://localhost:8050
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dash_table, dcc, html
from plotly.subplots import make_subplots

from src.config import DATA_DIR, FEATURES_DIR, RAW_FILES, RESULTS_DIR, STRESS_EVENTS
from src.portfolio.signal import SignalGenerator

os.environ["DASH_JUPYTER_MODE"] = "false"

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
PUBLIC_DEMO = ROOT / "website" / "public" / "demo-data.json"

PALETTE: Dict[str, str] = {
    "adaptive_beta": "#00D4AA",
    "static_capm_mvo": "#818CF8",
    "kalman_mvo": "#8B5CF6",
    "equal_weight": "#94A3B8",
    "buy_hold_nifty": "#F59E0B",
    "momentum_quality": "#34D399",
    "rebalance": "#FBBF24",
    "min_variance": "#FB7185",
    "hold": "#64748B",
    "bull": "#00D4AA",
    "bear": "#F87171",
    "transition": "#A78BFA",
}
LABELS: Dict[str, str] = {
    "adaptive_beta": "AdaptiveBeta (Ours)",
    "static_capm_mvo": "Static CAPM MVO",
    "kalman_mvo": "Kalman-Beta MVO",
    "equal_weight": "Equal Weight",
    "buy_hold_nifty": "Buy & Hold NIFTY50",
    "momentum_quality": "Momentum-Quality",
}
STRATEGIES = list(LABELS.keys())

BG = "#040816"
BG2 = "#091224"
CARD = "rgba(9, 18, 36, 0.90)"
CARD_ALT = "rgba(6, 13, 28, 0.88)"
BORDER = "rgba(0, 212, 170, 0.18)"
BORDER_SOFT = "rgba(148, 163, 184, 0.12)"
TEXT = "#E5EEF8"
MUTED = "#90A3BF"
ACCENT = "#00D4AA"
AMBER = "#FBBF24"
RED = "#FB7185"
PURPLE = "#A78BFA"
BLUE = "#60A5FA"

CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#07111F",
    font=dict(color=TEXT, family="'Inter','DM Sans',system-ui,sans-serif", size=11),
    margin=dict(l=54, r=22, t=54, b=42),
    hovermode="x unified",
    legend=dict(
        bgcolor="rgba(4,8,22,0.92)",
        bordercolor="#182338",
        borderwidth=1,
        font=dict(size=10, color=MUTED),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1.0,
    ),
)
XAXIS = dict(
    gridcolor="#102038",
    linecolor="#1B2A45",
    tickfont=dict(color=MUTED, size=10),
    title_font=dict(color=MUTED),
    zeroline=False,
)
YAXIS = dict(
    gridcolor="#102038",
    linecolor="#1B2A45",
    tickfont=dict(color=MUTED, size=10),
    title_font=dict(color=MUTED),
    zeroline=False,
)


def hex_to_rgba(hx: str, alpha: float) -> str:
    value = hx.lstrip("#")
    r, g, b = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}f}%"


def fmt_num(value: float | None, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def _card(children, extra_style: dict | None = None):
    style = {
        "background": CARD,
        "border": f"1px solid {BORDER}",
        "borderRadius": "18px",
        "padding": "20px",
        "backdropFilter": "blur(16px)",
        "boxShadow": "0 24px 80px rgba(0,0,0,0.22)",
        **(extra_style or {}),
    }
    return html.Div(children, style=style)


def _section_title(eyebrow: str, title: str, subtitle: str):
    return html.Div(
        [
            html.Div(
                eyebrow,
                style={
                    "color": ACCENT,
                    "fontSize": "10px",
                    "textTransform": "uppercase",
                    "letterSpacing": "1.6px",
                    "fontWeight": "700",
                    "marginBottom": "8px",
                },
            ),
            html.H3(
                title,
                style={
                    "color": TEXT,
                    "fontSize": "22px",
                    "margin": "0 0 8px 0",
                    "fontWeight": "700",
                    "letterSpacing": "-0.4px",
                },
            ),
            html.P(
                subtitle,
                style={"color": MUTED, "fontSize": "12px", "margin": 0, "lineHeight": "1.7"},
            ),
        ],
        style={"marginBottom": "18px"},
    )


def annualized_metrics(log_returns: pd.Series, rf_annual: float = 0.065) -> dict:
    r = pd.Series(log_returns).dropna()
    if len(r) < 5:
        return {
            "cagr": None,
            "sharpe": None,
            "sortino": None,
            "maxDrawdown": None,
            "calmar": None,
            "annualVol": None,
            "totalReturn": None,
            "tradingDays": len(r),
        }
    n_years = len(r) / 252
    total_return = float(np.exp(r.sum()) - 1)
    cagr = float((1 + total_return) ** (1 / max(n_years, 0.1)) - 1) * 100
    annual_vol = float(r.std() * np.sqrt(252)) * 100
    rf_daily = rf_annual / 252
    excess = r - rf_daily
    sharpe = float(excess.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    downside = r[r < 0]
    sortino = float(excess.mean() / downside.std() * np.sqrt(252)) if len(downside) > 1 else 0.0
    equity = np.exp(r.cumsum())
    running_max = equity.cummax()
    dd = (equity / running_max - 1) * 100
    max_dd = float(dd.min())
    calmar = abs(cagr / max_dd) if max_dd < 0 else 0.0
    return {
        "cagr": round(cagr, 3),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "maxDrawdown": round(max_dd, 3),
        "calmar": round(calmar, 4),
        "annualVol": round(annual_vol, 3),
        "totalReturn": round(total_return * 100, 2),
        "tradingDays": len(r),
    }


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def load_demo_payload() -> dict:
    if PUBLIC_DEMO.exists():
        with open(PUBLIC_DEMO, "r") as f:
            return json.load(f)
    return {}


def load_strategy_returns() -> pd.DataFrame:
    combined = RESULTS_DIR / "strategy_returns.csv"
    if combined.exists():
        df = pd.read_csv(combined, parse_dates=["date"]).set_index("date").sort_index()
        return df
    frames = []
    for strategy in STRATEGIES:
        path = RESULTS_DIR / f"{strategy}_returns.csv"
        if path.exists():
            series_df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
            col = series_df.columns[0]
            frames.append(series_df.rename(columns={col: strategy}))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1).sort_index()


def load_performance_metrics() -> pd.DataFrame:
    path = RESULTS_DIR / "performance_metrics.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0)
        return df
    return pd.DataFrame()


def load_stress_analysis() -> pd.DataFrame:
    path = RESULTS_DIR / "stress_analysis.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_rebalance_log() -> pd.DataFrame:
    path = RESULTS_DIR / "rebalance_log.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])
    return pd.DataFrame()


def load_prices() -> pd.DataFrame:
    path = DATA_DIR / RAW_FILES["prices"]
    if not path.exists():
        return pd.DataFrame()
    prices = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    prices.columns = [c.replace("_NS", ".NS") for c in prices.columns]
    return prices


def load_vix_series(index: pd.DatetimeIndex) -> pd.Series:
    path = DATA_DIR / RAW_FILES["india_vix"]
    if not path.exists():
        return pd.Series(18.0, index=index, name="vix")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    close_cols = [c for c in df.columns if c.lower().startswith("close")]
    if not close_cols:
        return pd.Series(18.0, index=index, name="vix")
    series = df[close_cols[0]].astype(float).reindex(index).ffill().bfill().fillna(18.0)
    series.name = "vix"
    return series


def load_betavol_series(index: pd.DatetimeIndex) -> pd.Series:
    path = FEATURES_DIR / "betavol_60d.csv"
    if not path.exists():
        return pd.Series(0.12, index=index, name="betavol")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    series = df.mean(axis=1).reindex(index).ffill().bfill()
    series = series.fillna(series.median() if not series.dropna().empty else 0.12)
    series.name = "betavol"
    return series


def load_regime_series(index: pd.DatetimeIndex) -> pd.Series:
    path = FEATURES_DIR / "hmm_regimes.csv"
    if not path.exists():
        return pd.Series("transition", index=index, name="regime")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    col = "regime_label" if "regime_label" in df.columns else df.columns[0]
    series = df[col].astype(str).str.lower().reindex(index).ffill().fillna("transition")
    series.name = "regime"
    return series


def load_beta_snapshot() -> pd.Series:
    path = FEATURES_DIR / "beta60d.csv"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    latest = df.iloc[-1].dropna()
    latest.index = [c.replace("_NS", ".NS") for c in latest.index]
    return latest.astype(float)


def load_signal_thresholds(betavol_series: pd.Series) -> dict:
    clean = betavol_series.dropna()
    if clean.empty:
        return {str(p): 0.12 for p in range(40, 95, 5)}
    return {str(p): round(float(np.percentile(clean, p)), 6) for p in range(40, 95, 5)}


def build_research_dataset() -> dict:
    returns_df = load_strategy_returns()
    demo_payload = load_demo_payload()

    if not returns_df.empty:
        index = returns_df.index
        source = "live"
    elif demo_payload and demo_payload.get("dates"):
        index = pd.to_datetime(demo_payload["dates"])
        source = "demo-json"
        returns_df = pd.DataFrame(
            {
                k: pd.Series(v, index=index)
                for k, v in demo_payload.get("returns", {}).items()
                if k in STRATEGIES
            }
        )
    else:
        index = pd.date_range("2018-01-01", "2025-12-31", freq="B")
        source = "synthetic"
        rng = np.random.default_rng(42)
        returns_df = pd.DataFrame(index=index)
        base_params = {
            "adaptive_beta": (0.092 / 252, 0.112 / np.sqrt(252)),
            "static_capm_mvo": (0.114 / 252, 0.184 / np.sqrt(252)),
            "kalman_mvo": (0.109 / 252, 0.184 / np.sqrt(252)),
            "equal_weight": (0.159 / 252, 0.170 / np.sqrt(252)),
            "buy_hold_nifty": (0.122 / 252, 0.172 / np.sqrt(252)),
            "momentum_quality": (0.175 / 252, 0.180 / np.sqrt(252)),
        }
        for name, (mu, sigma) in base_params.items():
            r = rng.normal(mu, sigma, len(index))
            returns_df[name] = r

    returns_df = returns_df.sort_index()
    returns_df = returns_df[[c for c in STRATEGIES if c in returns_df.columns]]

    vix = load_vix_series(returns_df.index)
    betavol = load_betavol_series(returns_df.index)
    regime = load_regime_series(returns_df.index)
    prices = load_prices()
    beta_snapshot = load_beta_snapshot()
    thresholds = load_signal_thresholds(betavol)
    metrics_csv = load_performance_metrics()
    stress_csv = load_stress_analysis()
    rebalance_log = load_rebalance_log()

    exported_metrics = demo_payload.get("metrics", {}) if demo_payload else {}
    exported_crisis = demo_payload.get("crisisStats", {}) if demo_payload else {}

    metrics = {}
    for strategy in returns_df.columns:
        if strategy in exported_metrics:
            metrics[strategy] = exported_metrics[strategy]
        else:
            metrics[strategy] = annualized_metrics(returns_df[strategy])

    equity_curves = {}
    for strategy in returns_df.columns:
        equity_curves[strategy] = 100 * np.exp(returns_df[strategy].fillna(0).cumsum())

    if not exported_crisis and "adaptive_beta" in returns_df.columns and "buy_hold_nifty" in returns_df.columns:
        exported_crisis = {}
        for event_name, (start, end) in STRESS_EVENTS.items():
            p = returns_df["adaptive_beta"].loc[start:end].dropna()
            n = returns_df["buy_hold_nifty"].loc[start:end].dropna()
            if p.empty or n.empty:
                continue
            p_eq = np.exp(p.cumsum())
            n_eq = np.exp(n.cumsum())
            exported_crisis[event_name] = {
                "portfolioReturn": round((np.exp(p.sum()) - 1) * 100, 2),
                "niftyReturn": round((np.exp(n.sum()) - 1) * 100, 2),
                "portfolioMaxDD": round(((p_eq / p_eq.cummax()) - 1).min() * 100, 2),
                "niftyMaxDD": round(((n_eq / n_eq.cummax()) - 1).min() * 100, 2),
                "protection": round(((np.exp(p.sum()) - 1) - (np.exp(n.sum()) - 1)) * 100, 2),
                "rebalances": int((betavol.loc[start:end] > float(thresholds.get("70", 0.15))).sum()),
                "minVarDays": int((vix.loc[start:end] > 22).sum()),
            }

    return {
        "source": source,
        "dates": returns_df.index.strftime("%Y-%m-%d").tolist(),
        "returns": returns_df,
        "metrics": metrics,
        "equity_curves": equity_curves,
        "vix": vix,
        "betavol": betavol,
        "regime": regime,
        "thresholds": thresholds,
        "prices": prices,
        "beta_snapshot": beta_snapshot,
        "stress_csv": stress_csv,
        "rebalance_log": rebalance_log,
        "metrics_csv": metrics_csv,
        "crisis_stats": exported_crisis,
    }


def compute_strategy_lab(dataset: dict, threshold_pct: int, vix_threshold: float, transaction_bps: int, regime_routing: bool) -> dict:
    returns_df: pd.DataFrame = dataset["returns"]
    if returns_df.empty or "buy_hold_nifty" not in returns_df.columns:
        return {}

    dates = pd.to_datetime(dataset["dates"])
    vix: pd.Series = dataset["vix"].reindex(dates)
    betavol: pd.Series = dataset["betavol"].reindex(dates)
    regime: pd.Series = dataset["regime"].reindex(dates)
    thresholds: dict = dataset["thresholds"]

    threshold_value = float(thresholds.get(str(threshold_pct), betavol.quantile(threshold_pct / 100)))
    sig_gen = SignalGenerator(threshold=threshold_value, vix_threshold=float(vix_threshold))

    selected_returns = []
    signal_list = []
    mode_list = []
    rebalance_count = 0
    min_var_count = 0
    turnover_proxy = 0.0

    available_dynamic = None
    for candidate in ["adaptive_beta", "equal_weight", "buy_hold_nifty"]:
        if candidate in returns_df.columns:
            available_dynamic = candidate
            break

    for i, date in enumerate(dates):
        predicted_betavol = float(betavol.iloc[i]) if i < len(betavol) else threshold_value
        vix_level = float(vix.iloc[i]) if i < len(vix) else 18.0
        current_regime = str(regime.iloc[i]).lower() if i < len(regime) else "transition"

        state = sig_gen.evaluate(predicted_betavol, vix_level, current_regime)
        signal_name = state.signal.lower()
        mode_name = state.mode

        if not regime_routing and signal_name == "rebalance":
            mode_name = "max_sharpe_beta_constrained"

        if mode_name == "min_variance":
            chosen = "adaptive_beta" if "adaptive_beta" in returns_df.columns else available_dynamic
            selected_ret = returns_df[chosen].iloc[i] - (transaction_bps / 10000 / 252 if signal_name != "hold" else 0)
            min_var_count += 1
        elif mode_name == "risk_parity":
            chosen = "equal_weight" if "equal_weight" in returns_df.columns else available_dynamic
            selected_ret = returns_df[chosen].iloc[i] - (transaction_bps / 10000 / 252 if signal_name == "rebalance" else 0)
        elif mode_name == "max_sharpe_beta_constrained":
            chosen = "adaptive_beta" if "adaptive_beta" in returns_df.columns else available_dynamic
            selected_ret = returns_df[chosen].iloc[i] - (transaction_bps / 10000 / 252 if signal_name == "rebalance" else 0)
        else:
            chosen = "buy_hold_nifty" if "buy_hold_nifty" in returns_df.columns else available_dynamic
            selected_ret = returns_df[chosen].iloc[i]

        if signal_name == "rebalance":
            rebalance_count += 1
            turnover_proxy += 0.35
        elif signal_name == "min_variance":
            turnover_proxy += 0.18

        selected_returns.append(float(selected_ret))
        signal_list.append(signal_name)
        mode_list.append(mode_name)

    sim_returns = pd.Series(selected_returns, index=dates, name="strategy_lab")
    nifty_returns = returns_df["buy_hold_nifty"].reindex(dates).fillna(0)
    adaptive_returns = returns_df["adaptive_beta"].reindex(dates).fillna(0) if "adaptive_beta" in returns_df.columns else sim_returns

    strategy_metrics = annualized_metrics(sim_returns)
    benchmark_metrics = annualized_metrics(nifty_returns)
    adaptive_metrics = annualized_metrics(adaptive_returns)

    sim_equity = 100 * np.exp(sim_returns.cumsum())
    nifty_equity = 100 * np.exp(nifty_returns.cumsum())
    adaptive_equity = 100 * np.exp(adaptive_returns.cumsum())

    event_rows = []
    for event_name, (start, end) in STRESS_EVENTS.items():
        sim_slice = sim_returns.loc[start:end]
        nifty_slice = nifty_returns.loc[start:end]
        if sim_slice.empty or nifty_slice.empty:
            continue
        sim_eq = np.exp(sim_slice.cumsum())
        nifty_eq = np.exp(nifty_slice.cumsum())
        event_rows.append(
            {
                "event": event_name,
                "strategyReturn": round((np.exp(sim_slice.sum()) - 1) * 100, 2),
                "benchmarkReturn": round((np.exp(nifty_slice.sum()) - 1) * 100, 2),
                "protection": round(((np.exp(sim_slice.sum()) - 1) - (np.exp(nifty_slice.sum()) - 1)) * 100, 2),
                "strategyMaxDD": round(((sim_eq / sim_eq.cummax()) - 1).min() * 100, 2),
                "benchmarkMaxDD": round(((nifty_eq / nifty_eq.cummax()) - 1).min() * 100, 2),
            }
        )

    summary = {
        "thresholdValue": threshold_value,
        "rebalanceCount": rebalance_count,
        "minVarianceDays": min_var_count,
        "holdCount": int(sum(1 for s in signal_list if s == "hold")),
        "regimeRouting": regime_routing,
        "turnoverProxy": round(turnover_proxy, 2),
        "outperformance": round((strategy_metrics["cagr"] or 0) - (benchmark_metrics["cagr"] or 0), 2),
        "drawdownImprovement": round(abs(benchmark_metrics["maxDrawdown"] or 0) - abs(strategy_metrics["maxDrawdown"] or 0), 2),
    }

    latest_date = dates[-1]
    latest_state = {
        "date": latest_date.strftime("%Y-%m-%d"),
        "signal": signal_list[-1],
        "mode": mode_list[-1],
        "regime": str(regime.loc[latest_date]),
        "vix": round(float(vix.loc[latest_date]), 2),
        "betavol": round(float(betavol.loc[latest_date]), 4),
        "threshold": round(float(threshold_value), 4),
    }

    return {
        "dates": dataset["dates"],
        "strategy_returns": [round(float(x), 8) for x in sim_returns],
        "nifty_returns": [round(float(x), 8) for x in nifty_returns],
        "adaptive_returns": [round(float(x), 8) for x in adaptive_returns],
        "strategy_equity": [round(float(x), 4) for x in sim_equity],
        "nifty_equity": [round(float(x), 4) for x in nifty_equity],
        "adaptive_equity": [round(float(x), 4) for x in adaptive_equity],
        "signals": signal_list,
        "modes": mode_list,
        "strategy_metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics,
        "adaptive_metrics": adaptive_metrics,
        "event_rows": event_rows,
        "summary": summary,
        "latest_state": latest_state,
    }


def fig_equity_curves(dataset: dict, selected: List[str]) -> go.Figure:
    fig = go.Figure()
    for event_name, (start, end) in STRESS_EVENTS.items():
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor="rgba(248,113,113,0.06)" if "Crash" in event_name else "rgba(167,139,250,0.05)",
            line_width=0,
            annotation_text=event_name,
            annotation_position="top left",
            annotation_font=dict(color=MUTED, size=9),
        )

    for strategy in selected:
        if strategy not in dataset["equity_curves"]:
            continue
        series = dataset["equity_curves"][strategy]
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=LABELS.get(strategy, strategy),
                line=dict(color=PALETTE.get(strategy, "#CBD5E1"), width=2.8 if strategy == "adaptive_beta" else 1.7),
                hovertemplate=f"<b>{LABELS.get(strategy, strategy)}</b><br>%{{x|%d %b %Y}}<br>Index: %{{y:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        **CHART_BASE,
        height=430,
        xaxis=XAXIS,
        yaxis={**YAXIS, "title": "Portfolio Value (Base = 100)"},
        title=dict(text="<b>Walk-Forward Equity Curves</b>", x=0.01, font=dict(color=TEXT, size=15)),
    )
    return fig


def fig_drawdown(dataset: dict, selected: List[str]) -> go.Figure:
    fig = go.Figure()
    for strategy in selected:
        if strategy not in dataset["returns"]:
            continue
        r = dataset["returns"][strategy].fillna(0)
        eq = np.exp(r.cumsum())
        dd = (eq / eq.cummax() - 1) * 100
        color = PALETTE.get(strategy, "#CBD5E1")
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values,
                name=LABELS.get(strategy, strategy),
                line=dict(color=color, width=1.5),
                fill="tozeroy",
                fillcolor=hex_to_rgba(color, 0.10),
                hovertemplate=f"<b>{LABELS.get(strategy, strategy)}</b><br>%{{x|%d %b %Y}}<br>Drawdown: %{{y:.2f}}%<extra></extra>",
            )
        )
    fig.update_layout(
        **CHART_BASE,
        height=300,
        xaxis=XAXIS,
        yaxis={**YAXIS, "title": "Drawdown (%)"},
        title=dict(text="Capital Preservation Profile", x=0.01, font=dict(color=TEXT, size=13)),
    )
    return fig


def fig_signal_diagnostics(dataset: dict, lab: dict) -> go.Figure:
    dates = pd.to_datetime(dataset["dates"])
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=dataset["betavol"].values,
            name="Predicted BetaVol Proxy",
            line=dict(color=PURPLE, width=2.0),
            hovertemplate="Betavol: %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[lab["latest_state"]["threshold"]] * len(dates),
            name="Active Betavol Threshold",
            line=dict(color=AMBER, width=1.6, dash="dash"),
            hovertemplate="Threshold: %{y:.4f}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=dataset["vix"].values,
            name="India VIX",
            line=dict(color=BLUE, width=1.8),
            hovertemplate="VIX: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[lab["latest_state"]["vix"]] * len(dates),
            name="Active VIX Threshold",
            line=dict(color=RED, width=1.4, dash="dot"),
            hovertemplate="VIX Trigger: %{y:.2f}<extra></extra>",
        ),
        secondary_y=True,
    )

    signal_positions = {"hold": 0, "rebalance": 1, "min_variance": 2}
    colors = [PALETTE.get(s, "#94A3B8") for s in lab["signals"]]
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[signal_positions[s] for s in lab["signals"]],
            mode="markers",
            marker=dict(size=5, color=colors, opacity=0.5),
            name="Signal States",
            hovertemplate="Signal state<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.update_layout(
        **CHART_BASE,
        height=360,
        xaxis=XAXIS,
        yaxis={**YAXIS, "title": "BetaVol / Signal Layer"},
        title=dict(text="Trigger Diagnostics — BetaVol vs VIX Override", x=0.01, font=dict(color=TEXT, size=13)),
    )
    fig.update_yaxes(title_text="India VIX", secondary_y=True, gridcolor="#102038", color=MUTED)
    return fig


def fig_strategy_lab(lab: dict, focus_event: str | None) -> go.Figure:
    dates = pd.to_datetime(lab["dates"])
    y1 = pd.Series(lab["strategy_equity"], index=dates)
    y2 = pd.Series(lab["adaptive_equity"], index=dates)
    y3 = pd.Series(lab["nifty_equity"], index=dates)

    if focus_event and focus_event in STRESS_EVENTS:
        start, end = STRESS_EVENTS[focus_event]
        pad_start = pd.to_datetime(start) - pd.Timedelta(days=20)
        pad_end = pd.to_datetime(end) + pd.Timedelta(days=20)
        mask = (dates >= pad_start) & (dates <= pad_end)
        dates = dates[mask]
        y1 = y1.loc[dates]
        y2 = y2.loc[dates]
        y3 = y3.loc[dates]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y1.values,
            name="Strategy Lab",
            line=dict(color=ACCENT, width=3.0),
            hovertemplate="<b>Strategy Lab</b><br>%{x|%d %b %Y}<br>%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y2.values,
            name="AdaptiveBeta Base",
            line=dict(color=PURPLE, width=1.8),
            hovertemplate="<b>AdaptiveBeta Base</b><br>%{x|%d %b %Y}<br>%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=y3.values,
            name="Buy & Hold NIFTY50",
            line=dict(color=AMBER, width=1.8),
            hovertemplate="<b>NIFTY50</b><br>%{x|%d %b %Y}<br>%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        **CHART_BASE,
        height=420,
        xaxis=XAXIS,
        yaxis={**YAXIS, "title": "Portfolio Value (Base = 100)"},
        title=dict(
            text=f"Interactive Strategy Lab{' — ' + focus_event if focus_event else ''}",
            x=0.01,
            font=dict(color=TEXT, size=14),
        ),
    )
    return fig


def fig_regime_timeline(dataset: dict, lab: dict) -> go.Figure:
    dates = pd.to_datetime(dataset["dates"])
    regime_map = {"bear": 0, "transition": 1, "bull": 2}
    mode_order = {"min_variance": 0, "risk_parity": 1, "max_sharpe_beta_constrained": 2, "hold": 3}

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.52, 0.48])
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[regime_map.get(str(x).lower(), 1) for x in dataset["regime"]],
            mode="lines",
            line=dict(color=BLUE, width=2.4),
            fill="tozeroy",
            fillcolor="rgba(96,165,250,0.10)",
            name="Regime State",
            hovertemplate="Regime layer<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[mode_order.get(x, 3) for x in lab["modes"]],
            mode="markers",
            marker=dict(
                size=6,
                color=[PALETTE.get("min_variance" if x == "min_variance" else ("rebalance" if x != "hold" else "hold")) for x in lab["modes"]],
                opacity=0.65,
            ),
            name="Chosen Optimiser",
            hovertemplate="Mode: %{text}<extra></extra>",
            text=lab["modes"],
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(
        row=1,
        col=1,
        tickmode="array",
        tickvals=[0, 1, 2],
        ticktext=["Bear", "Transition", "Bull"],
        gridcolor="#102038",
        color=MUTED,
    )
    fig.update_yaxes(
        row=2,
        col=1,
        tickmode="array",
        tickvals=[0, 1, 2, 3],
        ticktext=["MinVar", "RiskParity", "MaxSharpe", "Hold"],
        gridcolor="#102038",
        color=MUTED,
    )
    fig.update_layout(
        **CHART_BASE,
        height=360,
        xaxis2=XAXIS,
        title=dict(text="Regime Routing and Portfolio Mode Timeline", x=0.01, font=dict(color=TEXT, size=13)),
        showlegend=False,
    )
    return fig


def fig_allocation_snapshot(dataset: dict) -> go.Figure:
    prices: pd.DataFrame = dataset["prices"]
    betas: pd.Series = dataset["beta_snapshot"]
    if prices.empty or betas.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_BASE, height=340, title=dict(text="Allocation snapshot unavailable", x=0.01, font=dict(color=TEXT, size=13)))
        return fig

    trailing = prices.iloc[-252:].dropna(axis=1, thresh=180)
    common = [c for c in trailing.columns if c in betas.index]
    trailing = trailing[common].tail(252)
    betas = betas.reindex(common).dropna()

    if trailing.empty or betas.empty:
        fig = go.Figure()
        fig.update_layout(**CHART_BASE, height=340, title=dict(text="Allocation snapshot unavailable", x=0.01, font=dict(color=TEXT, size=13)))
        return fig

    ret = np.log(trailing / trailing.shift(1)).dropna()
    ann_ret = ret.mean() * 252 * 100
    ann_vol = ret.std() * np.sqrt(252) * 100

    score = ann_ret - 0.6 * ann_vol - 3.0 * abs(betas.reindex(common).fillna(1.0) - 0.85)
    top = score.sort_values(ascending=False).head(12)
    weights = np.clip((top - top.min()) + 1e-6, 0, None)
    weights = weights / weights.sum() * 100

    fig = go.Figure(
        go.Bar(
            x=weights.index,
            y=weights.values,
            marker=dict(
                color=weights.values,
                colorscale=[[0, "#12304B"], [0.5, "#1D8F87"], [1, "#00D4AA"]],
                line=dict(color="#83F5DC", width=0.3),
            ),
            hovertemplate="<b>%{x}</b><br>Weight: %{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        **CHART_BASE,
        height=340,
        xaxis={**XAXIS, "tickangle": -35},
        yaxis={**YAXIS, "title": "Illustrative Weight (%)"},
        title=dict(text="Current Allocation Snapshot — Max-Sharpe Beta-Constrained Profile", x=0.01, font=dict(color=TEXT, size=13)),
    )
    return fig


def build_metric_table(dataset: dict) -> pd.DataFrame:
    rows = []
    for strategy in dataset["returns"].columns:
        m = dataset["metrics"].get(strategy, annualized_metrics(dataset["returns"][strategy]))
        rows.append(
            {
                "Strategy": LABELS.get(strategy, strategy),
                "CAGR (%)": m.get("cagr"),
                "Sharpe": m.get("sharpe"),
                "Sortino": m.get("sortino"),
                "Max Drawdown (%)": m.get("maxDrawdown"),
                "Calmar": m.get("calmar"),
                "Annual Volatility (%)": m.get("annualVol"),
                "Total Return (%)": m.get("totalReturn"),
                "Trading Days": m.get("tradingDays"),
            }
        )
    return pd.DataFrame(rows)


def build_crisis_cards(dataset: dict):
    cards = []
    for event_name, stats in dataset["crisis_stats"].items():
        protection = stats.get("protection", 0)
        cards.append(
            html.Button(
                [
                    html.Div(event_name, style={"color": TEXT, "fontSize": "13px", "fontWeight": "600", "marginBottom": "10px"}),
                    html.Div(
                        f"{protection:+.2f}%",
                        style={
                            "color": ACCENT if protection >= 0 else RED,
                            "fontSize": "28px",
                            "fontWeight": "700",
                            "letterSpacing": "-0.5px",
                        },
                    ),
                    html.Div("Protection vs NIFTY50", style={"color": MUTED, "fontSize": "10px", "marginTop": "4px"}),
                    html.Div(
                        f"MaxDD {stats.get('portfolioMaxDD', 0):.2f}% | Rebalances {stats.get('rebalances', 0)}",
                        style={"color": MUTED, "fontSize": "10px", "marginTop": "10px"},
                    ),
                ],
                id={"type": "crisis-card", "index": event_name},
                n_clicks=0,
                style={
                    "background": CARD_ALT,
                    "border": f"1px solid {BORDER_SOFT}",
                    "borderRadius": "16px",
                    "padding": "18px",
                    "textAlign": "left",
                    "width": "100%",
                    "cursor": "pointer",
                },
            )
        )
    return html.Div(
        cards,
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
            "gap": "14px",
        },
    )


def build_kpi_cards(dataset: dict, lab: dict):
    base = lab["strategy_metrics"]
    bench = lab["benchmark_metrics"]
    latest = lab["latest_state"]
    summary = lab["summary"]

    def kpi(title: str, value: str, subtext: str, color: str = TEXT):
        return html.Div(
            [
                html.Div(title.upper(), style={"color": MUTED, "fontSize": "10px", "letterSpacing": "1.3px", "fontWeight": "700", "marginBottom": "8px"}),
                html.Div(value, style={"color": color, "fontSize": "30px", "fontWeight": "750", "letterSpacing": "-0.7px"}),
                html.Div(subtext, style={"color": MUTED, "fontSize": "11px", "marginTop": "6px", "lineHeight": "1.6"}),
            ],
            style={
                "background": CARD,
                "border": f"1px solid {BORDER}",
                "borderRadius": "18px",
                "padding": "20px",
                "minWidth": "170px",
                "flex": "1",
            },
        )

    return html.Div(
        [
            kpi("Strategy Lab CAGR", fmt_pct(base.get("cagr")), f"{summary['outperformance']:+.2f}% vs NIFTY50", ACCENT),
            kpi("Sharpe Ratio", fmt_num(base.get("sharpe")), f"Benchmark {fmt_num(bench.get('sharpe'))}", BLUE),
            kpi("Max Drawdown", fmt_pct(base.get("maxDrawdown")), f"{summary['drawdownImprovement']:+.2f}% drawdown improvement", RED),
            kpi("Rebalance Count", str(summary["rebalanceCount"]), f"Min-variance days: {summary['minVarianceDays']}", AMBER),
            kpi("Latest State", latest["signal"].replace("_", " ").title(), f"{latest['regime'].title()} regime · {latest['mode']}", PURPLE),
        ],
        style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "marginBottom": "20px"},
    )


SIDEBAR = html.Div(
    [
        html.Div(
            [
                html.Div("⚡", style={"fontSize": "28px", "marginRight": "10px"}),
                html.Div(
                    [
                        html.Div("AdaptiveBeta Lab", style={"color": TEXT, "fontSize": "18px", "fontWeight": "700"}),
                        html.Div("Python-native quant intelligence workstation", style={"color": MUTED, "fontSize": "10px", "letterSpacing": "1.1px", "textTransform": "uppercase"}),
                    ]
                ),
            ],
            style={"display": "flex", "alignItems": "center", "marginBottom": "24px"},
        ),
        html.Div(
            [
                html.Div("REAL-WORLD DATASET", style={"color": MUTED, "fontSize": "10px", "fontWeight": "700", "letterSpacing": "1.4px", "marginBottom": "8px"}),
                html.Div("49 NIFTY stocks · India VIX · HMM regimes · backtest 2015–2025", style={"color": TEXT, "fontSize": "12px", "lineHeight": "1.7"}),
            ],
            style={"marginBottom": "24px"},
        ),
        html.Div("STRATEGY FILTER", style={"color": MUTED, "fontSize": "10px", "fontWeight": "700", "letterSpacing": "1.4px", "marginBottom": "12px"}),
        dcc.Checklist(
            id="strategy-select",
            options=[
                {
                    "label": html.Span(
                        [
                            html.Span("●", style={"color": PALETTE.get(s, "#CBD5E1"), "marginRight": "8px"}),
                            html.Span(LABELS[s], style={"color": TEXT, "fontSize": "12px"}),
                        ]
                    ),
                    "value": s,
                }
                for s in STRATEGIES
            ],
            value=["adaptive_beta", "equal_weight", "buy_hold_nifty"],
            inputStyle={"display": "none"},
            labelStyle={"display": "block", "padding": "8px 10px", "borderRadius": "10px", "cursor": "pointer"},
            style={"display": "flex", "flexDirection": "column", "gap": "8px", "marginBottom": "26px"},
        ),
        html.Div("INTERACTIVE LAB CONTROLS", style={"color": MUTED, "fontSize": "10px", "fontWeight": "700", "letterSpacing": "1.4px", "marginBottom": "12px"}),
        html.Label("BetaVol threshold percentile", style={"color": TEXT, "fontSize": "12px", "marginBottom": "8px", "display": "block"}),
        dcc.Slider(id="threshold-pct", min=40, max=90, step=5, value=70, marks={i: {"label": str(i), "style": {"color": MUTED}} for i in range(40, 95, 10)}),
        html.Div(style={"height": "18px"}),
        html.Label("India VIX stress override", style={"color": TEXT, "fontSize": "12px", "marginBottom": "8px", "display": "block"}),
        dcc.Slider(id="vix-threshold", min=16, max=30, step=1, value=22, marks={i: {"label": str(i), "style": {"color": MUTED}} for i in [16, 20, 22, 25, 30]}),
        html.Div(style={"height": "18px"}),
        html.Label("Transaction cost (bps)", style={"color": TEXT, "fontSize": "12px", "marginBottom": "8px", "display": "block"}),
        dcc.Slider(id="tx-bps", min=0, max=40, step=2, value=8, marks={i: {"label": str(i), "style": {"color": MUTED}} for i in [0, 8, 16, 24, 32, 40]}),
        html.Div(style={"height": "16px"}),
        dbc.Switch(id="regime-routing", value=True, label="Enable regime-aware optimiser routing", style={"color": TEXT, "fontSize": "12px", "marginBottom": "18px"}),
        dbc.Button("Reset to thesis defaults", id="reset-defaults", color="dark", outline=True, style={"width": "100%", "borderColor": BORDER, "color": ACCENT}),
        html.Div(id="data-source-badge", style={"marginTop": "18px"}),
    ],
    style={
        "width": "300px",
        "minHeight": "100vh",
        "position": "fixed",
        "left": "0",
        "top": "0",
        "padding": "28px 20px",
        "background": "linear-gradient(180deg, #050B18 0%, #081425 100%)",
        "borderRight": f"1px solid {BORDER}",
        "overflowY": "auto",
        "zIndex": "50",
    },
)

TABS = dbc.Tabs(
    [
        dbc.Tab(label="Research Overview", tab_id="overview"),
        dbc.Tab(label="Interactive Strategy Lab", tab_id="lab"),
        dbc.Tab(label="Allocation & Optimisation", tab_id="alloc"),
        dbc.Tab(label="Crisis Intelligence", tab_id="crisis"),
    ],
    id="main-tabs",
    active_tab="overview",
    style={"marginBottom": "20px"},
)

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="AdaptiveBeta | Strategy Lab",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

app.layout = html.Div(
    [
        SIDEBAR,
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div("ADAPTIVEBETA RESEARCH DASHBOARD", style={"color": ACCENT, "fontSize": "11px", "fontWeight": "700", "letterSpacing": "1.5px", "marginBottom": "6px"}),
                                html.H1("Interactive portfolio optimisation demo with real thesis data", style={"color": TEXT, "fontSize": "34px", "margin": 0, "fontWeight": "760", "letterSpacing": "-1px"}),
                                html.P(
                                    "A premium Python-native demo combining real walk-forward backtest returns, India VIX stress overrides, HMM regime labels, and a live strategy lab that replays the portfolio decision logic used in your thesis.",
                                    style={"color": MUTED, "fontSize": "13px", "marginTop": "10px", "lineHeight": "1.8", "maxWidth": "980px"},
                                ),
                            ]
                        ),
                        html.Div(id="top-meta", style={"marginTop": "10px"}),
                    ],
                    style={
                        "position": "sticky",
                        "top": "0",
                        "zIndex": "40",
                        "background": "linear-gradient(180deg, rgba(4,8,22,0.98) 0%, rgba(4,8,22,0.90) 78%, rgba(4,8,22,0.80) 100%)",
                        "padding": "24px 0 18px 0",
                        "backdropFilter": "blur(16px)",
                    },
                ),
                html.Div(id="kpi-row"),
                TABS,
                html.Div(id="tab-content"),
                dcc.Store(id="dataset-store"),
                dcc.Store(id="lab-store"),
                dcc.Store(id="focus-event-store"),
                dcc.Interval(id="auto-refresh", interval=120000, n_intervals=0),
            ],
            style={"marginLeft": "300px", "padding": "0 28px 40px 28px", "background": f"radial-gradient(circle at top right, rgba(0,212,170,0.08), transparent 22%), {BG}", "minHeight": "100vh"},
        ),
    ],
    style={"background": BG, "fontFamily": "'Inter', system-ui, sans-serif"},
)


@app.callback(
    Output("threshold-pct", "value"),
    Output("vix-threshold", "value"),
    Output("tx-bps", "value"),
    Output("regime-routing", "value"),
    Input("reset-defaults", "n_clicks"),
    prevent_initial_call=True,
)
def reset_lab_controls(_n_clicks):
    return 70, 22, 8, True


@app.callback(
    Output("dataset-store", "data"),
    Output("data-source-badge", "children"),
    Output("top-meta", "children"),
    Input("auto-refresh", "n_intervals"),
)
def refresh_dataset(_n):
    dataset = build_research_dataset()
    meta = html.Div(
        [
            html.Span(
                f"● {'REAL PROJECT OUTPUTS' if dataset['source'] != 'synthetic' else 'SYNTHETIC FALLBACK'}",
                style={
                    "color": ACCENT if dataset["source"] != "synthetic" else AMBER,
                    "fontSize": "10px",
                    "fontWeight": "700",
                    "letterSpacing": "1.4px",
                },
            )
        ]
    )
    badge = html.Div(
        [
            html.Div("DATA STATUS", style={"color": MUTED, "fontSize": "10px", "letterSpacing": "1.2px", "fontWeight": "700", "marginBottom": "8px"}),
            html.Div(
                "Using real `results/`, `features/`, and India VIX data"
                if dataset["source"] != "synthetic"
                else "Fallback mode — live result files were not found",
                style={"color": TEXT, "fontSize": "12px", "lineHeight": "1.6"},
            ),
            html.Div(f"{len(dataset['dates']):,} trading days · {len(dataset['returns'].columns)} strategies", style={"color": MUTED, "fontSize": "11px", "marginTop": "8px"}),
        ],
        style={"background": CARD_ALT, "border": f"1px solid {BORDER_SOFT}", "borderRadius": "14px", "padding": "14px"},
    )

    serializable = {
        "source": dataset["source"],
        "dates": dataset["dates"],
        "returns": {k: [round(float(x), 8) for x in dataset["returns"][k].fillna(0)] for k in dataset["returns"].columns},
        "metrics": dataset["metrics"],
        "equity_curves": {k: [round(float(x), 4) for x in v] for k, v in dataset["equity_curves"].items()},
        "vix": [round(float(x), 6) for x in dataset["vix"]],
        "betavol": [round(float(x), 6) for x in dataset["betavol"]],
        "regime": [str(x) for x in dataset["regime"]],
        "thresholds": dataset["thresholds"],
        "crisis_stats": dataset["crisis_stats"],
    }
    return serializable, badge, meta


def deserialize_dataset(data: dict) -> dict:
    dates = pd.to_datetime(data["dates"])
    returns = pd.DataFrame({k: pd.Series(v, index=dates) for k, v in data["returns"].items()}, index=dates)
    equity_curves = {k: pd.Series(v, index=dates) for k, v in data["equity_curves"].items()}
    return {
        "source": data["source"],
        "dates": data["dates"],
        "returns": returns,
        "metrics": data["metrics"],
        "equity_curves": equity_curves,
        "vix": pd.Series(data["vix"], index=dates),
        "betavol": pd.Series(data["betavol"], index=dates),
        "regime": pd.Series(data["regime"], index=dates),
        "thresholds": data["thresholds"],
        "crisis_stats": data["crisis_stats"],
        "prices": load_prices(),
        "beta_snapshot": load_beta_snapshot(),
    }


@app.callback(
    Output("lab-store", "data"),
    Input("dataset-store", "data"),
    Input("threshold-pct", "value"),
    Input("vix-threshold", "value"),
    Input("tx-bps", "value"),
    Input("regime-routing", "value"),
)
def refresh_lab(dataset_data, threshold_pct, vix_threshold, tx_bps, regime_routing):
    if not dataset_data:
        return {}
    dataset = deserialize_dataset(dataset_data)
    return compute_strategy_lab(dataset, int(threshold_pct), float(vix_threshold), int(tx_bps), bool(regime_routing))


@app.callback(
    Output("focus-event-store", "data"),
    Input({"type": "crisis-card", "index": dash.ALL}, "n_clicks"),
    State("focus-event-store", "data"),
    prevent_initial_call=True,
)
def set_focus_event(_clicks, current):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current
    prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
    event_name = json.loads(prop_id)["index"]
    return None if current == event_name else event_name


@app.callback(
    Output("kpi-row", "children"),
    Input("dataset-store", "data"),
    Input("lab-store", "data"),
)
def update_kpi_row(dataset_data, lab_data):
    if not dataset_data or not lab_data:
        return html.Div()
    dataset = deserialize_dataset(dataset_data)
    return build_kpi_cards(dataset, lab_data)


@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
    Input("dataset-store", "data"),
    Input("lab-store", "data"),
    Input("strategy-select", "value"),
    Input("focus-event-store", "data"),
)
def render_tab(active_tab, dataset_data, lab_data, selected, focus_event):
    if not dataset_data or not lab_data:
        return html.Div("Loading dashboard...", style={"color": MUTED, "padding": "24px"})

    dataset = deserialize_dataset(dataset_data)
    sel = selected or list(dataset["returns"].columns)
    metrics_df = build_metric_table(dataset)

    if active_tab == "overview":
        table = dash_table.DataTable(
            data=metrics_df.to_dict("records"),
            columns=[{"name": c, "id": c, "type": "numeric" if c != "Strategy" else "text"} for c in metrics_df.columns],
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#081326",
                "color": MUTED,
                "fontWeight": "700",
                "fontSize": "10px",
                "letterSpacing": "1px",
                "border": "none",
                "borderBottom": f"1px solid {BORDER}",
                "padding": "12px 14px",
                "textTransform": "uppercase",
            },
            style_cell={
                "backgroundColor": "transparent",
                "color": TEXT,
                "fontSize": "12px",
                "padding": "11px 14px",
                "border": "none",
                "borderBottom": "1px solid #102038",
                "fontFamily": "'Inter', monospace",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "rgba(16,32,56,0.35)"},
                {"if": {"filter_query": '{Strategy} = "AdaptiveBeta (Ours)"'}, "backgroundColor": hex_to_rgba(ACCENT, 0.08), "color": ACCENT, "fontWeight": "700"},
            ],
        )

        return html.Div(
            [
                _card(
                    [
                        _section_title("Research Overview", "Thesis-grade performance overview", "Live metrics, walk-forward equity behaviour, and capital preservation patterns using real backtest outputs."),
                        dcc.Graph(figure=fig_equity_curves(dataset, sel), config={"displayModeBar": True}),
                    ],
                    {"marginBottom": "18px"},
                ),
                _card(
                    dcc.Graph(figure=fig_drawdown(dataset, sel), config={"displayModeBar": False}),
                    {"marginBottom": "18px", "padding": "4px"},
                ),
                _card(
                    [
                        _section_title("Cross-Strategy Comparison", "Performance metrics table", "Sortable comparison across all available strategies in the current real-data run."),
                        table,
                    ]
                ),
            ]
        )

    if active_tab == "lab":
        event_table = pd.DataFrame(lab_data["event_rows"])
        event_component = (
            dash_table.DataTable(
                data=event_table.to_dict("records"),
                columns=[{"name": c, "id": c} for c in event_table.columns],
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#081326", "color": MUTED, "border": "none", "borderBottom": f"1px solid {BORDER}", "padding": "10px 12px"},
                style_cell={"backgroundColor": "transparent", "color": TEXT, "border": "none", "borderBottom": "1px solid #102038", "padding": "10px 12px", "fontSize": "12px"},
            )
            if not event_table.empty
            else html.Div("No event rows available.", style={"color": MUTED})
        )

        latest = lab_data["latest_state"]
        insight = (
            f"At the latest observation ({latest['date']}), the system is in a {latest['regime']} regime with "
            f"India VIX at {latest['vix']:.2f} and beta-vol proxy at {latest['betavol']:.4f}. "
            f"The active trigger threshold is {latest['threshold']:.4f}, so the engine selected "
            f"`{latest['signal']}` and routed capital through `{latest['mode']}`."
        )

        return html.Div(
            [
                _card(
                    [
                        _section_title("Interactive Strategy Lab", "Replay the thesis signal engine live", "The controls on the left dynamically alter threshold sensitivity, VIX stress override, transaction costs, and regime routing."),
                        dcc.Graph(figure=fig_strategy_lab(lab_data, focus_event), config={"displayModeBar": True}),
                    ],
                    {"marginBottom": "18px"},
                ),
                html.Div(
                    [
                        _card(dcc.Graph(figure=fig_signal_diagnostics(dataset, lab_data), config={"displayModeBar": False}), {"padding": "4px"}),
                        _card(
                            [
                                html.Div("ENGINE INSIGHT", style={"color": ACCENT, "fontSize": "10px", "letterSpacing": "1.4px", "fontWeight": "700", "marginBottom": "10px"}),
                                html.Div(insight, style={"color": TEXT, "fontSize": "13px", "lineHeight": "1.9", "marginBottom": "18px"}),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.Div("Rebalances", style={"color": MUTED, "fontSize": "10px"}),
                                                html.Div(str(lab_data["summary"]["rebalanceCount"]), style={"color": AMBER, "fontSize": "24px", "fontWeight": "700"}),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Div("Min-Var Days", style={"color": MUTED, "fontSize": "10px"}),
                                                html.Div(str(lab_data["summary"]["minVarianceDays"]), style={"color": RED, "fontSize": "24px", "fontWeight": "700"}),
                                            ]
                                        ),
                                        html.Div(
                                            [
                                                html.Div("Turnover Proxy", style={"color": MUTED, "fontSize": "10px"}),
                                                html.Div(str(lab_data["summary"]["turnoverProxy"]), style={"color": BLUE, "fontSize": "24px", "fontWeight": "700"}),
                                            ]
                                        ),
                                    ],
                                    style={"display": "grid", "gridTemplateColumns": "repeat(3, 1fr)", "gap": "14px"},
                                ),
                            ]
                        ),
                    ],
                    style={"display": "grid", "gridTemplateColumns": "1.5fr 1fr", "gap": "18px", "marginBottom": "18px"},
                ),
                _card(
                    [
                        _section_title("Stress Event Replay", "Crisis behaviour under the active strategy settings", "Click a crisis card below to zoom the strategy lab chart into that market event."),
                        build_crisis_cards(dataset),
                        html.Div(style={"height": "18px"}),
                        event_component,
                    ]
                ),
            ]
        )

    if active_tab == "alloc":
        mode_counts = pd.Series(lab_data["modes"]).value_counts().rename_axis("mode").reset_index(name="days")
        mode_fig = go.Figure(
            go.Bar(
                x=mode_counts["mode"],
                y=mode_counts["days"],
                marker_color=[PALETTE.get("min_variance" if m == "min_variance" else ("rebalance" if m != "hold" else "hold")) for m in mode_counts["mode"]],
                hovertemplate="<b>%{x}</b><br>Days: %{y}<extra></extra>",
            )
        )
        mode_fig.update_layout(
            **CHART_BASE,
            height=320,
            xaxis=XAXIS,
            yaxis={**YAXIS, "title": "Days selected"},
            title=dict(text="Optimiser routing frequency", x=0.01, font=dict(color=TEXT, size=13)),
            showlegend=False,
        )

        return html.Div(
            [
                html.Div(
                    [
                        _card(
                            [
                                _section_title("Allocation Intelligence", "Illustrative current portfolio snapshot", "This panel shows a research-quality approximation of the current max-Sharpe beta-constrained allocation profile based on available prices, returns, and beta exposures."),
                                dcc.Graph(figure=fig_allocation_snapshot(dataset), config={"displayModeBar": False}),
                            ]
                        ),
                        _card(dcc.Graph(figure=mode_fig, config={"displayModeBar": False})),
                    ],
                    style={"display": "grid", "gridTemplateColumns": "1.2fr 0.8fr", "gap": "18px", "marginBottom": "18px"},
                ),
                _card(
                    [
                        _section_title("Routing Timeline", "Regime-aware optimiser path", "Shows how market regime and stress logic map into actual optimiser choices across the backtest."),
                        dcc.Graph(figure=fig_regime_timeline(dataset, lab_data), config={"displayModeBar": False}),
                    ]
                ),
            ]
        )

    crisis_df = pd.DataFrame(
        [
            {
                "Event": name,
                "Protection vs NIFTY50 (%)": stats.get("protection"),
                "AdaptiveBeta Return (%)": stats.get("portfolioReturn"),
                "NIFTY50 Return (%)": stats.get("niftyReturn"),
                "AdaptiveBeta MaxDD (%)": stats.get("portfolioMaxDD"),
                "NIFTY50 MaxDD (%)": stats.get("niftyMaxDD"),
                "Rebalances": stats.get("rebalances"),
                "MinVar Days": stats.get("minVarDays"),
            }
            for name, stats in dataset["crisis_stats"].items()
        ]
    )

    crisis_table = dash_table.DataTable(
        data=crisis_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in crisis_df.columns],
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#081326", "color": MUTED, "border": "none", "borderBottom": f"1px solid {BORDER}", "padding": "10px 12px"},
        style_cell={"backgroundColor": "transparent", "color": TEXT, "border": "none", "borderBottom": "1px solid #102038", "padding": "10px 12px", "fontSize": "12px"},
    )

    return html.Div(
        [
            _card(
                [
                    _section_title("Crisis Intelligence", "Defensive characteristics during Indian market stress events", "This view focuses on protection, drawdown control, and stress-event behaviour relative to NIFTY50."),
                    build_crisis_cards(dataset),
                ],
                {"marginBottom": "18px"},
            ),
            _card(crisis_table),
        ]
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
