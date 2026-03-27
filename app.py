"""
AdaptiveBeta — Production-Grade Quantitative Finance Dashboard
Framework: Plotly Dash + dash-bootstrap-components
Design:    Dark Glassmorphism | NIFTY50 Universe | 2018-2025

Run:  python app.py
URL:  http://localhost:8050
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

# ═══════════════════════════════════════════════════════════════
# CONSTANTS & THEME
# ═══════════════════════════════════════════════════════════════

ROOT     = Path(__file__).parent
RESULTS  = ROOT / "results"

PALETTE: Dict[str, str] = {
    "adaptive_beta":   "#00D4AA",
    "static_capm_mvo": "#818CF8",
    "kalman_mvo":      "#FBBF24",
    "equal_weight":    "#64748B",
    "buy_hold_nifty":  "#F87171",
    "momentum_quality":"#34D399",
}
LABELS: Dict[str, str] = {
    "adaptive_beta":   "AdaptiveBeta (Ours)",
    "static_capm_mvo": "Static CAPM MVO",
    "kalman_mvo":      "Kalman-Beta MVO",
    "equal_weight":    "Equal Weight",
    "buy_hold_nifty":  "Buy & Hold NIFTY50",
    "momentum_quality":"Momentum-Quality",
}
STRATEGIES = list(PALETTE.keys())

# Colour tokens
BG      = "#040A17"
CARD    = "rgba(8, 16, 42, 0.92)"
BORDER  = "rgba(0, 212, 170, 0.18)"
BORDER2 = "rgba(0, 212, 170, 0.06)"
TEXT    = "#E2ECF8"
MUTED   = "#7B8BAD"
ACCENT  = "#00D4AA"
RED     = "#F87171"
YELLOW  = "#FBBF24"
PURPLE  = "#818CF8"

# Shared chart theme (applied with **CHART_BASE)
CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#06101F",
    font=dict(color=TEXT, family="'Inter','DM Sans',system-ui,sans-serif", size=11),
    margin=dict(l=60, r=20, t=50, b=45),
    hovermode="x unified",
    legend=dict(
        bgcolor="rgba(4,10,23,0.92)", bordercolor="#1A2540", borderwidth=1,
        font=dict(size=10, color=MUTED),
    ),
)
XAXIS = dict(gridcolor="#0D1B35", linecolor="#1A2540", tickfont=dict(color=MUTED, size=10),
             title_font=dict(color=MUTED))
YAXIS = dict(gridcolor="#0D1B35", linecolor="#1A2540", tickfont=dict(color=MUTED, size=10),
             title_font=dict(color=MUTED))


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def hex_to_rgba(hx: str, a: float) -> str:
    """Safe hex → rgba conversion. Fixes the original dashboard bug."""
    h = hx.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"


def _card(children, extra_style: dict | None = None, cls: str = ""):
    style = {
        "background": CARD,
        "border": f"1px solid {BORDER}",
        "borderRadius": "14px",
        "padding": "22px",
        "backdropFilter": "blur(14px)",
        **(extra_style or {}),
    }
    return html.Div(children, style=style, className=cls)


# ═══════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════

def _read(fname: str, **kw) -> Optional[pd.DataFrame]:
    p = RESULTS / fname
    return pd.read_csv(p, **kw) if p.exists() else None


def load_all_returns() -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for s in STRATEGIES:
        df = _read(f"{s}_returns.csv", parse_dates=["date"], index_col="date")
        if df is not None and not df.empty:
            out[s] = df.iloc[:, 0].rename(s)
    # also check combined file
    combined = _read("strategy_returns.csv", parse_dates=["date"], index_col="date")
    if combined is not None:
        for col in combined.columns:
            if col not in out:
                out[col] = combined[col].rename(col)
    return out


def load_metrics() -> Optional[pd.DataFrame]:
    return _read("performance_metrics.csv", index_col=0)


def load_stress() -> Optional[pd.DataFrame]:
    return _read("stress_analysis.csv")


def load_rebalance() -> Optional[pd.DataFrame]:
    return _read("rebalance_log.csv", parse_dates=["date"])


def load_model_comparison() -> Optional[pd.DataFrame]:
    return _read("model_comparison.csv")


def calc_metrics(r: pd.Series) -> dict:
    r = r.dropna()
    if not len(r):
        return {"CAGR (%)": "—", "Sharpe": "—", "Sortino": "—",
                "Max DD (%)": "—", "Ann Vol (%)": "—", "Calmar": "—"}
    vol   = r.std() * 252**0.5
    ann   = r.mean() * 252
    n_yrs = len(r) / 252
    cum   = np.exp(r.cumsum())
    mdd   = (cum / cum.cummax() - 1).min()
    cagr  = np.exp(r.sum()) ** (1 / n_yrs) - 1 if n_yrs > 0 else 0
    ds    = r[r < 0].std() * 252**0.5
    return {
        "CAGR (%)":    round(cagr * 100, 2),
        "Sharpe":      round(ann / vol  if vol else 0, 3),
        "Sortino":     round(ann / ds   if ds  else 0, 3),
        "Max DD (%)":  round(mdd * 100, 2),
        "Ann Vol (%)": round(vol * 100, 2),
        "Calmar":      round(cagr / abs(mdd) if mdd else 0, 3),
    }


def demo_returns(seed: int = 42) -> Dict[str, pd.Series]:
    np.random.seed(seed)
    dates = pd.date_range("2018-01-02", "2024-12-31", freq="B")
    # calibrated to actual backtest numbers
    params = {
        "adaptive_beta":   (0.0924 / 252, 0.1501 / 252**0.5),
        "static_capm_mvo": (0.1184 / 252, 0.1916 / 252**0.5),
        "kalman_mvo":      (0.1118 / 252, 0.1928 / 252**0.5),
        "equal_weight":    (0.1605 / 252, 0.1763 / 252**0.5),
        "buy_hold_nifty":  (0.1245 / 252, 0.1783 / 252**0.5),
    }
    out: Dict[str, pd.Series] = {}
    for nm, (mu, sig) in params.items():
        r = np.random.normal(mu, sig, len(dates))
        crash = (dates >= "2020-02-20") & (dates <= "2020-03-23")
        r[crash] = np.random.normal(-0.022, sig * 2.5, crash.sum())
        out[nm] = pd.Series(r, index=dates, name=nm)
    return out


def serialize_returns(rd: Dict[str, pd.Series]) -> dict:
    return {k: {"idx": v.index.strftime("%Y-%m-%d").tolist(),
                "val": v.values.tolist()} for k, v in rd.items()}


def deserialize_returns(data: dict) -> Dict[str, pd.Series]:
    return {k: pd.Series(v["val"], index=pd.to_datetime(v["idx"]), name=k)
            for k, v in data.items()}


# ═══════════════════════════════════════════════════════════════
# CHART BUILDERS
# ═══════════════════════════════════════════════════════════════

STRESS_EVENTS = [
    ("COVID Crash",   "2020-02-20", "2020-05-01", "rgba(239,68,68,0.07)"),
    ("ADANI Crisis",  "2023-01-25", "2023-03-15", "rgba(251,191,36,0.07)"),
    ("IL&FS Crisis",  "2018-08-01", "2018-10-31", "rgba(139,92,246,0.06)"),
    ("2018 Hike",     "2018-09-01", "2018-11-30", "rgba(99,102,241,0.05)"),
]


def fig_equity(rd: Dict[str, pd.Series], selected: List[str]) -> go.Figure:
    fig = go.Figure()
    for ev, s, e, c in STRESS_EVENTS:
        fig.add_vrect(x0=s, x1=e, fillcolor=c, opacity=1, line_width=0,
                      annotation_text=ev, annotation_position="top left",
                      annotation_font=dict(color=MUTED, size=9))
    for nm in selected:
        if nm not in rd:
            continue
        cum = np.exp(rd[nm].cumsum()) * 100
        lw  = 2.8 if nm == "adaptive_beta" else 1.6
        fig.add_trace(go.Scatter(
            x=cum.index, y=cum.values,
            name=LABELS.get(nm, nm),
            line=dict(color=PALETTE.get(nm, "#AAA"), width=lw),
            hovertemplate=(f"<b style='color:{PALETTE.get(nm,'#AAA')}'>"
                           f"{LABELS.get(nm, nm)}</b>"
                           "<br>%{x|%d %b %Y}  ₹%{y:.1f}<extra></extra>"),
        ))
    fig.update_layout(
        **CHART_BASE, height=430,
        yaxis_title="Portfolio Value (₹100 base)",
        xaxis=XAXIS, yaxis=YAXIS,
        title=dict(text="Cumulative Returns — Walk-Forward Out-of-Sample",
                   font=dict(color=TEXT, size=14, weight=600), x=0.01),
    )
    return fig


def fig_drawdown(rd: Dict[str, pd.Series], selected: List[str]) -> go.Figure:
    fig = go.Figure()
    for nm in selected:
        if nm not in rd:
            continue
        cum = np.exp(rd[nm].cumsum())
        dd  = (cum / cum.cummax() - 1) * 100
        col = PALETTE.get(nm, "#AAA")
        fig.add_trace(go.Scatter(
            x=dd.index, y=dd.values,
            name=LABELS.get(nm, nm),
            line=dict(color=col, width=1.6),
            fill="tozeroy",
            fillcolor=hex_to_rgba(col, 0.09),
            hovertemplate=(f"<b style='color:{col}'>{LABELS.get(nm, nm)}</b>"
                           "<br>%{x|%d %b %Y}  %{y:.1f}%<extra></extra>"),
        ))
    fig.update_layout(
        **CHART_BASE, height=300,
        yaxis_title="Drawdown (%)",
        xaxis=XAXIS, yaxis=YAXIS,
        title=dict(text="Drawdown Analysis", font=dict(color=TEXT, size=13), x=0.01),
    )
    return fig


def fig_rolling_sharpe(rd: Dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    for nm, ret in rd.items():
        rs = (ret.rolling(252).mean() * 252) / (ret.rolling(252).std() * 252**0.5)
        fig.add_trace(go.Scatter(
            x=rs.index, y=rs.values,
            name=LABELS.get(nm, nm),
            line=dict(color=PALETTE.get(nm, "#AAA"),
                      width=2.5 if nm == "adaptive_beta" else 1.2),
            hovertemplate="<b>%{x|%d %b %Y}</b>  Sharpe=%{y:.2f}<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dot", line_color="#334155", line_width=1)
    fig.add_hline(y=1, line_dash="dash", line_color="#334155", line_width=1,
                  annotation_text="Sharpe = 1", annotation_font=dict(color=MUTED, size=9))
    fig.update_layout(
        **CHART_BASE, height=330,
        yaxis_title="Rolling 252-Day Sharpe",
        xaxis=XAXIS, yaxis=YAXIS,
        title=dict(text="Rolling 1-Year Sharpe Ratio", font=dict(color=TEXT, size=13), x=0.01),
    )
    return fig


def fig_stress_bar(rd: Dict[str, pd.Series]) -> go.Figure:
    events = {
        "COVID\nCrash":   ("2020-02-20", "2020-03-23"),
        "ADANI\nCrisis":  ("2023-01-25", "2023-03-15"),
        "2018\nRate Hike":("2018-09-01", "2018-11-30"),
        "IL&FS\nCrisis":  ("2018-08-01", "2018-10-31"),
    }
    fig = go.Figure()
    for nm in STRATEGIES:
        if nm not in rd:
            continue
        vals = []
        for s, e in events.values():
            try:
                r = rd[nm].loc[s:e]
                vals.append(float(np.exp(r.sum()) - 1) * 100 if len(r) else 0)
            except Exception:
                vals.append(0)
        fig.add_trace(go.Bar(
            name=LABELS.get(nm, nm), x=list(events.keys()), y=vals,
            marker_color=PALETTE.get(nm, "#AAA"),
            hovertemplate="<b>%{x}</b><br>Return: %{y:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        **CHART_BASE, height=380, barmode="group",
        yaxis_title="Cumulative Return (%)",
        xaxis=dict(**XAXIS, tickangle=0), yaxis=YAXIS,
        title=dict(text="Strategy Performance During Market Stress Events",
                   font=dict(color=TEXT, size=13), x=0.01),
    )
    return fig


def fig_monthly_heatmap(rd: Dict[str, pd.Series], strategy: str) -> go.Figure:
    if strategy not in rd:
        return go.Figure()
    r = rd[strategy].copy()
    r.index = pd.to_datetime(r.index)
    monthly = r.resample("ME").sum() * 100
    df = monthly.to_frame("ret")
    df["year"]  = df.index.year
    df["month"] = df.index.month
    pt = df.pivot_table(values="ret", index="year", columns="month", aggfunc="sum")
    pt.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
    text_vals = [[f"{v:.1f}%" if not np.isnan(v) else "" for v in row]
                 for row in pt.values]
    colorscale = [
        [0.0,  "#7F1D1D"],
        [0.3,  "#991B1B"],
        [0.45, "#1E293B"],
        [0.5,  "#0F172A"],
        [0.55, "#14532D"],
        [0.75, "#15803D"],
        [1.0,  "#064E3B"],
    ]
    fig = go.Figure(go.Heatmap(
        z=pt.values, x=pt.columns.tolist(), y=pt.index.tolist(),
        colorscale=colorscale, zmid=0,
        text=text_vals, texttemplate="%{text}",
        textfont=dict(size=9, color="white"),
        hovertemplate="<b>%{y} %{x}</b>: %{z:.2f}%<extra></extra>",
        colorbar=dict(title="Ret %", tickfont=dict(color=MUTED, size=9)),
    ))
    fig.update_layout(
        **CHART_BASE, height=320,
        title=dict(text=f"Monthly Returns — {LABELS.get(strategy, strategy)}",
                   font=dict(color=TEXT, size=13), x=0.01),
        xaxis=dict(tickfont=dict(color=MUTED)),
        yaxis=dict(tickfont=dict(color=MUTED), autorange="reversed"),
    )
    return fig


def fig_underwater(rd: Dict[str, pd.Series], strategy: str) -> go.Figure:
    """Detailed underwater equity (drawdown depth over time)."""
    if strategy not in rd:
        return go.Figure()
    ret = rd[strategy]
    cum = np.exp(ret.cumsum())
    peak = cum.cummax()
    uw  = (cum / peak - 1) * 100
    col = PALETTE.get(strategy, ACCENT)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=uw.index, y=uw.values,
        fill="tozeroy", fillcolor=hex_to_rgba(col, 0.18),
        line=dict(color=col, width=1.5),
        name="Drawdown",
        hovertemplate="%{x|%d %b %Y}  %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        **CHART_BASE, height=260,
        yaxis_title="Underwater Equity (%)",
        xaxis=XAXIS, yaxis=YAXIS,
        title=dict(text=f"Underwater Equity — {LABELS.get(strategy, strategy)}",
                   font=dict(color=TEXT, size=13), x=0.01),
    )
    return fig


def fig_return_distribution(rd: Dict[str, pd.Series], selected: List[str]) -> go.Figure:
    fig = go.Figure()
    for nm in selected:
        if nm not in rd:
            continue
        ret = rd[nm].dropna() * 100
        col = PALETTE.get(nm, "#AAA")
        fig.add_trace(go.Histogram(
            x=ret, name=LABELS.get(nm, nm),
            marker_color=hex_to_rgba(col, 0.7),
            marker_line=dict(color=col, width=0.5),
            opacity=0.75, nbinsx=80,
            hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
        ))
    fig.update_layout(
        **CHART_BASE, height=320, barmode="overlay",
        xaxis_title="Daily Return (%)", yaxis_title="Frequency",
        xaxis=XAXIS, yaxis=YAXIS,
        title=dict(text="Daily Return Distribution",
                   font=dict(color=TEXT, size=13), x=0.01),
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# KPI CARDS
# ═══════════════════════════════════════════════════════════════

def _kpi(icon: str, label: str, value: str, delta: str = "",
         delta_good: bool = True) -> html.Div:
    delta_col = ACCENT if delta_good else RED
    return html.Div([
        html.Div([
            html.Span(icon, style={"fontSize": "18px", "marginRight": "8px"}),
            html.Span(label, style={"color": MUTED, "fontSize": "10px",
                                    "fontWeight": "600", "letterSpacing": "1px",
                                    "textTransform": "uppercase"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"}),
        html.Div(value, style={"fontSize": "28px", "fontWeight": "700", "color": TEXT,
                               "letterSpacing": "-0.5px", "fontVariantNumeric": "tabular-nums"}),
        html.Div(delta, style={"fontSize": "10px", "color": delta_col, "marginTop": "5px",
                               "fontWeight": "500"}) if delta else None,
    ], style={
        "background": CARD,
        "border": f"1px solid {BORDER}",
        "borderRadius": "14px",
        "padding": "20px 22px",
        "flex": "1",
        "minWidth": "160px",
        "cursor": "default",
        "transition": "all 0.2s ease",
    }, className="kpi-card")


def build_kpi_row(rd: Dict[str, pd.Series],
                  mdf: Optional[pd.DataFrame]) -> html.Div:
    ab = calc_metrics(rd["adaptive_beta"]) if "adaptive_beta" in rd else {}
    nf = calc_metrics(rd["buy_hold_nifty"]) if "buy_hold_nifty" in rd else {}

    sharpe     = ab.get("Sharpe", "—")
    cagr       = ab.get("CAGR (%)", "—")
    mdd        = ab.get("Max DD (%)", "—")
    calmar     = ab.get("Calmar", "—")
    sortino    = ab.get("Sortino", "—")
    nifty_sh   = nf.get("Sharpe", None)

    delta_sh = ""
    if isinstance(sharpe, (int, float)) and isinstance(nifty_sh, (int, float)):
        diff = float(sharpe) - float(nifty_sh)
        delta_sh = f"{diff:+.3f} vs NIFTY50"

    return html.Div([
        _kpi("⚡", "Sharpe Ratio", str(sharpe), delta_sh,
             float(sharpe) >= float(nifty_sh) if nifty_sh and sharpe != "—" else True),
        _kpi("📈", "CAGR",         f"{cagr}%"),
        _kpi("📉", "Max Drawdown", f"{mdd}%",  delta_good=False),
        _kpi("🎯", "Calmar Ratio", str(calmar)),
        _kpi("🔬", "Sortino",      str(sortino)),
    ], style={"display": "flex", "gap": "14px", "flexWrap": "wrap",
              "marginBottom": "24px"})


# ═══════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════

SIDEBAR_CONTENT = html.Div([
    # ── Branding ─────────────────────────────────────────────
    html.Div([
        html.Div("⚡", style={"fontSize": "26px", "display": "inline-block",
                              "verticalAlign": "middle"}),
        html.Div([
            html.Div("AdaptiveBeta",
                     style={"fontSize": "17px", "fontWeight": "700", "color": TEXT,
                            "letterSpacing": "-0.3px"}),
            html.Div("Quant Portfolio System",
                     style={"fontSize": "9px", "color": MUTED, "letterSpacing": "1.2px",
                            "textTransform": "uppercase", "marginTop": "2px"}),
        ], style={"display": "inline-block", "verticalAlign": "middle", "marginLeft": "10px"}),
    ], style={"paddingBottom": "22px", "borderBottom": f"1px solid {BORDER}",
              "marginBottom": "22px"}),

    # ── Stats chips ──────────────────────────────────────────
    html.Div([
        html.Div("NIFTY50 Universe · 49 Stocks",
                 style={"color": MUTED, "fontSize": "10px", "marginBottom": "4px"}),
        html.Div("Walk-Forward 2018 – 2025",
                 style={"color": MUTED, "fontSize": "10px", "marginBottom": "4px"}),
        html.Div("M.Tech Thesis · SIT Pune 2026",
                 style={"color": MUTED, "fontSize": "10px"}),
    ], style={"marginBottom": "22px"}),

    html.Hr(style={"borderColor": BORDER, "margin": "0 0 18px 0"}),

    # ── Data source ──────────────────────────────────────────
    html.Div("DATA SOURCE", style={"color": MUTED, "fontSize": "9px",
                                    "letterSpacing": "1.4px", "fontWeight": "600",
                                    "marginBottom": "10px"}),
    dcc.RadioItems(
        id="data-source",
        options=[
            {"label": html.Span([
                html.Span("●", style={"color": ACCENT, "marginRight": "8px"}),
                html.Span("Live Results", style={"color": TEXT, "fontSize": "12px"}),
            ]), "value": "live"},
            {"label": html.Span([
                html.Span("●", style={"color": YELLOW, "marginRight": "8px"}),
                html.Span("Demo Data", style={"color": TEXT, "fontSize": "12px"}),
            ]), "value": "demo"},
        ],
        value="live" if (RESULTS / "performance_metrics.csv").exists() else "demo",
        style={"display": "flex", "flexDirection": "column", "gap": "8px"},
        inputStyle={"display": "none"},
        labelStyle={"cursor": "pointer", "padding": "7px 10px",
                    "borderRadius": "8px", "transition": "all 0.15s"},
    ),

    html.Div(id="data-status", style={"marginTop": "12px"}),

    html.Hr(style={"borderColor": BORDER, "margin": "18px 0"}),

    # ── Strategy toggles ─────────────────────────────────────
    html.Div("STRATEGIES", style={"color": MUTED, "fontSize": "9px",
                                   "letterSpacing": "1.4px", "fontWeight": "600",
                                   "marginBottom": "12px"}),
    dcc.Checklist(
        id="strategy-select",
        options=[{"label": html.Span([
            html.Span("●", style={"color": PALETTE.get(s, "#AAA"),
                                  "marginRight": "9px", "fontSize": "13px"}),
            html.Span(LABELS[s], style={"color": TEXT, "fontSize": "11px"}),
        ]), "value": s} for s in STRATEGIES],
        value=[s for s in ["adaptive_beta", "equal_weight", "buy_hold_nifty"]],
        style={"display": "flex", "flexDirection": "column", "gap": "8px"},
        inputStyle={"display": "none"},
        labelStyle={"display": "flex", "alignItems": "center", "cursor": "pointer",
                    "padding": "6px 10px", "borderRadius": "8px",
                    "transition": "background 0.15s"},
    ),

    html.Div(style={"flex": "1"}),  # spacer

    html.Hr(style={"borderColor": BORDER, "margin": "18px 0"}),
    html.Div([
        html.Div("Powered by Plotly Dash",
                 style={"color": MUTED, "fontSize": "9px"}),
        html.Div("© 2026 AdaptiveBeta Project",
                 style={"color": MUTED, "fontSize": "9px", "marginTop": "3px"}),
    ]),

], style={
    "width": "260px", "minHeight": "100vh",
    "background": "linear-gradient(175deg, #060C1F 0%, #030810 100%)",
    "borderRight": f"1px solid {BORDER}",
    "padding": "28px 18px",
    "position": "fixed", "top": "0", "left": "0",
    "display": "flex", "flexDirection": "column",
    "overflowY": "auto", "zIndex": "200",
})

NAV_TABS = dbc.Tabs([
    dbc.Tab(label="📈 Equity Curves",  tab_id="eq",
            label_style={"color": MUTED, "fontSize": "12px", "padding": "10px 18px"},
            active_label_style={"color": ACCENT, "fontWeight": "600",
                                "borderBottom": f"2px solid {ACCENT}"}),
    dbc.Tab(label="📊 Performance",    tab_id="perf",
            label_style={"color": MUTED, "fontSize": "12px", "padding": "10px 18px"},
            active_label_style={"color": ACCENT, "fontWeight": "600",
                                "borderBottom": f"2px solid {ACCENT}"}),
    dbc.Tab(label="⚠️ Risk & Stress", tab_id="risk",
            label_style={"color": MUTED, "fontSize": "12px", "padding": "10px 18px"},
            active_label_style={"color": ACCENT, "fontWeight": "600",
                                "borderBottom": f"2px solid {ACCENT}"}),
    dbc.Tab(label="⚡ Signals",        tab_id="sig",
            label_style={"color": MUTED, "fontSize": "12px", "padding": "10px 18px"},
            active_label_style={"color": ACCENT, "fontWeight": "600",
                                "borderBottom": f"2px solid {ACCENT}"}),
], id="nav-tabs", active_tab="eq",
   style={"borderBottom": f"1px solid {BORDER}", "marginBottom": "22px",
          "background": "transparent"})

MAIN_CONTENT = html.Div([
    # ── Top bar ──────────────────────────────────────────────
    html.Div([
        html.Div(id="topbar-title",
                 style={"color": MUTED, "fontSize": "12px", "flex": "1"}),
        html.Span("● LIVE DATA", id="live-dot",
                  style={"color": ACCENT, "fontSize": "9px", "fontWeight": "700",
                         "letterSpacing": "1.5px"}),
        dcc.Interval(id="auto-refresh", interval=60_000, n_intervals=0),
    ], style={
        "display": "flex", "alignItems": "center",
        "padding": "10px 28px",
        "background": "rgba(4,10,23,0.96)",
        "borderBottom": f"1px solid {BORDER}",
        "backdropFilter": "blur(16px)",
        "position": "sticky", "top": "0", "zIndex": "100",
    }),

    # ── Body ─────────────────────────────────────────────────
    html.Div([
        dcc.Loading(html.Div(id="kpi-row"),   type="circle", color=ACCENT),
        NAV_TABS,
        dcc.Loading(html.Div(id="tab-body"),  type="circle", color=ACCENT),
    ], style={"padding": "24px 28px"}),

], style={"marginLeft": "260px", "minHeight": "100vh", "background": BG})


app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="AdaptiveBeta | Quant Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

app.layout = html.Div([
    SIDEBAR_CONTENT,
    MAIN_CONTENT,
    dcc.Store(id="store-returns"),
    dcc.Store(id="store-metrics"),
], style={"background": BG, "minHeight": "100vh",
          "fontFamily": "'Inter', system-ui, sans-serif"})


# ═══════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════

@app.callback(
    Output("store-returns",  "data"),
    Output("store-metrics",  "data"),
    Output("data-status",    "children"),
    Output("topbar-title",   "children"),
    Output("live-dot",       "style"),
    Input("data-source",     "value"),
    Input("auto-refresh",    "n_intervals"),
)
def refresh_data(source, _n):
    dot_live   = {"color": ACCENT, "fontSize": "9px", "fontWeight": "700", "letterSpacing": "1.5px"}
    dot_demo   = {"color": YELLOW, "fontSize": "9px", "fontWeight": "700", "letterSpacing": "1.5px"}

    if source == "live":
        rd  = load_all_returns()
        mdf = load_metrics()
        if not rd:
            rd = demo_returns()
            status = html.Div("⚠  No results/ files found — showing demo",
                              style={"color": YELLOW, "fontSize": "10px"})
            dot = dot_demo
        else:
            n = len(next(iter(rd.values())))
            status = html.Div(f"✓  {n:,} trading days | {len(rd)} strategies",
                              style={"color": ACCENT, "fontSize": "10px"})
            dot = dot_live
        header = "NIFTY50 · Walk-Forward OOS · Live Results"
    else:
        rd     = demo_returns()
        mdf    = None
        status = html.Div("● Demo mode — run scripts 01–04 to load live results",
                          style={"color": YELLOW, "fontSize": "10px"})
        dot    = dot_demo
        header = "NIFTY50 · Walk-Forward OOS · Demo Data"

    return (serialize_returns(rd),
            mdf.to_dict() if mdf is not None else None,
            status, header, dot)


@app.callback(
    Output("kpi-row", "children"),
    Input("store-returns", "data"),
    Input("store-metrics", "data"),
)
def update_kpis(rd_data, mdf_data):
    if not rd_data:
        return html.Div()
    rd  = deserialize_returns(rd_data)
    mdf = pd.DataFrame(mdf_data) if mdf_data else None
    return build_kpi_row(rd, mdf)


@app.callback(
    Output("tab-body",        "children"),
    Input("nav-tabs",         "active_tab"),
    Input("store-returns",    "data"),
    Input("store-metrics",    "data"),
    Input("strategy-select",  "value"),
)
def render_tab(tab, rd_data, mdf_data, selected):
    if not rd_data:
        return html.Div("Loading…", style={"color": MUTED, "padding": "40px"})

    rd  = deserialize_returns(rd_data)
    mdf = pd.DataFrame(mdf_data) if mdf_data else None
    sel = selected or list(rd.keys())

    # ── EQUITY CURVES ────────────────────────────────────────
    if tab == "eq":
        return html.Div([
            _card(dcc.Graph(figure=fig_equity(rd, sel),
                            config={"displayModeBar": True,
                                    "modeBarButtonsToRemove": ["lasso2d","select2d"],
                                    "toImageButtonOptions": {"format":"png","scale":2}}),
                  extra_style={"padding": "4px", "marginBottom": "16px"}),
            _card(dcc.Graph(figure=fig_drawdown(rd, sel),
                            config={"displayModeBar": False}),
                  extra_style={"padding": "4px"}),
        ])

    # ── PERFORMANCE TABLE ────────────────────────────────────
    elif tab == "perf":
        rows = {LABELS.get(nm, nm): calc_metrics(ret) for nm, ret in rd.items()}
        df_t = pd.DataFrame(rows).T.reset_index().rename(columns={"index": "Strategy"})

        col_fmt = []
        for c in df_t.columns:
            if c == "Strategy":
                col_fmt.append({"name": c, "id": c, "type": "text"})
            else:
                col_fmt.append({"name": c, "id": c, "type": "numeric",
                                "format": {"specifier": ".3f"}})

        tbl = dash_table.DataTable(
            data=df_t.to_dict("records"),
            columns=col_fmt,
            sort_action="native",
            style_table={"overflowX": "auto", "borderRadius": "10px"},
            style_header={
                "backgroundColor": "#080F28",
                "color": MUTED,
                "fontWeight": "600",
                "fontSize": "10px",
                "letterSpacing": "0.8px",
                "textTransform": "uppercase",
                "border": "none",
                "borderBottom": f"1px solid {BORDER}",
                "padding": "12px 16px",
            },
            style_cell={
                "backgroundColor": "transparent",
                "color": TEXT,
                "fontSize": "12px",
                "border": "none",
                "borderBottom": "1px solid #0D1B35",
                "padding": "11px 16px",
                "fontFamily": "'Inter', monospace",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"},
                 "backgroundColor": "rgba(13,27,53,0.45)"},
                {"if": {"filter_query": '{Strategy} = "AdaptiveBeta (Ours)"'},
                 "backgroundColor": hex_to_rgba(ACCENT, 0.06),
                 "color": ACCENT, "fontWeight": "600"},
            ],
        )

        return html.Div([
            _card(tbl, extra_style={"padding": "0", "overflow": "hidden",
                                    "marginBottom": "18px"}),
            _card(dcc.Graph(figure=fig_rolling_sharpe(rd),
                            config={"displayModeBar": False}),
                  extra_style={"padding": "4px", "marginBottom": "18px"}),
            _card(dcc.Graph(figure=fig_return_distribution(rd, sel),
                            config={"displayModeBar": False}),
                  extra_style={"padding": "4px"}),
        ])

    # ── RISK & STRESS ────────────────────────────────────────
    elif tab == "risk":
        strategy_opts = [{"label": LABELS.get(s, s), "value": s}
                         for s in STRATEGIES if s in rd]
        default_strat = "adaptive_beta" if "adaptive_beta" in rd else (
            strategy_opts[0]["value"] if strategy_opts else None)

        return html.Div([
            _card(dcc.Graph(figure=fig_stress_bar(rd),
                            config={"displayModeBar": False}),
                  extra_style={"padding": "4px", "marginBottom": "18px"}),
            _card(html.Div([
                html.Div([
                    html.Div("SELECT STRATEGY FOR HEATMAP",
                             style={"color": MUTED, "fontSize": "9px",
                                    "letterSpacing": "1.2px", "fontWeight": "600",
                                    "marginBottom": "10px"}),
                    dcc.Dropdown(
                        id="heatmap-strat",
                        options=strategy_opts,
                        value=default_strat,
                        clearable=False,
                        style={"width": "300px"},
                        className="dark-dd",
                    ),
                ], style={"marginBottom": "16px"}),
                dcc.Graph(id="monthly-heat", config={"displayModeBar": False}),
                dcc.Graph(id="uw-chart",     config={"displayModeBar": False}),
            ]), extra_style={"padding": "22px"}),
        ])

    # ── SIGNALS ──────────────────────────────────────────────
    elif tab == "sig":
        reb = load_rebalance()
        if reb is not None and len(reb):
            sc = reb["signal"].value_counts().reset_index()
            sc.columns = ["signal", "count"]
            colors_map = {"REBALANCE": YELLOW, "MIN_VARIANCE": RED, "HOLD": "#64748B"}
            fig_pie = go.Figure(go.Pie(
                labels=sc["signal"], values=sc["count"],
                marker_colors=[colors_map.get(x, ACCENT) for x in sc["signal"]],
                hole=0.60,
                textinfo="label+percent",
                textfont=dict(color=TEXT, size=11),
                hovertemplate="<b>%{label}</b><br>Count: %{value}  (%{percent})<extra></extra>",
            ))
            fig_pie.update_layout(**CHART_BASE, height=320,
                                  title=dict(text="Rebalance Signal Distribution",
                                             font=dict(color=TEXT, size=13), x=0.01))
            tbl_reb = dash_table.DataTable(
                data=reb.sort_values("date", ascending=False).head(50).to_dict("records"),
                columns=[{"name": c, "id": c} for c in reb.columns],
                page_size=12, sort_action="native",
                style_table={"overflowX": "auto"},
                style_header={"backgroundColor": "#080F28", "color": MUTED,
                              "fontSize": "10px", "border": "none",
                              "borderBottom": f"1px solid {BORDER}", "padding": "10px 14px"},
                style_cell={"backgroundColor": "transparent", "color": TEXT,
                            "fontSize": "11px", "border": "none",
                            "borderBottom": "1px solid #0D1B35", "padding": "9px 14px"},
            )
            return html.Div([
                _card(dcc.Graph(figure=fig_pie, config={"displayModeBar": False}),
                      extra_style={"padding": "4px", "marginBottom": "18px"}),
                _card(tbl_reb, extra_style={"padding": "0", "overflow": "hidden"}),
            ])
        else:
            return _card(html.Div([
                html.Div("⚡", style={"fontSize": "52px", "textAlign": "center",
                                      "marginBottom": "20px"}),
                html.H4("AdaptiveBeta Signal Logic", style={"color": TEXT, "textAlign": "center",
                                                             "marginBottom": "20px"}),
                html.Div([
                    html.P([html.Span("Step 1  ", style={"color": ACCENT, "fontWeight": "700"}),
                            "Every 5 trading days — LSTM/XGBoost predicts 20d forward beta-volatility"]),
                    html.P([html.Span("Step 2  ", style={"color": YELLOW, "fontWeight": "700"}),
                            "If VIX > 25  →  MIN_VARIANCE override (capital protection)"]),
                    html.P([html.Span("Step 3  ", style={"color": ACCENT, "fontWeight": "700"}),
                            "If predicted betavol > Q55 threshold  →  REBALANCE"]),
                    html.P([html.Span("    Bull regime:", style={"color": ACCENT}),
                            "  Max-Sharpe beta-constrained portfolio"]),
                    html.P([html.Span("    Bear regime:", style={"color": RED}),
                            "  Risk-Parity (diversified defensive)"]),
                                        html.P([html.Span("    Transition:", style={"color": PURPLE}),
                            "  Max-Sharpe (opportunistic)"]),
                    html.P([html.Span("Step 4  ", style={"color": "#64748B", "fontWeight": "700"}),
                            "Otherwise  →  HOLD (save 0.08% round-trip cost)"]),
                ], style={"color": MUTED, "fontSize": "13px", "lineHeight": "2.2",
                          "maxWidth": "520px", "margin": "0 auto"}),
            ], style={"textAlign": "left", "maxWidth": "600px", "margin": "0 auto"}),
            extra_style={"padding": "40px"})

    return html.Div("Select a tab.", style={"color": MUTED})


@app.callback(
    Output("monthly-heat", "figure"),
    Output("uw-chart",     "figure"),
    Input("heatmap-strat", "value"),
    Input("store-returns", "data"),
)
def update_risk_charts(strategy, rd_data):
    if not rd_data or not strategy:
        return go.Figure(), go.Figure()
    rd = deserialize_returns(rd_data)
    return fig_monthly_heatmap(rd, strategy), fig_underwater(rd, strategy)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)

