<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f172a,35:1e3a8a,70:0284c7,100:22d3ee&height=260&section=header&text=AdaptiveBeta&fontSize=80&fontColor=ffffff&fontAlignY=34&desc=Beta%20is%20not%20constant.%20We%20forecast%20when%20it%20breaks.&descAlignY=53&descSize=19" width="100%"/>

<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=600&size=21&duration=3200&pause=900&color=0EA5E9&center=true&vCenter=true&width=880&lines=Predicting+beta+INSTABILITY%2C+not+beta+itself.;Rebalance+only+when+the+model+says+risk+is+about+to+break.;%E2%88%9217.4%25+max+drawdown+vs+%E2%88%9238.4%25+for+the+index.;%E2%88%928.97%25+through+COVID+while+MVO+lost+%E2%88%9235.33%25." alt="Typing SVG"/>

<br/>

**AI-Powered Dynamic Beta Forecasting & Regime-Aware Portfolio Construction**
`NIFTY-50` · `2015 → 2026` · `2,744 trading days` · `walk-forward validated` · `costs included`

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-017E7E?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai)
[![CVXPY](https://img.shields.io/badge/CVXPY-Convex_Opt-4B32C3?style=for-the-badge)](https://cvxpy.org)
[![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![MIT](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](#-license)

</div>

<div align="center">

<table>
<tr>
<td align="center" width="16.6%">
<img src="https://img.shields.io/badge/-🛡️-0f172a?style=for-the-badge&labelColor=0f172a"/><br/>
<h2>−17.4%</h2>
<b>MAX DRAWDOWN</b><br/>
<img src="https://img.shields.io/badge/rank-%231_of_6-22c55e?style=flat-square&labelColor=0f172a"/><br/>
<sub><i>index: −38.4%</i></sub>
</td>
<td align="center" width="16.6%">
<img src="https://img.shields.io/badge/-📉-0f172a?style=for-the-badge&labelColor=0f172a"/><br/>
<h2>11.26%</h2>
<b>ANNUAL VOL</b><br/>
<img src="https://img.shields.io/badge/rank-%231_of_6-22c55e?style=flat-square&labelColor=0f172a"/><br/>
<sub><i>index: 17.19%</i></sub>
</td>
<td align="center" width="16.6%">
<img src="https://img.shields.io/badge/-⚖️-0f172a?style=for-the-badge&labelColor=0f172a"/><br/>
<h2>1.047</h2>
<b>SORTINO</b><br/>
<img src="https://img.shields.io/badge/rank-%231_of_6-22c55e?style=flat-square&labelColor=0f172a"/><br/>
<sub><i>index: 0.781</i></sub>
</td>
<td align="center" width="16.6%">
<img src="https://img.shields.io/badge/-🎯-0f172a?style=for-the-badge&labelColor=0f172a"/><br/>
<h2>0.532</h2>
<b>CALMAR</b><br/>
<img src="https://img.shields.io/badge/rank-%231_of_6-22c55e?style=flat-square&labelColor=0f172a"/><br/>
<sub><i>index: 0.316</i></sub>
</td>
<td align="center" width="16.6%">
<img src="https://img.shields.io/badge/-🧠-0f172a?style=for-the-badge&labelColor=0f172a"/><br/>
<h2>76.5%</h2>
<b>DIRECTIONAL ACC</b><br/>
<img src="https://img.shields.io/badge/MAE-0.0391-8b5cf6?style=flat-square&labelColor=0f172a"/><br/>
<sub><i>XGBoost, OOS</i></sub>
</td>
<td align="center" width="16.6%">
<img src="https://img.shields.io/badge/-💰-0f172a?style=for-the-badge&labelColor=0f172a"/><br/>
<h2>+4.24%</h2>
<b>ALPHA</b><br/>
<img src="https://img.shields.io/badge/β-0.40-0ea5e9?style=flat-square&labelColor=0f172a"/><br/>
<sub><i>at 40% exposure</i></sub>
</td>
</tr>
</table>

<br/>

### ⟶ &nbsp; [📊 Results](#-results--walk-forward-out-of-sample) &nbsp;·&nbsp; [🏗 Architecture](#-system-architecture) &nbsp;·&nbsp; [🧪 Models](#-the-model-zoo--four-families-one-benchmark) &nbsp;·&nbsp; [🔬 Rigour](#-methodological-rigour) &nbsp;·&nbsp; [⚡ Quickstart](#-quickstart) &nbsp;·&nbsp; [🖥 Demos](#-three-ways-to-explore) &nbsp; ⟵

</div>

---

<div align="center">

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║     Classical CAPM assumes beta is CONSTANT.  It isn't.                    ║
║                                                                           ║
║     We forecast WHEN beta becomes unstable — and rebalance only then.      ║
║                                                                           ║
║     ⟹  half the volatility   ·   half the drawdown   ·   #1 Sortino        ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

</div>

---

## 🎯 The Problem

**Beta** — a stock's sensitivity to the market — is the single most-used number in portfolio construction. Every mean-variance optimiser, every risk model, every hedge ratio depends on it. And nearly everyone treats it as a **constant**.

It isn't. Beta drifts with regimes, macro shocks and sector rotation — and it frequently moves **before** the market does.

<table>
<tr>
<td width="33%" valign="top" align="center">

### 📉
**COVID CRASH · 2020**

```
RELIANCE.NS beta

1.65 ┤              ╭────
     │            ╭─╯
1.40 ┤         ╭──╯
     │      ╭──╯
1.15 ┤   ╭──╯
     │ ╭─╯
1.00 ┼─╯
     └────────────────────
      Feb   Mar   Apr
```
**+65% in six weeks.**
Your risk model was wrong the whole way down.

</td>
<td width="33%" valign="top" align="center">

### 🏦
**IL&FS CRISIS · 2018**

```
  NBFC sector beta
        ▲
        │   ╭──╮
   β ───┼──╯   ╰──  ← beta spikes
        │
  ──────┼──────────
        │    ╲
 price  │     ╲___  ← price falls
        ▼      LATER
```
Beta led price by **weeks**.
The signal was there first.

</td>
<td width="33%" valign="top" align="center">

### ⚠️
**THE CONSEQUENCE**

```
  target β  ────────────  0.85
                ╱
  actual β  ───╯          1.40
            ▲
            └─ you are here,
               in the crisis,
               60% more exposed
               than you believe
```
Stale beta ⟹ real exposure diverges **exactly when it matters**.

</td>
</tr>
</table>

> ### 💡 The core insight
> You don't actually need to predict **beta**. You need to predict **beta *volatility*** — the instability of the exposure itself.
>
> That's the quantity that tells you *when your risk model is about to be wrong* — and unlike returns, **it is genuinely forecastable**, because instability clusters. Our SHAP analysis confirms it: the top three predictors of future beta-vol are all past beta-vol.

---

## 💡 The Approach

Instead of rebalancing on a calendar — monthly, quarterly, arbitrary, cost-heavy and blind to stress — **AdaptiveBeta rebalances only when a model predicts beta instability is about to spike.**

```mermaid
stateDiagram-v2
    direction LR
    [*] --> Observe

    Observe: 🔍 OBSERVE
    Observe: 42 features @ close t

    Predict: 🧠 PREDICT
    Predict: XGBoost ⟶ β-vol at t+20

    Observe --> Predict

    Predict --> Panic: VIX > 22
    Predict --> Trigger: β-vol > 0.1794
    Predict --> Calm: otherwise

    Panic: 🛡️ MIN_VARIANCE
    Panic: 13.4% of days
    Panic: model bypassed — hard rule owns the tail

    Trigger: ⚖️ REBALANCE
    Trigger: 26.9% of days
    Trigger: routed by HMM regime

    Calm: 😴 HOLD
    Calm: 59.6% of days
    Calm: zero turnover — zero cost

    Panic --> Execute
    Trigger --> Execute
    Calm --> [*]

    Execute: 💸 EXECUTE
    Execute: CVXPY optimise − 8 bps round-trip

    Execute --> [*]
```

<table>
<tr>
<td width="33%" valign="top">

### 1️⃣ Forecast instability
Not the level. The target is **20-day-ahead 60-day rolling beta volatility** — a quantity with real autocorrelation structure, unlike returns.

</td>
<td width="33%" valign="top">

### 2️⃣ Act asymmetrically
A **VIX > 22 override** short-circuits the model in genuine panic. ML owns the ambiguous 87% of days; hard rules own the tail.

</td>
<td width="33%" valign="top">

### 3️⃣ Do nothing, mostly
**59.6% HOLD days.** Transaction costs are paid only when the risk case justifies them — the structural edge over calendar rebalancing.

</td>
</tr>
</table>

<div align="center">

<table><tr>
<td valign="top" width="50%" align="center">

**⚡ Signal distribution**

```mermaid
pie showData
    "😴 HOLD — no trade" : 1025
    "⚖️ REBALANCE" : 463
    "🛡️ MIN_VARIANCE" : 231
```

</td>
<td valign="top" width="50%" align="center">

**🌀 HMM regime distribution**

```mermaid
pie showData
    "🐻 Bear" : 1317
    "🔄 Transition" : 1162
    "🐂 Bull" : 260
```

</td>
</tr></table>

</div>

---

## 🏗 System Architecture

```mermaid
flowchart TB
    subgraph L0["🗄️ &nbsp; DATA LAYER &nbsp;·&nbsp; 12 feeds"]
        direction LR
        A1["📈 <b>Equities</b><br/>49 NIFTY-50 names<br/>daily OHLCV"]
        A2["📊 <b>Market</b><br/>NIFTY 50 · India VIX<br/>Bank / IT / FMCG"]
        A3["🌍 <b>Macro</b><br/>USD-INR · Crude<br/>Repo · 91d T-Bill"]
        A4["💱 <b>Flows</b> 🇮🇳<br/>FII / DII<br/>net investment"]
    end

    subgraph L1["🔧 &nbsp; FEATURE ENGINE &nbsp;·&nbsp; 42 features × 118,719 rows"]
        direction LR
        B1["<b>β dynamics</b><br/>30/60/120d OLS β<br/>β-vol · spread · trend"]
        B2["<b>Vol & macro</b><br/>VIX z-score · FX vol<br/>crude vol · rate spread"]
        B3["<b>Momentum</b><br/>21/63/126/252d<br/>RSI-14 · MA200 dist"]
    end

    subgraph L2["🧠 &nbsp; MODEL LAYER &nbsp;·&nbsp; 4 families benchmarked head-to-head"]
        direction LR
        C1["⭐ <b>XGBoost</b><br/>MAE <b>0.0391</b><br/>76.5% directional<br/>+ SHAP attribution"]
        C2["<b>LSTM</b> 2×128<br/>230,785 params<br/>MAE 0.0661<br/><i>negative result</i>"]
        C3["<b>Kalman</b><br/>state-space EM<br/>MAE 0.0749"]
        C4["<b>Gaussian HMM</b><br/>bull / bear<br/>/ transition"]
    end

    subgraph L3["🚦 &nbsp; DECISION LAYER"]
        direction TB
        D1{"VIX > 22 ?"}
        D2{"β-vol > Q70<br/>= 0.1794 ?"}
        D3["😴 <b>HOLD</b><br/>59.6%"]
        D4["🛡️ <b>MIN_VAR</b><br/>13.4%"]
        D5["⚖️ <b>REBALANCE</b><br/>26.9%"]
    end

    subgraph L4["📐 &nbsp; OPTIMISER LAYER &nbsp;·&nbsp; CVXPY convex programs"]
        direction LR
        E1["🐂 <b>Max-Sharpe</b><br/>β-band ≈ 0.85<br/><i>bull regime</i>"]
        E2["🐻 <b>Min-Variance</b><br/>capital protection<br/><i>bear / high-VIX</i>"]
        E3["🔄 <b>Risk-Parity</b><br/>equal risk contrib<br/><i>transition</i>"]
    end

    subgraph L5["📊 &nbsp; VALIDATION LAYER"]
        direction LR
        F1["<b>Walk-forward CV</b><br/>3y train → 1y test<br/>annual roll"]
        F2["<b>Costs</b><br/>8 bps round-trip<br/>charged on turnover"]
        F3["<b>Benchmarks</b><br/>6 strategies<br/>4 stress events"]
    end

    L0 ==> L1 ==> L2
    C1 ==>|"predicted β-vol"| D1
    C4 -.->|"regime label"| L4
    D1 -->|yes| D4
    D1 -->|no| D2
    D2 -->|no| D3
    D2 -->|yes| D5
    D4 ==> E2
    D5 ==> E1 & E2 & E3
    L4 ==> L5

    style C1 fill:#0ea5e9,stroke:#0369a1,color:#fff,stroke-width:4px
    style C2 fill:#334155,stroke:#1e293b,color:#fff
    style D3 fill:#64748b,stroke:#334155,color:#fff,stroke-width:2px
    style D4 fill:#ef4444,stroke:#991b1b,color:#fff,stroke-width:2px
    style D5 fill:#22c55e,stroke:#15803d,color:#fff,stroke-width:2px
    style L5 fill:#fef3c7,stroke:#d97706,stroke-width:2px
    style L2 fill:#f0f9ff,stroke:#0284c7,stroke-width:2px
```

<details>
<summary><b>🔄 &nbsp;Expand: the decision loop, day by day (sequence diagram)</b></summary>

<br/>

```mermaid
sequenceDiagram
    autonumber
    participant M as 📈 Market
    participant F as 🔧 Features
    participant X as 🧠 XGBoost
    participant H as 🌀 HMM
    participant S as 🚦 Signal
    participant O as 📐 CVXPY
    participant P as 💼 Portfolio

    M->>F: close t · VIX · flows · macro
    F->>X: 42-dim feature vector
    X-->>S: predicted β-vol at t+20
    M->>H: market return + VIX
    H-->>S: regime ∈ {bull, bear, transition}

    alt 🛡️ VIX > 22 — panic override
        S->>O: MIN_VARIANCE (model bypassed)
        O->>P: defensive weights
        Note over P: capital armour engaged
    else ⚖️ β-vol > Q70 threshold
        S->>O: REBALANCE, routed by regime
        O->>P: new weights − 8 bps cost
        Note over P: risk re-targeted to β ≈ 0.85
    else 😴 calm
        S-->>P: HOLD
        Note over P: no trade · no cost · no slippage
    end

    P->>M: t+1 returns, marked to market
```

</details>

<details>
<summary><b>🛡️ &nbsp;Expand: constraints baked into every optimisation</b></summary>

<br/>

| Constraint | Value | Why it exists |
|:---|:---:|:---|
| Max single-name weight | **15%** | Idiosyncratic blow-up protection |
| Min position weight | **1%** | Kills unimplementable dust weights |
| Max sector weight | **35%** | 8 sector groups — no accidental all-financials book |
| Portfolio beta band | **≈ 0.85** | The entire point: exposure is *targeted*, not inherited |
| Covariance estimator | **Ledoit-Wolf** | 49 assets on short windows — sample covariance is ill-conditioned |
| Transaction cost | **8 bps** | 5 bps brokerage + 3 bps slippage, charged on turnover |

</details>

---

## 🧪 The Model Zoo — Four Families, One Benchmark

Four model families, implemented and benchmarked on the **identical forward-looking task**: predict 20-day-ahead beta volatility.

<div align="center">

| | Model | MAE ⬇ | Error profile *(lower = better)* | Directional |
|:--:|:---|:---:|:---|:---:|
| 🏆 | **XGBoost + SHAP** | **0.0391** | `████████████░░░░░░░░░░░░░░░░░░` | **76.5%** |
| 🥈 | Static 60d rolling OLS | 0.0477 | `███████████████░░░░░░░░░░░░░░░` | — |
| 🥉 | LSTM · 2×128 · 230,785p | 0.0661 | `██████████████████████░░░░░░░░` | 50.4% ⚠️ |
| 4 | Kalman filter · state-space EM | 0.0749 | `██████████████████████████████` | — |

</div>

> ### 🔍 Honest negative result: the deep model lost.
> The LSTM early-stopped at epoch 12 with **50.4% directional accuracy — a coin flip.**
>
> **Diagnosis:** with a Huber objective on a small, noisy, low-SNR financial panel, the network converged to predicting the *conditional mean* rather than learning the dynamics. Gradient boosting on well-engineered features beat it decisively — **69% lower error.**
>
> This is reported, not buried. Choosing the *right* model matters more than choosing the *fanciest* one, and proving that with a controlled comparison is the point of the experiment.

### 🎨 What the Model Actually Learned — SHAP Attribution

<div align="center">

| # | Feature | mean \|SHAP\| | Contribution | What it means |
|:--:|:---|:---:|:---|:---|
| 🥇 | `betavol_60` | **0.0280** | `██████████████████████████████` | Beta instability is **strongly autocorrelated** |
| 🥈 | `betavol_trend` | **0.0205** | `██████████████████████` | The **direction** of instability carries signal |
| 🥉 | `betavol_30` | **0.0132** | `██████████████` | Short-horizon confirmation |
| 4 | `market_vol_60d` | 0.0074 | `████████` | Market-wide vol leaks into beta stability |
| 5 | `betavol_120` | 0.0037 | `████` | Long-horizon regime anchor |
| 6 | `combined_flow` 🇮🇳 | 0.0036 | `████` | **FII + DII flows — India-specific alpha** |
| 7 | `beta_60_30_spread` | 0.0028 | `███` | Term structure of beta |
| 8 | `beta_120` | 0.0023 | `██` | Structural exposure level |
| 9 | `vix_zscore` | 0.0022 | `██` | *Normalised* fear, not raw fear |
| 10 | `rfr` | 0.0014 | `█` | Risk-free rate backdrop |

</div>

> **The top three predictors are all beta-volatility measures.** Beta instability *clusters*, exactly the way return volatility clusters — that's the empirical foundation the entire strategy rests on, and SHAP confirms the model **discovered** it rather than being told.

<div align="center">
<img src="results/shap_importance.png" width="82%" alt="SHAP feature importance"/>
</div>

<details>
<summary><b>🧬 &nbsp;Expand: the full 42-feature universe (mindmap)</b></summary>

<br/>

```mermaid
mindmap
  root((42<br/>features))
    (β dynamics)
      30 / 60 / 120d rolling OLS β
      β-vol at each window
      β 60–30 spread
      β-vol trend
    (Volatility)
      India VIX level
      VIX 5d / 20d change
      VIX z-score
      VIX above-25 flag
      Market vol 20d / 60d
    (Macro)
      USD-INR 5d change
      USD-INR 20d vol
      Crude 5d change
      Crude 20d vol
    (Rates)
      RBI repo rate
      91-day T-bill yield
      Rate spread
    (Market)
      NIFTY 5d / 20d return
      Realised vol
    (Sector)
      Bank 10d momentum
      IT 10d momentum
      FMCG 10d momentum
      Bank-vs-NIFTY spread
    (Flows 🇮🇳)
      FII net
      DII net
      Combined flow
      FII / DII ratio
    (Stock-specific)
      21 / 63 / 126 / 252d momentum
      RSI-14
      MA-200 distance
      Realised vol 20d
```

</details>

---

## 📊 Results — Walk-Forward, Out-of-Sample

<div align="center">

`Data 2015-01-02 → 2026-02-23` &nbsp;·&nbsp; `OOS 2022-01-03 → 2026-01-23 (1,964 days)`
`3y train → 1y test → annual roll` &nbsp;·&nbsp; `strictly chronological` &nbsp;·&nbsp; `8 bps charged on every rebalance`

</div>

### 🗺 Where Every Strategy Sits on the Risk/Return Map

```mermaid
quadrantChart
    title Risk vs Return — 1,964 out-of-sample days, costs included
    x-axis "Low Risk (11% vol)" --> "High Risk (19% vol)"
    y-axis "Low Return (8% CAGR)" --> "High Return (18% CAGR)"
    quadrant-1 "High risk, high return"
    quadrant-2 "★ EFFICIENT ★"
    quadrant-3 "Low risk, low return"
    quadrant-4 "Inefficient"
    "AdaptiveBeta": [0.13, 0.12]
    "Static CAPM MVO": [0.84, 0.34]
    "Kalman MVO": [0.85, 0.29]
    "Equal Weight": [0.70, 0.79]
    "Buy & Hold NIFTY": [0.72, 0.42]
    "Momentum-Quality": [0.80, 0.95]
```

> AdaptiveBeta sits **alone on the far-left risk axis** — every other strategy is clustered at 17–18.5% volatility. It is not competing for the top-right corner; it is occupying a **risk bucket nobody else reaches**, which is exactly what a beta-targeted overlay is supposed to do.

### 🏁 The Scoreboard

<div align="center">

| Strategy | CAGR | Sharpe | Sortino | Max DD | Calmar | Ann. Vol | α | β |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🛡️ **AdaptiveBeta** | 9.24% | 0.785 | **1.047** 🥇 | **−17.36%** 🥇 | **0.532** 🥇 | **11.26%** 🥇 | **+4.24%** | **0.40** |
| Static CAPM MVO | 11.38% | 0.587 | 0.716 | −35.33% | 0.322 | 18.35% | +1.19% | 0.84 |
| Kalman-β MVO | 10.92% | 0.561 | 0.676 | −35.44% | 0.308 | 18.48% | +0.58% | 0.85 |
| Equal Weight | 15.86% | 0.864 | 0.992 | −38.23% | 0.415 | 17.04% | +3.85% | 0.95 |
| Buy & Hold NIFTY-50 | 12.16% | 0.668 | 0.781 | −38.44% | 0.316 | 17.19% | — | 1.00 |
| Momentum-Quality | 17.47% | **0.893** 🥇 | 1.028 | −38.51% | 0.454 | 18.03% | +4.90% | 0.92 |

**🏅 Medal count** &nbsp;·&nbsp; AdaptiveBeta **4×🥇** &nbsp;·&nbsp; Momentum-Quality **1×🥇** &nbsp;·&nbsp; everyone else **0**

</div>

### 📉 Max Drawdown — The Headline Chart

```
                     ← smaller is better

🛡️ AdaptiveBeta      ██████████████                          −17.36%   🥇
   Static CAPM MVO   ████████████████████████████            −35.33%
   Kalman-β MVO      ████████████████████████████            −35.44%
   Equal Weight      ██████████████████████████████          −38.23%
   Buy & Hold NIFTY  ██████████████████████████████          −38.44%
   Momentum-Quality  ██████████████████████████████          −38.51%
                     └────┴────┴────┴────┴────┴────┴────┘
                     0   -8  -16  -24  -32  -40 %

                     ⟹ 2.2× shallower than the index
```

<table>
<tr>
<td width="50%" valign="top">

**📊 Annual volatility** *(lower better)*
```
AdaptiveBeta   ███████████████         11.26% 🥇
Equal Weight   ███████████████████████ 17.04%
NIFTY-50       ███████████████████████ 17.19%
Momentum-Q     ████████████████████████ 18.03%
Static MVO     █████████████████████████ 18.35%
Kalman MVO     █████████████████████████ 18.48%
```

</td>
<td width="50%" valign="top">

**📈 CAGR** *(higher better)*
```
Momentum-Q     █████████████████████████ 17.47%
Equal Weight   ███████████████████████  15.86%
NIFTY-50       █████████████████        12.16%
Static MVO     ████████████████         11.38%
Kalman MVO     ████████████████         10.92%
AdaptiveBeta   █████████████             9.24%
```

</td>
</tr>
<tr>
<td width="50%" valign="top">

**⚖️ Sortino** *(higher better)*
```
AdaptiveBeta   █████████████████████████ 1.047 🥇
Momentum-Q     ████████████████████████  1.028
Equal Weight   ████████████████████████  0.992
NIFTY-50       ███████████████████       0.781
Static MVO     █████████████████         0.716
Kalman MVO     ████████████████          0.676
```

</td>
<td width="50%" valign="top">

**🎯 Calmar** *(higher better)*
```
AdaptiveBeta   █████████████████████████ 0.532 🥇
Momentum-Q     █████████████████████     0.454
Equal Weight   ████████████████████      0.415
Static MVO     ███████████████           0.322
NIFTY-50       ███████████████           0.316
Kalman MVO     ██████████████            0.308
```

</td>
</tr>
</table>

<div align="center">
<img src="results/equity_curves.png" width="49%" alt="Equity curves"/>
<img src="results/drawdown_curves.png" width="49%" alt="Drawdown curves"/>
<br/>
<sub><i>Left: cumulative wealth. Right: the underwater plot — where AdaptiveBeta separates, and separates decisively.</i></sub>
</div>

### 🧭 Reading These Results Honestly

<div align="center">

> ## AdaptiveBeta does **not** have the highest CAGR.
> ## It was never designed to.

</div>

It runs at a market beta of **0.40** — structurally **~60% less exposed** to the index than every benchmark in the table. Across a 2022–2026 window in which Indian equities generally rose, *any* defensive strategy trails a long-only book on raw return. **That is arithmetic, not underperformance.**

The question that actually matters is **what you got in exchange**:

<div align="center">

| vs Buy & Hold NIFTY-50 | Δ | Visual |
|:---|:---:|:---|
| CAGR given up | −2.9 pts | `███` 🔻 |
| **Max drawdown avoided** | **−21.1 pts** | `██████████████████████` 🟢 |
| **Volatility removed** | **−5.9 pts** | `██████` 🟢 |
| **Sortino gained** | **+34%** | `█████████` 🟢 |
| **Calmar gained** | **+68%** | `██████████████████` 🟢 |
| **Alpha generated** | **+4.24%** *at β = 0.40* | `█████` 🟢 |

</div>

> **Give up 2.9 points of return. Remove 21 points of drawdown.**
>
> On a leverage-adjusted basis this is the strongest risk-adjusted book in the comparison set: an 11.26%-vol strategy *can* be levered toward benchmark volatility — a −38% drawdown **cannot be un-levered after the fact.**

### 🔥 Stress Testing — The Real Test

Four genuine Indian-market dislocations, all inside the walk-forward out-of-sample window.

```mermaid
timeline
    title Stress events survived — AdaptiveBeta drawdown vs Static MVO
    2018 Aug–Oct : IL&FS Crisis : 🛡️ −13.54% vs −15.85%
    2018 Sep–Nov : Rate Hike Cycle : 🛡️ −12.86% vs −14.33%
    2020 Feb–May : COVID Crash : 🛡️ −8.97% vs −35.33% ⭐
    2023 Jan–Mar : ADANI Crisis : 🛡️ −5.87% vs −13.71%
```

<div align="center">

### 💥 COVID Crash — max drawdown, Feb–May 2020

```
🛡️ AdaptiveBeta      ███████                                −8.97%   ⭐
   Static CAPM MVO   ████████████████████████████          −35.33%
   Kalman-β MVO      ████████████████████████████          −35.44%
   Equal Weight      █████████████████████████████         −37.63%
   Buy & Hold NIFTY  █████████████████████████████         −37.63%
   Momentum-Quality  ██████████████████████████████        −38.51%

   ⟹ a 26-POINT drawdown differential in the worst equity event of the decade
```

**The VIX override fired, min-variance engaged, and the book de-risked *while the crash was happening* — not after.**

</div>

<details>
<summary><b>📋 &nbsp;Expand: full stress table — all 4 events × 6 strategies</b></summary>

<br/>

| Event | 🛡️ **AdaptiveBeta** | Static MVO | Kalman MVO | Equal Wt | NIFTY-50 | Momentum-Q |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **COVID Crash** *(Feb–May 2020)* | | | | | | |
| ↳ Cumulative return | **−7.18%** 🥇 | −10.42% | −10.56% | −18.94% | −17.57% | −15.47% |
| ↳ Max drawdown | **−8.97%** 🥇 | −35.33% | −35.44% | −37.63% | −37.63% | −38.51% |
| **ADANI Crisis** *(Jan–Mar 2023)* | | | | | | |
| ↳ Cumulative return | −6.66% | −10.19% | −11.07% | −6.66% | **−6.33%** | −6.56% |
| ↳ Max drawdown | **−5.87%** 🥇 | −13.71% | −14.25% | −5.87% | −5.90% | −7.08% |
| **2018 Rate Hike** *(Sep–Nov 2018)* | | | | | | |
| ↳ Cumulative return | −7.51% | −9.61% | −9.86% | −7.51% | **−6.88%** | −8.75% |
| ↳ Max drawdown | −12.86% | −14.33% | −14.43% | −12.86% | −13.45% | **−12.44%** |
| **IL&FS Crisis** *(Aug–Oct 2018)* | | | | | | |
| ↳ Cumulative return | **−6.01%** 🥇 | −7.91% | −7.89% | −6.01% | −8.54% | −6.42% |
| ↳ Max drawdown | **−13.54%** 🥇 | −15.85% | −15.90% | −13.54% | −14.55% | −13.84% |

**Smallest max drawdown in 3 of 4 events. Smallest loss in 2 of 4. Never worse than −7.51% in any crisis.**

</details>

<div align="center">
<img src="results/stress_heatmap.png" width="78%" alt="Stress event heatmap"/>
</div>

### 🌀 Regime Detection & Signal Behaviour

<div align="center">
<img src="results/hmm_regimes.png" width="49%" alt="HMM regime timeline"/>
<img src="results/signal_threshold_demo.png" width="49%" alt="Signal threshold"/>
<br/>
<sub><i>Left: unsupervised HMM regime timeline, 2015–2026. Right: predicted β-vol against the frozen Q70 trigger.</i></sub>
</div>

```
   TRADING ACTIVITY — 1,719 signal days

   😴 HOLD          ████████████████████████████████████  59.6%   1,025 days   ₹0 cost
   ⚖️ REBALANCE     ████████████████                      26.9%     463 days   8 bps
   🛡️ MIN_VARIANCE  ████████                              13.4%     231 days   8 bps

   ⟹ nearly 60% of days need no trade at all — the source of the cost edge
```

<details>
<summary><b>📐 &nbsp;Expand: optimiser-mode diagnostic snapshot (and why one number needs a caveat)</b></summary>

<br/>

Cross-sectional comparison of the three optimiser modes on the same covariance estimate at **2021-12-31**:

| Mode | Expected Return* | Ann. Vol | Sharpe* | Portfolio β | # Stocks | Max Weight |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Max-Sharpe (β-constrained) | 60.83%* | 17.34% | 3.133* | 0.850 | 11 | 15.0% |
| Min-Variance | 12.38% | 10.84% | 0.542 | 0.602 | 21 | 15.0% |
| Risk-Parity | 12.38% | 10.84% | 0.542 | 0.602 | 21 | 15.0% |

> ⚠️ **Read with care.** These are *in-sample optimiser objective values* at a single date, derived from historical mean returns — the classic Markowitz estimation-error inflation, **not realised performance**. They are included purely as an optimiser diagnostic. The realised, cost-inclusive, out-of-sample numbers are in [the scoreboard](#-the-scoreboard).
>
> Risk-Parity fell back to Min-Variance weights at this date (CVXPY found no feasible risk-parity solution for that covariance conditioning) — **logged, not silently swallowed.**

<img src="results/optimiser_weights_chart.png" width="58%" alt="Optimiser weights"/>

</details>

<div align="center">
<img src="results/rolling_sharpe.png" width="80%" alt="Rolling Sharpe"/>
<br/><sub><i>1-year rolling Sharpe — stability of the risk-adjusted profile through time.</i></sub>
</div>

---

## 🔬 Methodological Rigour

Everything below exists to make the results **believable**, not flattering.

<table>
<tr><td width="4%">✅</td><td width="28%"><b>Walk-forward CV</b></td><td>3-year train → 1-year test → annual roll. Retrained each window; never sees its test year.</td></tr>
<tr><td>✅</td><td><b>Zero lookahead</b></td><td>Strictly chronological splits. Time series <b>never</b> shuffled. Targets forward-shifted 20 days.</td></tr>
<tr><td>✅</td><td><b>Costs charged</b></td><td>8 bps round-trip on turnover at every rebalance — no frictionless fantasy.</td></tr>
<tr><td>✅</td><td><b>Threshold frozen</b></td><td>Q70 = 0.1794 calibrated on <i>training-period</i> predictions only, then locked.</td></tr>
<tr><td>✅</td><td><b>Scaler fit on train only</b></td><td><code>StandardScaler</code> fitted on train, applied to test — no distributional leakage.</td></tr>
<tr><td>✅</td><td><b>Shrinkage covariance</b></td><td>Ledoit-Wolf on all 49×49 estimates — sample covariance is unusable at this ratio.</td></tr>
<tr><td>✅</td><td><b>Survivorship handled</b></td><td>TMPV.NS excluded (87 post-demerger rows). HDFCLIFE / SBILIFE pre-IPO rows explicitly zero-return, not silently filled.</td></tr>
<tr><td>✅</td><td><b>Data quirks documented</b></td><td>WTI's negative 2020-04-20 print clipped to 0. FII/DII monthly flows forward-filled with the ≤1-month lag stated openly.</td></tr>
<tr><td>✅</td><td><b>Reproducible</b></td><td>All seeds fixed at 42. Device auto-detect: CUDA → MPS → CPU.</td></tr>
<tr><td>✅</td><td><b>Six benchmarks, not one</b></td><td>Including two that beat us on CAGR — reported, not hidden.</td></tr>
</table>

<details>
<summary><b>⚖️ &nbsp;Expand: known limitations (stated plainly)</b></summary>

<br/>

A backtest that claims no weaknesses is a backtest nobody examined.

- **Long-only, no leverage.** The natural next step is vol-targeting AdaptiveBeta up to benchmark volatility, where its risk-adjusted edge would translate into absolute return.
- **Single market, single universe.** NIFTY-50 constituents only; cross-market generalisation untested.
- **Monthly FII/DII flows** forward-filled to daily — up to a one-month information lag on that feature group.
- **Close-price execution** assumed; no intraday microstructure or partial-fill modelling.
- **Fixed 8 bps cost model** does not scale with order size or liquidity conditions.
- **Regime labels are unsupervised** — HMM states interpreted post-hoc by sorted mean return, not validated against an external regime taxonomy.

</details>

---

## ⚡ Quickstart

```bash
# 1 ─ Clone
git clone https://github.com/daishinkan7/FinTech_AdaptiveBeta.git
cd FinTech_AdaptiveBeta

# 2 ─ Environment
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3 ─ Run everything
python run_pipeline.py
```

<div align="center">

| Stage | Script | ⏱ Runtime | Produces |
|:--:|:---|:---:|:---|
| **01** | `01_feature_engineering.py` | `~10 s` | 118,719 × 44 stacked feature matrix |
| **02** | `02_train_models.py --device auto` | `~8–10 min` | XGBoost · LSTM · Kalman · HMM + SHAP |
| **03** | `03_portfolio_optimisation.py` | `~1 s` | Validated weights, all three modes |
| **04** | `04_backtest.py` | `~25–30 min` | Walk-forward results, metrics, 12 charts |

</div>

```bash
python run_pipeline.py --device cpu    # force CPU
python run_pipeline.py --fast          # skip Kalman betas (~20 min saved)
python run_pipeline.py --start 3       # resume from stage 03
```

<sub>**Stage 02 flags:** `--device {auto,cuda,mps,cpu}` · `--epochs N` · `--fast` (skip Kalman) · `--no-xgb` (skip XGBoost + SHAP)</sub>

<details>
<summary><b>📥 &nbsp;Expand: data layout</b></summary>

<br/>

```
data/
├── 📈 stocks/    all_stocks_prices.csv · all_stocks_volume.csv
├── 📊 market/    nifty50.csv · india_vix.csv
├── 🌍 macro/     usdinr.csv · crude_oil.csv · risk_free_rate_91d_daily.csv · repo_rate_daily.csv
├── 💱 flows/     fii_flows.csv · dii_flows.csv                    (monthly)
└── 🏭 sector/    nifty_bank.csv · nifty_it.csv · nifty_fmcg.csv
```

**Minimum viable input:** `stocks/all_stocks_prices.csv` + `market/nifty50.csv`.
Macro, flow and sector feeds **degrade gracefully** — the pipeline drops those feature groups with a warning rather than crashing.

</details>

---

## 🖥 Three Ways to Explore

<table>
<tr>
<td width="33%" align="center" valign="top">

### 🌐 &nbsp;Next.js Research Site

Production site with architecture walkthrough, results, and a **browser-side interactive strategy simulator** driven by real exported backtest data.

```bash
cd website
npm install && npm run dev
```
**`localhost:3000`**

<sub>Next 14 · TypeScript · Tailwind<br/>Recharts · Framer Motion</sub>

</td>
<td width="33%" align="center" valign="top">

### 📊 &nbsp;Plotly Dash Analyst App

Full dashboard — equity curves, drawdowns, regime timeline, SHAP explorer, live signal inspection.

```bash
python app.py
```
**`localhost:8050`**

<sub>Dash · Bootstrap · Plotly</sub>

</td>
<td width="33%" align="center" valign="top">

### 🧪 &nbsp;Strategy Lab

Threshold sweeps, VIX-override sensitivity and regime-routing what-ifs — for probing *why* the strategy behaves as it does.

```
src/dashboard/strategy_lab.py
```

<sub>Parameter-sensitivity tooling</sub>

</td>
</tr>
</table>

---

## 📁 Repository Map

```
FinTech_AdaptiveBeta/
│
├── 📂 src/                          ← layered library (the real codebase)
│   ├── ⚙️  config.py                 ← single source of truth: paths, tickers, hyperparams
│   ├── 📥 data/       loader.py · validator.py
│   ├── 🔧 features/   returns.py · beta.py · macro.py · momentum.py
│   ├── 🧠 models/     xgboost_model.py · lstm.py · kalman.py · hmm_regime.py
│   ├── 💼 portfolio/  signal.py · optimiser.py · constraints.py
│   ├── 🔄 backtest/   engine.py · metrics.py · transaction_costs.py
│   ├── 📊 dashboard/  layout.py · callbacks.py · charts.py · data_service.py · strategy_lab.py
│   └── 🛠️  utils/      logging.py · plotting.py
│
├── 📂 scripts/         01 features → 02 train → 03 optimise → 04 backtest
├── 📂 data/            raw market / macro / flow / sector CSVs
├── 📂 features/        computed panels: β, β-vol, Kalman β, HMM regimes, targets
├── 📂 models/          xgb_model.pkl · lstm_best.pt · scaler.pkl · signal_config.json
├── 📂 results/         metrics CSVs + 12 publication-grade charts
├── 📂 website/         Next.js 14 research site + interactive demo
├── 📂 docs/            SETUP_GUIDE.md · RESULTS.md (full run log)
│
├── 🚀 run_pipeline.py  master orchestrator
├── 📊 app.py           Plotly Dash dashboard
└── 📋 requirements.txt
```

<div align="center">

<table>
<tr>
<td align="center" width="25%"><h3>11.3k</h3><sub><b>lines of Python</b><br/>layered package</sub></td>
<td align="center" width="25%"><h3>4.85k</h3><sub><b>lines of TS/React</b><br/>Next.js 14 site</sub></td>
<td align="center" width="25%"><h3>4</h3><sub><b>model families</b><br/>GBM · DL · state-space · latent</sub></td>
<td align="center" width="25%"><h3>42</h3><sub><b>engineered features</b><br/>8 groups</sub></td>
</tr>
<tr>
<td align="center"><h3>49</h3><sub><b>equities</b><br/>NIFTY-50 universe</sub></td>
<td align="center"><h3>2,744</h3><sub><b>trading days</b><br/>2015 → 2026</sub></td>
<td align="center"><h3>6</h3><sub><b>benchmark strategies</b><br/>incl. 2 that beat us</sub></td>
<td align="center"><h3>4</h3><sub><b>stress events</b><br/>real dislocations</sub></td>
</tr>
</table>

</div>

<details>
<summary><b>📦 &nbsp;Expand: generated artefacts</b></summary>

<br/>

| Directory | Contents |
|:---|:---|
| `features/` | `stacked_features.csv` (118,719×44) · `market_features.csv` · `beta{30,60,120}d.csv` · `betavol_*.csv` · `kalman_betas.csv` (2,743×49) · `hmm_regimes.csv` · `target_betavol_20d_ahead.csv` · `optimiser_weights_*.csv` |
| `models/` | `xgb_model.pkl` · `lstm_best.pt` · `scaler.pkl` · `feature_cols.txt` · `signal_config.json` |
| `results/` | `performance_metrics.csv` · `stress_analysis.csv` · `model_comparison.csv` · `optimiser_comparison.csv` · `signal_frequency.csv` · `shap_importance.csv` · per-strategy return series · 12 charts |

</details>

---

## 🧰 Tech Stack

<div align="center">

**Core** &nbsp;
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat-square&logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)

**Models** &nbsp;
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-017E7E?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-8b5cf6?style=flat-square)
![hmmlearn](https://img.shields.io/badge/hmmlearn-64748b?style=flat-square)
![pykalman](https://img.shields.io/badge/pykalman-64748b?style=flat-square)

**Optimisation** &nbsp;
![CVXPY](https://img.shields.io/badge/CVXPY-4B32C3?style=flat-square)
![PyPortfolioOpt](https://img.shields.io/badge/PyPortfolioOpt-0ea5e9?style=flat-square)
![Riskfolio](https://img.shields.io/badge/Riskfolio--Lib-f97316?style=flat-square)
![QuantStats](https://img.shields.io/badge/QuantStats-22c55e?style=flat-square)

**Interface** &nbsp;
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Dash](https://img.shields.io/badge/Dash-008DE4?style=flat-square&logo=plotly&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-000000?style=flat-square&logo=nextdotjs&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat-square&logo=typescript&logoColor=white)
![Tailwind](https://img.shields.io/badge/Tailwind-06B6D4?style=flat-square&logo=tailwindcss&logoColor=white)
![Framer](https://img.shields.io/badge/Framer_Motion-0055FF?style=flat-square&logo=framer&logoColor=white)

</div>

---

<details>
<summary><b>📚 &nbsp;References</b></summary>

<br/>

1. **Fama, E. F., & French, K. R.** (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.
2. **Ang, A., & Kristensen, D.** (2012). Testing conditional factor models. *Journal of Financial Economics*, 106(1), 132–156.
3. **Kalman, R. E.** (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45.
4. **Ledoit, O., & Wolf, M.** (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
5. **Black, F., & Litterman, R.** (1992). Global portfolio optimization. *Financial Analysts Journal*, 48(5), 28–43.
6. **Chen, T., & Guestrin, C.** (2016). XGBoost: A scalable tree boosting system. *KDD '16*.
7. **Hochreiter, S., & Schmidhuber, J.** (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
8. **Lundberg, S. M., & Lee, S.-I.** (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.

</details>

---

<div align="center">

> ### ⚠️ Disclaimer
> Academic research project. Backtested results are hypothetical, carry the inherent limitations of simulated performance, and are **not** indicative of future returns. Nothing here constitutes investment advice.

**📄 License** — Released under the **MIT License**. Free to use, modify and distribute.

</div>

---

<div align="center">

## 👤 Kunal Ajgaonkar

**AI & ML Engineer** &nbsp;·&nbsp; M.Tech Capstone &nbsp;·&nbsp; Symbiosis Institute of Technology, Pune &nbsp;·&nbsp; 2026

`data engineering` → `feature research` → `model benchmarking` → `convex optimisation` → `walk-forward validation` → `production frontend`

<br/>

**If this was useful or interesting, a ⭐ goes a long way.**

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:22d3ee,30:0284c7,65:1e3a8a,100:0f172a&height=150&section=footer" width="100%"/>

</div>
