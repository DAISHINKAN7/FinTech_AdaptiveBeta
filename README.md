# AdaptiveBeta: AI-Powered Dynamic Beta Prediction & Portfolio Optimisation
 
> **Kunal Ajgaonkar** | 25070149010 | M.Tech AI & ML, Symbiosis Institute of Technology, Pune | 2026

> **Krish Patel** | 25070149009 | M.Tech AI & ML, Symbiosis Institute of Technology, Pune | 2026
 
---
 
## The Problem

Classical CAPM assumes beta — the sensitivity of  stock's returns to market moves — is **constant**. It isn't.

Beta shifts with market regimes, macro conditions, and sector rotations, often **before** the market itself moves. During the COVID crash of 2020, RELIANCE.NS beta jumped from ~1.0 to ~1.65 in six weeks. During the IL&FS crisis of 2018, NBFC sector betas surged weeks before the equity selloff.

Using stale beta estimates for portfolio construction means your actual risk exposure diverges from your target precisely when it matters most — in crises.

 
Classical CAPM assumes beta — the sensitivity of a stock's returns to market moves — is **constant**. It isn't.
 
Beta shifts with market regimes, macro conditions, and sector rotations, often **before** the market itself moves. During the COVID crash of 2020, RELIANCE.NS beta jumped from ~1.0 to ~1.65 in six weeks. During the IL&FS crisis of 2018, NBFC sector betas surged weeks before the equity selloff. Using stale beta estimates for portfolio construction means your actual risk exposure diverges from your target precisely when it matters most — in crises.
 
---
 
## Our Approach
 
We train an **XGBoost model** (primary) and a **2-layer LSTM** (deep learning baseline) to predict **20-day forward beta volatility** using India-specific signals. Instead of rebalancing on a fixed calendar schedule, rebalancing is triggered **only when** the model predicts that beta instability is about to spike above a calibrated threshold — saving transaction costs while protecting capital during turbulent regimes.
 
### Feature Groups (42 total features)
 
| Feature Group | Features |
|---|---|
| Beta dynamics | Rolling 30d/60d/120d OLS beta, beta volatility, beta spread, beta trend |
| VIX signals | India VIX level, 5d/20d change, z-score, above-25 flag |
| Macro | USD/INR rate change & vol, crude oil change & vol |
| Rates | RBI repo rate, 91-day T-bill yield, rate spread |
| Market | NIFTY50 5d/20d return, 20d/60d volatility |
| Sectors | NIFTY Bank/IT/FMCG 10d momentum, Bank-vs-NIFTY spread |
| Flows | FII/DII net investment, combined flow, FII/DII ratio |
| Stock-specific | 5d/21d/63d/126d/252d momentum, RSI-14, MA200 distance, realised vol |
 
### Top 10 Features by SHAP Importance (XGBoost)
 
| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | betavol_60 | 0.0280 |
| 2 | betavol_trend | 0.0205 |
| 3 | betavol_30 | 0.0132 |
| 4 | market_vol_60d | 0.0074 |
| 5 | betavol_120 | 0.0037 |
| 6 | combined_flow | 0.0036 |
| 7 | beta_60_30_spread | 0.0028 |
| 8 | beta_120 | 0.0023 |
| 9 | vix_zscore | 0.0022 |
| 10 | rfr | 0.0014 |
 
### The Threshold Trigger (Key Innovation)
 
**Priority order:**
1. **VIX override**: If India VIX > 22 → force **MIN_VARIANCE** (market stress)
2. **Beta threshold**: If predicted betavol > Q70 of training predictions → **REBALANCE**
3. **Otherwise**: **HOLD** (save transaction costs)
 
**Signal distribution over training history:**
 
| Signal | Count | Frequency |
|---|---|---|
| HOLD | 1,025 | 59.6% |
| REBALANCE | 463 | 26.9% |
| MIN_VARIANCE | 231 | 13.4% |
 
**Regime routing on REBALANCE:**
- Bull market → max-Sharpe MVO with beta band constraint (β ≈ 0.85)
- Bear market → min-variance (capital protection)
- Transition → risk parity
 
**HMM Regime distribution (2015–2021 training period):**
 
| Regime | Days |
|---|---|
| Bear | 1,317 |
| Transition | 1,162 |
| Bull | 260 |
 
---
 
## Results — Walk-Forward Backtest (April 2026 Run)
 
**Data range:** 2015-01-02 → 2026-02-23 (2,744 trading days)
**Out-of-sample period:** 2022-01-03 → 2026-01-23 (1,964 OOS days for walk-forward strategies)
**Transaction costs:** 0.08% round-trip (0.05% brokerage + 0.03% slippage)
**Train/test split:** chronological, no lookahead bias
 
### Strategy Performance Summary
 
| Strategy | CAGR | Sharpe | Sortino | Max Drawdown | Calmar | Annual Vol |
|---|---|---|---|---|---|---|
| **AdaptiveBeta (ours)** | **9.24%** | **0.785** | **1.047** | **−17.36%** | **0.532** | **11.26%** |
| Static CAPM MVO | 11.38% | 0.587 | 0.716 | −35.33% | 0.322 | 18.35% |
| Kalman-Beta MVO | 10.92% | 0.561 | 0.676 | −35.44% | 0.308 | 18.48% |
| Equal Weight | 15.86% | 0.864 | 0.992 | −38.23% | 0.415 | 17.04% |
| Buy & Hold NIFTY50 | 12.16% | 0.668 | 0.781 | −38.44% | 0.316 | 17.19% |
| Momentum-Quality | 17.47% | 0.893 | 1.028 | −38.51% | 0.454 | 18.03% |
 
### Key Takeaways
 
- **AdaptiveBeta achieves the lowest max drawdown by a wide margin**: −17.36% vs −35% to −38.5% for all other strategies
- **Best Sortino ratio (1.047)** — highest downside risk-adjusted return
- **Best Calmar ratio (0.532)** — highest return per unit of drawdown
- **Lowest volatility (11.26%)** — roughly half the vol of all competitors (~17–18%)
- The strategy trades less (59.6% HOLD days) → lower transaction cost drag
- The primary objective of AdaptiveBeta is **capital preservation with stable risk-adjusted returns**, not maximum absolute return — confirmed by the results
 
### Stress Event Analysis
 
| Event | AdaptiveBeta | Static MVO | Kalman MVO | Equal Weight | NIFTY50 | Momentum-Q |
|---|---|---|---|---|---|---|
| **COVID Crash** (Feb–May 2020) | | | | | | |
| Cumulative Return | **−7.18%** | −10.42% | −10.56% | −18.94% | −17.57% | −15.47% |
| Max Drawdown | **−8.97%** | −35.33% | −35.44% | −37.63% | −37.63% | −38.51% |
| **ADANI Crisis** (Jan–Mar 2023) | | | | | | |
| Cumulative Return | −6.66% | −10.19% | −11.07% | −6.66% | **−6.33%** | −6.56% |
| Max Drawdown | **−5.87%** | −13.71% | −14.25% | −5.87% | −5.90% | −7.08% |
| **2018 Rate Hike** (Sep–Nov 2018) | | | | | | |
| Cumulative Return | −7.51% | −9.61% | −9.86% | −7.51% | **−6.88%** | −8.75% |
| Max Drawdown | −12.86% | −14.33% | −14.43% | −12.86% | −13.45% | **−12.44%** |
| **IL&FS Crisis** (Aug–Oct 2018) | | | | | | |
| Cumulative Return | **−6.01%** | −7.91% | −7.89% | −6.01% | −8.54% | −6.42% |
| Max Drawdown | **−13.54%** | −15.85% | −15.90% | −13.54% | −14.55% | −13.84% |
 
AdaptiveBeta has the smallest or near-smallest drawdown in **all 4 stress events**.
 
### Model Comparison
 
| Model | MAE | Direction Accuracy | Notes |
|---|---|---|---|
| XGBoost | **0.0391** | **76.5%** | Best model — used for signal generation |
| Static OLS Beta (60d) | 0.0477 | N/A | Rolling baseline, no direction prediction |
| LSTM (proposed deep model) | 0.0661 | 50.4% | 230,785 params, early stopped at epoch 12 |
| Kalman Filter | 0.0749 | N/A | State-space EM, no direction prediction |
 
**Calibrated betavol threshold:** `0.1794` (Q70 of XGBoost training predictions)
**VIX stress threshold:** `22.0`
 
### Portfolio Optimiser Comparison (as of 2021-12-31)
 
| Mode | Expected Return | Annual Vol | Sharpe | Portfolio Beta | Stocks | Max Weight |
|---|---|---|---|---|---|---|
| Max-Sharpe (beta-constrained) | 60.83% | 17.34% | 3.133 | 0.850 | 11 | 15.0% |
| Min-Variance | 12.38% | 10.84% | 0.542 | 0.602 | 21 | 15.0% |
| Risk-Parity | 12.38% | 10.84% | 0.542 | 0.602 | 21 | 15.0% |
 
---
 
## Setup
 
### 1. Clone the repository
 
```bash
git clone https://github.com/daishinkan7/fintech_adaptivebeta.git
cd FinTech_AdaptiveBeta
```
 
### 2. Create virtual environment
 
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows
```
 
### 3. Install Python dependencies
 
```bash
pip install -r requirements.txt
 
# Optional: GPU support for faster LSTM training
pip install torch --index-url https://download.pytorch.org/whl/cu118
```
 
### 4. Data directory structure
 
Create the following under the project root:
 
```
data/
├── stocks/
│   ├── all_stocks_prices.csv          ← daily close prices (date, RELIANCE.NS, TCS.NS, ...)
│   └── all_stocks_volume.csv          ← daily volume (same format)
├── market/
│   ├── nifty50.csv                    ← daily (date, Open, High, Low, Close, Volume)
│   └── india_vix.csv                  ← daily (date, Open, High, Low, Close)
├── macro/
│   ├── usdinr.csv                     ← daily (date, Open, High, Low, Close)
│   ├── crude_oil.csv                  ← daily WTI (date, Open, High, Low, Close)
│   ├── risk_free_rate_91d_daily.csv   ← daily (date, risk_free_rate)
│   └── repo_rate_daily.csv            ← daily (date, repo_rate)
├── flows/
│   ├── fii_flows.csv                  ← monthly (date, fii_net_investment)
│   └── dii_flows.csv                  ← monthly (date, dii_net_investment)
└── sector/
    ├── nifty_bank.csv                 ← daily (date, Open, High, Low, Close)
    ├── nifty_it.csv                   ← daily (date, Open, High, Low, Close)
    └── nifty_fmcg.csv                 ← daily (date, Open, High, Low, Close)
```
 
Minimum required: `data/stocks/all_stocks_prices.csv` + `data/market/nifty50.csv`
(macro, flows, sectors degrade gracefully if missing)
 
---
 
## Running the Pipeline
 
### Option A — One-shot master runner
 
```bash
python run_pipeline.py               # full pipeline, auto device
python run_pipeline.py --device cpu  # force CPU
python run_pipeline.py --fast        # skip Kalman betas (~20 min saved)
python run_pipeline.py --start 3     # resume from script 03
```
 
### Option B — Step by step
 
```bash
python scripts/01_feature_engineering.py          # ~10 sec
python scripts/02_train_models.py --device auto   # ~8–10 min on Apple MPS
python scripts/03_portfolio_optimisation.py       # ~1 sec
python scripts/04_backtest.py                     # ~25–30 min
```
 
### Script 02 CLI flags
 
| Flag | Default | Description |
|---|---|---|
| `--device` | `auto` | `auto` → CUDA → MPS (Apple Silicon) → CPU |
| `--epochs N` | 50 | Override LSTM training epochs |
| `--fast` | off | Skip Kalman betas |
| `--no-xgb` | off | Skip XGBoost + SHAP |
 
---
 
## Dashboards
 
### Option A — Plotly Dash (app.py)
 
```bash
python app.py
# Opens at http://localhost:8050
```
 
### Option B — Streamlit
 
```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```
 
### Option C — Next.js Website
 
```bash
cd website
npm install
npm run dev
# Opens at http://localhost:3000
```
 
See [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) for a comprehensive step-by-step guide covering all three interfaces.
 
---
 
## Output Files
 
```
features/
  stacked_features.csv           ← 118,719 rows × 44 cols (per-stock LSTM input)
  market_features.csv            ← 24 market-level features
  beta{30,60,120}d.csv           ← rolling OLS beta panels
  betavol_{30,60,120}d.csv       ← beta volatility panels
  target_betavol_20d_ahead.csv   ← LSTM prediction target
  kalman_betas.csv               ← Kalman-filtered betas (2,743 × 49)
  hmm_regimes.csv                ← bull / bear / transition daily labels
  optimiser_weights_*.csv        ← validated portfolio weights
 
models/
  lstm_best.pt                   ← best LSTM checkpoint (early stop epoch 12)
  xgb_model.pkl                  ← fitted XGBoost model
  scaler.pkl                     ← StandardScaler (fit on train data)
  feature_cols.txt               ← 42 ordered feature names
  signal_config.json             ← betavol threshold=0.1794, vix_threshold=22.0
 
results/
  feature_overview.png           ← beta / betavol time series
  shap_importance.csv / .png     ← XGBoost SHAP feature ranking
  lstm_training_curves.png       ← train/val loss over 12 epochs
  hmm_regimes.png                ← regime timeline 2015–2026
  signal_threshold_demo.png      ← betavol vs threshold visualisation
  optimiser_weights_chart.png    ← top holdings per optimisation mode
  optimiser_comparison.csv       ← expected return / vol / Sharpe by mode
  signal_frequency.csv           ← HOLD / REBALANCE / MIN_VARIANCE counts
  strategy_returns.csv           ← daily returns (all 6 strategies)
  performance_metrics.csv        ← CAGR / Sharpe / Sortino / MaxDD / Calmar
  stress_analysis.csv            ← 4 stress event metrics per strategy
  equity_curves.png              ← cumulative wealth chart
  drawdown_curves.png            ← underwater equity chart
  rolling_sharpe.png             ← 1-year rolling Sharpe ratio
  stress_heatmap.png             ← heatmap of stress performance
  monthly_returns_heatmap_*.png  ← calendar return heatmap per strategy
```
 
---
 
## Repository Structure
 
```
FinTech_AdaptiveBeta/
├── data/                               ← PUT YOUR CSV FILES HERE
├── scripts/
│   ├── 01_feature_engineering.py
│   ├── 02_train_models.py
│   ├── 03_portfolio_optimisation.py
│   └── 04_backtest.py
├── src/
│   ├── config.py
│   ├── data/loader.py
│   ├── data/validator.py
│   ├── features/returns.py
│   ├── features/beta.py
│   ├── features/macro.py
│   ├── features/momentum.py
│   ├── models/lstm.py
│   ├── models/xgboost_model.py
│   ├── models/kalman.py
│   ├── models/hmm_regime.py
│   ├── portfolio/signal.py
│   ├── portfolio/optimiser.py
│   ├── portfolio/constraints.py
│   ├── backtest/engine.py
│   ├── backtest/metrics.py
│   ├── backtest/transaction_costs.py
│   └── utils/
├── features/                           ← computed feature matrices
├── models/                             ← trained model artifacts
├── results/                            ← charts and CSVs
├── website/                            ← Next.js portfolio site
├── docs/
│   ├── SETUP_GUIDE.md                  ← comprehensive how-to-run guide
│   └── RESULTS.md                      ← detailed latest run results
├── run_pipeline.py                     ← master orchestrator
├── app.py                              ← Plotly Dash dashboard
├── requirements.txt
└── README.md
```
 
---
 
## Implementation Notes
 
- **TMPV.NS excluded** — only 87 rows post-demerger (Oct 2025)
- **HDFCLIFE.NS, SBILIFE.NS** — pre-IPO NaN rows filled with 0 returns (706 and 674 rows respectively)
- **Crude oil 2020-04-20** — WTI went negative; clipped to 0
- **FII/DII flows** — monthly data, forward-filled to daily (up to 1-month signal lag)
- **Never shuffle time series** — all splits are strictly chronological
- **Ledoit-Wolf shrinkage** used for all covariance estimates (stable with 49 assets)
- **All random seeds fixed at 42** for reproducibility
- **Device auto-detection**: CUDA → MPS (Apple Silicon) → CPU
- **LSTM trained on MPS** in latest run: 230,785 parameters, early stopped at epoch 12
 
---
 
## Key References
 
- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Black, F., & Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal*, 48(5), 28–43.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
- Ang, A., & Kristensen, D. (2012). Testing conditional factor models. *Journal of Financial Economics*, 106(1), 132–156.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
 
---
 
## License
 
MIT License — free to use, modify, and distribute.
 
*Built as M.Tech AI & ML capstone, Symbiosis Institute of Technology, Pune, 2026.*