# AdaptiveBeta: AI-Powered Dynamic Beta Prediction & Portfolio Optimisation

> **Kunal** | M.Tech AI & ML, Symbiosis Institute of Technology, Pune | 2026

---

## The Problem

Classical CAPM assumes beta — the sensitivity of  stock's returns to market moves — is **constant**. It isn't.

Beta shifts with market regimes, macro conditions, and sector rotations, often **before** the market itself moves. During the COVID crash of 2020, RELIANCE.NS beta jumped from ~1.0 to ~1.65 in six weeks. During the IL&FS crisis of 2018, NBFC sector betas surged weeks before the equity selloff.

Using stale beta estimates for portfolio construction means your actual risk exposure diverges from your target precisely when it matters most — in crises.

---

## Our Approach

We train an **LSTM** to predict **20-day forward beta volatility** using India-specific signals:

| Feature Group | Features |
|---------------|----------|
| Beta dynamics | Rolling 30d/60d/120d OLS beta, beta volatility, beta spread, beta trend |
| VIX signals | India VIX level, 5d/20d change, z-score, above-25 flag |
| Macro | USD/INR rate change & vol, crude oil change & vol |
| Rates | RBI repo rate, 91-day T-bill yield, rate spread |
| Market | NIFTY50 5d/20d return, 20d/60d volatility |
| Sectors | NIFTY Bank/IT/FMCG 10d momentum, Bank-vs-NIFTY spread |
| Flows | FII/DII net investment, combined flow, FII/DII ratio |
| Stock-specific | Stock 5d return, 20d vol, return vs. market |

### The Threshold Trigger (Key Innovation)

Instead of rebalancing on a fixed calendar schedule, we trigger rebalancing **only when** the LSTM predicts that beta instability is about to spike above the 75th percentile threshold.

**Priority order:**
1. **VIX override**: If India VIX > 25 → force **MIN_VARIANCE** (market stress)
2. **Beta threshold**: If predicted betavol > 75th pct → **REBALANCE**
3. **Otherwise**: **HOLD** (save transaction costs)

**Regime routing:**
- Bull market + REBALANCE → max-Sharpe MVO with beta band constraint
- Bear market + REBALANCE → min-variance (capital protection)
- Transition + REBALANCE → risk parity

---

## Results (Walk-Forward Backtest, 2018–2024, NIFTY50)

No lookahead. 6 out-of-sample years. 0.08% round-trip transaction costs.

| Strategy | Annual Return | Sharpe | Sortino | Max Drawdown | Calmar |
|---|---|---|---|---|---|
| **AdaptiveBeta (ours)** | **18.4%** | **1.42** | **2.10** | **−18.2%** | **1.01** |
| Kalman-Beta MVO | 15.8% | 1.19 | 1.60 | −22.1% | 0.71 |
| Static CAPM MVO | 14.1% | 1.08 | 1.40 | −24.6% | 0.57 |
| Equal Weight | 13.2% | 0.94 | 1.10 | −31.4% | 0.42 |
| Buy & Hold NIFTY50 | 12.7% | 0.89 | 0.90 | −38.7% | 0.33 |

**Key insights:**
- AdaptiveBeta outperforms NIFTY50 by **+5.7% annually** with only **−18.2% max drawdown** (vs −38.7%)
- During COVID crash: AdaptiveBeta −12.4% vs NIFTY50 −32.1% — VIX trigger protected capital
- Sharpe improvement of **+0.53** vs benchmark; Calmar of **3× better** than NIFTY50

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/daishinkan7/ai_fintech.git
cd ai_fintech
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt

# For GPU support (faster LSTM training — recommended):
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 3. Put your data files locally

Create the following directory structure under the project root:

```
ai_fintech/
└── data/
    ├── stocks/
    │   ├── all_stocks_prices.csv          ← daily close prices (date, RELIANCE.NS, TCS.NS, ...)
    │   └── all_stocks_volume.csv          ← daily volume (same format)
    ├── market/
    │   ├── nifty50.csv                    ← daily (date, Open, High, Low, Close, Volume)
    │   └── india_vix.csv                  ← daily (date, Open, High, Low, Close)
    ├── macro/
    │   ├── usdinr.csv                     ← daily (date, Open, High, Low, Close)
    │   ├── crude_oil.csv                  ← daily WTI crude (date, Open, High, Low, Close)
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

**All CSV files must have a `date` column** (parsed as the index). The minimum required files to start are:
- `data/stocks/all_stocks_prices.csv`
- `data/market/nifty50.csv`

The pipeline degrades gracefully if optional files (macro, flows, sectors) are missing.

---

## Running the Pipeline

### Option A — Run all 4 scripts at once (recommended)

```bash
# Full pipeline (GPU auto-detected):
python run_pipeline.py

# Full pipeline on CPU only:
python run_pipeline.py --device cpu

# Fast mode — skips Kalman betas (saves ~20 min):
python run_pipeline.py --fast

# Resume from script 3 if script 1 & 2 already ran:
python run_pipeline.py --start 3
```

### Option B — Run scripts individually (step by step)

```bash
# Step 1: Feature engineering (~5 min)
python scripts/01_feature_engineering.py

# Step 2: Train models (~30–60 min with GPU, ~2–4 hrs CPU)
python scripts/02_train_models.py --device auto
python scripts/02_train_models.py --device auto --fast          # skip Kalman
python scripts/02_train_models.py --device cuda --epochs 30     # GPU, 30 epochs

# Step 3: Portfolio optimisation validation (~2 min)
python scripts/03_portfolio_optimisation.py

# Step 4: Walk-forward backtest (~30 min)
python scripts/04_backtest.py
python scripts/04_backtest.py --fast                            # skip Kalman strategy
```

### Script 02 CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--device` | `auto` | `auto` probes CUDA → MPS (Apple) → CPU |
| `--epochs N` | 50 | Override LSTM training epochs |
| `--fast` | off | Skip Kalman filter betas (slow step) |
| `--no-xgb` | off | Skip XGBoost training (and SHAP) |

---

## Output Files

After running the full pipeline, these directories will be populated:

```
features/
  beta30d.csv, beta60d.csv, beta120d.csv       ← rolling OLS betas
  betavol_30d.csv, betavol_60d.csv, ...        ← beta volatility panels
  target_betavol_20d_ahead.csv                 ← LSTM prediction target
  market_features.csv                          ← macro + VIX + sector features
  stacked_features.csv                         ← per-stock features (LSTM input)
  kalman_betas.csv                             ← Kalman-filtered betas
  hmm_regimes.csv                              ← bull/bear/transition labels
  optimiser_weights_*.csv                      ← validated portfolio weights

models/
  lstm_best.pt                                 ← best LSTM checkpoint
  scaler.pkl                                   ← StandardScaler for features
  feature_cols.txt                             ← ordered feature column list
  signal_config.json                           ← betavol threshold + VIX level

results/
  feature_overview.png                         ← beta / betavol time series
  shap_importance.csv / .png                   ← XGBoost SHAP feature ranks
  lstm_training_curves.png                     ← loss / val_loss over epochs
  hmm_regimes.png                              ← market regime timeline
  model_comparison.csv                         ← XGB vs LSTM vs Kalman
  signal_threshold_demo.png                    ← betavol vs threshold over time
  optimiser_weights_chart.png                  ← top-20 holdings per mode
  optimiser_comparison.csv                     ← expected return / vol / Sharpe
  strategy_returns.csv                         ← daily returns (all strategies)
  performance_metrics.csv                      ← Sharpe, Sortino, CAGR, etc.
  stress_analysis.csv                          ← COVID / ADANI / IL&FS metrics
  equity_curves.png                            ← cumulative wealth chart
  drawdown_curves.png                          ← drawdown time series
  rolling_sharpe.png                           ← 1-year rolling Sharpe
  stress_heatmap.png                           ← heatmap of stress performance
  monthly_returns_heatmap.png                  ← AdaptiveBeta monthly calendar
  pipeline_run.log                             ← full pipeline log
```

---

## Launch the Streamlit Dashboard

After running the backtest (script 04):

```bash
streamlit run dashboard.py
# Opens at http://localhost:8501
```

---

## Launch the Portfolio Website

```bash
cd website
npm install
npm run dev
# Opens at http://localhost:3000
```

---

## Repository Structure

```
adaptivebeta/
├── data/                               ← PUT YOUR CSV FILES HERE
│   ├── stocks/
│   ├── market/
│   ├── macro/
│   ├── flows/
│   └── sector/
├── scripts/
│   ├── 01_feature_engineering.py      ← data loading, beta computation
│   ├── 02_train_models.py             ← XGBoost, Kalman, LSTM, HMM
│   ├── 03_portfolio_optimisation.py   ← optimiser validation
│   └── 04_backtest.py                 ← walk-forward backtest engine
├── src/
│   ├── config.py                      ← all constants + tickers
│   ├── data/loader.py                 ← data loading + alignment
│   ├── data/validator.py              ← data quality checks
│   ├── features/returns.py            ← log return utilities
│   ├── features/beta.py               ← rolling OLS beta + betavol
│   ├── features/macro.py              ← market + macro feature builder
│   ├── models/lstm.py                 ← BetaLSTM + training loop
│   ├── models/xgboost_model.py        ← XGBoost + SHAP
│   ├── models/kalman.py               ← Kalman filter beta
│   ├── models/hmm_regime.py           ← HMM regime classifier
│   ├── portfolio/signal.py            ← threshold trigger + routing
│   ├── portfolio/optimiser.py         ← MVO, min-var, risk parity
│   ├── portfolio/constraints.py       ← weight utilities
│   ├── backtest/engine.py             ← WalkForwardBacktester
│   ├── backtest/metrics.py            ← performance metrics + stress
│   ├── backtest/transaction_costs.py  ← cost model
│   └── utils/                         ← logging + plotting
├── run_pipeline.py                    ← master runner (start here)
├── dashboard.py                       ← Streamlit results dashboard
├── requirements.txt                   ← Python dependencies
└── website/                           ← Next.js portfolio site
```

---

## Notes

- **TMPV.NS excluded** — only 87 rows post-demerger (Oct 2025)
- **HDFCLIFE.NS, SBILIFE.NS** — pre-IPO NaN rows handled with fillna(0) for returns
- **Crude oil 2020-04-20** — WTI went negative; clipped to 0
- **FII/DII flows** — monthly data, forward-filled to daily
- **Never shuffle time series** — all train/test splits are strictly chronological
- **Ledoit-Wolf shrinkage** used for all covariance estimates (more stable than sample covariance with 49 assets)
- All random seeds fixed at 42 for reproducibility

---

## Key References

- Fama, E. F., & French, K. R. (1993). Common risk factors in the returns on stocks and bonds. *Journal of Financial Economics*, 33(1), 3–56.
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35–45.
- Black, F., & Litterman, R. (1992). Global portfolio optimization. *Financial Analysts Journal*, 48(5), 28–43.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365–411.
- Ang, A., & Kristensen, D. (2012). Testing conditional factor models. *Journal of Financial Economics*, 106(1), 132–156.

---

## License

MIT License — free to use, modify, and distribute.

---

*Built as M.Tech AI & ML thesis, Symbiosis Institute of Technology, Pune, 2026.*
