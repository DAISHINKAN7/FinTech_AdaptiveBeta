# AdaptiveBeta: AI-Powered Dynamic Beta Prediction & Portfolio Optimisation

> **Kunal** | M.Tech AI & ML, Symbiosis Institute of Technology, Pune | 2026

---

## The Problem

Classical CAPM assumes beta — the sensitivity of a stock's returns to market moves — is **constant**. It isn't.

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

## Architecture

```
Raw Data (Google Drive)
    │
    ▼
01_feature_engineering.ipynb
    ├── Align all data to NIFTY50 trading calendar
    ├── Compute log returns
    ├── Rolling OLS beta (30d, 60d, 120d)
    ├── Beta volatility panels
    └── Market + per-stock feature matrices
    │
    ▼
02_beta_prediction_models.ipynb
    ├── XGBoost baseline + SHAP explainability
    ├── Kalman filter beta estimation
    ├── LSTM training (HuberLoss, AdamW, early stopping)
    └── HMM regime classification (3 states)
    │
    ▼
03_portfolio_optimisation.ipynb
    ├── Threshold calibration (75th pct of training betavol)
    ├── Rebalance signal logic (VIX override + beta threshold)
    └── Portfolio optimisers (max-Sharpe MVO, min-var, risk parity)
    │
    ▼
04_backtesting.ipynb
    ├── Walk-forward engine (train 3y, test 1y, roll 1y)
    ├── 5-strategy comparison with transaction costs
    ├── Performance metrics (Sharpe, Sortino, Calmar, max DD)
    ├── Stress event analysis (COVID, ADANI, IL&FS, 2018 hike)
    └── QuantStats HTML tearsheet
```

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
# For GPU support (recommended for LSTM):
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 3. Set up Google Drive

Your raw data is already at `/content/drive/MyDrive/AI_Finance_Project/raw_data/`.

Expected structure:
```
AI_Finance_Project/
├── raw_data/
│   ├── stocks/
│   │   ├── all_stocks_prices.csv
│   │   └── all_stocks_volume.csv
│   ├── market/
│   │   ├── nifty50.csv
│   │   └── india_vix.csv
│   ├── macro/
│   │   ├── usdinr.csv
│   │   ├── crude_oil.csv
│   │   ├── risk_free_rate_91d_daily.csv
│   │   └── repo_rate_daily.csv
│   ├── flows/
│   │   ├── fii_flows.csv
│   │   └── dii_flows.csv
│   └── sector/
│       ├── nifty_bank.csv
│       ├── nifty_it.csv
│       └── nifty_fmcg.csv
├── features/   ← created by notebooks
├── models/     ← created by notebooks
└── results/    ← created by notebooks
```

### 4. Run the notebooks in order

Open in **Google Colab** (GPU runtime recommended):

1. `notebooks/01_feature_engineering.ipynb` — ~5 min
2. `notebooks/02_beta_prediction_models.ipynb` — ~30–60 min (LSTM training)
3. `notebooks/03_portfolio_optimisation.ipynb` — ~5 min
4. `notebooks/04_backtesting.ipynb` — ~30–60 min

### 5. Launch the Streamlit dashboard

```bash
# Locally
streamlit run dashboard.py

# In Colab
!pip install pyngrok
from pyngrok import ngrok
import subprocess
subprocess.Popen(['streamlit', 'run', 'dashboard.py'])
public_url = ngrok.connect(8501)
print(public_url)
```

### 6. Run the portfolio website

```bash
cd website
npm install
npm run dev
# Open http://localhost:3000
```

---

## Repository Structure

```
adaptivebeta/
├── notebooks/                          ← 4 sequential Colab notebooks
├── src/
│   ├── config.py                       ← All constants + tickers
│   ├── data/loader.py                  ← Data loading + alignment
│   ├── data/validator.py               ← Data quality checks
│   ├── features/returns.py             ← Log return utilities
│   ├── features/beta.py                ← Rolling OLS beta + betavol
│   ├── features/macro.py               ← Market + macro feature builder
│   ├── models/lstm.py                  ← BetaLSTM + training loop
│   ├── models/xgboost_model.py         ← XGBoost + SHAP
│   ├── models/kalman.py                ← Kalman filter beta
│   ├── models/hmm_regime.py            ← HMM regime classifier
│   ├── portfolio/signal.py             ← Threshold trigger + routing
│   ├── portfolio/optimiser.py          ← MVO, min-var, risk parity
│   ├── portfolio/constraints.py        ← Weight utilities
│   ├── backtest/engine.py              ← WalkForwardBacktester
│   ├── backtest/metrics.py             ← Performance metrics + stress
│   ├── backtest/transaction_costs.py   ← Cost model
│   └── utils/                          ← Logging + plotting
├── dashboard.py                        ← Streamlit results dashboard
├── requirements.txt                    ← Python dependencies
└── website/                            ← Next.js portfolio site
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
