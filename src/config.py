"""Central configuration for the AdaptiveBeta project.

All constants, paths, tickers, and hyperparameters live here.
Import from this module everywhere — never hardcode in notebooks.
"""

from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Google Drive root (used in Colab notebooks)
# ---------------------------------------------------------------------------
DRIVE_ROOT: Final[str] = "/content/drive/MyDrive/AI_Finance_Project"

# Local root (used when running outside Colab)
LOCAL_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Sub-directories (relative to whichever root is active)
# ---------------------------------------------------------------------------
RAW_DATA_DIR: Final[str] = "raw_data"
FEATURES_DIR: Final[str] = "features"
MODELS_DIR: Final[str] = "models"
RESULTS_DIR: Final[str] = "results"

# ---------------------------------------------------------------------------
# Stock universe — 49 tickers (TMPV.NS excluded: only 87 rows post-demerger)
# ---------------------------------------------------------------------------
TICKERS: Final[list[str]] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "BAJFINANCE.NS",
    "HCLTECH.NS",
    "SUNPHARMA.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "NESTLEIND.NS",
    "WIPRO.NS",
    "POWERGRID.NS",
    "NTPC.NS",
    "ONGC.NS",
    "TECHM.NS",
    "JSWSTEEL.NS",
    "TATASTEEL.NS",
    "M&M.NS",
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "COALINDIA.NS",
    "BAJAJFINSV.NS",
    "HDFCLIFE.NS",
    "SBILIFE.NS",
    "DRREDDY.NS",
    "DIVISLAB.NS",
    "CIPLA.NS",
    "EICHERMOT.NS",
    "HEROMOTOCO.NS",
    "APOLLOHOSP.NS",
    "BAJAJ-AUTO.NS",
    "BRITANNIA.NS",
    "GRASIM.NS",
    "INDUSINDBK.NS",
    "TATACONSUM.NS",
    "UPL.NS",
    "BPCL.NS",
    "IOC.NS",
    "HINDALCO.NS",
]

# Tickers with pre-IPO NaN rows — handle with fillna(0) for returns
LATE_IPO_TICKERS: Final[list[str]] = ["HDFCLIFE.NS", "SBILIFE.NS"]

# ---------------------------------------------------------------------------
# Raw data file paths (relative to RAW_DATA_DIR)
# ---------------------------------------------------------------------------
RAW_FILES: Final[dict[str, str]] = {
    "prices": "stocks/all_stocks_prices.csv",
    "volume": "stocks/all_stocks_volume.csv",
    "nifty50": "market/nifty50.csv",
    "india_vix": "market/india_vix.csv",
    "usdinr": "macro/usdinr.csv",
    "crude_oil": "macro/crude_oil.csv",
    "risk_free_rate": "macro/risk_free_rate_91d_daily.csv",
    "repo_rate": "macro/repo_rate_daily.csv",
    "fii_flows": "flows/fii_flows.csv",
    "dii_flows": "flows/dii_flows.csv",
    "nifty_bank": "sector/nifty_bank.csv",
    "nifty_it": "sector/nifty_it.csv",
    "nifty_fmcg": "sector/nifty_fmcg.csv",
}

# ---------------------------------------------------------------------------
# Feature engineering constants
# ---------------------------------------------------------------------------
BETA_WINDOWS: Final[list[int]] = [30, 60, 120]
TARGET_HORIZON: Final[int] = 20          # predict betavol 20 trading days ahead
BETAVOL_BASE_WINDOW: Final[int] = 60     # use 60d betavol as prediction target

# ---------------------------------------------------------------------------
# Threshold & regime constants
# ---------------------------------------------------------------------------
THRESHOLD_QUANTILE: Final[float] = 0.75  # 75th pct of training betavol
VIX_STRESS_LEVEL: Final[float] = 25.0   # VIX override level

# ---------------------------------------------------------------------------
# Walk-forward backtest constants
# ---------------------------------------------------------------------------
TRAIN_END: Final[str] = "2021-12-31"
TEST_START: Final[str] = "2022-01-01"
BACKTEST_START_YEAR: Final[int] = 2015
BACKTEST_END_YEAR: Final[int] = 2024
TRAIN_YEARS: Final[int] = 3
TEST_YEARS: Final[int] = 1
STEP_MONTHS: Final[int] = 12

# ---------------------------------------------------------------------------
# Transaction cost model
# ---------------------------------------------------------------------------
BROKERAGE: Final[float] = 0.0005   # 0.05% per trade
SLIPPAGE: Final[float] = 0.0003    # 0.03% market impact

# ---------------------------------------------------------------------------
# Portfolio construction constraints
# ---------------------------------------------------------------------------
MAX_WEIGHT: Final[float] = 0.15    # max single-stock weight
MIN_WEIGHT: Final[float] = 0.01    # minimum non-zero weight
DEFAULT_BETA_TARGET: Final[float] = 0.85   # target portfolio beta

# ---------------------------------------------------------------------------
# LSTM hyperparameters
# ---------------------------------------------------------------------------
LSTM_SEQ_LEN: Final[int] = 30
LSTM_HIDDEN: Final[int] = 64
LSTM_LAYERS: Final[int] = 2
LSTM_DROPOUT: Final[float] = 0.2
LSTM_BATCH: Final[int] = 256
LSTM_EPOCHS: Final[int] = 50
LSTM_LR: Final[float] = 1e-3
LSTM_PATIENCE: Final[int] = 10

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED: Final[int] = 42

# ---------------------------------------------------------------------------
# Strategies included in backtest
# ---------------------------------------------------------------------------
STRATEGIES: Final[list[str]] = [
    "adaptive_beta",
    "static_capm_mvo",
    "kalman_mvo",
    "equal_weight",
    "buy_hold_nifty",
]

# ---------------------------------------------------------------------------
# Stress events for analysis
# ---------------------------------------------------------------------------
STRESS_EVENTS: Final[dict[str, tuple[str, str]]] = {
    "COVID Crash": ("2020-02-01", "2020-05-01"),
    "ADANI Crisis": ("2023-01-25", "2023-03-15"),
    "2018 Rate Hike": ("2018-09-01", "2018-11-30"),
    "IL&FS Crisis": ("2018-08-01", "2018-10-31"),
}
