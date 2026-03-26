"""Dataset loading and calendar alignment utilities.

All raw CSV files are loaded through this module. Every dataset is
reindexed to the NIFTY50 trading calendar and forward-filled to handle
weekends, holidays, and gaps (e.g. monthly FII/DII data).

Typical usage:
    from src.data.loader import DataLoader
    loader = DataLoader("/content/drive/MyDrive/AI_Finance_Project")
    prices = loader.prices
    market_ret = loader.market_returns
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import RAW_FILES, TICKERS

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and align all raw data files for the AdaptiveBeta pipeline.

    Args:
        root: Root directory of the project (contains raw_data/, features/, etc.)

    Raises:
        FileNotFoundError: If the root directory does not exist.
    """

    def __init__(self, root: str) -> None:
        self.root = Path(root)
        raw = self.root / "raw_data"
        if not raw.exists():
            raise FileNotFoundError(f"raw_data directory not found at {raw}")
        self._raw = raw

        # Lazily cached attributes
        self._trading_days: Optional[pd.DatetimeIndex] = None
        self._prices: Optional[pd.DataFrame] = None
        self._volume: Optional[pd.DataFrame] = None
        self._nifty: Optional[pd.Series] = None
        self._vix: Optional[pd.Series] = None
        self._usdinr: Optional[pd.Series] = None
        self._crude: Optional[pd.Series] = None
        self._rfr: Optional[pd.Series] = None
        self._repo: Optional[pd.Series] = None
        self._fii: Optional[pd.Series] = None
        self._dii: Optional[pd.Series] = None
        self._bank: Optional[pd.Series] = None
        self._nifty_it: Optional[pd.Series] = None
        self._fmcg: Optional[pd.Series] = None

        logger.info("DataLoader initialised at root=%s", self.root)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read(self, rel_path: str, date_col: str = "date") -> pd.DataFrame:
        """Read a CSV with a date index, sorted ascending.

        Args:
            rel_path: Path relative to raw_data/.
            date_col: Name of the date column.

        Returns:
            DataFrame with DatetimeIndex sorted ascending.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        full = self._raw / rel_path
        if not full.exists():
            raise FileNotFoundError(f"Data file not found: {full}")
        df = pd.read_csv(full, parse_dates=[date_col])
        df = df.set_index(date_col).sort_index()
        logger.debug("Loaded %s — shape %s", rel_path, df.shape)
        return df

    def _align(
        self,
        series_or_df: pd.DataFrame | pd.Series,
        cols: Optional[list[str]] = None,
    ) -> pd.DataFrame | pd.Series:
        """Reindex to the NIFTY50 trading calendar and forward-fill.

        Args:
            series_or_df: Input data with a DatetimeIndex.
            cols: Optional subset of columns to return (DataFrame only).

        Returns:
            Data aligned to the trading calendar.
        """
        out = series_or_df.reindex(self.trading_days).ffill()
        if cols is not None and isinstance(out, pd.DataFrame):
            out = out[cols]
        return out

    # ------------------------------------------------------------------
    # Core datasets
    # ------------------------------------------------------------------

    @property
    def trading_days(self) -> pd.DatetimeIndex:
        """NIFTY50 trading calendar (master reference index)."""
        if self._trading_days is None:
            nifty_raw = self._read(RAW_FILES["nifty50"])
            self._trading_days = nifty_raw.index
        return self._trading_days

    @property
    def nifty_close(self) -> pd.Series:
        """NIFTY50 daily close prices, aligned to trading calendar."""
        if self._nifty is None:
            df = self._read(RAW_FILES["nifty50"])
            s = df["Close"].reindex(self.trading_days).ffill()
            s.name = "NIFTY50"
            self._nifty = s
        return self._nifty

    @property
    def prices(self) -> pd.DataFrame:
        """Stock close prices for all 49 tickers, aligned to trading calendar.

        HDFCLIFE and SBILIFE have pre-IPO NaN rows — these are left as NaN
        so that returns can be computed correctly (returns will be NaN before IPO).
        """
        if self._prices is None:
            df = self._read(RAW_FILES["prices"])
            # Keep only known tickers that exist in the file
            available = [t for t in TICKERS if t in df.columns]
            missing = set(TICKERS) - set(available)
            if missing:
                logger.warning("Tickers not found in prices file: %s", missing)
            df = df[available].reindex(self.trading_days).ffill()
            self._prices = df
        return self._prices

    @property
    def volume(self) -> pd.DataFrame:
        """Stock volume data aligned to trading calendar."""
        if self._volume is None:
            df = self._read(RAW_FILES["volume"])
            available = [t for t in TICKERS if t in df.columns]
            df = df[available].reindex(self.trading_days).ffill()
            self._volume = df
        return self._volume

    # ------------------------------------------------------------------
    # Market / macro series
    # ------------------------------------------------------------------

    @property
    def vix(self) -> pd.Series:
        """India VIX close, aligned to trading calendar."""
        if self._vix is None:
            df = self._read(RAW_FILES["india_vix"])
            self._vix = self._align(df["Close"]).rename("india_vix")
        return self._vix

    @property
    def usdinr(self) -> pd.Series:
        """USD/INR exchange rate close, aligned to trading calendar."""
        if self._usdinr is None:
            df = self._read(RAW_FILES["usdinr"])
            self._usdinr = self._align(df["Close"]).rename("usdinr")
        return self._usdinr

    @property
    def crude(self) -> pd.Series:
        """WTI crude oil close (USD/barrel), clipped at 0 for Apr-2020 negative day.

        On 2020-04-20 WTI went negative (-37.63). We clip to 0 to avoid log
        return blowup. The feature engineering uses log-differences anyway so
        one clipped day has minimal impact.
        """
        if self._crude is None:
            df = self._read(RAW_FILES["crude_oil"])
            s = self._align(df["Close"]).rename("crude_oil")
            s = s.clip(lower=0)
            self._crude = s
        return self._crude

    @property
    def risk_free_rate(self) -> pd.Series:
        """91-day T-bill yield (% p.a.), forward-filled daily, aligned to calendar."""
        if self._rfr is None:
            df = self._read(RAW_FILES["risk_free_rate"])
            # Column is named 'risk_free_rate' in the CSV
            col = "risk_free_rate" if "risk_free_rate" in df.columns else df.columns[0]
            self._rfr = self._align(df[col]).rename("risk_free_rate")
        return self._rfr

    @property
    def repo_rate(self) -> pd.Series:
        """RBI repo rate (% p.a.), forward-filled daily, aligned to calendar."""
        if self._repo is None:
            df = self._read(RAW_FILES["repo_rate"])
            col = "repo_rate" if "repo_rate" in df.columns else df.columns[0]
            self._repo = self._align(df[col]).rename("repo_rate")
        return self._repo

    @property
    def fii_flows(self) -> pd.Series:
        """FII net investment (Rs Crores, monthly), forward-filled to daily."""
        if self._fii is None:
            df = self._read(RAW_FILES["fii_flows"])
            col = "fii_net_investment" if "fii_net_investment" in df.columns else df.columns[0]
            self._fii = self._align(df[col]).rename("fii_net_investment")
        return self._fii

    @property
    def dii_flows(self) -> pd.Series:
        """DII net investment (Rs Crores, monthly), forward-filled to daily."""
        if self._dii is None:
            df = self._read(RAW_FILES["dii_flows"])
            col = "dii_net_investment" if "dii_net_investment" in df.columns else df.columns[0]
            self._dii = self._align(df[col]).rename("dii_net_investment")
        return self._dii

    # ------------------------------------------------------------------
    # Sector indices
    # ------------------------------------------------------------------

    @property
    def nifty_bank(self) -> pd.Series:
        """NIFTY Bank index close, aligned to trading calendar."""
        if self._bank is None:
            df = self._read(RAW_FILES["nifty_bank"])
            self._bank = self._align(df["Close"]).rename("nifty_bank")
        return self._bank

    @property
    def nifty_it(self) -> pd.Series:
        """NIFTY IT index close, aligned to trading calendar."""
        if self._nifty_it is None:
            df = self._read(RAW_FILES["nifty_it"])
            self._nifty_it = self._align(df["Close"]).rename("nifty_it")
        return self._nifty_it

    @property
    def nifty_fmcg(self) -> pd.Series:
        """NIFTY FMCG index close, aligned to trading calendar."""
        if self._fmcg is None:
            df = self._read(RAW_FILES["nifty_fmcg"])
            self._fmcg = self._align(df["Close"]).rename("nifty_fmcg")
        return self._fmcg

    # ------------------------------------------------------------------
    # Return series
    # ------------------------------------------------------------------

    @property
    def market_returns(self) -> pd.Series:
        """Daily log returns of NIFTY50."""
        ret = np.log(self.nifty_close / self.nifty_close.shift(1))
        return ret.rename("market_ret")

    @property
    def stock_returns(self) -> pd.DataFrame:
        """Daily log returns for all stocks.

        HDFCLIFE and SBILIFE: NaN before IPO, then valid returns after listing.
        """
        ret = np.log(self.prices / self.prices.shift(1))
        return ret
