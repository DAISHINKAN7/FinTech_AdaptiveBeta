"""Data quality validation for the AdaptiveBeta pipeline.

Run these checks after loading data and before feature engineering to
catch shape mismatches, excessive NaN fractions, and date range issues
early — financial pipelines must not silently propagate bad data.

Typical usage:
    from src.data.validator import validate_prices, validate_feature_matrix
    validate_prices(prices, trading_days)
    validate_feature_matrix(features_df)
"""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Raised when a data quality check fails."""


def validate_prices(
    prices: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    max_nan_fraction: float = 0.05,
    expected_tickers: Optional[list[str]] = None,
) -> None:
    """Validate the stock price panel.

    Args:
        prices: DataFrame with stocks as columns and DatetimeIndex.
        trading_days: Reference NIFTY50 trading calendar.
        max_nan_fraction: Maximum tolerated NaN fraction per ticker (excluding
            known pre-IPO tickers which may have more NaNs).
        expected_tickers: Optional list of expected column names.

    Raises:
        DataValidationError: On any critical quality failure.
    """
    logger.info("Validating prices panel — shape %s", prices.shape)

    # Index alignment
    if not prices.index.equals(trading_days):
        raise DataValidationError(
            f"Prices index does not match trading calendar. "
            f"Prices: {prices.index[0]} → {prices.index[-1]}, "
            f"Calendar: {trading_days[0]} → {trading_days[-1]}"
        )

    # Expected tickers
    if expected_tickers is not None:
        missing = set(expected_tickers) - set(prices.columns)
        if missing:
            logger.warning("Missing tickers in prices: %s", missing)

    # NaN check — skip HDFCLIFE and SBILIFE (pre-IPO NaNs expected)
    ipo_tickers = {"HDFCLIFE.NS", "SBILIFE.NS"}
    for col in prices.columns:
        nan_frac = prices[col].isna().mean()
        if col not in ipo_tickers and nan_frac > max_nan_fraction:
            raise DataValidationError(
                f"Ticker {col} has {nan_frac:.1%} NaN — exceeds threshold {max_nan_fraction:.1%}"
            )

    # No future dates
    if prices.index.max() > pd.Timestamp.today():
        logger.warning("Prices contain future dates: %s", prices.index.max())

    # All prices non-negative
    neg_counts = (prices < 0).sum().sum()
    if neg_counts > 0:
        raise DataValidationError(f"Found {neg_counts} negative price values")

    logger.info("Prices validation PASSED — %d stocks, %d rows", prices.shape[1], prices.shape[0])


def validate_feature_matrix(
    features: pd.DataFrame,
    max_nan_fraction: float = 0.10,
    required_cols: Optional[list[str]] = None,
) -> None:
    """Validate the feature matrix before model training.

    Args:
        features: Feature DataFrame with DatetimeIndex.
        max_nan_fraction: Max tolerated NaN fraction across the whole matrix.
        required_cols: Columns that must be present.

    Raises:
        DataValidationError: On critical quality failure.
    """
    logger.info("Validating feature matrix — shape %s", features.shape)

    if features.empty:
        raise DataValidationError("Feature matrix is empty")

    if required_cols:
        missing = set(required_cols) - set(features.columns)
        if missing:
            raise DataValidationError(f"Missing required feature columns: {missing}")

    nan_frac = features.isna().mean().mean()
    if nan_frac > max_nan_fraction:
        raise DataValidationError(
            f"Feature matrix has {nan_frac:.1%} NaN — exceeds {max_nan_fraction:.1%}"
        )

    # Check for infinities
    inf_count = np.isinf(features.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        raise DataValidationError(f"Feature matrix contains {inf_count} infinite values")

    # Date range sanity
    if not isinstance(features.index, pd.DatetimeIndex):
        raise DataValidationError("Feature matrix must have a DatetimeIndex")

    if not features.index.is_monotonic_increasing:
        raise DataValidationError("Feature matrix index is not sorted ascending")

    logger.info(
        "Feature matrix validation PASSED — %d features, %d rows, "
        "%.2f%% NaN, date range %s → %s",
        features.shape[1],
        features.shape[0],
        nan_frac * 100,
        features.index[0].date(),
        features.index[-1].date(),
    )


def validate_returns(
    returns: pd.Series | pd.DataFrame,
    max_abs_daily_return: float = 0.30,
) -> None:
    """Validate log return series for outliers.

    Args:
        returns: Log return series or DataFrame.
        max_abs_daily_return: Alert threshold for |daily return|.

    Raises:
        DataValidationError: If returns contain infinities.
    """
    if isinstance(returns, pd.Series):
        returns = returns.to_frame()

    inf_count = np.isinf(returns).sum().sum()
    if inf_count > 0:
        raise DataValidationError(f"Returns contain {inf_count} infinite values")

    extreme = (returns.abs() > max_abs_daily_return).sum().sum()
    if extreme > 0:
        logger.warning(
            "Found %d return observations with |ret| > %.0f%% — verify these are real",
            extreme,
            max_abs_daily_return * 100,
        )


def log_data_summary(loader: object) -> None:
    """Print a human-readable summary of all loaded datasets.

    Args:
        loader: A DataLoader instance.
    """
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Trading calendar : {loader.trading_days[0].date()} → {loader.trading_days[-1].date()} ({len(loader.trading_days)} days)")
    print(f"Stocks           : {loader.prices.shape[1]} tickers, {loader.prices.shape[0]} rows")
    print(f"NIFTY50          : {loader.nifty_close.shape[0]} rows")
    print(f"India VIX        : {loader.vix.notna().sum()} non-NaN rows")
    print(f"USD/INR          : {loader.usdinr.notna().sum()} rows")
    print(f"Crude Oil        : {loader.crude.notna().sum()} rows")
    print(f"Risk-free rate   : {loader.risk_free_rate.notna().sum()} rows")
    print(f"Repo rate        : {loader.repo_rate.notna().sum()} rows")
    print(f"FII flows        : {loader.fii_flows.notna().sum()} rows (monthly → daily ff)")
    print(f"DII flows        : {loader.dii_flows.notna().sum()} rows (monthly → daily ff)")
    print("=" * 60 + "\n")
