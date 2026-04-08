"""Walk-forward backtesting engine.

Implements strict walk-forward cross-validation (no lookahead bias):
- Training window: 3 years
- Test window: 1 year
- Roll: 1 year forward
- First OOS year: 2018 (after 3 years of training from 2015)

Strategies simulated:
1. adaptive_beta: LSTM threshold trigger + regime routing
2. static_capm_mvo: Standard monthly MVO with static betas
3. kalman_mvo: Monthly MVO with Kalman-filtered betas
4. equal_weight: Monthly equal-weight rebalance
5. buy_hold_nifty: Buy & hold NIFTY50 index

Typical usage:
    from src.backtest.engine import WalkForwardBacktester
    backtester = WalkForwardBacktester(loader, signal_gen, optimiser)
    all_returns, rebalance_log = backtester.run()
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import (
    BACKTEST_END_YEAR,
    BACKTEST_START_YEAR,
    LSTM_SEQ_LEN,
    STRATEGIES,
    TEST_YEARS,
    TICKERS,
    TRAIN_YEARS,
)
from src.backtest.transaction_costs import CostModel
from src.models.lstm import BetaLSTM, train_lstm, predict_lstm
from src.portfolio.signal import SignalGenerator
from src.portfolio.optimiser import PortfolioOptimiser

logger = logging.getLogger(__name__)


@dataclass
class RebalanceEvent:
    """Records a single rebalance event for audit trail."""

    date: pd.Timestamp
    strategy: str
    signal: str
    mode: str
    regime: str
    predicted_betavol: float
    vix_level: float
    turnover: float
    cost: float


@dataclass
class BacktestResults:
    """Container for all backtest output."""

    returns: Dict[str, pd.Series] = field(default_factory=dict)
    rebalance_log: list[RebalanceEvent] = field(default_factory=list)
    fold_metrics: list[dict] = field(default_factory=list)


class WalkForwardBacktester:
    """Walk-forward backtesting engine for all strategies.

    Args:
        loader: DataLoader instance.
        signal_gen: Calibrated SignalGenerator.
        optimiser: PortfolioOptimiser instance.
        kalman_betas: DataFrame of Kalman filter betas (optional).
        feature_matrix_path: Path to stacked features CSV.
    """

    def __init__(
        self,
        loader: object,
        signal_gen: SignalGenerator,
        optimiser: PortfolioOptimiser,
        kalman_betas: Optional[pd.DataFrame] = None,
        feature_matrix_path: Optional[str] = None,
    ) -> None:
        self.loader = loader
        self.signal_gen = signal_gen
        self.optimiser = optimiser
        self.kalman_betas = kalman_betas
        self.feature_matrix_path = feature_matrix_path
        self.cost_model = CostModel()

        # Cached data
        self._prices = loader.prices
        self._market_ret = loader.market_returns
        self._vix = loader.vix
        self._regimes: Optional[pd.Series] = None
        self._stacked: Optional[pd.DataFrame] = None
        self._feature_cols: Optional[list[str]] = None

    def _load_stacked_features(self) -> None:
        """Load stacked feature matrix from disk (built by Notebook 1)."""
        if self._stacked is None and self.feature_matrix_path:
            logger.info("Loading stacked features from %s", self.feature_matrix_path)
            self._stacked = pd.read_csv(
                self.feature_matrix_path, parse_dates=["date"]
            ).set_index("date")
            self._feature_cols = [
                c for c in self._stacked.columns if c not in ["target", "ticker"]
            ]

    def set_regimes(self, regime_series: pd.Series) -> None:
        """Provide HMM regime labels.

        Args:
            regime_series: Series with values 'bull'|'bear'|'transition'.
        """
        self._regimes = regime_series

    def _get_regime(self, date: pd.Timestamp) -> str:
        """Look up regime for a date, defaulting to 'transition'.

        Args:
            date: Date to query.

        Returns:
            Regime string.
        """
        if self._regimes is None:
            return "transition"
        if date in self._regimes.index:
            return str(self._regimes.loc[date])
        return "transition"

    def _get_vix(self, date: pd.Timestamp) -> float:
        """Look up VIX for a date, defaulting to 20.

        Args:
            date: Date to query.

        Returns:
            VIX level.
        """
        if date in self._vix.index and not np.isnan(self._vix.loc[date]):
            return float(self._vix.loc[date])
        return 20.0

    def _train_lstm_for_fold(
        self,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        save_path: Optional[str] = None,
    ) -> Optional[BetaLSTM]:
        """Retrain the LSTM on the given training window.

        Args:
            train_start: Start of training window.
            train_end: End of training window.
            save_path: Path to save best model checkpoint.

        Returns:
            Trained BetaLSTM or None if features unavailable.
        """
        if self._stacked is None:
            logger.warning("Stacked features not loaded — LSTM unavailable for this fold")
            return None

        mask_train = (self._stacked.index >= train_start) & (self._stacked.index <= train_end)
        train_data = self._stacked[mask_train]

        if len(train_data) < LSTM_SEQ_LEN * 2:
            logger.warning("Insufficient training data for fold — skipping LSTM")
            return None

        # Chronological val split: last 20% of training data
        val_split = int(len(train_data) * 0.8)
        train_part = train_data.iloc[:val_split]
        val_part = train_data.iloc[val_split:]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_part[self._feature_cols])
        y_train = train_part["target"].values
        X_val = scaler.transform(val_part[self._feature_cols])
        y_val = val_part["target"].values

        n_features = len(self._feature_cols)
        model = train_lstm(
            X_train, y_train, X_val, y_val,
            n_features=n_features,
            save_path=save_path,
        )
        # Store scaler on model for prediction use
        model._scaler = scaler
        model._feature_cols = self._feature_cols
        return model

    def _lstm_predict_for_date(
        self,
        model: BetaLSTM,
        date: pd.Timestamp,
    ) -> float:
        """Get LSTM portfolio-level betavol prediction for a given date.

        Args:
            model: Trained LSTM with _scaler and _feature_cols attributes.
            date: Date for which to predict.

        Returns:
            Predicted portfolio-level beta volatility (scalar).
        """
        if self._stacked is None or not hasattr(model, "_scaler"):
            return float(self.signal_gen.threshold)   # neutral prediction

        # Get last SEQ_LEN rows of features up to `date`
        seq_len = LSTM_SEQ_LEN
        mask = self._stacked.index <= date
        subset = self._stacked[mask].tail(seq_len + 1)

        if len(subset) < seq_len:
            return float(self.signal_gen.threshold)

        # Average across stocks at each date (cross-sectional mean features)
        by_date = subset.groupby(subset.index)[model._feature_cols].mean()
        if len(by_date) < seq_len:
            return float(self.signal_gen.threshold)

        X_seq = by_date.tail(seq_len).values
        X_scaled = model._scaler.transform(X_seq)
        preds = predict_lstm(model, X_scaled, seq_len=seq_len - 1)
        return float(preds[-1]) if len(preds) > 0 else float(self.signal_gen.threshold)

    def _equal_weight(self) -> pd.Series:
        """Equal weight across all tickers."""
        n = len(TICKERS)
        return pd.Series({t: 1.0 / n for t in TICKERS})

    def _simulate_strategy(
        self,
        strategy: str,
        test_dates: pd.DatetimeIndex,
        lstm_model: Optional[BetaLSTM],
        betas_60: pd.DataFrame,
    ) -> pd.Series:
        """Simulate one strategy over a test period.

        Args:
            strategy: Strategy name from STRATEGIES.
            test_dates: DatetimeIndex of the test period.
            lstm_model: Trained LSTM (may be None for non-LSTM strategies).
            betas_60: 60d rolling beta panel.

        Returns:
            Daily return Series for the test period.
        """
        prices = self._prices
        market_ret = self._market_ret
        n_stocks = len(TICKERS)
        weights = self._equal_weight()
        daily_ret: list[float] = []

        for i, date in enumerate(test_dates[1:], 1):
            prev_date = test_dates[i - 1]

            # Daily stock returns
            if date in prices.index and prev_date in prices.index:
                p_now = prices.loc[date].reindex(TICKERS).fillna(method="ffill")
                p_prev = prices.loc[prev_date].reindex(TICKERS).fillna(method="ffill")
                stock_returns = np.log((p_now / p_prev).replace([np.inf, -np.inf], 0))
            else:
                stock_returns = pd.Series(0.0, index=TICKERS)

            # --- Buy & Hold NIFTY ---
            if strategy == "buy_hold_nifty":
                r = float(market_ret.loc[date]) if date in market_ret.index else 0.0
                daily_ret.append(r)
                continue

            # --- Equal weight ---
            if strategy == "equal_weight":
                # Monthly rebalance
                if date.day <= 5 or i == 1:
                    weights = self._equal_weight()
                r = float((weights * stock_returns.reindex(weights.index).fillna(0)).sum())
                daily_ret.append(r)
                continue

            # --- Adaptive Beta: VIX override fires EVERY trading day (critical bug fix) ---
            # Old code checked VIX only every 5 days. During COVID, VIX went 20→82 in 3 days —
            # the 5-day gate caused up to 4 days of unprotected crash exposure per spike.
            if strategy == "adaptive_beta":
                vix_daily = self._get_vix(prev_date)
                if vix_daily > self.signal_gen.vix_threshold:
                    pw = prices.loc[:prev_date].tail(252)
                    if len(pw) < 30:
                        pw = prices.loc[:prev_date]
                    cb = (
                        betas_60.loc[prev_date]
                        if not betas_60.empty and prev_date in betas_60.index
                        else betas_60.iloc[-1] if not betas_60.empty
                        else pd.Series({t: 1.0 for t in TICKERS})
                    )
                    new_weights = self.optimiser.optimise(pw, cb, "min_variance")
                    if not new_weights.empty:
                        cost = self.cost_model.compute(weights, new_weights)
                        weights = new_weights
                        daily_ret.append(-cost)
                        continue

            # --- ML strategies: check LSTM threshold signal every 5 trading days ---
            should_check = (i % 5 == 0) or (i == 1)
            if should_check:
                price_window = prices.loc[:prev_date].tail(252)
                if len(price_window) < 30:
                    price_window = prices.loc[:prev_date]

                current_betas = (
                    betas_60.loc[prev_date]
                    if prev_date in betas_60.index
                    else betas_60.iloc[-1]
                )
                vix_now = self._get_vix(prev_date)
                regime_now = self._get_regime(prev_date)

                if strategy == "adaptive_beta":
                    pred_betavol = self._lstm_predict_for_date(lstm_model, prev_date) if lstm_model else float(self.signal_gen.threshold)
                    state = self.signal_gen.evaluate(pred_betavol, vix_now, regime_now)
                    mode = state.mode

                    if mode != "hold":
                        beta_target = max(0.5, min(1.2, pred_betavol * 0.8))
                        new_weights = self.optimiser.optimise(
                            price_window, current_betas, mode, beta_target
                        )
                        if not new_weights.empty:
                            cost = self.cost_model.compute(weights, new_weights)
                            weights = new_weights
                            daily_ret.append(-cost)
                            continue

                elif strategy == "static_capm_mvo":
                    if date.day <= 5:
                        new_weights = self.optimiser.optimise(
                            price_window, current_betas, "max_sharpe_beta_constrained"
                        )
                        if not new_weights.empty:
                            cost = self.cost_model.compute(weights, new_weights)
                            weights = new_weights
                            daily_ret.append(-cost)
                            continue

                elif strategy == "kalman_mvo":
                    if self.kalman_betas is not None and date.day <= 5:
                        kb = (
                            self.kalman_betas.loc[prev_date]
                            if prev_date in self.kalman_betas.index
                            else current_betas
                        )
                        new_weights = self.optimiser.optimise(
                            price_window, kb, "max_sharpe_beta_constrained"
                        )
                        if not new_weights.empty:
                            cost = self.cost_model.compute(weights, new_weights)
                            weights = new_weights
                            daily_ret.append(-cost)
                            continue

            # Portfolio return for this day
            r = float((weights * stock_returns.reindex(weights.index).fillna(0)).sum())
            daily_ret.append(r)

        return pd.Series(daily_ret, index=test_dates[1:], name=strategy)

    def run(
        self,
        betas_60: Optional[pd.DataFrame] = None,
    ) -> BacktestResults:
        """Execute the full walk-forward backtest.

        Args:
            betas_60: 60d rolling beta panel (if not provided, uses equal weights).

        Returns:
            BacktestResults with returns dict and rebalance log.
        """
        self._load_stacked_features()

        results = BacktestResults()
        for s in STRATEGIES:
            results.returns[s] = pd.Series(dtype=float)

        fold_starts = pd.date_range(
            start=f"{BACKTEST_START_YEAR + TRAIN_YEARS}-01-01",
            end=f"{BACKTEST_END_YEAR}-01-01",
            freq="YS",
        )

        for fold_start in fold_starts:
            fold_end = fold_start + pd.DateOffset(years=TEST_YEARS) - pd.Timedelta(days=1)
            train_start = fold_start - pd.DateOffset(years=TRAIN_YEARS)

            logger.info(
                "Fold: train %s → %s | test %s → %s",
                train_start.date(),
                fold_start.date(),
                fold_start.date(),
                fold_end.date(),
            )

            # Retrain LSTM on this fold's training window
            lstm_model = self._train_lstm_for_fold(train_start, fold_start)

            # Test period dates
            price_idx = self._prices.index
            test_dates = price_idx[(price_idx >= fold_start) & (price_idx <= fold_end)]

            if len(test_dates) < 2:
                logger.warning("Insufficient test dates for fold starting %s — skipping", fold_start.date())
                continue

            beta_panel = betas_60 if betas_60 is not None else pd.DataFrame()

            for strategy in STRATEGIES:
                fold_returns = self._simulate_strategy(
                    strategy, test_dates, lstm_model, beta_panel
                )
                results.returns[strategy] = pd.concat(
                    [results.returns[strategy], fold_returns]
                )
                logger.info(
                    "Strategy '%s' fold %s: %d trading days, mean_ret=%.4f%%",
                    strategy,
                    fold_start.year,
                    len(fold_returns),
                    fold_returns.mean() * 100,
                )

        return results
