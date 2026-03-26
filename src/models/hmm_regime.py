"""Hidden Markov Model for market regime classification.

Identifies 3 latent market states: bull, bear, and transition.
Uses daily market returns + VIX + short-term vol as observation signals.

Regimes are mapped to labels by sorting by mean market return in each
hidden state: lowest mean → 'bear', middle → 'transition', highest → 'bull'.

Typical usage:
    from src.models.hmm_regime import fit_hmm, predict_regimes
    hmm_model, regime_labels = fit_hmm(market_ret, vix)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed — HMM regime model unavailable")


def fit_hmm(
    market_ret: pd.Series,
    vix: pd.Series,
    n_states: int = 3,
    n_iter: int = 100,
    random_state: int = 42,
) -> tuple:
    """Fit a Gaussian HMM on market return + VIX observations.

    Args:
        market_ret: Log returns of the market index.
        vix: India VIX level aligned to market_ret.
        n_states: Number of hidden states (default 3: bull/bear/transition).
        n_iter: Number of EM iterations for HMM training.
        random_state: Reproducibility seed.

    Returns:
        Tuple of (fitted_hmm_model, regime_labels_series).
        regime_labels_series has values 'bull' | 'bear' | 'transition'
        indexed by date.

    Raises:
        ImportError: If hmmlearn is not installed.
    """
    if not HMM_AVAILABLE:
        raise ImportError("hmmlearn is not installed. Run: pip install hmmlearn")

    # Build observation matrix: [market_ret, vix, rolling_vol_5d]
    rolling_vol = market_ret.rolling(5).std()
    features = pd.DataFrame({
        "market_ret": market_ret,
        "vix": vix.reindex(market_ret.index).ffill(),
        "vol_5d": rolling_vol,
    }).dropna()

    X = features.values

    # Standardise for numerical stability
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
    )
    model.fit(X_norm)

    hidden_states = model.predict(X_norm)

    # Map state IDs to labels by mean market return per state
    state_mean_ret = {
        state: market_ret.loc[features.index][hidden_states == state].mean()
        for state in range(n_states)
    }
    sorted_states = sorted(state_mean_ret, key=state_mean_ret.get)

    # 3-state map: lowest return → bear, middle → transition, highest → bull
    label_map: dict[int, str] = {}
    labels_list = ["bear", "transition", "bull"]
    for rank, state_id in enumerate(sorted_states):
        label_map[state_id] = labels_list[rank]

    regime_series = pd.Series(
        [label_map[s] for s in hidden_states],
        index=features.index,
        name="regime",
    )

    logger.info("HMM regime distribution:\n%s", regime_series.value_counts().to_string())
    return model, regime_series


def predict_regimes(
    hmm_model,
    market_ret: pd.Series,
    vix: pd.Series,
    n_states: int = 3,
) -> pd.Series:
    """Predict regimes on new data using a fitted HMM model.

    Args:
        hmm_model: Trained hmmlearn.hmm.GaussianHMM instance.
        market_ret: Log returns of the market.
        vix: VIX level.
        n_states: Number of states in the model.

    Returns:
        Series of integer state IDs. Caller must apply the same label_map
        used during fit_hmm.
    """
    rolling_vol = market_ret.rolling(5).std()
    features = pd.DataFrame({
        "market_ret": market_ret,
        "vix": vix.reindex(market_ret.index).ffill(),
        "vol_5d": rolling_vol,
    }).dropna()

    X = features.values
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    states = hmm_model.predict(X_norm)
    return pd.Series(states, index=features.index, name="regime_id")
