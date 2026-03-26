"""Portfolio weight constraint helpers.

Provides pure-function utilities for weight cleaning, normalization,
and validation used across the optimiser and backtesting engine.

Typical usage:
    from src.portfolio.constraints import clean_weights, validate_weights
    w = clean_weights(raw_weights, min_weight=0.01, max_weight=0.15)
    validate_weights(w)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def clean_weights(
    weights: pd.Series,
    min_weight: float = 0.01,
    max_weight: float = 0.15,
) -> pd.Series:
    """Remove tiny weights and re-normalise to sum to 1.

    1. Zero out weights below min_weight
    2. Clip remaining weights at max_weight
    3. Re-normalise so weights sum exactly to 1.0

    Args:
        weights: Raw portfolio weights (may not sum to 1).
        min_weight: Minimum non-zero weight threshold.
        max_weight: Maximum weight per position.

    Returns:
        Cleaned weights summing to 1.0.

    Raises:
        ValueError: If all weights are zero after cleaning.
    """
    w = weights.copy()
    w[w < min_weight] = 0.0
    w = w.clip(upper=max_weight)
    total = w.sum()
    if total == 0:
        raise ValueError("All weights zeroed out during cleaning — check inputs")
    w = w / total
    return w


def validate_weights(
    weights: pd.Series,
    tol: float = 1e-4,
) -> None:
    """Validate portfolio weight constraints.

    Args:
        weights: Portfolio weights Series.
        tol: Tolerance for sum-to-one check.

    Raises:
        ValueError: If weights violate long-only or sum-to-one constraints.
    """
    if (weights < -tol).any():
        neg = weights[weights < -tol]
        raise ValueError(f"Negative weights detected: {neg.to_dict()}")

    total = weights.sum()
    if abs(total - 1.0) > tol:
        raise ValueError(f"Weights sum to {total:.6f} — expected 1.0 ± {tol}")


def compute_turnover(old_weights: pd.Series, new_weights: pd.Series) -> float:
    """Compute one-way portfolio turnover.

    Turnover = sum(|w_new - w_old|) / 2

    Args:
        old_weights: Previous portfolio weights.
        new_weights: New portfolio weights.

    Returns:
        One-way turnover in [0, 1].
    """
    all_tickers = old_weights.index.union(new_weights.index)
    old = old_weights.reindex(all_tickers).fillna(0.0)
    new = new_weights.reindex(all_tickers).fillna(0.0)
    return float((new - old).abs().sum() / 2)


def weight_concentration(weights: pd.Series) -> dict[str, float]:
    """Compute weight concentration metrics.

    Args:
        weights: Portfolio weights.

    Returns:
        Dict with 'herfindahl' (HHI), 'effective_n', 'max_weight'.
    """
    w = weights[weights > 0]
    hhi = float((w**2).sum())
    effective_n = 1.0 / hhi if hhi > 0 else 0.0
    return {
        "herfindahl": hhi,
        "effective_n": effective_n,
        "max_weight": float(w.max()) if len(w) > 0 else 0.0,
        "n_positions": int((w > 0).sum()),
    }
