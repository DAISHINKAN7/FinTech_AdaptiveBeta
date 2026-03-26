"""XGBoost baseline model with SHAP feature importance.

Used as the gradient-boosting baseline against our LSTM. Also provides
SHAP explainability to understand which features drive beta volatility
predictions.

Typical usage:
    from src.models.xgboost_model import train_xgboost, compute_shap
    model, preds = train_xgboost(X_train, y_train, X_test, y_test)
    shap_df = compute_shap(model, X_test, feature_cols)
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    import shap
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("xgboost or shap not installed — XGBoost model unavailable")


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    early_stopping_rounds: int = 20,
    random_state: int = 42,
) -> tuple:
    """Train an XGBoost regressor for beta volatility prediction.

    Args:
        X_train: Scaled training features (n_train, n_features).
        y_train: Training targets (n_train,).
        X_test: Scaled test features.
        y_test: Test targets (for early stopping evaluation set).
        n_estimators: Maximum number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: XGBoost eta.
        subsample: Row subsampling ratio.
        colsample_bytree: Column subsampling ratio.
        early_stopping_rounds: Stop if eval metric doesn't improve.
        random_state: Random seed.

    Returns:
        Tuple of (trained_model, test_predictions_array).

    Raises:
        ImportError: If xgboost is not installed.
    """
    if not XGB_AVAILABLE:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        early_stopping_rounds=early_stopping_rounds,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",   # faster for large datasets
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )
    preds = model.predict(X_test)
    mae = np.mean(np.abs(preds - y_test))
    logger.info("XGBoost — best_iteration=%d, test_MAE=%.4f", model.best_iteration, mae)
    return model, preds


def compute_shap(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    max_display: int = 20,
) -> pd.DataFrame:
    """Compute SHAP values and return a feature importance DataFrame.

    Args:
        model: Trained XGBoost model.
        X_test: Test feature matrix.
        feature_names: List of feature column names.
        max_display: Number of top features to display in plot.

    Returns:
        DataFrame with columns ['feature', 'mean_abs_shap'] sorted descending.

    Raises:
        ImportError: If shap is not installed.
    """
    if not XGB_AVAILABLE:
        raise ImportError("shap is not installed. Run: pip install shap")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    logger.info("Top 5 SHAP features: %s", importance["feature"].head(5).tolist())
    return importance, shap_values


def direction_accuracy(preds: np.ndarray, actuals: np.ndarray) -> float:
    """Compute directional accuracy: fraction of correct up/down predictions.

    Args:
        preds: Predicted values.
        actuals: Actual values.

    Returns:
        Directional accuracy in [0, 1].
    """
    pred_direction = np.sign(np.diff(preds))
    actual_direction = np.sign(np.diff(actuals))
    valid = actual_direction != 0
    if valid.sum() == 0:
        return 0.5
    return (pred_direction[valid] == actual_direction[valid]).mean()
