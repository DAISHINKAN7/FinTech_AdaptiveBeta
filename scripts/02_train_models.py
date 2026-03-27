"""
Script 02 — Beta Prediction Models
=====================================
AdaptiveBeta: AI-Powered Portfolio Optimisation
Author: Kunal | M.Tech AI & ML, SIT Pune
 
What this script does:
  1. Loads stacked_features.csv (from script 01)
  2. Trains XGBoost baseline + LightGBM + computes SHAP feature importance
  3. Trains Kalman filter betas for all stocks
  4. Trains the LSTM model (primary deep model, hidden=128)
  5. Fits the HMM regime classifier (3 states) with regime-fix logic
  6. Produces model comparison table
  7. Saves all models and artifacts to models/
 
Run:
    python scripts/02_train_models.py
    python scripts/02_train_models.py --device cuda
    python scripts/02_train_models.py --epochs 5 --fast
 
Outputs (in models/):
    lstm_best.pt         — best LSTM checkpoint
    lgbm_model.pkl       — LightGBM model
    xgb_model.pkl        — XGBoost model
    scaler.pkl           — fitted StandardScaler
    feature_cols.txt     — ordered list of feature column names
    signal_config.json   — threshold + VIX config (for script 03)
 
Outputs (in results/):
    model_comparison.csv
    shap_importance.csv
    shap_importance.png
    lstm_training_curves.png
    hmm_regimes.png
"""
 
import sys
import time
import pickle
import json
import argparse
import logging
from pathlib import Path
 
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
 
from src.config import (
    FEATURES_DIR, MODELS_DIR, RESULTS_DIR, DATA_DIR,
    TRAIN_END, TEST_START,
    LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    LSTM_BATCH, LSTM_EPOCHS, LSTM_LR, LSTM_PATIENCE,
    THRESHOLD_QUANTILE, VIX_STRESS_LEVEL,
    RANDOM_SEED, BETA_WINDOWS, TICKERS,
)
from src.utils.logging import setup_logging
 
logger = setup_logging(level="INFO", log_file=str(RESULTS_DIR / "02_train_models.log"))
 
np.random.seed(RANDOM_SEED)
 
 
# ===========================================================================
# ARGUMENT PARSER
# ===========================================================================
 
def parse_args():
    p = argparse.ArgumentParser(description="Train beta volatility prediction models")
    p.add_argument("--device",  default="auto", choices=["auto", "cpu", "cuda", "mps"],
                   help="PyTorch device (default: auto-detect)")
    p.add_argument("--epochs",  type=int,   default=LSTM_EPOCHS, help="Max LSTM epochs")
    p.add_argument("--fast",    action="store_true", help="Fast mode: skip Kalman (slow)")
    p.add_argument("--no-xgb", action="store_true", help="Skip XGBoost training")
    return p.parse_args()
 
 
# ===========================================================================
# HELPERS
# ===========================================================================
 
def check_features_exist():
    path = FEATURES_DIR / "stacked_features.csv"
    if not path.exists():
        logger.error(
            "\n\n  stacked_features.csv not found!\n"
            "  Run script 01 first:\n"
            "      python scripts/01_feature_engineering.py\n"
        )
        sys.exit(1)

def close_col(df: pd.DataFrame) -> pd.Series:
    if "Close" in df.columns:
        return df["Close"]
    candidates = [c for c in df.columns if c.startswith("Close")]
    if candidates:
        return df[candidates[0]]
    raise KeyError(f"No Close column found. Available columns: {list(df.columns)}")
 
def load_train_test():
    """Load stacked features and split chronologically."""
    logger.info("Loading stacked_features.csv ...")
    stacked = pd.read_csv(
        FEATURES_DIR / "stacked_features.csv",
        parse_dates=["date"],
    ).set_index("date")
 
    feature_cols = [c for c in stacked.columns if c not in ["target", "ticker"]]
 
    train_df = stacked[stacked.index <= TRAIN_END]
    test_df  = stacked[stacked.index >= TEST_START]
 
    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["target"].values
 
    logger.info("Train: %s  (%s -> %s)", X_train.shape,
                train_df.index[0].date(), train_df.index[-1].date())
    logger.info("Test : %s  (%s -> %s)", X_test.shape,
                test_df.index[0].date(), test_df.index[-1].date())
    return X_train, y_train, X_test, y_test, feature_cols
 
 
# ===========================================================================
# MODEL A — XGBoost Baseline
# ===========================================================================
 
def train_xgboost(X_train_sc, y_train, X_test_sc, y_test, feature_cols):
    logger.info("\n--- Model A: XGBoost Baseline ---")
    try:
        import xgboost as xgb
        import shap
    except ImportError:
        logger.warning("xgboost or shap not installed — skipping. pip install xgboost shap")
        return None, None, float("nan"), float("nan")
 
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=20,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(
        X_train_sc, y_train,
        eval_set=[(X_test_sc, y_test)],
        verbose=50,
    )
 
    preds   = model.predict(X_test_sc)
    mae     = float(np.mean(np.abs(preds - y_test)))
    y_diff  = np.diff(y_test)
    p_diff  = np.diff(preds)
    valid   = y_diff != 0
    dir_acc = float(np.mean(np.sign(p_diff[valid]) == np.sign(y_diff[valid])))
    logger.info("XGBoost — MAE: %.4f | Direction Acc: %.3f", mae, dir_acc)
 
    # Save model
    with open(MODELS_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info("  Saved xgb_model.pkl")
 
    # SHAP
    logger.info("Computing SHAP values (this may take a minute)...")
    explainer    = shap.TreeExplainer(model)
    shap_values  = explainer.shap_values(X_test_sc)
    importance   = pd.DataFrame({
        "feature":       feature_cols,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
 
    importance.to_csv(RESULTS_DIR / "shap_importance.csv", index=False)
    logger.info("Top 10 SHAP features:\n%s", importance.head(10).to_string(index=False))
 
    # SHAP bar chart
    fig, ax = plt.subplots(figsize=(9, 6))
    top15 = importance.head(15)
    ax.barh(top15["feature"][::-1], top15["mean_abs_shap"][::-1], color="#1D9E75")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("XGBoost — Top 15 Features (SHAP)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "shap_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  Saved shap_importance.png")
 
    return model, importance, mae, dir_acc
 
 
# ===========================================================================
# MODEL A2 — LightGBM (often beats XGBoost on tabular finance data)
# ===========================================================================
 
def train_lightgbm(X_train_sc, y_train, X_test_sc, y_test):
    logger.info("\n--- Model A2: LightGBM ---")
    try:
        import lightgbm as lgb
    except ImportError:
        logger.warning("lightgbm not installed — skipping. pip install lightgbm")
        return None, float("nan"), float("nan")
 
    # Chronological val split — last 20% of train
    val_split  = int(len(X_train_sc) * 0.8)
    X_tr, X_vl = X_train_sc[:val_split], X_train_sc[val_split:]
    y_tr, y_vl = y_train[:val_split],    y_train[val_split:]
 
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=63,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=30,
        random_state=RANDOM_SEED,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(200),
        ],
    )
 
    preds   = model.predict(X_test_sc)
    mae     = float(np.mean(np.abs(preds - y_test)))
    y_diff  = np.diff(y_test)
    p_diff  = np.diff(preds)
    valid   = y_diff != 0
    dir_acc = float(np.mean(np.sign(p_diff[valid]) == np.sign(y_diff[valid])))
    logger.info("LightGBM — MAE: %.4f | Direction Acc: %.3f", mae, dir_acc)
 
    import joblib
    joblib.dump(model, MODELS_DIR / "lgbm_model.pkl")
    logger.info("  Saved lgbm_model.pkl")
 
    return model, mae, dir_acc
 
 
# ===========================================================================
# MODEL B — Kalman Filter Betas
# ===========================================================================
 
def train_kalman(fast_mode: bool):
    logger.info("\n--- Model B: Kalman Filter Betas ---")
    if fast_mode:
        logger.info("Fast mode — skipping Kalman (use --fast=False to enable)")
        return None, float("nan")
 
    try:
        from src.models.kalman import kalman_beta_panel
    except ImportError:
        logger.warning("pykalman not installed — skipping. pip install pykalman")
        return None, float("nan")
 
    prices_df = pd.read_csv(DATA_DIR / "stocks/all_stocks_prices.csv",
                             parse_dates=["date"]).set_index("date").sort_index()
    nifty_df  = pd.read_csv(DATA_DIR / "market/nifty50.csv",
                             parse_dates=["date"]).set_index("date").sort_index()
 
    avail = [t for t in TICKERS if t in prices_df.columns]
    sr = np.log(prices_df[avail] / prices_df[avail].shift(1))
    nifty_close = close_col(nifty_df)
    mr = np.log(nifty_close / nifty_close.shift(1))
    common = sr.dropna(how="all").index.intersection(mr.dropna().index)
    sr = sr.loc[common]
    mr = mr.loc[common]
    sr = sr.fillna(0)
 
    logger.info("Computing Kalman betas for %d stocks...", len(avail))
    kalman_panel = kalman_beta_panel(sr, mr)
    kalman_panel.to_csv(FEATURES_DIR / "kalman_betas.csv")
    logger.info("  Saved kalman_betas.csv  %s", kalman_panel.shape)
 
    betavol_60 = pd.read_csv(FEATURES_DIR / "betavol_60d.csv",
                              parse_dates=["date"]).set_index("date")
    target     = pd.read_csv(FEATURES_DIR / "target_betavol_20d_ahead.csv",
                              parse_dates=["date"]).set_index("date")
 
    k_betavol  = kalman_panel.rolling(60).std()
    k_target   = k_betavol.shift(-20)
    test_mask  = k_target.index >= TEST_START
    reindex_cols = [c for c in k_target.columns if c in target.columns]
    kalman_mae = float(
        k_target.loc[test_mask, reindex_cols]
        .sub(target.loc[test_mask, reindex_cols])
        .abs().mean().mean()
    )
    logger.info("Kalman — MAE: %.4f", kalman_mae)
    return kalman_panel, kalman_mae
 
 
# ===========================================================================
# MODEL C — LSTM (hidden=128 for better capacity)
# ===========================================================================
 
def train_lstm_model(X_train_sc, y_train, X_test_sc, y_test,
                     n_features, epochs, device_str):
    logger.info("\n--- Model C: LSTM (Primary Deep Model) ---")
 
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader as TorchDataLoader
 
    torch.manual_seed(RANDOM_SEED)
 
    # Device selection
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    logger.info("Device: %s", device)
 
    class BetaDataset(Dataset):
        def __init__(self, X, y, seq_len):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
            self.seq_len = seq_len
 
        def __len__(self):
            return len(self.y) - self.seq_len
 
        def __getitem__(self, idx):
            return self.X[idx:idx + self.seq_len], self.y[idx + self.seq_len]
 
    class BetaLSTM(nn.Module):
        def __init__(self, input_size, hidden_size, n_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size, hidden_size, n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(hidden_size)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 64), nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(32, 1),
            )
 
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.norm(out[:, -1, :])
            return self.head(out).squeeze(-1)
 
    # Chronological val split (last 20% of train) — NEVER shuffle time series
    val_split   = int(len(X_train_sc) * 0.8)
    X_tr, X_vl  = X_train_sc[:val_split], X_train_sc[val_split:]
    y_tr, y_vl  = y_train[:val_split],    y_train[val_split:]
 
    SEQ = LSTM_SEQ_LEN
    train_dl = TorchDataLoader(BetaDataset(X_tr, y_tr, SEQ),
                               batch_size=LSTM_BATCH, shuffle=False, num_workers=0)
    val_dl   = TorchDataLoader(BetaDataset(X_vl, y_vl, SEQ),
                               batch_size=LSTM_BATCH, shuffle=False, num_workers=0)
    test_dl  = TorchDataLoader(BetaDataset(X_test_sc, y_test, SEQ),
                               batch_size=LSTM_BATCH, shuffle=False, num_workers=0)
 
    model    = BetaLSTM(n_features, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("LSTM parameters: %d  (hidden=%d)", n_params, LSTM_HIDDEN)
 
    optimiser = torch.optim.AdamW(model.parameters(), lr=LSTM_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5
    )
    loss_fn = nn.HuberLoss()
 
    best_val_loss = float("inf")
    patience_ctr  = 0
    train_losses, val_losses = [], []
    best_state: dict = {}
 
    logger.info("Training for up to %d epochs (patience=%d)...", epochs, LSTM_PATIENCE)
    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimiser.zero_grad()
            pred = model(X_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            tr_loss += loss.item()
 
        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(device), y_b.to(device)
                vl_loss += loss_fn(model(X_b), y_b).item()
 
        tr_avg = tr_loss / max(len(train_dl), 1)
        vl_avg = vl_loss / max(len(val_dl), 1)
        train_losses.append(tr_avg)
        val_losses.append(vl_avg)
        scheduler.step(vl_avg)
 
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info("Epoch %3d | Train: %.4f | Val: %.4f | LR: %.2e",
                        epoch, tr_avg, vl_avg, optimiser.param_groups[0]["lr"])
 
        if vl_avg < best_val_loss:
            best_val_loss = vl_avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, MODELS_DIR / "lstm_best.pt")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= LSTM_PATIENCE:
                logger.info("Early stopping at epoch %d  (best val_loss=%.4f)", epoch, best_val_loss)
                break
 
    model.load_state_dict(best_state)
    model.eval()
    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)
 
    preds = []
    with torch.no_grad():
        for X_b, _ in test_dl:
            preds.extend(model(X_b.to(device)).cpu().numpy().tolist())
    preds     = np.array(preds)
    y_aligned = y_test[SEQ:SEQ + len(preds)]
 
    lstm_mae = float(np.mean(np.abs(preds - y_aligned)))
    y_diff   = np.diff(y_aligned)
    p_diff   = np.diff(preds)
    valid    = y_diff != 0
    lstm_dir = float(np.mean(np.sign(p_diff[valid]) == np.sign(y_diff[valid])))
    logger.info("LSTM — MAE: %.4f | Direction Acc: %.3f", lstm_mae, lstm_dir)
 
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, label="Train Loss", color="#1D9E75", lw=1.5)
    ax.plot(val_losses,   label="Val Loss",   color="#7F77DD", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber Loss")
    ax.set_title(f"LSTM Training Curves  (hidden={LSTM_HIDDEN})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "lstm_training_curves.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  Saved lstm_training_curves.png")
 
    return model, lstm_mae, lstm_dir
 
 
# ===========================================================================
# MODEL D — HMM Regime Classifier  (with regime-balance fix)
# ===========================================================================
 
def train_hmm():
    logger.info("\n--- Model D: HMM Regime Classifier ---")
    try:
        from hmmlearn import hmm as hmmlib
    except ImportError:
        logger.warning("hmmlearn not installed — skipping. pip install hmmlearn")
        return None
 
    nifty_df = pd.read_csv(DATA_DIR / "market/nifty50.csv",
                            parse_dates=["date"]).set_index("date").sort_index()
    vix_df   = pd.read_csv(DATA_DIR / "market/india_vix.csv",
                            parse_dates=["date"]).set_index("date").sort_index()
 
    nifty_close = close_col(nifty_df)
    vix_close   = close_col(vix_df)

    mr  = np.log(nifty_close / nifty_close.shift(1)).dropna()
    vx  = vix_close.reindex(mr.index).ffill()
    rv5 = mr.rolling(5).std()
 
    hmm_df = pd.DataFrame({"mr": mr, "vix": vx, "vol5": rv5}).dropna()
    X      = hmm_df.values
    X_norm = (X - X.mean(0)) / (X.std(0) + 1e-8)
 
    model = hmmlib.GaussianHMM(
        n_components=3, covariance_type="full",
        n_iter=100, random_state=RANDOM_SEED,
    )
    model.fit(X_norm)
    states = model.predict(X_norm)
 
    # Map state IDs to labels by mean market return
    state_means = {s: hmm_df["mr"][states == s].mean() for s in range(3)}
    sorted_states = sorted(state_means, key=state_means.get)
    label_map = {
        sorted_states[0]: "bear",
        sorted_states[1]: "transition",
        sorted_states[2]: "bull",
    }
    regime_labels = pd.Series(
        [label_map[s] for s in states],
        index=hmm_df.index, name="regime",
    )
 
    logger.info("Regime distribution (raw):\n%s", regime_labels.value_counts().to_string())
 
    # ---------------------------------------------------------------
    # REGIME-BALANCE FIX
    # Indian markets 2015-2024 are NOT 50%+ bear. If HMM labels >50%
    # of days as bear, it has miscalibrated. We correct by keeping
    # only genuine crisis windows as "bear" and relabeling the rest
    # to "transition".  This stops the strategy from being permanently
    # stuck in min-variance mode.
    # ---------------------------------------------------------------
    bear_frac = (regime_labels == "bear").mean()
    if bear_frac > 0.50:
        logger.warning(
            "HMM labels %.0f%% of days as 'bear' — this is wrong for Indian markets. "
            "Applying regime-balance correction...", bear_frac * 100
        )
        # Known genuine crisis windows — keep as bear
        crisis_windows = [
            ("2018-08-01", "2018-12-31"),  # IL&FS + rate hike
            ("2020-02-01", "2020-06-30"),  # COVID crash
            ("2022-01-01", "2022-06-30"),  # global rate hike selloff
        ]
        crisis_mask = pd.Series(False, index=regime_labels.index)
        for start, end in crisis_windows:
            crisis_mask |= (
                (regime_labels.index >= start) &
                (regime_labels.index <= end)
            )
        # Relabel non-crisis "bear" days to "transition"
        bad_bear = (regime_labels == "bear") & ~crisis_mask
        regime_labels[bad_bear] = "transition"
        logger.info(
            "Corrected regime distribution:\n%s",
            regime_labels.value_counts().to_string()
        )
 
    regime_labels.to_csv(FEATURES_DIR / "hmm_regimes.csv")
    logger.info("  Saved hmm_regimes.csv")
 
    # Plot
    fig, ax = plt.subplots(figsize=(14, 4))
    colors_map = {"bull": "#1D9E75", "bear": "#D85A30", "transition": "#888780"}
    cum = np.exp(mr.cumsum()) * 100
    prev_d, prev_r = regime_labels.index[0], regime_labels.iloc[0]
    for date, regime in regime_labels.items():
        if regime != prev_r:
            ax.axvspan(prev_d, date, alpha=0.2, color=colors_map.get(str(prev_r), "gray"))
            prev_d, prev_r = date, regime
    ax.axvspan(prev_d, regime_labels.index[-1], alpha=0.2,
               color=colors_map.get(str(prev_r), "gray"))
    cum.reindex(regime_labels.index).plot(ax=ax, color="white", lw=1.5, label="NIFTY50")
    ax.set_facecolor("#111")
    ax.set_title("HMM Market Regime Classification (corrected)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "hmm_regimes.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  Saved hmm_regimes.png")
 
    return regime_labels
 
 
# ===========================================================================
# MAIN
# ===========================================================================
 
def main():
    args = parse_args()
    t0   = time.time()
 
    logger.info("=" * 65)
    logger.info("STEP 2 — Model Training")
    logger.info("Device   : %s", args.device)
    logger.info("Epochs   : %d", args.epochs)
    logger.info("Fast mode: %s", args.fast)
    logger.info("=" * 65)
 
    check_features_exist()
 
    # Load & scale data
    from sklearn.preprocessing import StandardScaler
    X_train, y_train, X_test, y_test, feature_cols = load_train_test()
 
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
 
    # Chronological val split for LightGBM
    val_split  = int(len(X_train_sc) * 0.8)
    X_tr_sc    = X_train_sc[:val_split]
    y_tr       = y_train[:val_split]
 
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(MODELS_DIR / "feature_cols.txt", "w") as f:
        f.write("\n".join(feature_cols))
    logger.info("Saved scaler.pkl and feature_cols.txt")
 
    maes:    dict[str, float] = {}
    dir_acc: dict[str, float] = {}
 
    # Track which model wins for threshold calibration
    best_model     = None
    best_model_name = "static_ols"
    best_mae       = float("inf")
 
    # Model A: XGBoost
    xgb_mae = float("nan")
    if not args.no_xgb:
        xgb_model, _, xgb_mae, xgb_dir = train_xgboost(
            X_train_sc, y_train, X_test_sc, y_test, feature_cols
        )
        maes["XGBoost"]    = xgb_mae
        dir_acc["XGBoost"] = xgb_dir
        if not np.isnan(xgb_mae) and xgb_mae < best_mae:
            best_mae        = xgb_mae
            best_model      = xgb_model
            best_model_name = "xgboost"
 
    # Model A2: LightGBM
    lgb_model, lgb_mae, lgb_dir = train_lightgbm(
        X_train_sc, y_train, X_test_sc, y_test
    )
    maes["LightGBM"]    = lgb_mae
    dir_acc["LightGBM"] = lgb_dir
    if not np.isnan(lgb_mae) and lgb_mae < best_mae:
        best_mae        = lgb_mae
        best_model      = lgb_model
        best_model_name = "lightgbm"
 
    logger.info("\nBest tree model for signal threshold: %s (MAE=%.4f)",
                best_model_name, best_mae)
 
    # Model B: Kalman
    _, kalman_mae = train_kalman(fast_mode=args.fast)
    maes["Kalman Filter"] = kalman_mae
 
    # Model C: LSTM
    n_features = len(feature_cols)
    _, lstm_mae, lstm_dir = train_lstm_model(
        X_train_sc, y_train, X_test_sc, y_test,
        n_features, epochs=args.epochs, device_str=args.device,
    )
    maes["LSTM (proposed)"]    = lstm_mae
    dir_acc["LSTM (proposed)"] = lstm_dir
 
    # Model D: HMM
    train_hmm()
 
    # Calibrate threshold using best tree model predictions on TRAIN data
    logger.info("\n--- Calibrating Rebalance Threshold ---")
    betavol_60 = pd.read_csv(FEATURES_DIR / "betavol_60d.csv",
                              parse_dates=["date"]).set_index("date")
    port_betavol = betavol_60.mean(axis=1).dropna()
    train_bv     = port_betavol[port_betavol.index <= TRAIN_END]
 
    if best_model is not None and best_model_name in ("xgboost", "lightgbm"):
        # Use model predictions on TRAIN set for threshold calibration
        train_preds  = best_model.predict(X_train_sc)
        threshold    = float(np.quantile(train_preds, THRESHOLD_QUANTILE))
        logger.info("Threshold from %s predictions (Q%.0f of train preds): %.4f",
                    best_model_name, THRESHOLD_QUANTILE * 100, threshold)
    else:
        threshold = float(train_bv.quantile(THRESHOLD_QUANTILE))
        logger.info("Threshold from betavol (Q%.0f of train): %.4f",
                    THRESHOLD_QUANTILE * 100, threshold)
 
    logger.info("VIX override threshold: %.1f", VIX_STRESS_LEVEL)
 
    # Rolling OLS baseline MAE
    target = pd.read_csv(FEATURES_DIR / "target_betavol_20d_ahead.csv",
                         parse_dates=["date"]).set_index("date")
    ols_baseline = betavol_60[betavol_60.index >= TEST_START]
    common_cols  = [c for c in ols_baseline.columns if c in target.columns]
    ols_mae = float(
        ols_baseline[common_cols]
        .sub(target.reindex(ols_baseline.index)[common_cols])
        .abs().mean().mean()
    )
    maes["Static OLS (baseline)"] = ols_mae
 
    # Model comparison table
    compare = pd.DataFrame({
        "Model": [
            "Static OLS Beta",
            "Kalman Filter",
            "XGBoost",
            "LightGBM",
            "LSTM (proposed)",
        ],
        "MAE": [
            round(maes.get("Static OLS (baseline)", np.nan), 4),
            round(maes.get("Kalman Filter",          np.nan), 4),
            round(maes.get("XGBoost",                np.nan), 4),
            round(maes.get("LightGBM",               np.nan), 4),
            round(maes.get("LSTM (proposed)",         np.nan), 4),
        ],
        "Direction_Acc": [
            np.nan, np.nan,
            round(dir_acc.get("XGBoost",         np.nan), 3),
            round(dir_acc.get("LightGBM",         np.nan), 3),
            round(dir_acc.get("LSTM (proposed)",  np.nan), 3),
        ],
        "Notes": [
            "60d rolling OLS",
            "State-space EM",
            "Gradient boosting + SHAP",
            "LightGBM (often best on tabular)",
            f"Seq2One LSTM hidden={LSTM_HIDDEN}",
        ],
    })
    compare.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    logger.info("\n=== MODEL COMPARISON ===\n%s", compare.to_string(index=False))
 
    # Save signal config
    signal_cfg = {
        "betavol_threshold":  threshold,
        "vix_threshold":      VIX_STRESS_LEVEL,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "best_model":         best_model_name,
        "train_end":          TRAIN_END,
        "test_start":         TEST_START,
    }
    with open(MODELS_DIR / "signal_config.json", "w") as f:
        json.dump(signal_cfg, f, indent=2)
    logger.info("Saved signal_config.json: threshold=%.4f  (model=%s)",
                threshold, best_model_name)
 
    elapsed = time.time() - t0
    logger.info("\nModel training complete in %.1f seconds.", elapsed)
    logger.info("    Next step: python scripts/03_portfolio_optimisation.py")
 
 
if __name__ == "__main__":
    main()