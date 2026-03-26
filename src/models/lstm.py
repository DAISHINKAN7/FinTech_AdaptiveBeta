"""LSTM model for forward beta volatility prediction.

Architecture:
    Input sequence (seq_len × n_features)
    → LSTM (hidden=64, layers=2, dropout=0.2)
    → LayerNorm
    → Linear(64→32) → ReLU → Dropout → Linear(32→1)

Key design choices:
- HuberLoss: more robust than MSE to betavol spikes
- AdamW + ReduceLROnPlateau: good default for financial time series
- Gradient clipping (max_norm=1.0): prevents exploding gradients
- NO shuffle on DataLoader: preserves temporal order
- Early stopping with patience=10

Typical usage:
    from src.models.lstm import BetaLSTM, BetaDataset, train_lstm, predict_lstm
    model = train_lstm(X_train_sc, y_train, X_val_sc, y_val, n_features, save_path)
    preds = predict_lstm(model, X_test_sc, device)
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.config import (
    LSTM_BATCH,
    LSTM_DROPOUT,
    LSTM_EPOCHS,
    LSTM_HIDDEN,
    LSTM_LAYERS,
    LSTM_LR,
    LSTM_PATIENCE,
    LSTM_SEQ_LEN,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)

# Reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class BetaDataset(Dataset):
    """Sliding-window dataset for sequential LSTM input.

    Args:
        X: Feature array of shape (n_samples, n_features).
        y: Target array of shape (n_samples,).
        seq_len: Number of time steps per sample.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = LSTM_SEQ_LEN) -> None:
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.y) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len]


class BetaLSTM(nn.Module):
    """LSTM network for forward beta volatility regression.

    Args:
        input_size: Number of input features.
        hidden_size: LSTM hidden state dimension.
        n_layers: Number of stacked LSTM layers.
        dropout: Dropout probability (applied between LSTM layers and in head).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = LSTM_HIDDEN,
        n_layers: int = LSTM_LAYERS,
        dropout: float = LSTM_DROPOUT,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, seq_len, input_size).

        Returns:
            Predictions of shape (batch,).
        """
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])   # take last time step
        return self.head(out).squeeze(-1)


def train_lstm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    save_path: Optional[str] = None,
    seq_len: int = LSTM_SEQ_LEN,
    hidden_size: int = LSTM_HIDDEN,
    n_layers: int = LSTM_LAYERS,
    dropout: float = LSTM_DROPOUT,
    batch_size: int = LSTM_BATCH,
    epochs: int = LSTM_EPOCHS,
    lr: float = LSTM_LR,
    patience: int = LSTM_PATIENCE,
) -> BetaLSTM:
    """Train the LSTM model with early stopping.

    Args:
        X_train: Scaled training features (n_train, n_features).
        y_train: Training targets (n_train,).
        X_val: Scaled validation features (n_val, n_features).
        y_val: Validation targets (n_val,).
        n_features: Number of input features.
        save_path: Path to save the best model checkpoint. If None, skips saving.
        seq_len: LSTM sequence length.
        hidden_size: LSTM hidden units.
        n_layers: LSTM layers.
        dropout: Dropout rate.
        batch_size: Batch size.
        epochs: Maximum training epochs.
        lr: Initial learning rate for AdamW.
        patience: Early stopping patience (epochs without val improvement).

    Returns:
        Trained BetaLSTM with weights from the best validation epoch loaded.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training LSTM on device=%s", device)

    train_ds = BetaDataset(X_train, y_train, seq_len)
    val_ds = BetaDataset(X_val, y_val, seq_len)
    # CRITICAL: shuffle=False to preserve temporal order
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = BetaLSTM(n_features, hidden_size, n_layers, dropout).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5, verbose=True
    )
    loss_fn = nn.HuberLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state: dict = {}

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimiser.zero_grad()
            pred = model(X_b)
            loss = loss_fn(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += loss_fn(model(X_b), y_b).item()

        train_avg = train_loss / max(len(train_dl), 1)
        val_avg = val_loss / max(len(val_dl), 1)
        scheduler.step(val_avg)

        if epoch % 5 == 0:
            logger.info(
                "Epoch %3d | Train: %.4f | Val: %.4f | LR: %.2e",
                epoch,
                train_avg,
                val_avg,
                optimiser.param_groups[0]["lr"],
            )

        if val_avg < best_val_loss:
            best_val_loss = val_avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_state, save_path)
                logger.debug("Model checkpoint saved to %s", save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (best val_loss=%.4f)", epoch, best_val_loss)
                break

    # Load best weights
    model.load_state_dict(best_state)
    model.eval()
    logger.info("Training complete. Best val_loss=%.4f", best_val_loss)
    return model


def predict_lstm(
    model: BetaLSTM,
    X_scaled: np.ndarray,
    seq_len: int = LSTM_SEQ_LEN,
    batch_size: int = LSTM_BATCH,
) -> np.ndarray:
    """Generate predictions from a trained LSTM.

    Args:
        model: Trained BetaLSTM (already in eval mode).
        X_scaled: Scaled feature array (n_samples, n_features).
        seq_len: Sequence length used during training.
        batch_size: Inference batch size.

    Returns:
        Predictions array of shape (n_samples - seq_len,).
    """
    device = next(model.parameters()).device
    dummy_y = np.zeros(len(X_scaled))
    ds = BetaDataset(X_scaled, dummy_y, seq_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    preds: list[float] = []
    with torch.no_grad():
        for X_b, _ in dl:
            out = model(X_b.to(device))
            preds.extend(out.cpu().numpy().tolist())
    return np.array(preds)
