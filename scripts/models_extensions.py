"""Extended model variants for multi-asset experiments.

HigherOrderStateConditionedNet: k-step Markov conditioning.
build_loaders_higher_order: DataLoader factory that provides full state history.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_DEFAULT_HIDDEN = (64, 128, 256, 128, 64)
_DEFAULT_DROPOUT = 0.2


class HigherOrderStateConditionedNet(nn.Module):
    """[one_hot(X_t); one_hot(X_{t-1}); ...; one_hot(X_{t-k+1}); F_t] -> logits.

    For k=1 the architecture and input dimensionality are identical to
    StateConditionedNet, so results are directly comparable.
    """

    def __init__(
        self,
        n_feat: int,
        n_xt_states: int,
        n_output: int,
        k: int = 1,
        hidden_dims: tuple = _DEFAULT_HIDDEN,
        dropout: float = _DEFAULT_DROPOUT,
    ):
        super().__init__()
        self.n_xt_states = n_xt_states
        self.k = k
        layers = []
        in_dim = k * n_xt_states + n_feat
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, n_output)

    def forward(self, features: torch.Tensor, x_history: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        features  : (B, n_feat)
        x_history : (B, k) integer bin indices [X_t, X_{t-1}, ..., X_{t-k+1}]
                    or (B,) for k=1 compatibility

        Returns
        -------
        logits : (B, n_output)
        """
        if x_history.dim() == 1:
            x_history = x_history.unsqueeze(1)  # (B,) → (B, 1) for k=1

        parts = [F.one_hot(x_history[:, i], num_classes=self.n_xt_states).float()
                 for i in range(self.k)]
        z = torch.cat(parts + [features], dim=-1)
        return self.out(self.mlp(z))


# ---------------------------------------------------------------------------
# Higher-order DataLoader
# ---------------------------------------------------------------------------

class _HigherOrderDataset(Dataset):
    """Returns (F_t, x_history_k, y) where x_history_k shape is (k,)."""

    def __init__(
        self,
        F_normed: np.ndarray,
        X_t_all: np.ndarray,
        y_all: np.ndarray,
        indices: np.ndarray,
        k: int,
    ):
        self.k = k
        # Filter indices to ensure full k-step history is available
        valid = indices[indices >= k - 1]
        self.F = torch.tensor(F_normed[valid], dtype=torch.float32)
        self.y = torch.tensor(y_all[valid], dtype=torch.long)
        # x_history[i] = [X_t[t], X_t[t-1], ..., X_t[t-k+1]]
        hist = np.stack([X_t_all[valid - lag] for lag in range(k)], axis=1)
        self.x_history = torch.tensor(hist, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.F[idx], self.x_history[idx], self.y[idx]


def build_loaders_higher_order(
    cfg_dict: dict,
    F_normed: np.ndarray,
    X_t_all: np.ndarray,
    k: int,
    batch_train: int = 256,
    batch_eval: int = 512,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders for k-step Markov conditioning.

    Drops the first k-1 indices from each split so every sample has full
    history available. No zero-padding is applied.

    Returns (train_loader, val_loader, test_loader).
    """
    y_all = cfg_dict["y_all"]
    train_ds = _HigherOrderDataset(F_normed, X_t_all, y_all, cfg_dict["idx_train"], k)
    val_ds   = _HigherOrderDataset(F_normed, X_t_all, y_all, cfg_dict["idx_val"],   k)
    test_ds  = _HigherOrderDataset(F_normed, X_t_all, y_all, cfg_dict["idx_test"],  k)
    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_eval,  shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_eval,  shuffle=False)
    return train_loader, val_loader, test_loader
