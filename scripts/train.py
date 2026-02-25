"""Training utilities.

train_one_run is lifted verbatim from MasterNotebook.ipynb (cell 25).
New additions: build_A_t_neural, build_A_t_statefree, caching helpers,
and an optional train_continuous for GaussHeteroNet / MDNNet.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

EPS = 1e-8

# ---------------------------------------------------------------------------
# Dataset (lifted from master notebook cell 20/21)
# ---------------------------------------------------------------------------

class MasterDataset(Dataset):
    def __init__(self, F_normed, X_t, y, indices):
        self.F = torch.tensor(F_normed[indices], dtype=torch.float32)
        self.X_t = torch.tensor(X_t[indices], dtype=torch.long)
        self.y = torch.tensor(y[indices], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.F[idx], self.X_t[idx], self.y[idx]


def build_loaders(cfg_dict: dict, F_normed, X_t_all, batch_train=256, batch_eval=512):
    """Build train/val/test DataLoaders for a given (h, N) config dict."""
    train_ds = MasterDataset(F_normed, X_t_all, cfg_dict["y_all"], cfg_dict["idx_train"])
    val_ds = MasterDataset(F_normed, X_t_all, cfg_dict["y_all"], cfg_dict["idx_val"])
    test_ds = MasterDataset(F_normed, X_t_all, cfg_dict["y_all"], cfg_dict["idx_test"])
    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_eval, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_eval, shuffle=False)
    return train_loader, val_loader, test_loader


# CK-specific dataset (y is X_{t+h}, not cumulative return bin)
class CKDataset(Dataset):
    def __init__(self, F_normed, X_t_all, y_ck, indices):
        self.F = torch.tensor(F_normed[indices], dtype=torch.float32)
        self.X_t = torch.tensor(X_t_all[indices], dtype=torch.long)
        self.y = torch.tensor(y_ck[indices], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.F[idx], self.X_t[idx], self.y[idx]


def build_ck_loaders(F_normed, X_t_all, y_ck, idx_train, idx_val, idx_test,
                     batch_train=256, batch_eval=512):
    """Build train/val/test loaders for CK task (label = X_{t+h})."""
    train_ds = CKDataset(F_normed, X_t_all, y_ck, idx_train)
    val_ds   = CKDataset(F_normed, X_t_all, y_ck, idx_val)
    test_ds  = CKDataset(F_normed, X_t_all, y_ck, idx_test)
    train_loader = DataLoader(train_ds, batch_size=batch_train, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_eval,  shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_eval,  shuffle=False)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Soft labels (lifted from master notebook cell 15)
# ---------------------------------------------------------------------------

def create_soft_labels_batch(y_hard, n_states, kernel="gaussian", sigma=2.0, eps=EPS):
    """Convert hard integer labels to soft probability distributions over bins."""
    batch_size = y_hard.shape[0]
    device = y_hard.device
    j = torch.arange(n_states, device=device, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
    y_exp = y_hard.unsqueeze(1).float()
    if kernel == "gaussian":
        unnorm = torch.exp(-((j - y_exp) ** 2) / (2 * sigma ** 2))
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    return unnorm / (unnorm.sum(dim=1, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# Training loop (lifted from master notebook cell 25, minor signature change)
# ---------------------------------------------------------------------------

def train_one_run(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    N: int,
    sigma: float,
    device: torch.device,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 10,
    grad_clip: float = 1.0,
    verbose: bool = False,
) -> Tuple[Dict, Dict]:
    """Train model with AdamW, ReduceLROnPlateau, early stopping.

    Returns (best_state_dict, history_dict).
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, max_epochs + 1):
        # ── Train ──
        model.train()
        t_loss, t_correct, t_n = 0.0, 0, 0
        for F_b, xt_b, y_b in train_loader:
            F_b, xt_b, y_b = F_b.to(device), xt_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(F_b, xt_b)
            soft = create_soft_labels_batch(y_b, N, sigma=sigma)
            log_probs = F.log_softmax(logits, dim=1)
            loss = F.kl_div(log_probs, soft, reduction="batchmean")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            t_loss += loss.item() * len(y_b)
            t_correct += (logits.argmax(1) == y_b).sum().item()
            t_n += len(y_b)

        # ── Validate ──
        model.eval()
        v_loss, v_correct, v_n = 0.0, 0, 0
        with torch.no_grad():
            for F_b, xt_b, y_b in val_loader:
                F_b, xt_b, y_b = F_b.to(device), xt_b.to(device), y_b.to(device)
                logits = model(F_b, xt_b)
                soft = create_soft_labels_batch(y_b, N, sigma=sigma)
                log_probs = F.log_softmax(logits, dim=1)
                loss = F.kl_div(log_probs, soft, reduction="batchmean")
                v_loss += loss.item() * len(y_b)
                v_correct += (logits.argmax(1) == y_b).sum().item()
                v_n += len(y_b)

        avg_t, avg_v = t_loss / t_n, v_loss / v_n
        scheduler.step(avg_v)
        history["train_loss"].append(avg_t)
        history["val_loss"].append(avg_v)
        history["train_acc"].append(t_correct / t_n)
        history["val_acc"].append(v_correct / v_n)

        if avg_v < best_val_loss:
            best_val_loss = avg_v
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"  epoch {epoch:>3}: train={avg_t:.4f} val={avg_v:.4f} "
                  f"t_acc={t_correct/t_n:.3f} v_acc={v_correct/v_n:.3f}")

        if epochs_no_improve >= patience:
            if verbose:
                print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.to(device)
    return best_state, history


# ---------------------------------------------------------------------------
# Optional continuous training
# ---------------------------------------------------------------------------

def train_continuous(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    loss_type: str = "gaussian",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    max_epochs: int = 100,
    patience: int = 10,
    grad_clip: float = 1.0,
    verbose: bool = False,
) -> Tuple[Dict, Dict]:
    """Train GaussHeteroNet or MDNNet on continuous return targets.

    loss_type: 'gaussian' | 'mdn'
    DataLoader must return (F_b, xt_b, r_b) where r_b is continuous return.
    """
    from .models import GaussHeteroNet, MDNNet

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        model.train()
        t_loss, t_n = 0.0, 0
        for F_b, xt_b, r_b in train_loader:
            F_b, r_b = F_b.to(device), r_b.to(device).float()
            optimizer.zero_grad()
            if loss_type == "gaussian":
                mu, log_sigma = model(F_b, None)
                loss = GaussHeteroNet.gaussian_nll(mu, log_sigma, r_b)
            elif loss_type == "mdn":
                pi, mu, log_sigma = model(F_b, None)
                loss = MDNNet.mdn_nll(pi, mu, log_sigma, r_b)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            t_loss += loss.item() * len(r_b)
            t_n += len(r_b)

        model.eval()
        v_loss, v_n = 0.0, 0
        with torch.no_grad():
            for F_b, xt_b, r_b in val_loader:
                F_b, r_b = F_b.to(device), r_b.to(device).float()
                if loss_type == "gaussian":
                    mu, log_sigma = model(F_b, None)
                    loss = GaussHeteroNet.gaussian_nll(mu, log_sigma, r_b)
                else:
                    pi, mu, log_sigma = model(F_b, None)
                    loss = MDNNet.mdn_nll(pi, mu, log_sigma, r_b)
                v_loss += loss.item() * len(r_b)
                v_n += len(r_b)

        avg_t, avg_v = t_loss / t_n, v_loss / v_n
        scheduler.step(avg_v)
        history["train_loss"].append(avg_t)
        history["val_loss"].append(avg_v)

        if avg_v < best_val_loss:
            best_val_loss = avg_v
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:>3}: train={avg_t:.4f} val={avg_v:.4f}")

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_state)
    return best_state, history


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def is_cached(path: str | Path) -> bool:
    return Path(path).exists()


def cache_model(state_dict: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)


def load_cached_model(model: nn.Module, path: str | Path) -> nn.Module:
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


# ---------------------------------------------------------------------------
# Build time-varying A_t matrices for CK + operator diagnostics
# ---------------------------------------------------------------------------

def build_A_t_neural(
    model: nn.Module,
    F_normed: np.ndarray,
    time_indices: np.ndarray,
    N_XT: int,
    N_output: int,
    device: torch.device,
) -> np.ndarray:
    """Build A_t[i, j] = P(output=j | X_t=i, F_t) for all states i at each t.

    For state-conditioned models.  A_t is square when N_output = N_XT = 55 (CK tasks).

    Parameters
    ----------
    time_indices : np.ndarray
        Time points t at which to evaluate (e.g., test indices).
    N_XT : int
        Number of possible input states.
    N_output : int
        Number of output bins.

    Returns
    -------
    A : np.ndarray, shape (len(time_indices), N_XT, N_output)
    """
    model.eval()
    T = len(time_indices)
    A = np.zeros((T, N_XT, N_output), dtype=np.float32)
    all_states = torch.arange(N_XT, dtype=torch.long, device=device)

    with torch.no_grad():
        for pos, t in enumerate(time_indices):
            F_t = torch.tensor(F_normed[t], dtype=torch.float32).unsqueeze(0)
            F_t_rep = F_t.expand(N_XT, -1).to(device)
            logits = model(F_t_rep, all_states)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            A[pos] = probs

    return A


def build_A_t_statefree(
    model: nn.Module,
    F_normed: np.ndarray,
    time_indices: np.ndarray,
    N_XT: int,
    N_output: int,
    device: torch.device,
) -> np.ndarray:
    """Build A_t for state-free model: all rows identical (degenerate dynamics).

    For state-free models, the output distribution depends only on F_t, not on
    the input state.  We compute the softmax once per t, then tile across all
    N_XT rows.

    Returns
    -------
    A : np.ndarray, shape (len(time_indices), N_XT, N_output)
        Every row A[pos, i, :] is identical for a given pos (time step t).
    """
    model.eval()
    T = len(time_indices)
    A = np.zeros((T, N_XT, N_output), dtype=np.float32)
    dummy_state = torch.zeros(1, dtype=torch.long, device=device)

    with torch.no_grad():
        for pos, t in enumerate(time_indices):
            F_t = torch.tensor(F_normed[t], dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(F_t, dummy_state)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]   # shape (N_output,)
            A[pos] = probs[None, :]                              # broadcast to all N_XT rows

    return A
