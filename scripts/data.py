"""Data loading and preprocessing utilities.

All functions lifted verbatim from MasterNotebook.ipynb cells 4, 6, 10, 12
with minor wrapping for reuse.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

EPS = 1e-8


# ---------------------------------------------------------------------------
# Master dataset
# ---------------------------------------------------------------------------

def load_master_dataset(data_dir: str | Path = "dataset"):
    """Load train_diagnostic.csv.

    Returns
    -------
    prices : np.ndarray, shape (T,)
    F_raw  : np.ndarray, shape (T, n_features)
    feature_cols : list[str]
    """
    data_dir = Path(data_dir)
    train_df = pd.read_csv(data_dir / "train_diagnostic.csv")

    # Drop Opinion (contains NaN)
    if "Opinion" in train_df.columns:
        train_df = train_df.drop(columns=["Opinion"])

    assert "Price" in train_df.columns, "Price column not found"
    prices = train_df["Price"].values.astype(np.float64)

    # Features: everything except index/auxiliary columns
    drop_cols = {"index", "Percent_change_forward", "Backward_Bin", "Price"}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    F_raw = train_df[feature_cols].values.astype(np.float32)

    return prices, F_raw, feature_cols


def compute_returns(prices: np.ndarray, h: int) -> np.ndarray:
    """R_t^(h) = (P_{t+h} - P_t) / P_t,  length = len(prices) - h."""
    return (prices[h:] - prices[:-h]) / prices[:-h]


def preprocess_features(
    F_raw: np.ndarray,
    train_idx: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """Fit z-score normalisation on train_idx rows; apply globally.

    Returns F_normed with NaN/Inf replaced by 0.
    """
    F_train = F_raw[train_idx]
    feat_mean = F_train.mean(axis=0, keepdims=True)
    feat_std = F_train.std(axis=0, keepdims=True) + eps
    F_normed = ((F_raw - feat_mean) / feat_std).astype(np.float32)
    n_bad = np.isnan(F_normed).sum() + np.isinf(F_normed).sum()
    if n_bad > 0:
        F_normed = np.nan_to_num(F_normed, nan=0.0, posinf=0.0, neginf=0.0)
    return F_normed


def make_splits(
    T: int,
    fracs: Tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chronological 70/15/15 split.

    Best-config selection is on val_idx only; test_idx touched once at the end.

    Returns (idx_train, idx_val, idx_test).
    """
    train_end = int(fracs[0] * T)
    val_end = int((fracs[0] + fracs[1]) * T)
    idx_train = np.arange(0, train_end)
    idx_val = np.arange(train_end, val_end)
    idx_test = np.arange(val_end, T)
    assert idx_train[-1] < idx_val[0], "train/val overlap"
    assert idx_val[-1] < idx_test[0], "val/test overlap"
    return idx_train, idx_val, idx_test


def build_all_splits(prices: np.ndarray, horizons: list) -> dict:
    """Build chronological splits for each horizon h."""
    splits = {}
    for h in horizons:
        # y_all uses compute_returns(prices[1:], h) → length = len(prices) - 1 - h
        T_h = len(prices) - 1 - h
        idx_train, idx_val, idx_test = make_splits(T_h)
        splits[h] = {
            "T_h": T_h,
            "idx_train": idx_train,
            "idx_val": idx_val,
            "idx_test": idx_test,
        }
    return splits


def build_all_ck_splits(X_t_all: np.ndarray, horizons: list) -> dict:
    """Build chronological splits for CK tasks.

    For CK tasks, y_t^(h) = X_{t+h} so the valid index range is
    t = 0, ..., len(X_t_all) - h - 1.  This is one shorter than the
    cumulative-return task (which uses len(prices) - h = len(X_t_all) - h + 1).

    Parameters
    ----------
    X_t_all : np.ndarray, shape (len(prices) - 1,)
        1-day return bins for all time points.
    horizons : list of int

    Returns
    -------
    dict : h -> {"T_ck": int, "idx_train": ndarray, "idx_val": ndarray, "idx_test": ndarray}
    """
    T_xt = len(X_t_all)
    ck_splits = {}
    for h in horizons:
        T_ck = T_xt - h           # valid sample count: t in [0, T_ck - 1]
        idx_train, idx_val, idx_test = make_splits(T_ck)
        ck_splits[h] = {
            "T_ck": T_ck,
            "idx_train": idx_train,
            "idx_val": idx_val,
            "idx_test": idx_test,
        }
    return ck_splits
