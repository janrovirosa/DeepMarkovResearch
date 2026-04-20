"""Data loading and preprocessing utilities.

All functions lifted verbatim from MasterNotebook.ipynb cells 4, 6, 10, 12
with minor wrapping for reuse.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

EPS = 1e-8


# ---------------------------------------------------------------------------
# Master dataset
# ---------------------------------------------------------------------------

def load_master_dataset(data_dir: str | Path = "dataset", ticker: str | None = None):
    """Load single-asset or multi-asset dataset.

    When *ticker* is given, loads from
    ``dataset_multiasset/prices_{ticker}.csv`` and
    ``dataset_multiasset/features_{ticker}.csv``.
    Features are already standardised; F_raw is returned as-is (float32).

    When *ticker* is None (default), loads the original single-asset
    ``train_diagnostic.csv`` from *data_dir* — unchanged behaviour.

    Returns
    -------
    prices : np.ndarray, shape (T,)
    F_raw  : np.ndarray, shape (T, n_features)
    feature_cols : list[str]
    """
    if ticker is not None:
        base = Path(data_dir).parent / "dataset_multiasset"
        prices_df  = pd.read_csv(base / f"prices_{ticker}.csv",   parse_dates=["date"])
        features_df = pd.read_csv(base / f"features_{ticker}.csv", parse_dates=["date"])

        assert (prices_df["date"].values == features_df["date"].values).all(), \
            f"{ticker}: prices and features date columns are misaligned"

        prices       = prices_df["price"].values.astype(np.float64)
        feature_cols = [c for c in features_df.columns if c != "date"]
        F_raw        = features_df[feature_cols].values.astype(np.float32)
        return prices, F_raw, feature_cols

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


# ---------------------------------------------------------------------------
# Feature subset selection (regime conditioning)
# ---------------------------------------------------------------------------

_MACRO_FEATURE_NAMES = ["DFF", "T10Y2Y", "BAA10Y", "VIXCLS", "DCOILWTICO", "USREC"]

_BANK_TICKERS = {"JPM", "C", "WFC", "GS", "MS", "PNC", "USB", "FITB", "MTB", "BAC"}
_INDEX_TICKERS = {"SPY", "XLF"}


def build_feature_subset(
    F_raw: np.ndarray,
    feature_cols: List[str],
    regime: str,
    target_ticker: str | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Return (F_subset, subset_feature_cols) for the given conditioning regime.

    Parameters
    ----------
    F_raw        : (T, n_feat) feature matrix (already standardised for multi-asset)
    feature_cols : ordered list of feature column names matching F_raw columns
    regime       : 'full' | 'macro_only' | 'own_only' | 'other_banks_only'
    target_ticker: required for 'own_only' and 'other_banks_only'

    Returns
    -------
    F_subset         : (T, n_subset)
    subset_feat_cols : list[str] of length n_subset
    """
    if regime == "full":
        return F_raw, list(feature_cols)

    if regime == "macro_only":
        keep = [c for c in feature_cols if c in _MACRO_FEATURE_NAMES]
        if not keep:
            raise ValueError("No macro feature columns found in feature_cols")
        idx = [feature_cols.index(c) for c in keep]
        return F_raw[:, idx], keep

    if regime == "own_only":
        if target_ticker is None:
            raise ValueError("target_ticker is required for regime='own_only'")
        pat = re.compile(rf"^lag\d+_return_{re.escape(target_ticker)}$")
        keep = [c for c in feature_cols if pat.match(c) or c in ("volume", "log_volume")]
        if not keep:
            raise ValueError(
                f"No 'own_only' columns found for ticker={target_ticker!r} in feature_cols"
            )
        idx = [feature_cols.index(c) for c in keep]
        return F_raw[:, idx], keep

    if regime == "other_banks_only":
        if target_ticker is None:
            raise ValueError("target_ticker is required for regime='other_banks_only'")
        # Keep contemporaneous returns and all lag features for any bank/index ticker
        # other than the target. Drop macro features and all own-stock features.
        keep_tickers = (_BANK_TICKERS | _INDEX_TICKERS) - {target_ticker}
        cont_pat = re.compile(r"^return_(.+)$")
        lag_pat = re.compile(r"^lag\d+_return_(.+)$")
        keep = []
        for c in feature_cols:
            m = cont_pat.match(c)
            if m and m.group(1) in keep_tickers:
                keep.append(c)
                continue
            m = lag_pat.match(c)
            if m and m.group(1) in keep_tickers:
                keep.append(c)
        if not keep:
            raise ValueError(
                f"No 'other_banks_only' columns found for ticker={target_ticker!r} in feature_cols"
            )
        print(f"other_banks_only [{target_ticker}]: {len(keep)} features")
        idx = [feature_cols.index(c) for c in keep]
        return F_raw[:, idx], keep

    raise ValueError(
        f"Unknown regime {regime!r}; choose from 'full', 'macro_only', 'own_only', 'other_banks_only'"
    )


# ---------------------------------------------------------------------------
# Volatility-state binning
# ---------------------------------------------------------------------------

def compute_X_t_vol(
    prices: np.ndarray,
    vol_window: int,
    n_bins: int,
    train_end: int,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray, int]:
    """Compute rolling-volatility bin state X_t_vol, dropping NaN warmup rows.

    sigma_t = std(r_{t-vol_window+1}, ..., r_t)  (vol_window daily log-returns)

    The first vol_window-1 entries lack sufficient history and are DROPPED
    (not forward-filled) to avoid lookahead bias.  Returned arrays have
    length len(prices) - vol_window.

    Bin edges are fit on training data only (train_end indexes into the
    returned truncated arrays, i.e. sigma_valid[:train_end] is used).

    Returns
    -------
    X_t_vol   : (len(prices) - vol_window,)  bin assignments in [0, N_XT-1]
    sigma_t   : (len(prices) - vol_window,)  raw volatility values (unstandardised)
    N_XT      : int  actual bin count (≤ n_bins; may be reduced by get_edges)
    edges_vol : (N_XT+1,)  quantile edges with ±inf boundaries
    warmup    : int  rows dropped from start (= vol_window - 1)
    """
    from .bins import get_edges, assign_bins  # deferred to avoid circular import

    warmup = vol_window - 1
    r = compute_returns(prices, 1)                                  # (T-1,)
    sigma_raw = pd.Series(r).rolling(vol_window).std(ddof=1).values # (T-1,) with warmup NaNs
    sigma_valid = sigma_raw[warmup:]                                # (T - vol_window,)
    assert np.isnan(sigma_valid).sum() == 0, (
        f"compute_X_t_vol: unexpected NaN after dropping {warmup} warmup rows"
    )
    N_XT, edges_vol = get_edges(sigma_valid[:train_end], n_bins)
    X_t_vol = assign_bins(sigma_valid, edges_vol)
    return X_t_vol, sigma_valid, N_XT, edges_vol, warmup
