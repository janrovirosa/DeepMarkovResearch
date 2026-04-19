"""Per-timestep operator diagnostics and CK error computation.

All heavy lifting delegated to eval.py and train.py — this module only
orchestrates and returns DataFrames with no file I/O.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .eval import (
    build_ck_composed,
    compute_ck_errors,
    compute_dobrushin,
    compute_row_entropy,
    compute_row_heterogeneity,
)
from .train import build_A_t_neural, build_A_t_statefree


def build_A_t_series(
    model: nn.Module,
    F_normed: np.ndarray,
    time_indices: np.ndarray,
    N_XT: int,
    N_out: int,
    device: torch.device,
    is_statefree: bool = False,
) -> np.ndarray:
    """Build per-timestep operator matrices.

    Returns
    -------
    A : np.ndarray, shape (len(time_indices), N_XT, N_out)
    """
    if is_statefree:
        return build_A_t_statefree(model, F_normed, time_indices, N_XT, N_out, device)
    return build_A_t_neural(model, F_normed, time_indices, N_XT, N_out, device)


def compute_operator_ts(A_t_series: np.ndarray) -> pd.DataFrame:
    """Compute per-timestep row_heterogeneity, row_entropy, dobrushin.

    Parameters
    ----------
    A_t_series : (T, N_XT, N_out)

    Returns
    -------
    DataFrame with columns: timestep, row_heterogeneity, row_entropy, dobrushin
    """
    T = len(A_t_series)
    row_het = compute_row_heterogeneity(A_t_series)   # (T,)
    row_ent = compute_row_entropy(A_t_series)          # (T,)
    dobr    = compute_dobrushin(A_t_series)            # (T,)

    return pd.DataFrame({
        "timestep":          np.arange(T),
        "row_heterogeneity": row_het,
        "row_entropy":       row_ent,
        "dobrushin":         dobr,
    })


def compute_ck_ts(
    A_t_1step: np.ndarray,
    A_t_hstep: np.ndarray,
    h: int,
) -> pd.DataFrame:
    """Compare direct h-step operator to composed 1-step products, per timestep.

    Both arrays must be square (N_XT == N_out) because composed 1-step matrices
    live in the same state space as the direct h-step matrix.

    Parameters
    ----------
    A_t_1step  : (T_1step, N_XT, N_XT)  — 1-step operators over test period
    A_t_hstep  : (T_hstep, N_XT, N_XT)  — direct h-step operators over test period
                 T_hstep = T_1step - h + 1  (shorter because composition needs look-ahead)
    h          : int, composition depth

    Returns
    -------
    DataFrame with columns: timestep, ck_kl, ck_tv
        timestep is aligned to A_t_hstep (i.e., 0 … T_hstep-1)
    """
    composed = build_ck_composed(A_t_1step, h)    # (T_hstep, N, N)
    T_out = min(len(composed), len(A_t_hstep))
    errors = compute_ck_errors(
        A_t_hstep[:T_out].astype(np.float64),
        composed[:T_out].astype(np.float64),
    )
    per_t_kl = errors["per_time_kl"]  # (T_out,)
    per_t_tv = errors["per_time_tv"]  # (T_out,)

    return pd.DataFrame({
        "timestep": np.arange(T_out),
        "ck_kl":    per_t_kl,
        "ck_tv":    per_t_tv,
    })


def attach_dates(
    df: pd.DataFrame,
    prices_path: str | Path,
    test_indices: List[int],
) -> pd.DataFrame:
    """Add a 'date' column by reading the date column from prices_{ticker}.csv.

    Aligns df.timestep (0-based position in test split) to the calendar date
    at prices_df.iloc[test_indices[timestep]].

    Parameters
    ----------
    df          : DataFrame with a 'timestep' column (0-based)
    prices_path : path to prices_{ticker}.csv (must have 'date' column)
    test_indices: list/array of integer row indices in prices_df that form the test split

    Returns
    -------
    df with an additional 'date' column (datetime64); rows where timestep exceeds
    len(test_indices) get NaT.
    """
    prices_df = pd.read_csv(prices_path, parse_dates=["date"])
    dates = prices_df["date"].values
    test_indices = np.asarray(test_indices)

    def _lookup(ts):
        if ts < len(test_indices):
            idx = test_indices[ts]
            if idx < len(dates):
                return dates[idx]
        return pd.NaT

    df = df.copy()
    df["date"] = df["timestep"].apply(_lookup)
    return df


def select_snapshot_timesteps(
    A_t_series: np.ndarray,
) -> dict:
    """Select calm, bearish, bullish snapshot timestep indices for Fig 6.

    calm    : timestep with Dobrushin coefficient at 10th percentile (most uniform)
    bearish : timestep where weighted mean output bin is lowest
    bullish : timestep where weighted mean output bin is highest

    Parameters
    ----------
    A_t_series : (T, N_XT, N_out)  — operator series (state_cond, h=1, N=55)

    Returns
    -------
    dict: {'calm': int, 'bearish': int, 'bullish': int}
    """
    T, N_XT, N_out = A_t_series.shape
    dobr = compute_dobrushin(A_t_series)  # (T,)

    # Weighted mean output bin averaged across input states
    bins = np.arange(N_out, dtype=np.float64)
    # A_t_series[t]: (N_XT, N_out) → mean over rows of dot with bins
    wmean = (A_t_series.astype(np.float64) * bins[None, None, :]).sum(axis=2).mean(axis=1)  # (T,)

    calm_t    = int(np.where(dobr == np.percentile(dobr, 10, method="nearest"))[0][0])
    bearish_t = int(np.argmin(wmean))
    bullish_t = int(np.argmax(wmean))

    return {"calm": calm_t, "bearish": bearish_t, "bullish": bullish_t}
