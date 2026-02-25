"""Bin construction utilities.

Core functions lifted verbatim from MasterNotebook.ipynb (cells 14/15, 17).
New function: get_xt_labels_for_ck — builds CK-compatible X_{t+h} labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core bin utilities (lifted from master notebook)
# ---------------------------------------------------------------------------

def compute_quantile_edges(returns_train: np.ndarray, N: int) -> np.ndarray:
    """Fit N quantile bin edges on training returns.

    Returns array of N+1 edges with -inf/+inf at boundaries.
    """
    edges = np.quantile(returns_train, np.linspace(0, 1, N + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_bins(returns: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Digitize returns into bins [0, N-1]."""
    N = len(edges) - 1
    return np.clip(np.digitize(returns, edges) - 1, 0, N - 1)


def _effective_bins_raw(R_train: np.ndarray, N: int) -> int:
    """Count duplicate-free bins achievable with N quantile bins."""
    raw = np.quantile(R_train, np.linspace(0, 1, N + 1))
    return len(np.unique(raw)) - 1


def get_edges(R_train: np.ndarray, N_target: int) -> Tuple[int, np.ndarray]:
    """Return (N_actual, edges) for up to N_target quantile bins.

    Auto-reduces N if duplicate quantile edges are present.
    Edges have ±inf boundaries.
    """
    eff = _effective_bins_raw(R_train, N_target)
    N_actual = min(N_target, eff)
    raw = np.quantile(R_train, np.linspace(0, 1, N_actual + 1))
    eff2 = len(np.unique(raw)) - 1
    if eff2 < N_actual:
        N_actual = eff2
        raw = np.quantile(R_train, np.linspace(0, 1, N_actual + 1))
    edges = raw.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return N_actual, edges


def compute_sigma(
    edges_target: np.ndarray,
    edges_anchor: np.ndarray,
    sigma_anchor: float = 1.0,
) -> float:
    """Sigma scaling: sigma_N = sigma_anchor * (median_width_anchor / median_width_target)."""
    finite_t = edges_target[np.isfinite(edges_target)]
    finite_a = edges_anchor[np.isfinite(edges_anchor)]
    widths_t = np.diff(finite_t)
    widths_a = np.diff(finite_a)
    delta_t = float(np.median(widths_t)) if len(widths_t) > 0 else 1.0
    delta_a = float(np.median(widths_a)) if len(widths_a) > 0 else 1.0
    return sigma_anchor * (delta_a / delta_t)


# ---------------------------------------------------------------------------
# Config builder (lifted + wrapped)
# ---------------------------------------------------------------------------

def build_all_configs(
    prices: np.ndarray,
    F_normed: np.ndarray,
    X_t_all: np.ndarray,
    horizons: list,
    n_bins_list: list,
    N_XT: int,
    edges_xt: np.ndarray,
    splits: dict,
    sigma_anchor: float = 1.0,
    results_dir: Path | None = None,
) -> dict:
    """Build (h, N) config dicts mirroring master notebook cell 17.

    Returns dict keyed by (h, N_requested) with all arrays + metadata.
    """
    from .data import compute_returns

    configs = {}

    for h in horizons:
        # Y_t^(h) = bin of return from t+1 to t+1+h (strictly forward-looking relative to X_t)
        # prices[1:] shifts the base price by one day so h=1 label is NOT identical to X_t
        R_h = compute_returns(prices[1:], h)
        sp = splits[h]
        R_train = R_h[sp["idx_train"]]

        # Anchor edges for sigma computation
        anchor_N = max(n_bins_list)
        _, edges_anchor_h = get_edges(R_train, anchor_N)

        for N in n_bins_list:
            N_actual, edges = get_edges(R_train, N)
            y_all = assign_bins(R_h, edges)
            sigma = compute_sigma(edges, edges_anchor_h, sigma_anchor)

            configs[(h, N)] = {
                "R_h": R_h,
                "y_all": y_all,
                "edges": edges,
                "sigma": sigma,
                "N_actual": N_actual,
                "T_h": sp["T_h"],
                "idx_train": sp["idx_train"],
                "idx_val": sp["idx_val"],
                "idx_test": sp["idx_test"],
            }

            if results_dir is not None:
                out = Path(results_dir)
                out.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"edge": edges}).to_csv(
                    out / f"bin_edges_h{h}_N{N_actual}.csv", index=False
                )

    return configs


# ---------------------------------------------------------------------------
# CK label builder (NEW)
# ---------------------------------------------------------------------------

def get_xt_labels_for_ck(X_t_all: np.ndarray, horizons: list) -> Dict[int, np.ndarray]:
    """Build CK-compatible labels y_t^(h) := X_{t+h}.

    The label X_{t+h} lives in the SAME 55-bin state space as X_t.
    This makes A_t^(h) square [55×55], enabling matrix multiplication
    for the time-inhomogeneous CK product.

    Parameters
    ----------
    X_t_all : np.ndarray, shape (T_total,)
        Full 1-day return bin sequence.
    horizons : list of int

    Returns
    -------
    dict : h -> np.ndarray of shape (T_total - h,)
        y_t^(h) = X_{t+h} for t = 0, ..., T_total-h-1
        Both the input X_t (first T-h values) and this label have length T-h.
    """
    return {h: X_t_all[h:] for h in horizons}


# ---------------------------------------------------------------------------
# X_t computation helper
# ---------------------------------------------------------------------------

def compute_X_t(prices: np.ndarray, N_XT_target: int, train_end: int):
    """Compute 1-day return bins X_t for all time points.

    Fits quantile edges ONCE on h=1 train window (first train_end returns).

    Returns
    -------
    X_t_all  : np.ndarray, shape (len(prices)-1,)
    N_XT     : int  (actual number of bins, ≤ N_XT_target)
    edges_xt : np.ndarray, shape (N_XT+1,)
    """
    from .data import compute_returns

    r_1day_all = compute_returns(prices, 1)
    r_train = r_1day_all[:train_end]

    N_XT, edges_xt = get_edges(r_train, N_XT_target)
    X_t_all = assign_bins(r_1day_all, edges_xt)
    return X_t_all, N_XT, edges_xt
