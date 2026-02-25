"""Evaluation utilities.

evaluate_model, evaluate_baselines, mean_log_likelihood lifted from MasterNotebook.ipynb.
All other functions are new.
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

EPS = 1e-8


# ---------------------------------------------------------------------------
# Lifted from master notebook
# ---------------------------------------------------------------------------

def mean_log_likelihood(pred_dist: np.ndarray, y_true: np.ndarray, eps: float = EPS) -> float:
    """Mean log P(y_true) under predicted distribution.

    pred_dist : (N_test, N_bins)
    y_true    : (N_test,) integer array
    """
    N = len(y_true)
    probs = pred_dist[np.arange(N), y_true]
    return float(np.log(probs + eps).mean())


def evaluate_model(
    model,
    loader: DataLoader,
    N: int,
    device: torch.device,
    eps: float = EPS,
) -> Dict:
    """Compute mean_ll, accuracy, severity on a DataLoader."""
    model.eval()
    all_lp, all_preds, all_true, all_exp = [], [], [], []
    bins_t = torch.arange(N, dtype=torch.float32, device=device)

    with torch.no_grad():
        for F_b, xt_b, y_b in loader:
            F_b, xt_b, y_b = F_b.to(device), xt_b.to(device), y_b.to(device)
            logits = model(F_b, xt_b)
            probs = F.softmax(logits, dim=1)
            lp = torch.log(probs[torch.arange(len(y_b), device=device), y_b] + eps)
            all_lp.append(lp.cpu())
            all_preds.append(logits.argmax(1).cpu())
            all_true.append(y_b.cpu())
            all_exp.append((probs * bins_t).sum(1).cpu())

    all_lp = torch.cat(all_lp)
    all_preds = torch.cat(all_preds)
    all_true = torch.cat(all_true)
    all_exp = torch.cat(all_exp)

    return {
        "mean_ll": all_lp.mean().item(),
        "accuracy": (all_preds == all_true).float().mean().item(),
        "severity": torch.abs(all_exp - all_true.float()).mean().item(),
        "loglik_per_sample": all_lp.numpy(),
    }


def evaluate_baselines(
    cfg_dict: dict,
    X_t_all: np.ndarray,
    N_XT: int,
    alpha_grid: list,
    tau_grid: list,
    eps: float = EPS,
) -> Dict:
    """Evaluate marginal, additive, and backoff baselines.

    Returns dict with per-baseline results including per-sample loglik for bootstrap.
    """
    y_all = cfg_dict["y_all"]
    idx_train = cfg_dict["idx_train"]
    idx_val   = cfg_dict["idx_val"]
    idx_test  = cfg_dict["idx_test"]
    N = cfg_dict["N_actual"]

    y_tr, y_va, y_te = y_all[idx_train], y_all[idx_val], y_all[idx_test]
    s_tr, s_va, s_te = X_t_all[idx_train], X_t_all[idx_val], X_t_all[idx_test]

    # Marginal
    marginal = _compute_marginal(y_tr, N)
    ll_marg_val  = float(np.log(marginal[y_va] + eps).mean())
    ll_marg_test = float(np.log(marginal[y_te] + eps).mean())
    lp_marg_test = np.log(marginal[y_te] + eps)

    # Additive — tune on val
    best_ll, best_alpha = -np.inf, None
    for alpha in alpha_grid:
        P = _compute_conditional_additive(s_tr, y_tr, N_XT, N, alpha, marginal)
        ll = float(np.log(P[s_va, y_va] + eps).mean())
        if ll > best_ll:
            best_ll, best_alpha = ll, alpha
    P_add = _compute_conditional_additive(s_tr, y_tr, N_XT, N, best_alpha, marginal)
    lp_add = np.log(P_add[s_te, y_te] + eps)

    # Backoff — tune (alpha, tau) on val
    best_bk_ll, best_bk_alpha, best_tau = -np.inf, None, None
    for alpha in alpha_grid:
        for tau in tau_grid:
            A_bk, _, _ = _build_backoff_matrix(s_tr, y_tr, N_XT, N, alpha, tau, marginal)
            ll = float(np.log(A_bk[s_va, y_va] + eps).mean())
            if ll > best_bk_ll:
                best_bk_ll, best_bk_alpha, best_tau = ll, alpha, tau
    A_bk, _, _ = _build_backoff_matrix(s_tr, y_tr, N_XT, N, best_bk_alpha, best_tau, marginal)
    lp_bk = np.log(A_bk[s_te, y_te] + eps)

    return {
        "marginal": {
            "test_ll": float(lp_marg_test.mean()),
            "val_ll": ll_marg_val,
            "accuracy": float((np.argmax(marginal) == y_te).mean()),
            "severity": np.nan,
            "matrix": marginal,
            "loglik_per_sample": lp_marg_test,
        },
        "additive": {
            "test_ll": float(lp_add.mean()),
            "val_ll": best_ll,
            "accuracy": float((P_add[s_te].argmax(1) == y_te).mean()),
            "severity": float(np.abs((P_add[s_te] * np.arange(N)).sum(1) - y_te).mean()),
            "alpha": best_alpha,
            "matrix": P_add,
            "loglik_per_sample": lp_add,
        },
        "backoff": {
            "test_ll": float(lp_bk.mean()),
            "val_ll": best_bk_ll,
            "accuracy": float((A_bk[s_te].argmax(1) == y_te).mean()),
            "severity": float(np.abs((A_bk[s_te] * np.arange(N)).sum(1) - y_te).mean()),
            "alpha": best_bk_alpha,
            "tau": best_tau,
            "matrix": A_bk,
            "loglik_per_sample": lp_bk,
        },
    }


def _compute_marginal(y: np.ndarray, N: int) -> np.ndarray:
    counts = np.bincount(y, minlength=N).astype(np.float64)
    return counts / counts.sum()


def _compute_joint_counts(s, y, n_x, n_y):
    C = np.zeros((n_x, n_y), dtype=np.float64)
    for si, yi in zip(s, y):
        C[si, yi] += 1
    return C


def _compute_conditional_additive(s, y, n_x, n_y, alpha, marginal_fallback=None):
    C = _compute_joint_counts(s, y, n_x, n_y)
    if alpha == 0:
        row_sums = C.sum(axis=1)
        P = np.zeros_like(C)
        for i in range(n_x):
            if row_sums[i] > 0:
                P[i] = C[i] / row_sums[i]
            elif marginal_fallback is not None:
                P[i] = marginal_fallback
            else:
                P[i] = 1.0 / n_y
        return P
    C_alpha = C + alpha
    return C_alpha / C_alpha.sum(axis=1, keepdims=True)


def _build_backoff_matrix(s_train, y_train, n_x, n_y, alpha, tau, marginal):
    C = _compute_joint_counts(s_train, y_train, n_x, n_y)
    C_alpha = C + alpha
    P_cond = C_alpha / C_alpha.sum(axis=1, keepdims=True)
    state_counts = C.sum(axis=1)
    lam = state_counts / (state_counts + tau)
    A = np.zeros((n_x, n_y), dtype=np.float64)
    for i in range(n_x):
        A[i] = lam[i] * P_cond[i] + (1 - lam[i]) * marginal
    return A, lam, P_cond


# ---------------------------------------------------------------------------
# Per-sample log-likelihood (for block bootstrap CIs)
# ---------------------------------------------------------------------------

def get_loglik_per_sample_model(
    model, loader: DataLoader, N: int, device: torch.device, eps: float = EPS
) -> np.ndarray:
    """Return per-day log P(y_t | model) array for all samples in loader."""
    return evaluate_model(model, loader, N, device, eps)["loglik_per_sample"]


def get_loglik_per_sample_baseline(
    pred_matrix: np.ndarray,
    X_t_test: np.ndarray,
    y_test: np.ndarray,
    eps: float = EPS,
) -> np.ndarray:
    """Return per-day log P(y_t) for count/backoff baseline.

    pred_matrix[i, j] = P(y=j | X_t=i).
    """
    return np.log(pred_matrix[X_t_test, y_test] + eps)


# ---------------------------------------------------------------------------
# Degeneracy statistics
# ---------------------------------------------------------------------------

def compute_degeneracy_stats(
    y_train: np.ndarray,
    N_Y: int,
    thresholds: tuple = (5, 10),
) -> Dict:
    """Visitation counts for label Y_t in training set.

    Returns
    -------
    dict with:
        counts        : np.ndarray (N_Y,) — count per bin
        effective     : int — number of bins with count > 0
        frac_below    : dict threshold -> float fraction of bins below threshold
    """
    counts = np.bincount(y_train, minlength=N_Y).astype(np.int64)
    result = {
        "counts": counts,
        "effective": int((counts > 0).sum()),
        "frac_below": {},
    }
    for k in thresholds:
        result["frac_below"][k] = float((counts < k).mean())
    return result


def compute_transition_sparsity(
    X_t: np.ndarray,
    Y_t: np.ndarray,
    N_X: int,
    N_Y: int,
    thresholds: tuple = (5, 10),
) -> Dict:
    """Build joint count matrix C[N_X, N_Y] and compute cell-level degeneracy metrics.

    ALIGNMENT: both arrays must have the same length (call site is responsible).
    For horizon h:
        X_t_aligned = X_t_all[:T_h][idx_train]
        Y_t_aligned = y_all[:T_h][idx_train]

    Returns
    -------
    dict with:
        C                            : np.ndarray (N_X, N_Y) — joint count matrix
        row_sums                     : np.ndarray (N_X,)
        frac_cells_zero              : float — fraction of C[i,j] == 0
        frac_cells_lt5               : float — fraction of C[i,j] < 5
        median_nonzero_per_row       : float — median # of j with C[i,j] > 0
        p90_nonzero_per_row          : float — 90th pctile of nonzero count per row
        median_row_entropy_empirical : float — median entropy of empirical row dists
        median_row_maxprob_empirical : float — median max_j C[i,j]/row_sum_i
    """
    assert len(X_t) == len(Y_t), "X_t and Y_t must have the same length"
    valid = (X_t >= 0) & (X_t < N_X) & (Y_t >= 0) & (Y_t < N_Y)
    C = np.zeros((N_X, N_Y), dtype=np.int64)
    np.add.at(C, (X_t[valid], Y_t[valid]), 1)
    row_sums = C.sum(axis=1)

    # Cell-level sparsity
    frac_cells_zero = float((C == 0).mean())
    frac_cells_lt5  = float((C < 5).mean())
    nonzero_per_row = (C > 0).sum(axis=1)
    median_nonzero_per_row = float(np.median(nonzero_per_row))
    p90_nonzero_per_row    = float(np.percentile(nonzero_per_row, 90))

    # Row-distribution statistics (only for rows with at least one observation)
    row_entropy_vals, row_maxprob_vals = [], []
    for i in np.where(row_sums > 0)[0]:
        p = C[i].astype(np.float64) / row_sums[i]
        p_nz = p[p > 0]
        row_entropy_vals.append(-float(np.sum(p_nz * np.log(p_nz))))
        row_maxprob_vals.append(float(p.max()))
    median_row_entropy_empirical = float(np.median(row_entropy_vals)) if row_entropy_vals else np.nan
    median_row_maxprob_empirical = float(np.median(row_maxprob_vals)) if row_maxprob_vals else np.nan

    return {
        "C": C,
        "row_sums": row_sums,
        "frac_cells_zero": frac_cells_zero,
        "frac_cells_lt5": frac_cells_lt5,
        "median_nonzero_per_row": median_nonzero_per_row,
        "p90_nonzero_per_row": p90_nonzero_per_row,
        "median_row_entropy_empirical": median_row_entropy_empirical,
        "median_row_maxprob_empirical": median_row_maxprob_empirical,
    }


# ---------------------------------------------------------------------------
# CK consistency
# ---------------------------------------------------------------------------

def build_ck_composed(A1_matrices: np.ndarray, h: int) -> np.ndarray:
    """Compose A1[t] × A1[t+1] × ... × A1[t+h-1].

    A1_matrices: (T, N_XT, N_XT) — square because label = X_{t+h} in same 55-bin space.

    Returns
    -------
    composed : np.ndarray, shape (T-h+1, N_XT, N_XT)
    """
    T, N, _ = A1_matrices.shape
    T_out = T - h + 1
    composed = np.zeros((T_out, N, N), dtype=np.float64)
    for pos in range(T_out):
        mat = A1_matrices[pos].astype(np.float64)
        for k in range(1, h):
            mat = mat @ A1_matrices[pos + k].astype(np.float64)
        composed[pos] = mat
    return composed


def compute_ck_errors(
    A_h: np.ndarray,
    A_composed: np.ndarray,
    state_weights: Optional[np.ndarray] = None,
    eps: float = EPS,
) -> Dict:
    """Compare A_h (direct) vs A_composed (product of 1-step matrices).

    Parameters
    ----------
    A_h, A_composed : (T, N_XT, N_XT)
    state_weights   : (N_XT,) optional weight per state (e.g., train visitation counts)

    Returns
    -------
    dict: mean_kl, mean_tv, frobenius, per_time_kl (np.ndarray T,)
    """
    T, N, _ = A_h.shape
    assert A_h.shape == A_composed.shape

    P = A_h.astype(np.float64)
    Q = A_composed.astype(np.float64)
    Q = np.clip(Q, eps, None)
    P_clip = np.clip(P, eps, None)

    # KL per row: sum_j p*log(p/q)
    kl_rows = (P_clip * np.log(P_clip / Q)).sum(axis=2)  # (T, N_XT)

    # TV per row: 0.5 * sum |p-q|
    tv_rows = 0.5 * np.abs(P - Q).sum(axis=2)            # (T, N_XT)

    # Frobenius per time step
    frob_t = np.sqrt(((P - Q) ** 2).sum(axis=(1, 2)))    # (T,)

    if state_weights is not None:
        w = state_weights / (state_weights.sum() + eps)
        mean_kl = float((kl_rows * w[None, :]).sum(axis=1).mean())
        mean_tv = float((tv_rows * w[None, :]).sum(axis=1).mean())
    else:
        mean_kl = float(kl_rows.mean())
        mean_tv = float(tv_rows.mean())

    return {
        "mean_kl": mean_kl,
        "mean_tv": mean_tv,
        "frobenius": float(frob_t.mean()),
        "per_time_kl": kl_rows.mean(axis=1),  # (T,)
        "per_time_tv": tv_rows.mean(axis=1),
    }


# ---------------------------------------------------------------------------
# Operator interpretability diagnostics
# ---------------------------------------------------------------------------

def compute_dobrushin(A: np.ndarray) -> np.ndarray:
    """delta(A) = 0.5 * max_{i,j} ||A[i,:] - A[j,:]||_1.

    A : (N, N) or (T, N, N)
    Returns scalar or (T,) array.
    """
    if A.ndim == 2:
        diff = np.abs(A[:, None, :] - A[None, :, :]).sum(axis=2)
        return float(0.5 * diff.max())
    # (T, N, N)
    diff = np.abs(A[:, :, None, :] - A[:, None, :, :]).sum(axis=3)
    return 0.5 * diff.max(axis=(1, 2))


def compute_row_heterogeneity(A: np.ndarray) -> np.ndarray:
    """Average pairwise TV distance between rows.

    rho(A) = (2 / N(N-1)) * sum_{i<j} TV(A[i,:], A[j,:])

    Measures state-dependence strength. For StateFreeNet, rho ≈ 0 (sanity check).
    A : (N, N) or (T, N, N).  Returns scalar or (T,) array.
    """
    if A.ndim == 2:
        N = A.shape[0]
        if N <= 1:
            return 0.0
        tv_sum = 0.0
        n_pairs = 0
        for i in range(N):
            for j in range(i + 1, N):
                tv_sum += 0.5 * np.abs(A[i] - A[j]).sum()
                n_pairs += 1
        return float(tv_sum / n_pairs)

    T, N, _ = A.shape
    result = np.zeros(T)
    for t in range(T):
        result[t] = compute_row_heterogeneity(A[t])
    return result


def compute_row_entropy(A: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Mean Shannon entropy per row: mean_i H(A[i,:]).

    A : (N, N) or (T, N, N).  Returns scalar or (T,) array.
    """
    if A.ndim == 2:
        A_c = np.clip(A, eps, None)
        return float(-(A_c * np.log(A_c)).sum(axis=1).mean())
    A_c = np.clip(A, eps, None)
    return -(A_c * np.log(A_c)).sum(axis=2).mean(axis=1)


def compute_spectral_mixing_proxy(
    A: np.ndarray,
    lazy_eps: float = 0.1,
    n_power_iter: int = 200,
) -> np.ndarray:
    """Robust spectral mixing proxy (no symmetry / reversibility assumption).

    Algorithm (per matrix):
    1. A_lazy = (1-lazy_eps)*A + lazy_eps*(1/N)*ones[N,N]
    2. Stationary dist pi via power iteration on A_lazy^T
    3. M = A_lazy - ones[:,None] @ pi[None,:]  (rank-1 deflation)
    4. proxy = sigma_max(M) = numpy.linalg.svd(M, compute_uv=False)[0]
       Lower sigma_max => faster mixing / stronger contraction.
    5. Return NaN on any failure (never crash).

    A : (N, N) or (T, N, N).  Returns scalar or (T,) array.
    Figure should only be shown if ≥50% of values are finite.
    """
    if A.ndim == 2:
        return _spectral_proxy_single(A, lazy_eps, n_power_iter)
    T = A.shape[0]
    result = np.full(T, np.nan)
    for t in range(T):
        result[t] = _spectral_proxy_single(A[t], lazy_eps, n_power_iter)
    return result


def _spectral_proxy_single(A: np.ndarray, lazy_eps: float, n_power_iter: int) -> float:
    try:
        N = A.shape[0]
        A_lazy = (1 - lazy_eps) * A + lazy_eps * np.ones((N, N)) / N
        # Power iteration for stationary distribution
        pi = np.ones(N) / N
        for _ in range(n_power_iter):
            pi = A_lazy.T @ pi
            pi /= pi.sum()
        M = A_lazy - np.ones((N, 1)) @ pi[None, :]
        svs = np.linalg.svd(M, compute_uv=False)
        return float(svs[0])
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def compute_pit(probs: np.ndarray, y_true: np.ndarray, rng=None) -> np.ndarray:
    """Randomized PIT for discrete distributions.

    PIT_i ~ Uniform(F(y_i - 1), F(y_i)) where F is the CDF of the predicted distribution.

    Returns
    -------
    pit : np.ndarray, shape (N_test,) — should be ~Uniform(0,1) if calibrated.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    N_test = len(y_true)
    pit = np.zeros(N_test)
    cdf = np.cumsum(probs, axis=1)
    for i in range(N_test):
        y = y_true[i]
        F_below = float(cdf[i, y - 1]) if y > 0 else 0.0
        F_at = float(cdf[i, y])
        pit[i] = rng.uniform(F_below, F_at)
    return pit


def compute_ece(
    probs: np.ndarray,
    y_true: np.ndarray,
    event_fn: Callable,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """ECE for a binary event.

    event_fn(y) -> bool/int array, e.g. lambda y: y < N//2 for negative return.

    Returns
    -------
    ece : float
    conf_bins : np.ndarray (n_bins,) — mean predicted probability per bin
    acc_bins  : np.ndarray (n_bins,) — empirical frequency per bin
    """
    # Predicted probability of event
    N = probs.shape[1]
    all_bins = np.arange(N)
    event_mask = np.array([bool(event_fn(b)) for b in all_bins])
    pred_prob_event = probs[:, event_mask].sum(axis=1)  # (N_test,)
    actual_event = event_fn(y_true).astype(float)        # (N_test,)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    conf_bins = np.zeros(n_bins)
    acc_bins = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for b in range(n_bins):
        mask = (pred_prob_event >= bin_edges[b]) & (pred_prob_event < bin_edges[b + 1])
        if b == n_bins - 1:
            mask |= (pred_prob_event == 1.0)
        counts[b] = mask.sum()
        if counts[b] > 0:
            conf_bins[b] = pred_prob_event[mask].mean()
            acc_bins[b] = actual_event[mask].mean()

    ece = float(np.abs(conf_bins - acc_bins).dot(counts) / len(y_true))
    return ece, conf_bins, acc_bins


def compute_brier(
    probs: np.ndarray,
    y_true: np.ndarray,
    event_fn: Callable,
) -> float:
    """Brier score for a binary event."""
    N = probs.shape[1]
    event_mask = np.array([bool(event_fn(b)) for b in range(N)])
    pred_p = probs[:, event_mask].sum(axis=1)
    actual = event_fn(y_true).astype(float)
    return float(((pred_p - actual) ** 2).mean())


# ---------------------------------------------------------------------------
# Block bootstrap CI
# ---------------------------------------------------------------------------

def block_bootstrap_ci(
    loglik_t: np.ndarray,
    block_size: int = 21,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Circular block bootstrap CI for mean log-likelihood.

    Input
    -----
    loglik_t : np.ndarray — per-day log P(y_t) on the test set.
               Produced by get_loglik_per_sample_model() or _baseline().

    Scope: compute ONLY for key configs (h=1,N=55) and (h=10,N=55),
           optionally val-best per horizon if enabled.

    Returns
    -------
    (mean, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    T = len(loglik_t)
    means = np.zeros(n_boot)
    for b in range(n_boot):
        n_blocks = int(np.ceil(T / block_size))
        starts = rng.integers(0, T, size=n_blocks)
        indices = np.concatenate([
            (np.arange(block_size) + s) % T for s in starts
        ])[:T]
        means[b] = loglik_t[indices].mean()
    return (
        float(loglik_t.mean()),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )
