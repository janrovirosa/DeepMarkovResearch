#!/usr/bin/env python3
"""Extended multi-asset experiment runner (A-EXT through F-NEW).

Does NOT modify or replace run_multiasset_experiments.py (Experiments A-E).
Writes to *_ext experiment subdirectories inside results_multiasset/.

Experiments
-----------
A-EXT  Full factorial baseline: 10 tickers Ã— 5h Ã— 5N Ã— 2 models Ã— 3 seeds
B-EXT  Sigma sweep: JPM Ã— hâˆˆ{1,21} Ã— 5Ïƒ Ã— 3 seeds
C-EXT  SWA + best sigma: same grid as A-EXT
D-EXT  Higher-order k: JPM Ã— 5k Ã— 4h Ã— N=55 Ã— 3 seeds
E-EXT  Conditioning regime: 10 tickers Ã— 4 regimes Ã— hâˆˆ{1,21} Ã— N=55 Ã— 3 seeds
F-NEW  Cross-asset synchrony: post-hoc from C-EXT weights, no training

Entry points
------------
  run_all_experiments_ext(cfg)  â€” importable callable for notebook use
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.bins import build_all_configs, compute_X_t
from scripts.config import ExperimentConfig, make_config
from scripts.data import (
    build_all_splits,
    build_feature_subset,
    load_master_dataset,
)
from scripts.diagnostics_ts import (
    attach_dates,
    build_A_t_series,
    compute_ck_ts,
    compute_operator_ts,
    select_snapshot_timesteps,
)
from scripts.eval import evaluate_model
from scripts.models import StateConditionedNet, StateFreeNet
from scripts.models_extensions import (
    HigherOrderStateConditionedNet,
    build_loaders_higher_order,
)
from scripts.results_writer import save_results
from scripts.sweeps import sweep_sigma
from scripts.train import (
    MasterDataset,
    build_loaders,
    cache_model,
    is_cached,
    load_cached_model,
    train_one_run,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

HORIZONS_EXT = [1, 5, 10, 21, 63]
BINS_EXT     = [4, 10, 20, 35, 55]
TICKERS      = ["JPM", "C", "WFC", "GS", "MS", "PNC", "USB", "FITB", "MTB", "BAC"]
REGIMES_EXT  = ["full", "macro_only", "own_only", "other_banks_only"]
K_LIST       = [1, 2, 3, 5, 10]
SIGMA_LIST_EXT = [0.5, 1.0, 1.5, 2.0, 3.0]
D_HORIZONS   = [1, 5, 21, 63]
E_HORIZONS   = [1, 21]
N_XT_TARGET  = 55          # fixed state bins (never changes)

RESULTS_DIR  = Path("results_multiasset")
CACHE_DIR    = RESULTS_DIR / "_cache_ext"

try:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"[device] AMD GPU via DirectML: {DEVICE}")
except ImportError:
    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"[device] {DEVICE}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _marginal_nll(N: int) -> float:
    return float(np.log(N))


def _cache_path_ext(
    exp_prefix: str,
    ticker: str,
    model_type: str,
    h: int,
    N: int,
    seed: int,
    extra: str = "",
) -> Path:
    suffix = f"_{extra}" if extra else ""
    fname = f"{exp_prefix}_{ticker}_{model_type}_h{h}_N{N}_seed{seed}{suffix}.pt"
    return CACHE_DIR / fname


def setup_ticker_ext(ticker: str, cfg: ExperimentConfig) -> Dict:
    """Load ticker data for extended grid (HORIZONS_EXT Ã— BINS_EXT)."""
    prices, F_raw, feature_cols = load_master_dataset(ticker=ticker)
    F_normed = F_raw  # already standardised

    splits = build_all_splits(prices, HORIZONS_EXT)
    train_end = int(splits[min(HORIZONS_EXT)]["idx_train"][-1]) + 1
    X_t_all, N_XT, edges_xt = compute_X_t(prices, N_XT_target=N_XT_TARGET, train_end=train_end)

    configs = build_all_configs(
        prices, F_normed, X_t_all,
        horizons=HORIZONS_EXT,
        n_bins_list=BINS_EXT,
        N_XT=N_XT,
        edges_xt=edges_xt,
        splits=splits,
        sigma_anchor=cfg.sigma_anchor,
    )
    prices_path = (
        Path(__file__).resolve().parent / "dataset_multiasset" / f"prices_{ticker}.csv"
    )
    return {
        "prices": prices,
        "F_normed": F_normed,
        "feature_cols": feature_cols,
        "n_feat": F_normed.shape[1],
        "X_t_all": X_t_all,
        "N_XT": N_XT,
        "splits": splits,
        "configs": configs,
        "prices_path": prices_path,
    }


def _build_model(model_type: str, n_feat: int, N_XT: int, N_actual: int) -> torch.nn.Module:
    if model_type == "state_cond":
        return StateConditionedNet(n_feat, N_XT, N_actual)
    return StateFreeNet(n_feat, N_actual)


def _save_operator_ts(
    model: torch.nn.Module,
    F_normed: np.ndarray,
    test_indices: np.ndarray,
    N_XT: int,
    N_actual: int,
    device: torch.device,
    is_statefree: bool,
    out_dir: Path,
    model_type: str,
    h: int,
    N: int,
    seed: int,
    prices_path: Path,
) -> np.ndarray:
    """Compute and save per-timestep operator diagnostics CSV. Returns A_t series."""
    A_t = build_A_t_series(model, F_normed, test_indices, N_XT, N_actual, device, is_statefree)
    df = compute_operator_ts(A_t)
    df = attach_dates(df, prices_path, list(test_indices))
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"operator_ts_{model_type}_h{h}_N{N}_seed{seed}.csv", index=False)
    return A_t


def _save_ck_ts(
    A_t_1step_extended: Optional[np.ndarray],
    A_t_hstep: np.ndarray,
    h: int,
    N: int,
    out_dir: Path,
    model_type: str,
    seed: int,
    hstep_test_indices: np.ndarray,
    prices_path: Path,
) -> None:
    """Compute and save per-timestep CK error CSV.

    CK composition is only well-defined for N == N_XT (square matrices).
    For N != N_XT, writes NaN-filled CSV.
    A_t_1step_extended has shape (len(hstep_test_indices) + h - 1, N_XT, N_XT).
    """
    if N != N_XT_TARGET or A_t_1step_extended is None:
        T_ck = max(0, len(A_t_hstep) - h + 1)
        df_ck = pd.DataFrame({
            "timestep": np.arange(T_ck),
            "ck_kl": np.full(T_ck, np.nan),
            "ck_tv": np.full(T_ck, np.nan),
        })
    else:
        df_ck = compute_ck_ts(A_t_1step_extended, A_t_hstep, h)
    df_ck = attach_dates(df_ck, prices_path, list(hstep_test_indices))
    out_dir.mkdir(parents=True, exist_ok=True)
    df_ck.to_csv(out_dir / f"ck_ts_{model_type}_h{h}_N{N}_seed{seed}.csv", index=False)


def _save_snapshots(
    A_t_series: np.ndarray,
    out_dir: Path,
    seed: int,
) -> None:
    """Save calm/bearish/bullish 55x55 operator snapshots for Fig 6."""
    snaps = select_snapshot_timesteps(A_t_series)
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for label, ts in snaps.items():
        arr = A_t_series[ts].astype(np.float32)
        np.save(snap_dir / f"operator_snapshot_{label}_seed{seed}.npy", arr)


def _build_extended_1step_indices(
    hstep_test_indices: np.ndarray,
    h: int,
    F_len: int,
) -> np.ndarray:
    """Return indices for h=1 model eval needed for CK composition.

    Covers hstep_test_indices[0] to hstep_test_indices[-1] + h - 1,
    clamped to F_len - 1.
    """
    start = int(hstep_test_indices[0])
    end   = min(int(hstep_test_indices[-1]) + h, F_len)
    return np.arange(start, end)


# ---------------------------------------------------------------------------
# CK post-pass
# ---------------------------------------------------------------------------

def _ck_post_pass(
    ticker: str,
    N: int,
    td: Dict,
    exp_tag: str,
    exp_prefix: str,
    sigma: float,
    use_swa: bool,
    cfg: ExperimentConfig,
    a_t_cache: Dict,    # {(model_type, h, seed): A_t_series} â€” operator series computed during training
) -> int:
    """Compute CK time series for all h > 1 using cached h=1 operators.

    Returns the number of deferred (skipped) cells due to missing h=1 cache.
    """
    deferred = 0
    out_dir = RESULTS_DIR / ticker / exp_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    F_normed     = td["F_normed"]
    N_XT         = td["N_XT"]
    n_feat       = td["n_feat"]
    prices_path  = td["prices_path"]

    for model_type in ["state_cond", "state_free"]:
        is_statefree = model_type == "state_free"
        for seed in cfg.seeds:
            # Load / retrieve h=1 A_t series
            A_t_h1 = a_t_cache.get((model_type, 1, seed))
            if A_t_h1 is None:
                cp_h1 = _cache_path_ext(exp_prefix, ticker, model_type, 1, N, seed)
                if not is_cached(cp_h1):
                    print(f"  WARNING: CK deferred for ({ticker}, {model_type}, "
                          f"N={N}, seed={seed}) â€” h=1 model not cached")
                    for h in HORIZONS_EXT:
                        if h > 1:
                            deferred += 1
                    continue
                # Rebuild h=1 model and build A_t
                cfg_h1   = td["configs"][(1, N)]
                model_h1 = _build_model(model_type, n_feat, N_XT, cfg_h1["N_actual"])
                model_h1 = load_cached_model(model_h1, cp_h1).to(DEVICE)
                A_t_h1   = build_A_t_series(
                    model_h1, F_normed, cfg_h1["idx_test"],
                    N_XT, cfg_h1["N_actual"], DEVICE, is_statefree,
                )

            for h in HORIZONS_EXT:
                if h == 1:
                    continue
                try:
                    cfg_h  = td["configs"][(h, N)]
                    A_t_h  = a_t_cache.get((model_type, h, seed))
                    if A_t_h is None:
                        cp_h = _cache_path_ext(exp_prefix, ticker, model_type, h, N, seed)
                        if not is_cached(cp_h):
                            print(f"  WARNING: CK deferred for ({ticker}, {model_type}, "
                                  f"h={h}, N={N}, seed={seed}) â€” h={h} model not cached")
                            deferred += 1
                            continue
                        model_h = _build_model(model_type, n_feat, N_XT, cfg_h["N_actual"])
                        model_h = load_cached_model(model_h, cp_h).to(DEVICE)
                        A_t_h   = build_A_t_series(
                            model_h, F_normed, cfg_h["idx_test"],
                            N_XT, cfg_h["N_actual"], DEVICE, is_statefree,
                        )

                    # Build extended h=1 indices for composition
                    ext_idx = _build_extended_1step_indices(
                        cfg_h["idx_test"], h, len(F_normed)
                    )
                    cfg_h1_any = td["configs"][(1, N)]
                    model_h1_reload = _build_model(
                        model_type, n_feat, N_XT, cfg_h1_any["N_actual"])
                    cp_h1 = _cache_path_ext(exp_prefix, ticker, model_type, 1, N, seed)
                    model_h1_reload = load_cached_model(model_h1_reload, cp_h1).to(DEVICE)
                    A_t_h1_ext = build_A_t_series(
                        model_h1_reload, F_normed, ext_idx,
                        N_XT, cfg_h1_any["N_actual"], DEVICE, is_statefree,
                    )

                    _save_ck_ts(
                        A_t_h1_ext, A_t_h, h, N,
                        out_dir, model_type, seed,
                        cfg_h["idx_test"], prices_path,
                    )
                except Exception:
                    print(f"  CK post-pass FAILED ({ticker}, {model_type}, h={h}, N={N}, seed={seed})")
                    traceback.print_exc()
                    deferred += 1

    return deferred


# ---------------------------------------------------------------------------
# Core per-cell training / evaluation
# ---------------------------------------------------------------------------

def run_one_ext(
    ticker: str,
    model_type: str,
    h: int,
    N: int,
    seed: int,
    td: Dict,
    F_normed: np.ndarray,
    n_feat: int,
    sigma: float,
    use_swa: bool,
    cfg: ExperimentConfig,
    exp_prefix: str,
    exp_tag: str,
    extra_cache_tag: str = "",
) -> Tuple[Dict, np.ndarray]:
    """Train or load one model; save operator_ts CSV; return (metrics_row, A_t_series)."""
    cfg_dict  = td["configs"][(h, N)]
    N_actual  = cfg_dict["N_actual"]
    N_XT      = td["N_XT"]
    X_t_all   = td["X_t_all"]
    prices_path = td["prices_path"]
    out_dir   = RESULTS_DIR / ticker / exp_tag
    is_sf     = model_type == "state_free"

    train_loader, val_loader, test_loader = build_loaders(cfg_dict, F_normed, X_t_all)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = _build_model(model_type, n_feat, N_XT, N_actual)
    cp    = _cache_path_ext(exp_prefix, ticker, model_type, h, N, seed, extra=extra_cache_tag)

    if is_cached(cp):
        model = load_cached_model(model, cp)
    else:
        state_dict, _ = train_one_run(
            model, train_loader, val_loader,
            N=N_actual, sigma=sigma, device=DEVICE,
            use_swa=use_swa, swa_epochs=cfg.swa_epochs,
            max_epochs=cfg.max_epochs, patience=cfg.patience,
        )
        cache_model(state_dict, cp)

    model = model.to(DEVICE)

    train_m = evaluate_model(model, train_loader, N_actual, DEVICE)
    val_m   = evaluate_model(model, val_loader,   N_actual, DEVICE)
    test_m  = evaluate_model(model, test_loader,  N_actual, DEVICE)

    nll_train = -train_m["mean_ll"]
    nll_val   = -val_m["mean_ll"]
    nll_test  = -test_m["mean_ll"]
    delta     = nll_test - _marginal_nll(N_actual)

    print(f"  [{exp_prefix}] {ticker} {model_type:12s} h={h:2d} N={N:2d} seed={seed} "
          f"nll_test={nll_test:.4f}  Î”={delta:+.4f}")

    # Operator diagnostics time series (always)
    A_t = _save_operator_ts(
        model, F_normed, cfg_dict["idx_test"],
        N_XT, N_actual, DEVICE, is_sf,
        out_dir, model_type, h, N, seed, prices_path,
    )

    # Operator snapshots (h=1, N=55, state_cond only)
    if h == 1 and N == 55 and model_type == "state_cond":
        _save_snapshots(A_t, out_dir, seed)

    row = {
        "model": model_type, "h": h, "N": N, "seed": seed,
        "nll_train": nll_train, "nll_val": nll_val,
        "nll_test": nll_test, "delta_marginal": delta,
    }
    return row, A_t


# ---------------------------------------------------------------------------
# Experiment A-EXT â€” Full factorial baseline
# ---------------------------------------------------------------------------

def run_experiment_A_ext(cfg: ExperimentConfig):
    print("\n" + "=" * 60)
    print("EXPERIMENT A-EXT â€” Full factorial baseline")
    print("=" * 60)

    total_deferred = 0

    for ticker in TICKERS:
        try:
            td = setup_ticker_ext(ticker, cfg)
            F_normed = td["F_normed"]
            n_feat   = td["n_feat"]
        except Exception:
            print(f"  [A-EXT] FAILED setup for {ticker}")
            traceback.print_exc()
            continue

        for N in BINS_EXT:
            all_rows: List[Dict] = []
            # {(model_type, h, seed): A_t_series} â€” for CK post-pass
            a_t_cache: Dict[Tuple, np.ndarray] = {}

            for h in HORIZONS_EXT:          # h=1 FIRST â†’ populates cache for CK
                for model_type in ["state_cond", "state_free"]:
                    for seed in cfg.seeds:
                        try:
                            row, A_t = run_one_ext(
                                ticker, model_type, h, N, seed,
                                td, F_normed, n_feat,
                                sigma=cfg.sigma_anchor, use_swa=False,
                                cfg=cfg, exp_prefix="Aext", exp_tag="baseline_ext",
                            )
                            all_rows.append(row)
                            a_t_cache[(model_type, h, seed)] = A_t
                        except Exception:
                            print(f"  [A-EXT] FAILED {ticker} {model_type} h={h} N={N} seed={seed}")
                            traceback.print_exc()
                            all_rows.append({
                                "model": model_type, "h": h, "N": N, "seed": seed,
                                "nll_train": None, "nll_val": None,
                                "nll_test": None, "delta_marginal": None,
                            })

            # Save NLL rows
            if all_rows:
                save_results(
                    RESULTS_DIR, ticker, "baseline_ext",
                    config_dict={"sigma": cfg.sigma_anchor, "use_swa": False,
                                 "N": N, "horizons": HORIZONS_EXT},
                    metrics_dict={"rows": all_rows},
                )

            # CK post-pass for this (ticker, N)
            d = _ck_post_pass(
                ticker, N, td,
                exp_tag="baseline_ext", exp_prefix="Aext",
                sigma=cfg.sigma_anchor, use_swa=False,
                cfg=cfg, a_t_cache=a_t_cache,
            )
            total_deferred += d

    print(f"\n[A-EXT] CK deferred: {total_deferred} cells. Check run.log for details.")


# ---------------------------------------------------------------------------
# Experiment B-EXT â€” Sigma sweep (JPM, hâˆˆ{1,21})
# ---------------------------------------------------------------------------

def run_experiment_B_ext(cfg: ExperimentConfig) -> float:
    print("\n" + "=" * 60)
    print("EXPERIMENT B-EXT â€” Sigma sweep on JPM (h=1 and h=21)")
    print("=" * 60)

    sweep_cache = CACHE_DIR / "sweep_sigma_ext"

    results_by_h = {}
    for h_sweep in [1, 21]:
        try:
            df = sweep_sigma(
                "JPM", h=h_sweep, N=55,
                sigma_list=SIGMA_LIST_EXT, n_seeds=len(cfg.seeds),
                device=DEVICE,
                max_epochs=cfg.max_epochs, patience=cfg.patience,
                cache_dir=sweep_cache,
            )
            out_p = RESULTS_DIR / "JPM" / "sigma_sweep_ext" / f"sweep_h{h_sweep}.csv"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_p, index=False)
            best = float(df.groupby("sigma")["nll_val"].mean().idxmin())
            print(f"  [B-EXT] h={h_sweep}: best_sigma = {best}")
            results_by_h[h_sweep] = (best, df)
        except Exception:
            print(f"  [B-EXT] FAILED for h={h_sweep}")
            traceback.print_exc()
            results_by_h[h_sweep] = (cfg.sigma_anchor, pd.DataFrame())

    best_h1 = results_by_h.get(1, (cfg.sigma_anchor,))[0]
    best_h21 = results_by_h.get(21, (cfg.sigma_anchor,))[0]
    if best_h1 != best_h21:
        print(f"  [B-EXT] NOTE: h=1 best_sigma={best_h1} differs from h=21 best_sigma={best_h21}")
        print(f"  [B-EXT] Using h=1 result (BEST_SIGMA={best_h1}) for downstream experiments")
    print(f"\n[B-EXT] BEST_SIGMA = {best_h1}")
    return best_h1


# ---------------------------------------------------------------------------
# Experiment C-EXT â€” SWA + best sigma, full factorial
# ---------------------------------------------------------------------------

def run_experiment_C_ext(cfg: ExperimentConfig, best_sigma: float):
    print("\n" + "=" * 60)
    print(f"EXPERIMENT C-EXT â€” SWA + best_sigma={best_sigma}, full factorial")
    print("=" * 60)

    total_deferred = 0

    for ticker in TICKERS:
        try:
            td = setup_ticker_ext(ticker, cfg)
            F_normed = td["F_normed"]
            n_feat   = td["n_feat"]
        except Exception:
            print(f"  [C-EXT] FAILED setup for {ticker}")
            traceback.print_exc()
            continue

        for N in BINS_EXT:
            all_rows: List[Dict] = []
            a_t_cache: Dict[Tuple, np.ndarray] = {}

            for h in HORIZONS_EXT:
                for model_type in ["state_cond", "state_free"]:
                    for seed in cfg.seeds:
                        try:
                            row, A_t = run_one_ext(
                                ticker, model_type, h, N, seed,
                                td, F_normed, n_feat,
                                sigma=best_sigma, use_swa=True,
                                cfg=cfg, exp_prefix=f"Cext_s{best_sigma}",
                                exp_tag="swa_bestsigma_ext",
                            )
                            all_rows.append(row)
                            a_t_cache[(model_type, h, seed)] = A_t
                        except Exception:
                            print(f"  [C-EXT] FAILED {ticker} {model_type} h={h} N={N} seed={seed}")
                            traceback.print_exc()
                            all_rows.append({
                                "model": model_type, "h": h, "N": N, "seed": seed,
                                "nll_train": None, "nll_val": None,
                                "nll_test": None, "delta_marginal": None,
                            })

            if all_rows:
                save_results(
                    RESULTS_DIR, ticker, "swa_bestsigma_ext",
                    config_dict={"sigma": best_sigma, "use_swa": True,
                                 "swa_epochs": cfg.swa_epochs,
                                 "N": N, "horizons": HORIZONS_EXT},
                    metrics_dict={"rows": all_rows},
                )

            d = _ck_post_pass(
                ticker, N, td,
                exp_tag="swa_bestsigma_ext", exp_prefix=f"Cext_s{best_sigma}",
                sigma=best_sigma, use_swa=True,
                cfg=cfg, a_t_cache=a_t_cache,
            )
            total_deferred += d

    print(f"\n[C-EXT] CK deferred: {total_deferred} cells. Check run.log for details.")


# ---------------------------------------------------------------------------
# Experiment D-EXT â€” Higher-order k, full horizon sweep (JPM only)
# ---------------------------------------------------------------------------

def run_experiment_D_ext(cfg: ExperimentConfig, best_sigma: float):
    print("\n" + "=" * 60)
    print("EXPERIMENT D-EXT â€” Higher-order k ablation (JPM, N=55)")
    print("=" * 60)

    ticker = "JPM"
    N = 55
    total_deferred = 0

    try:
        td = setup_ticker_ext(ticker, cfg)
    except Exception:
        print(f"  [D-EXT] FAILED setup for {ticker}")
        traceback.print_exc()
        return

    F_normed   = td["F_normed"]
    n_feat     = td["n_feat"]
    N_XT       = td["N_XT"]
    X_t_all    = td["X_t_all"]
    prices_path = td["prices_path"]
    out_dir    = RESULTS_DIR / ticker / "higher_order_ext"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    a_t_k1_cache: Dict[int, np.ndarray] = {}  # seed -> A_t for k=1 (same as state_cond h=1)

    for k in K_LIST:
        for h in D_HORIZONS:
            cfg_dict = td["configs"][(h, N)]
            N_actual = cfg_dict["N_actual"]

            for seed in cfg.seeds:
                try:
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    cp = CACHE_DIR / f"Dext_{ticker}_ho_k{k}_h{h}_N{N}_seed{seed}.pt"
                    train_lo, val_lo, test_lo = build_loaders_higher_order(
                        cfg_dict, F_normed, X_t_all, k=k)
                    model = HigherOrderStateConditionedNet(n_feat, N_XT, N_actual, k=k)

                    if is_cached(cp):
                        model = load_cached_model(model, cp)
                    else:
                        state_dict, _ = train_one_run(
                            model, train_lo, val_lo,
                            N=N_actual, sigma=best_sigma, device=DEVICE,
                            use_swa=True, swa_epochs=cfg.swa_epochs,
                            max_epochs=cfg.max_epochs, patience=cfg.patience,
                        )
                        cache_model(state_dict, cp)

                    model = model.to(DEVICE)
                    val_m  = evaluate_model(model, val_lo,  N_actual, DEVICE)
                    test_m = evaluate_model(model, test_lo, N_actual, DEVICE)
                    nll_val  = -val_m["mean_ll"]
                    nll_test = -test_m["mean_ll"]
                    delta = nll_test - _marginal_nll(N_actual)

                    print(f"  [D-EXT] k={k} h={h} seed={seed} "
                          f"nll_test={nll_test:.4f}  Î”={delta:+.4f}")

                    # Operator diagnostics â€” HigherOrderStateConditionedNet has same
                    # forward(F, x_t) interface as StateConditionedNet (k=1 is identical)
                    A_t = build_A_t_series(
                        model, F_normed, cfg_dict["idx_test"],
                        N_XT, N_actual, DEVICE, is_statefree=False,
                    )
                    op_df = compute_operator_ts(A_t)
                    op_df = attach_dates(op_df, prices_path, list(cfg_dict["idx_test"]))
                    op_df.to_csv(
                        out_dir / f"operator_ts_ho_k{k}_h{h}_N{N}_seed{seed}.csv",
                        index=False,
                    )

                    # CK (N=55=N_XT: well-defined)
                    if h > 1 and k == 1:
                        # Use k=1 operator as "1-step" for composition (k>1 CK not well-defined)
                        k1_A = a_t_k1_cache.get(seed)
                        if k1_A is not None:
                            ext_idx = _build_extended_1step_indices(
                                cfg_dict["idx_test"], h, len(F_normed))
                            # Reload k=1 h=1 model for extended range
                            cp_k1_h1 = CACHE_DIR / f"Dext_{ticker}_ho_k1_h1_N{N}_seed{seed}.pt"
                            if is_cached(cp_k1_h1):
                                m_k1 = HigherOrderStateConditionedNet(n_feat, N_XT, N_actual, k=1)
                                m_k1 = load_cached_model(m_k1, cp_k1_h1).to(DEVICE)
                                A_k1_ext = build_A_t_series(
                                    m_k1, F_normed, ext_idx, N_XT, N_actual, DEVICE, False)
                                df_ck = compute_ck_ts(A_k1_ext, A_t, h)
                                df_ck = attach_dates(df_ck, prices_path, list(cfg_dict["idx_test"]))
                                df_ck.to_csv(
                                    out_dir / f"ck_ts_ho_k{k}_h{h}_N{N}_seed{seed}.csv",
                                    index=False,
                                )

                    if k == 1 and h == 1:
                        a_t_k1_cache[seed] = A_t

                    all_rows.append({
                        "model": f"ho_k{k}", "h": h, "N": N, "seed": seed,
                        "nll_train": None, "nll_val": nll_val,
                        "nll_test": nll_test, "delta_marginal": delta,
                    })

                except Exception:
                    print(f"  [D-EXT] FAILED k={k} h={h} seed={seed}")
                    traceback.print_exc()
                    all_rows.append({
                        "model": f"ho_k{k}", "h": h, "N": N, "seed": seed,
                        "nll_train": None, "nll_val": None,
                        "nll_test": None, "delta_marginal": None,
                    })

    if all_rows:
        save_results(
            RESULTS_DIR, ticker, "higher_order_ext",
            config_dict={"sigma": best_sigma, "use_swa": True,
                         "k_list": K_LIST, "horizons": D_HORIZONS, "N": N},
            metrics_dict={"rows": all_rows},
        )
    print(f"\n[D-EXT] CK deferred: {total_deferred} cells.")


# ---------------------------------------------------------------------------
# Experiment E-EXT â€” Conditioning regime, all stocks, hâˆˆ{1,21}
# ---------------------------------------------------------------------------

def run_experiment_E_ext(cfg: ExperimentConfig, best_sigma: float):
    print("\n" + "=" * 60)
    print("EXPERIMENT E-EXT â€” Conditioning regime (all tickers, hâˆˆ{1,21})")
    print("=" * 60)

    N = 55
    total_deferred = 0

    for ticker in TICKERS:
        try:
            td = setup_ticker_ext(ticker, cfg)
        except Exception:
            print(f"  [E-EXT] FAILED setup for {ticker}")
            traceback.print_exc()
            continue

        F_normed_full = td["F_normed"]
        feature_cols  = td["feature_cols"]
        N_XT          = td["N_XT"]
        X_t_all       = td["X_t_all"]
        prices_path   = td["prices_path"]
        out_dir = RESULTS_DIR / ticker / "conditioning_regime_ext"
        out_dir.mkdir(parents=True, exist_ok=True)

        all_rows: List[Dict] = []

        for regime in REGIMES_EXT:
            try:
                F_sub, _ = build_feature_subset(
                    F_normed_full, feature_cols, regime, target_ticker=ticker)
            except Exception:
                print(f"  [E-EXT] FAILED build_feature_subset {ticker} {regime}")
                traceback.print_exc()
                continue
            n_feat_sub = F_sub.shape[1]

            for h in E_HORIZONS:
                cfg_dict  = td["configs"][(h, N)]
                N_actual  = cfg_dict["N_actual"]

                # Build regime-specific loaders
                tr_ds = MasterDataset(F_sub, X_t_all, cfg_dict["y_all"], cfg_dict["idx_train"])
                va_ds = MasterDataset(F_sub, X_t_all, cfg_dict["y_all"], cfg_dict["idx_val"])
                te_ds = MasterDataset(F_sub, X_t_all, cfg_dict["y_all"], cfg_dict["idx_test"])
                train_lo = DataLoader(tr_ds, 256, shuffle=True)
                val_lo   = DataLoader(va_ds, 512, shuffle=False)
                test_lo  = DataLoader(te_ds, 512, shuffle=False)

                a_t_cache_e: Dict[int, np.ndarray] = {}

                for seed in cfg.seeds:
                    try:
                        torch.manual_seed(seed)
                        np.random.seed(seed)

                        cp = CACHE_DIR / f"Eext_{ticker}_regime{regime}_h{h}_N{N}_seed{seed}.pt"
                        model = StateConditionedNet(n_feat_sub, N_XT, N_actual)

                        if is_cached(cp):
                            model = load_cached_model(model, cp)
                        else:
                            state_dict, _ = train_one_run(
                                model, train_lo, val_lo,
                                N=N_actual, sigma=best_sigma, device=DEVICE,
                                use_swa=True, swa_epochs=cfg.swa_epochs,
                                max_epochs=cfg.max_epochs, patience=cfg.patience,
                            )
                            cache_model(state_dict, cp)

                        model = model.to(DEVICE)
                        val_m  = evaluate_model(model, val_lo,  N_actual, DEVICE)
                        test_m = evaluate_model(model, test_lo, N_actual, DEVICE)
                        nll_val  = -val_m["mean_ll"]
                        nll_test = -test_m["mean_ll"]
                        delta = nll_test - _marginal_nll(N_actual)

                        print(f"  [E-EXT] {ticker} {regime:20s} h={h} seed={seed} "
                              f"nll_test={nll_test:.4f}  Î”={delta:+.4f}")

                        # Operator diagnostics (using regime features)
                        A_t = build_A_t_series(
                            model, F_sub, cfg_dict["idx_test"],
                            N_XT, N_actual, DEVICE, is_statefree=False,
                        )
                        op_df = compute_operator_ts(A_t)
                        op_df = attach_dates(op_df, prices_path, list(cfg_dict["idx_test"]))
                        op_df.to_csv(
                            out_dir / f"operator_ts_state_cond_{regime}_h{h}_N{N}_seed{seed}.csv",
                            index=False,
                        )
                        a_t_cache_e[seed] = A_t

                        all_rows.append({
                            "model": f"state_cond_{regime}", "h": h, "N": N, "seed": seed,
                            "nll_train": None, "nll_val": nll_val,
                            "nll_test": nll_test, "delta_marginal": delta,
                        })

                    except Exception:
                        print(f"  [E-EXT] FAILED {ticker} {regime} h={h} seed={seed}")
                        traceback.print_exc()
                        all_rows.append({
                            "model": f"state_cond_{regime}", "h": h, "N": N, "seed": seed,
                            "nll_train": None, "nll_val": None,
                            "nll_test": None, "delta_marginal": None,
                        })

                # CK for h=21 (using h=1 operators as composition base)
                if h == 21:
                    for seed in cfg.seeds:
                        try:
                            A_t_h = a_t_cache_e.get(seed)
                            cp_h1  = CACHE_DIR / f"Eext_{ticker}_regime{regime}_h1_N{N}_seed{seed}.pt"
                            if A_t_h is not None and is_cached(cp_h1):
                                cfg_h1 = td["configs"][(1, N)]
                                m_h1   = StateConditionedNet(n_feat_sub, N_XT, cfg_h1["N_actual"])
                                m_h1   = load_cached_model(m_h1, cp_h1).to(DEVICE)
                                ext_idx = _build_extended_1step_indices(
                                    cfg_dict["idx_test"], h, len(F_normed_full))
                                A_h1_ext = build_A_t_series(
                                    m_h1, F_sub, ext_idx, N_XT,
                                    cfg_h1["N_actual"], DEVICE, False,
                                )
                                df_ck = compute_ck_ts(A_h1_ext, A_t_h, h)
                                df_ck = attach_dates(
                                    df_ck, prices_path, list(cfg_dict["idx_test"]))
                                df_ck.to_csv(
                                    out_dir / f"ck_ts_state_cond_{regime}_h{h}_N{N}_seed{seed}.csv",
                                    index=False,
                                )
                        except Exception:
                            print(f"  [E-EXT] CK FAILED {ticker} {regime} h={h} seed={seed}")
                            traceback.print_exc()
                            total_deferred += 1

        if all_rows:
            save_results(
                RESULTS_DIR, ticker, "conditioning_regime_ext",
                config_dict={"sigma": best_sigma, "use_swa": True,
                             "swa_epochs": cfg.swa_epochs,
                             "regimes": REGIMES_EXT, "horizons": E_HORIZONS, "N": N},
                metrics_dict={"rows": all_rows},
            )

    print(f"\n[E-EXT] CK deferred: {total_deferred} cells.")


# ---------------------------------------------------------------------------
# Experiment F-NEW â€” Cross-asset synchrony (post-hoc, no training)
# ---------------------------------------------------------------------------

def run_experiment_F_new(cfg: ExperimentConfig, best_sigma: float):
    print("\n" + "=" * 60)
    print("EXPERIMENT F-NEW â€” Cross-asset synchrony (post-hoc)")
    print("=" * 60)

    N = 55
    h_ck = 5   # CK error at h=5 for the cross-asset CSV
    all_stock_rows: List[Dict] = []

    for ticker in TICKERS:
        try:
            td = setup_ticker_ext(ticker, cfg)
            F_normed  = td["F_normed"]
            n_feat    = td["n_feat"]
            N_XT      = td["N_XT"]
            prices_path = td["prices_path"]
            cfg_h1    = td["configs"][(1, N)]
            cfg_h5    = td["configs"][(h_ck, N)]
            N_actual  = cfg_h1["N_actual"]
        except Exception:
            print(f"  [F-NEW] FAILED setup for {ticker}")
            traceback.print_exc()
            continue

        # Load C-EXT state_cond h=1 N=55 models (use mean over seeds via arithmetic avg)
        h1_At_list = []
        h5_At_list = []
        for seed in cfg.seeds:
            try:
                cp_h1 = _cache_path_ext(f"Cext_s{best_sigma}", ticker, "state_cond", 1, N, seed)
                cp_h5 = _cache_path_ext(f"Cext_s{best_sigma}", ticker, "state_cond", h_ck, N, seed)
                if not is_cached(cp_h1):
                    print(f"  [F-NEW] WARNING: C-EXT h=1 N=55 state_cond cache missing "
                          f"for {ticker} seed={seed} â€” skipping seed")
                    continue
                model_h1 = StateConditionedNet(n_feat, N_XT, N_actual)
                model_h1 = load_cached_model(model_h1, cp_h1).to(DEVICE)
                A_t_h1 = build_A_t_series(
                    model_h1, F_normed, cfg_h1["idx_test"],
                    N_XT, N_actual, DEVICE, False,
                )
                h1_At_list.append(A_t_h1)

                if is_cached(cp_h5):
                    model_h5 = StateConditionedNet(n_feat, N_XT, N_actual)
                    model_h5 = load_cached_model(model_h5, cp_h5).to(DEVICE)
                    A_t_h5 = build_A_t_series(
                        model_h5, F_normed, cfg_h5["idx_test"],
                        N_XT, N_actual, DEVICE, False,
                    )
                    h5_At_list.append(A_t_h5)
            except Exception:
                print(f"  [F-NEW] FAILED loading {ticker} seed={seed}")
                traceback.print_exc()

        if not h1_At_list:
            print(f"  [F-NEW] WARNING: no models loaded for {ticker} â€” skipping")
            continue

        # Average operator diagnostics over seeds
        A_mean = np.mean(h1_At_list, axis=0)
        op_df  = compute_operator_ts(A_mean)
        op_df  = attach_dates(op_df, prices_path, list(cfg_h1["idx_test"]))

        # CK error at h=h_ck (averaged over seeds)
        ck_kl_vals = []
        for A_t_h1, A_t_h5 in zip(h1_At_list, h5_At_list):
            ext_idx = _build_extended_1step_indices(
                cfg_h5["idx_test"], h_ck, len(F_normed))
            # Reload extended 1-step operators from extended indices
            # Use mean-based A_t_h1 (approximate) for simplicity in F-NEW
            A_t_h1_ext = A_t_h1[:len(ext_idx)]  # use available data
            if len(A_t_h1_ext) >= len(A_t_h5) + h_ck - 1:
                df_ck = compute_ck_ts(A_t_h1_ext, A_t_h5, h_ck)
                ck_kl_vals.append(df_ck["ck_kl"].values)

        ck_kl_mean = np.mean(ck_kl_vals, axis=0) if ck_kl_vals else np.full(len(op_df), np.nan)

        # Pad or truncate ck_kl to match op_df length
        T_op = len(op_df)
        T_ck = len(ck_kl_mean)
        ck_kl_aligned = np.full(T_op, np.nan)
        ck_kl_aligned[:min(T_op, T_ck)] = ck_kl_mean[:min(T_op, T_ck)]

        for i, row in op_df.iterrows():
            all_stock_rows.append({
                "date": row.get("date", pd.NaT),
                "ticker": ticker,
                "row_heterogeneity": row["row_heterogeneity"],
                "row_entropy": row["row_entropy"],
                "dobrushin": row["dobrushin"],
                "ck_error_h5": ck_kl_aligned[i],
            })
        print(f"  [F-NEW] {ticker}: {len(op_df)} timesteps processed")

    if all_stock_rows:
        df_out = pd.DataFrame(all_stock_rows)
        out_p  = RESULTS_DIR / "cross_asset_diagnostics.csv"
        df_out.to_csv(out_p, index=False)
        print(f"  [F-NEW] Written: {out_p}  ({len(df_out)} rows)")
    else:
        print("  [F-NEW] WARNING: no data written to cross_asset_diagnostics.csv")


# ---------------------------------------------------------------------------
# Summary extension
# ---------------------------------------------------------------------------

def write_summary_ext(best_sigma: float):
    print("\n" + "=" * 60)
    print("Writing extended summary.md sections")
    print("=" * 60)

    summary_path = RESULTS_DIR / "summary.md"
    existing = summary_path.read_text() if summary_path.exists() else "# Multi-Asset Summary\n"

    lines = ["\n\n---\n# Extended Experiment Summary (A-EXT through F-NEW)\n"]

    # Per-stock best config table
    lines.append("## Per-Stock Best Configuration (argmin delta_marginal over all h,N,regime)\n")
    best_rows = []
    for ticker in TICKERS:
        frames = []
        for exp_tag in ["swa_bestsigma_ext", "conditioning_regime_ext"]:
            p = RESULTS_DIR / ticker / exp_tag / "nll_metrics.csv"
            if p.exists():
                df = pd.read_csv(p)
                df["ticker"] = ticker
                df["exp_tag"] = exp_tag
                frames.append(df)
        if frames:
            df_all = pd.concat(frames).reset_index(drop=True)
            df_valid = df_all.dropna(subset=["delta_marginal"])
            if not df_valid.empty:
                br = df_valid.loc[df_valid["delta_marginal"].idxmin()]
                # ensure scalar (guard against duplicate index returning DataFrame)
                if isinstance(br, pd.DataFrame):
                    br = br.iloc[0]
                best_rows.append({
                    "ticker": ticker,
                    "exp_tag": br.get("exp_tag", ""),
                    "model":   str(br.get("model", "")),
                    "h":       br.get("h", ""),
                    "N":       br.get("N", ""),
                    "delta_marginal": br.get("delta_marginal", ""),
                })
    if best_rows:
        lines.append(pd.DataFrame(best_rows).to_markdown(index=False))
    else:
        lines.append("_No extended results found._")
    lines.append("\n")

    # Pooled delta_marginal by horizon and by N
    lines.append("## Pooled delta_marginal (meanÂ±std across 10 stocks)\n")
    all_c_ext = []
    for ticker in TICKERS:
        p = RESULTS_DIR / ticker / "swa_bestsigma_ext" / "nll_metrics.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["ticker"] = ticker
            all_c_ext.append(df)
    if all_c_ext:
        df_pool = pd.concat(all_c_ext)
        by_h = df_pool.groupby("h")["delta_marginal"].agg(["mean", "std"])
        by_N = df_pool.groupby("N")["delta_marginal"].agg(["mean", "std"])
        lines.append("### By horizon h\n")
        lines.append(by_h.to_markdown())
        lines.append("\n### By N bins\n")
        lines.append(by_N.to_markdown())
    lines.append("\n")

    # Cross-asset synchrony headline
    lines.append("## Cross-Asset Synchrony\n")
    cross_p = RESULTS_DIR / "cross_asset_diagnostics.csv"
    if cross_p.exists():
        df_cross = pd.read_csv(cross_p, parse_dates=["date"])
        pivot = df_cross.pivot_table(
            index="date", columns="ticker", values="row_entropy")
        corr = pivot.corr()
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        off_diag = corr.values[mask]
        off_diag_finite = off_diag[np.isfinite(off_diag)]
        mean_corr = float(np.mean(off_diag_finite)) if len(off_diag_finite) > 0 else np.nan
        lines.append(f"Mean off-diagonal Pearson correlation of row entropy: **{mean_corr:.4f}**\n")
    else:
        lines.append("_cross_asset_diagnostics.csv not found._\n")

    # Failure diagnostic
    lines.append("## Failure Diagnostic (delta_marginal > 0)\n")
    fail_rows = []
    for ticker in TICKERS:
        for exp_tag in ["baseline_ext", "swa_bestsigma_ext", "conditioning_regime_ext"]:
            p = RESULTS_DIR / ticker / exp_tag / "nll_metrics.csv"
            if p.exists():
                df = pd.read_csv(p)
                bad = df[df["delta_marginal"].fillna(0) > 0]
                for _, r in bad.iterrows():
                    fail_rows.append({
                        "ticker": ticker, "exp_tag": exp_tag,
                        "model": r.get("model",""), "h": r.get("h",""),
                        "N": r.get("N",""), "seed": r.get("seed",""),
                        "delta_marginal": r.get("delta_marginal",""),
                    })
    if fail_rows:
        lines.append(pd.DataFrame(fail_rows).to_markdown(index=False))
    else:
        lines.append("_No positive delta_marginal results found._\n")
    lines.append("\n")

    with open(summary_path, "w") as f:
        f.write(existing + "\n".join(lines))
    print(f"  Extended summary appended to: {summary_path}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_all_experiments_ext(cfg: ExperimentConfig):
    """Run the extended multi-asset experiment suite (A-EXT â†’ F-NEW).

    Parameters
    ----------
    cfg : ExperimentConfig
        Controls seeds, swa_epochs, sigma_anchor, max_epochs, patience.
        horizons and n_bins_list from cfg are ignored â€” HORIZONS_EXT and
        BINS_EXT are used instead.
    """
    print(f"\nResults directory: {RESULTS_DIR.resolve()}")
    print(f"Device: {DEVICE}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    run_experiment_A_ext(cfg)
    best_sigma = run_experiment_B_ext(cfg)
    run_experiment_C_ext(cfg, best_sigma)
    run_experiment_D_ext(cfg, best_sigma)
    run_experiment_E_ext(cfg, best_sigma)
    run_experiment_F_new(cfg, best_sigma)
    write_summary_ext(best_sigma)

    print("\nAll extended experiments complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_experiments_ext(make_config())
