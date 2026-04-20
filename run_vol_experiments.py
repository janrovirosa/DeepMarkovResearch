#!/usr/bin/env python3
"""Volatility-state conditioning experiment runner.

Tests whether conditioning on realized-volatility bin state (σ_t) is more
predictive than return-bin state (X_t) for forward return distributions.

Model variants
--------------
  state_free          — StateFreeNet,          F=full,  X_t unused
  state_cond_return   — StateConditionedNet,   F=full,  X_t=return-bin
  state_cond_vol_w5   — StateConditionedNet,   F=full,  X_t=vol-bin(w=5)
  state_cond_vol_w10  — StateConditionedNet,   F=full,  X_t=vol-bin(w=10)
  state_cond_vol_w21  — StateConditionedNet,   F=full,  X_t=vol-bin(w=21)
  state_cond_macro    — StateConditionedNet,   F=macro, X_t=return-bin

Warmup alignment
----------------
All models are trained on the SAME date range.  The maximum vol warmup
(vol_window=21 → warmup=20) is applied to every array before any model sees
data, so NLL comparisons are apples-to-apples.

CK non-square note
------------------
For N∈{35,45} (N ≠ N_XT=55), the operator is non-square and CK composition
is undefined.  Those CSVs contain NaN values and a ck_unavailable_reason
column.  Figures notebooks must render a visible "(N/A: non-square)" label
rather than silently dropping those rows.

Entry point
-----------
  run_vol_experiments(cfg)   — importable callable for notebook use
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.bins import build_all_configs, compute_X_t
from scripts.config import ExperimentConfig, make_config
from scripts.data import (
    build_all_splits,
    build_feature_subset,
    compute_X_t_vol,
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
from scripts.results_writer import save_results
from scripts.train import (
    build_loaders,
    cache_model,
    is_cached,
    load_cached_model,
    train_one_run,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HORIZONS_VOL = [1, 2, 3, 5]
BINS_VOL     = [35, 45, 55]
TICKERS      = ["JPM", "C", "WFC", "GS", "MS", "PNC", "USB", "FITB", "MTB", "BAC"]
VOL_WINDOWS  = [5, 10, 21]
MODELS       = [
    "state_free",
    "state_cond_return",
    "state_cond_vol_w5",
    "state_cond_vol_w10",
    "state_cond_vol_w21",
    "state_cond_macro",
]
BEST_SIGMA   = 1.0       # from existing sigma sweep (B-EXT)
N_XT_TARGET  = 55        # fixed state bins
WARMUP_MAX   = max(VOL_WINDOWS) - 1   # = 20; applied to all models for fair comparison

RESULTS_DIR  = Path("results_vol")
CACHE_DIR    = RESULTS_DIR / "_cache"

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


def _cache_path(ticker: str, model: str, h: int, N: int, seed: int) -> Path:
    return CACHE_DIR / f"{ticker}_{model}_h{h}_N{N}_seed{seed}.pt"


def _vol_window_for_model(model: str) -> Optional[int]:
    if model == "state_cond_vol_w5":  return 5
    if model == "state_cond_vol_w10": return 10
    if model == "state_cond_vol_w21": return 21
    return None


def _is_statefree(model: str) -> bool:
    return model == "state_free"


def _build_model(model: str, n_feat: int, N_XT: int, N_actual: int) -> torch.nn.Module:
    if model == "state_free":
        return StateFreeNet(n_feat, N_actual)
    return StateConditionedNet(n_feat, N_XT, N_actual)


# ---------------------------------------------------------------------------
# Per-ticker setup
# ---------------------------------------------------------------------------

def setup_ticker_vol(ticker: str, cfg: ExperimentConfig) -> Dict:
    """Load ticker data and apply warmup truncation consistently to all arrays.

    All model variants will see the same date range (warmup_max rows dropped
    from the start of every array).  Volatility states for all three windows
    are precomputed and stored in the returned dict.
    """
    prices_raw, F_raw, feature_cols = load_master_dataset(ticker=ticker)
    # F_raw is already standardised for multi-asset; keep as-is
    F_normed_raw = F_raw

    # ── Compute vol states on FULL price series before truncation ──────────
    # We need a representative train_end to fit vol bin edges.  Use the h=1
    # split on the full series as a proxy; vol edges are only used for state
    # assignment, not for label construction.
    T_full = len(prices_raw)
    splits_full = build_all_splits(prices_raw, HORIZONS_VOL)
    train_end_full = int(splits_full[1]["idx_train"][-1]) + 1   # h=1 train end

    X_t_vols_raw: Dict[int, np.ndarray] = {}   # keyed by vol_window
    sigma_ts_raw: Dict[int, np.ndarray] = {}

    for w in VOL_WINDOWS:
        warmup_w = w - 1
        # train_end in the truncated (post-drop) space for this window
        train_end_w = max(0, train_end_full - warmup_w)
        X_t_vol_w, sigma_w, _N, _edges, _wm = compute_X_t_vol(
            prices_raw, vol_window=w, n_bins=N_XT_TARGET, train_end=train_end_w
        )
        X_t_vols_raw[w] = X_t_vol_w   # length T_full - w
        sigma_ts_raw[w] = sigma_w

    # ── Apply uniform warmup truncation (WARMUP_MAX = 20) to all arrays ────
    # After truncation:
    #   prices_eff  has length T_full - WARMUP_MAX
    #   F_normed_eff has length T_full - WARMUP_MAX
    #   X_t_return_eff has length T_full - 1 - WARMUP_MAX
    #   X_t_vol_wN_eff has length T_full - 1 - WARMUP_MAX  (extra trim for smaller windows)
    prices_eff   = prices_raw[WARMUP_MAX:]
    F_normed_eff = F_normed_raw[WARMUP_MAX:]

    # X_t_return: bins on 1-day returns from the FULL series, then truncate
    # train_end for return bins must also be in the truncated space
    splits_eff   = build_all_splits(prices_eff, HORIZONS_VOL)
    train_end_eff = int(splits_eff[min(HORIZONS_VOL)]["idx_train"][-1]) + 1

    # Compute return-state on the effective (truncated) prices
    X_t_return_eff, N_XT_ret, edges_xt = compute_X_t(
        prices_eff, N_XT_target=N_XT_TARGET, train_end=train_end_eff
    )
    # Length: len(prices_eff) - 1 = T_full - WARMUP_MAX - 1

    # Align vol arrays: X_t_vols_raw[w] has length T_full - w
    # After WARMUP_MAX drop, we need length T_full - 1 - WARMUP_MAX
    # X_t_vols_raw[w][WARMUP_MAX - (w-1) :] trims the extra head so start dates align
    X_t_vols_eff: Dict[int, np.ndarray] = {}
    sigma_ts_eff: Dict[int, np.ndarray] = {}
    for w in VOL_WINDOWS:
        extra_trim = WARMUP_MAX - (w - 1)   # ≥ 0 by construction (WARMUP_MAX = max warmup)
        X_t_vols_eff[w] = X_t_vols_raw[w][extra_trim:]
        sigma_ts_eff[w]  = sigma_ts_raw[w][extra_trim:]
        assert len(X_t_vols_eff[w]) == len(X_t_return_eff), (
            f"{ticker} w={w}: vol array length {len(X_t_vols_eff[w])} "
            f"≠ return array length {len(X_t_return_eff)}"
        )

    # ── Build bin configs on effective dataset ─────────────────────────────
    configs = build_all_configs(
        prices_eff, F_normed_eff, X_t_return_eff,
        horizons=HORIZONS_VOL,
        n_bins_list=BINS_VOL,
        N_XT=N_XT_ret,
        edges_xt=edges_xt,
        splits=splits_eff,
        sigma_anchor=cfg.sigma_anchor,
    )

    # ── Macro feature subset ───────────────────────────────────────────────
    F_macro, macro_cols = build_feature_subset(
        F_normed_eff, feature_cols, "macro_only"
    )

    # ── Save vol state time series CSV ────────────────────────────────────
    prices_csv = (
        Path(__file__).resolve().parent / "dataset_multiasset" / f"prices_{ticker}.csv"
    )
    prices_df = pd.read_csv(prices_csv, parse_dates=["date"])
    # Align dates to the effective truncated range
    dates_eff = prices_df["date"].values[WARMUP_MAX + 1:]   # +1 because states = diffs of prices
    ts_len = min(len(dates_eff), len(X_t_return_eff))

    ts_df = pd.DataFrame({"date": dates_eff[:ts_len]})
    for w in VOL_WINDOWS:
        ts_df[f"sigma_t_w{w}"]  = sigma_ts_eff[w][:ts_len]
        ts_df[f"X_t_vol_w{w}"]  = X_t_vols_eff[w][:ts_len]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts_df.to_csv(RESULTS_DIR / f"vol_state_ts_{ticker}.csv", index=False)

    return {
        "prices":           prices_eff,
        "F_normed":         F_normed_eff,
        "F_macro":          F_macro,
        "feature_cols":     feature_cols,
        "macro_cols":       macro_cols,
        "n_feat":           F_normed_eff.shape[1],
        "n_feat_macro":     F_macro.shape[1],
        "X_t_return":       X_t_return_eff,
        "X_t_vols":         X_t_vols_eff,
        "N_XT":             N_XT_ret,
        "splits":           splits_eff,
        "configs":          configs,
        "prices_path":      prices_csv,
    }


# ---------------------------------------------------------------------------
# Operator and CK helpers
# ---------------------------------------------------------------------------

def _save_operator_ts_vol(
    model: torch.nn.Module,
    F_matrix: np.ndarray,
    test_indices: np.ndarray,
    N_XT: int,
    N_actual: int,
    is_sf: bool,
    out_dir: Path,
    model_name: str,
    h: int,
    N: int,
    seed: int,
    prices_path: Path,
) -> np.ndarray:
    A_t = build_A_t_series(model, F_matrix, test_indices, N_XT, N_actual, DEVICE, is_sf)
    df = compute_operator_ts(A_t)
    df = attach_dates(df, prices_path, list(test_indices))
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"operator_ts_h{h}_N{N}_seed{seed}.csv", index=False)
    return A_t


def _save_ck_ts_vol(
    A_t_1step_ext: Optional[np.ndarray],
    A_t_hstep: np.ndarray,
    h: int,
    N: int,
    out_dir: Path,
    seed: int,
    hstep_test_indices: np.ndarray,
    prices_path: Path,
) -> None:
    if N != N_XT_TARGET or A_t_1step_ext is None:
        T_ck = max(0, len(A_t_hstep) - h + 1)
        df_ck = pd.DataFrame({
            "timestep": np.arange(T_ck),
            "ck_kl":    np.full(T_ck, np.nan),
            "ck_tv":    np.full(T_ck, np.nan),
            "ck_unavailable_reason": (
                f"N_out={N} ≠ N_XT=55 (non-square operator; CK composition undefined)"
            ),
        })
    else:
        df_ck = compute_ck_ts(A_t_1step_ext, A_t_hstep, h)
    df_ck = attach_dates(df_ck, prices_path, list(hstep_test_indices))
    out_dir.mkdir(parents=True, exist_ok=True)
    df_ck.to_csv(out_dir / f"ck_ts_h{h}_N{N}_seed{seed}.csv", index=False)


def _save_snapshots_vol(A_t_series: np.ndarray, out_dir: Path, seed: int) -> None:
    snaps = select_snapshot_timesteps(A_t_series)
    snap_dir = out_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    for label, ts in snaps.items():
        arr = A_t_series[ts].astype(np.float32)
        np.save(snap_dir / f"operator_snapshot_{label}_seed{seed}.npy", arr)


def _build_ext_1step_indices(
    hstep_test_indices: np.ndarray, h: int, F_len: int
) -> np.ndarray:
    start = int(hstep_test_indices[0])
    end   = min(int(hstep_test_indices[-1]) + h, F_len)
    return np.arange(start, end)


# ---------------------------------------------------------------------------
# CK post-pass
# ---------------------------------------------------------------------------

def _ck_post_pass_vol(
    ticker: str,
    model_name: str,
    N: int,
    td: Dict,
    a_t_cache: Dict,   # {(model_name, h, seed): A_t}
    cfg: ExperimentConfig,
) -> int:
    """Compute per-timestep CK errors for h > 1 using cached h=1 operators."""
    deferred = 0
    out_dir = RESULTS_DIR / ticker / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    N_XT     = td["N_XT"]
    prices_p = td["prices_path"]

    F_matrix, n_feat = _select_F_and_n(model_name, td)
    X_t_array        = _select_X_t(model_name, td)
    is_sf            = _is_statefree(model_name)

    for seed in cfg.seeds:
        A_t_h1 = a_t_cache.get((model_name, 1, seed))
        if A_t_h1 is None:
            cp_h1 = _cache_path(ticker, model_name, 1, N, seed)
            if not is_cached(cp_h1):
                print(f"  WARNING: CK deferred ({ticker}, {model_name}, N={N}, "
                      f"seed={seed}) — h=1 model not cached")
                deferred += len([h for h in HORIZONS_VOL if h > 1])
                continue
            cfg_h1 = td["configs"][(1, N)]
            m_h1   = _build_model(model_name, n_feat, N_XT, cfg_h1["N_actual"]).to(DEVICE)
            m_h1   = load_cached_model(m_h1, cp_h1)
            A_t_h1 = build_A_t_series(
                m_h1, F_matrix, cfg_h1["idx_test"], N_XT, cfg_h1["N_actual"], DEVICE, is_sf
            )

        for h in HORIZONS_VOL:
            if h == 1:
                continue
            try:
                cfg_h  = td["configs"][(h, N)]
                A_t_h  = a_t_cache.get((model_name, h, seed))
                if A_t_h is None:
                    cp_h = _cache_path(ticker, model_name, h, N, seed)
                    if not is_cached(cp_h):
                        print(f"  WARNING: CK deferred ({ticker}, {model_name}, "
                              f"h={h}, N={N}, seed={seed}) — h={h} model not cached")
                        deferred += 1
                        continue
                    m_h = _build_model(model_name, n_feat, N_XT, cfg_h["N_actual"]).to(DEVICE)
                    m_h = load_cached_model(m_h, cp_h)
                    A_t_h = build_A_t_series(
                        m_h, F_matrix, cfg_h["idx_test"],
                        N_XT, cfg_h["N_actual"], DEVICE, is_sf,
                    )

                ext_idx = _build_ext_1step_indices(
                    cfg_h["idx_test"], h, len(F_matrix)
                )
                cfg_h1_any = td["configs"][(1, N)]
                m_h1_rel   = _build_model(
                    model_name, n_feat, N_XT, cfg_h1_any["N_actual"]
                ).to(DEVICE)
                cp_h1 = _cache_path(ticker, model_name, 1, N, seed)
                m_h1_rel   = load_cached_model(m_h1_rel, cp_h1)
                A_t_h1_ext = build_A_t_series(
                    m_h1_rel, F_matrix, ext_idx,
                    N_XT, cfg_h1_any["N_actual"], DEVICE, is_sf,
                )
                _save_ck_ts_vol(
                    A_t_h1_ext, A_t_h, h, N,
                    out_dir, seed, cfg_h["idx_test"], prices_p,
                )
            except Exception:
                print(f"  CK post-pass FAILED ({ticker}, {model_name}, h={h}, "
                      f"N={N}, seed={seed})")
                traceback.print_exc()
                deferred += 1

    return deferred


# ---------------------------------------------------------------------------
# Feature / state selection helpers
# ---------------------------------------------------------------------------

def _select_F_and_n(model_name: str, td: Dict) -> Tuple[np.ndarray, int]:
    if model_name == "state_cond_macro":
        return td["F_macro"], td["n_feat_macro"]
    return td["F_normed"], td["n_feat"]


def _select_X_t(model_name: str, td: Dict) -> np.ndarray:
    w = _vol_window_for_model(model_name)
    if w is not None:
        return td["X_t_vols"][w]
    return td["X_t_return"]


# ---------------------------------------------------------------------------
# Core per-run function
# ---------------------------------------------------------------------------

def run_one_vol(
    ticker: str,
    model_name: str,
    h: int,
    N: int,
    seed: int,
    td: Dict,
    cfg: ExperimentConfig,
) -> Tuple[Dict, np.ndarray]:
    """Train (or load from cache) one model; save diagnostics; return metrics."""
    F_matrix, n_feat = _select_F_and_n(model_name, td)
    X_t_array        = _select_X_t(model_name, td)
    is_sf            = _is_statefree(model_name)
    N_XT             = td["N_XT"]
    prices_p         = td["prices_path"]
    out_dir          = RESULTS_DIR / ticker / model_name

    cfg_dict = td["configs"][(h, N)]
    N_actual = cfg_dict["N_actual"]

    train_loader, val_loader, test_loader = build_loaders(cfg_dict, F_matrix, X_t_array)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = _build_model(model_name, n_feat, N_XT, N_actual)
    cp    = _cache_path(ticker, model_name, h, N, seed)

    if is_cached(cp):
        model = load_cached_model(model, cp)
    else:
        state_dict, _ = train_one_run(
            model, train_loader, val_loader,
            N=N_actual, sigma=BEST_SIGMA, device=DEVICE,
            use_swa=True, swa_epochs=cfg.swa_epochs,
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

    w = _vol_window_for_model(model_name)
    print(f"  [VOL] {ticker} {model_name:22s} h={h:2d} N={N:2d} seed={seed} "
          f"nll_test={nll_test:.4f}  Δ={delta:+.4f}")

    A_t = _save_operator_ts_vol(
        model, F_matrix, cfg_dict["idx_test"],
        N_XT, N_actual, is_sf,
        out_dir, model_name, h, N, seed, prices_p,
    )

    # Snapshots: h=1, N=55, state_cond variants only
    if h == 1 and N == 55 and not is_sf:
        _save_snapshots_vol(A_t, out_dir, seed)

    row = {
        "ticker":        ticker,
        "model":         model_name,
        "vol_window":    w,
        "h":             h,
        "N":             N,
        "seed":          seed,
        "nll_train":     nll_train,
        "nll_val":       nll_val,
        "nll_test":      nll_test,
        "delta_marginal": delta,
    }
    return row, A_t


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_vol_experiments(cfg: ExperimentConfig) -> None:
    """Run all volatility-state conditioning experiments."""
    print("\n" + "=" * 70)
    print("VOLATILITY-STATE EXPERIMENT")
    print(f"  Tickers: {TICKERS}")
    print(f"  Horizons: {HORIZONS_VOL}  Bins: {BINS_VOL}  Models: {len(MODELS)}")
    print(f"  Seeds: {cfg.seeds}  WARMUP_MAX: {WARMUP_MAX}")
    print("=" * 70)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    total_deferred = 0

    for ticker in TICKERS:
        print(f"\n{'─'*60}\n  Ticker: {ticker}\n{'─'*60}")
        try:
            td = setup_ticker_vol(ticker, cfg)
        except Exception:
            print(f"  [VOL] FAILED setup for {ticker}")
            traceback.print_exc()
            continue

        for N in BINS_VOL:
            all_rows: List[Dict] = []
            a_t_cache: Dict[Tuple, np.ndarray] = {}

            for h in HORIZONS_VOL:      # h=1 FIRST — needed by CK post-pass
                for model_name in MODELS:
                    for seed in cfg.seeds:
                        try:
                            row, A_t = run_one_vol(ticker, model_name, h, N, seed, td, cfg)
                            all_rows.append(row)
                            a_t_cache[(model_name, h, seed)] = A_t
                        except Exception:
                            print(f"  [VOL] FAILED {ticker} {model_name} "
                                  f"h={h} N={N} seed={seed}")
                            traceback.print_exc()
                            w = _vol_window_for_model(model_name)
                            all_rows.append({
                                "ticker": ticker, "model": model_name,
                                "vol_window": w, "h": h, "N": N, "seed": seed,
                                "nll_train": None, "nll_val": None,
                                "nll_test": None, "delta_marginal": None,
                            })

            # Save NLL rows per model
            for model_name in MODELS:
                model_rows = [r for r in all_rows if r["model"] == model_name]
                if model_rows:
                    out_dir = RESULTS_DIR / ticker / model_name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    metrics_path = out_dir / "nll_metrics.csv"
                    new_df = pd.DataFrame(model_rows)
                    if metrics_path.exists():
                        existing = pd.read_csv(metrics_path)
                        new_df = pd.concat([existing, new_df], ignore_index=True)
                    new_df.to_csv(metrics_path, index=False)

            # CK post-pass per (model, N)
            for model_name in MODELS:
                d = _ck_post_pass_vol(ticker, model_name, N, td, a_t_cache, cfg)
                total_deferred += d

    write_summary_vol()
    print(f"\n[VOL] Finished. CK deferred: {total_deferred} cells.")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def write_summary_vol() -> None:
    """Write results_vol/summary_vol.md with 5 summary tables."""
    from scripts.results_writer import _has_bad_metrics

    summary_lines = ["# Volatility-State Experiment Summary\n"]

    # Table 1: NLL by model (mean across all tickers, h=1, N=55)
    rows_all: List[Dict] = []
    for ticker in TICKERS:
        for model_name in MODELS:
            p = RESULTS_DIR / ticker / model_name / "nll_metrics.csv"
            if p.exists():
                df = pd.read_csv(p)
                df["ticker"] = ticker
                rows_all.append(df)

    if rows_all:
        df_all = pd.concat(rows_all, ignore_index=True)

        summary_lines.append("## Table 1: Mean NLL (h=1, N=55, all tickers)\n")
        t1 = (
            df_all[df_all["h"].eq(1) & df_all["N"].eq(55)]
            .groupby("model")[["nll_test", "delta_marginal"]]
            .mean()
            .round(4)
            .reset_index()
        )
        summary_lines.append(t1.to_markdown(index=False) + "\n\n")

        summary_lines.append("## Table 2: Mean NLL by horizon (N=55, all tickers)\n")
        t2 = (
            df_all[df_all["N"].eq(55)]
            .groupby(["model", "h"])["nll_test"]
            .mean()
            .round(4)
            .unstack("h")
            .reset_index()
        )
        summary_lines.append(t2.to_markdown(index=False) + "\n\n")

        summary_lines.append("## Table 3: Mean NLL by N bins (h=1, all tickers)\n")
        t3 = (
            df_all[df_all["h"].eq(1)]
            .groupby(["model", "N"])["nll_test"]
            .mean()
            .round(4)
            .unstack("N")
            .reset_index()
        )
        summary_lines.append(t3.to_markdown(index=False) + "\n\n")

        summary_lines.append("## Table 4: Per-ticker NLL (h=1, N=55, state_cond_return vs state_cond_vol_w21)\n")
        mask = (
            df_all["h"].eq(1) & df_all["N"].eq(55)
            & df_all["model"].isin(["state_cond_return", "state_cond_vol_w21"])
        )
        t4 = (
            df_all[mask]
            .groupby(["ticker", "model"])["nll_test"]
            .mean()
            .round(4)
            .unstack("model")
            .reset_index()
        )
        summary_lines.append(t4.to_markdown(index=False) + "\n\n")

        summary_lines.append("## Table 5: FAILED cells\n")
        failed_tickers = []
        for ticker in TICKERS:
            for model_name in MODELS:
                failed_marker = RESULTS_DIR / ticker / model_name / "FAILED"
                if failed_marker.exists():
                    failed_tickers.append({"ticker": ticker, "model": model_name})
        if failed_tickers:
            summary_lines.append(pd.DataFrame(failed_tickers).to_markdown(index=False) + "\n\n")
        else:
            summary_lines.append("No FAILED markers found.\n\n")

    (RESULTS_DIR / "summary_vol.md").write_text("".join(summary_lines))
    print(f"  [VOL] Summary written to {RESULTS_DIR / 'summary_vol.md'}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = make_config()
    run_vol_experiments(cfg)
