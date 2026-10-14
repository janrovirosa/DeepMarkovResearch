#!/usr/bin/env python3
"""
Tasks 1-4: operator arrays, CK discrepancy, diagnostics, volatility correlations.
No retraining — forward passes only.
"""

import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.data import load_master_dataset, preprocess_features, compute_returns
from scripts.data import make_splits
from scripts.bins import compute_X_t
from scripts.models import StateConditionedNet, StateFreeNet
from scripts.train import build_A_t_neural, build_A_t_statefree
from scripts.eval import (
    build_ck_composed,
    compute_dobrushin, compute_row_heterogeneity, compute_row_entropy,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_DIR  = ROOT / "results/paper_upgrade/2026-03-07/cache"
OUT_DIR    = ROOT / "results/paper_upgrade/2026-04-07"
OUT_CACHE  = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CACHE.mkdir(parents=True, exist_ok=True)

DEVICE   = torch.device("cpu")
N_XT     = 55
HORIZONS = [2, 5, 10]
SEED     = 42
EPS      = 1e-10

# ---------------------------------------------------------------------------
# Load & preprocess data
# ---------------------------------------------------------------------------
print("=== Loading data ===")
prices, F_raw_orig, feature_cols_orig = load_master_dataset(ROOT / "dataset")

# Model was trained with Opinion INCLUDED (n_feat=194) — reload without dropping it
train_df = pd.read_csv(ROOT / "dataset/train_diagnostic.csv")
if "Opinion" in train_df.columns:
    train_df = train_df.drop(columns=["Opinion"])  # still drop Opinion but...

# Actually: model input dim = 249 = 55 + 194, so n_feat=194.
# load_master_dataset already gives 193 (drops Opinion). Let's verify:
# If model first layer is 249 wide, then n_feat=194 with Opinion.
# Re-load keeping Opinion in the features:
train_df_full = pd.read_csv(ROOT / "dataset/train_diagnostic.csv")
drop_cols = {"index", "Percent_change_backward", "Backward_Bin", "Price"}
feature_cols_194 = [c for c in train_df_full.columns if c not in drop_cols]
F_raw = train_df_full[feature_cols_194].values.astype(np.float32)
n_feat = F_raw.shape[1]
print(f"  n_feat={n_feat}, prices T={len(prices)}")
assert n_feat == 194, f"Expected 194 features, got {n_feat}"

# X_t_all: 1-day return bins, length = len(prices)-1
T_xt = len(prices) - 1  # = 2368
train_end_xt = int(0.70 * T_xt)  # ~1657
X_t_all, N_XT_actual, edges_xt = compute_X_t(prices, N_XT, train_end_xt)
assert N_XT_actual == N_XT, f"N_XT mismatch: {N_XT_actual}"
print(f"  T_xt={T_xt}, train_end_xt={train_end_xt}, N_XT={N_XT}")

# Feature normalization — use same train_idx as h=1 CK training split
T_ck_h1 = T_xt - 1  # 2367
idx_train_h1, _, _ = make_splits(T_ck_h1)
F_normed = preprocess_features(F_raw, idx_train_h1)
print(f"  F_normed shape: {F_normed.shape}, NaN: {np.isnan(F_normed).sum()}")


# ---------------------------------------------------------------------------
# TASK 1: Direct h-step operator arrays (forward passes)
# ---------------------------------------------------------------------------
print("\n=== TASK 1: Direct h-step operators ===")

def load_ck_model(model_type, h, n_feat, n_xt, device):
    """Load CK model weights from cache."""
    fname = CACHE_DIR / f"ck_{model_type}_h{h}_seed{SEED}.pt"
    assert fname.exists(), f"Missing weights: {fname}"
    if model_type == "state_cond":
        model = StateConditionedNet(
            n_feat=n_feat, n_xt_states=n_xt, n_output=n_xt,
            hidden_dims=(64, 128, 256, 128, 64), dropout=0.2,
        )
    else:
        model = StateFreeNet(
            n_feat=n_feat, n_output=n_xt,
            hidden_dims=(64, 128, 256, 128, 64), dropout=0.2,
        )
    sd = torch.load(fname, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    return model.to(device)


for h in HORIZONS:
    T_ck = T_xt - h   # number of valid CK time steps
    time_idx = np.arange(T_ck)

    for mtype in ["state_cond", "state_free"]:
        out_path = OUT_CACHE / f"A_t_direct_{mtype}_h{h}.npy"
        if out_path.exists():
            print(f"  [SKIP] {out_path.name} already exists")
            continue

        print(f"  Building A_t_direct_{mtype}_h{h}  (T={T_ck}) ...")
        model = load_ck_model(mtype, h, n_feat, N_XT, DEVICE)

        if mtype == "state_cond":
            A = build_A_t_neural(model, F_normed, time_idx, N_XT, N_XT, DEVICE)
        else:
            A = build_A_t_statefree(model, F_normed, time_idx, N_XT, N_XT, DEVICE)

        np.save(out_path, A)
        row_sums_ok = np.allclose(A.sum(2), 1.0, atol=1e-4)
        print(f"    saved {out_path.name}  shape={A.shape}  "
              f"range=[{A.min():.4f},{A.max():.4f}]  "
              f"row_sums_ok={row_sums_ok}")

print("  Task 1 done.")


# ---------------------------------------------------------------------------
# TASK 2: Per-timestep CK discrepancy time series
# ---------------------------------------------------------------------------
print("\n=== TASK 2: CK discrepancy time series ===")

# Load h=1 operators (already cached in 2026-03-07)
A1_sc = np.load(CACHE_DIR / "A_t_ck_state_cond_h1.npy").astype(np.float64)  # (T1, 55, 55)
A1_sf = np.load(CACHE_DIR / "A_t_ck_state_free_h1.npy").astype(np.float64)
T1 = A1_sc.shape[0]
print(f"  A1 state_cond shape: {A1_sc.shape}")
print(f"  A1 state_free  shape: {A1_sf.shape}")

rows = []
for h in HORIZONS:
    T_ck = T_xt - h
    # Composed: A_composed[t] = A1[t] @ A1[t+1] @ ... @ A1[t+h-1]
    # T_composed = T1 - h + 1 = T_xt - 1 - h + 1 = T_xt - h = T_ck  ✓
    T_composed = T1 - h + 1
    assert T_composed == T_ck, f"T_composed={T_composed} != T_ck={T_ck}"

    # Load direct arrays from Task 1
    A_direct_sc = np.load(OUT_CACHE / f"A_t_direct_state_cond_h{h}.npy").astype(np.float64)
    A_direct_sf = np.load(OUT_CACHE / f"A_t_direct_state_free_h{h}.npy").astype(np.float64)

    for mtype, A1, A_direct in [
        ("state_cond", A1_sc, A_direct_sc),
        ("state_free", A1_sf, A_direct_sf),
    ]:
        print(f"  Computing CK discrepancy: {mtype} h={h}  T={T_ck} ...")

        # Compose h matrices
        A_comp = np.zeros((T_composed, N_XT, N_XT), dtype=np.float64)
        for pos in range(T_composed):
            mat = A1[pos].copy()
            for k in range(1, h):
                mat = mat @ A1[pos + k]
            A_comp[pos] = mat

        # Per-timestep KL and TV  (mean over rows)
        P = np.clip(A_direct, EPS, None)   # direct
        Q = np.clip(A_comp,   EPS, None)   # composed

        # KL(P || Q) = sum_j P*log(P/Q),  mean over rows i
        kl_t = (P * np.log(P / Q)).sum(axis=2).mean(axis=1)   # (T_ck,)
        tv_t = (0.5 * np.abs(A_direct - A_comp)).sum(axis=2).mean(axis=1)  # (T_ck,)

        print(f"    KL mean={kl_t.mean():.4f} max={kl_t.max():.4f} | "
              f"TV mean={tv_t.mean():.4f} max={tv_t.max():.4f}")

        for t_pos in range(T_ck):
            rows.append({
                "t":     t_pos,
                "model": mtype,
                "h":     h,
                "KL_t":  float(kl_t[t_pos]),
                "TV_t":  float(tv_t[t_pos]),
            })

ck_df = pd.DataFrame(rows, columns=["t", "model", "h", "KL_t", "TV_t"])
ck_path = OUT_DIR / "ck_timeseries.csv"
ck_df.to_csv(ck_path, index=False)
print(f"  Saved {ck_path}  shape={ck_df.shape}")


# ---------------------------------------------------------------------------
# TASK 3: Per-timestep operator diagnostics (state_cond h=1)
# ---------------------------------------------------------------------------
print("\n=== TASK 3: Operator diagnostics (state_cond h=1) ===")

A1 = A1_sc  # already loaded, float64, shape (T1, 55, 55)
T1 = A1.shape[0]

print(f"  Computing Dobrushin for {T1} time steps ...")
dobrushin_t = compute_dobrushin(A1)  # (T1,)

print(f"  Computing row heterogeneity ...")
row_hetero_t = compute_row_heterogeneity(A1)  # (T1,)

print(f"  Computing row entropy ...")
row_entropy_t = compute_row_entropy(A1)  # (T1,)

diag_df = pd.DataFrame({
    "t":           np.arange(T1),
    "dobrushin":   dobrushin_t,
    "row_hetero":  row_hetero_t,
    "row_entropy": row_entropy_t,
})
diag_path = OUT_DIR / "operator_diagnostics_timeseries.csv"
diag_df.to_csv(diag_path, index=False)
print(f"  Saved {diag_path}  shape={diag_df.shape}")
print(f"  dobrushin:  mean={dobrushin_t.mean():.4f}  range=[{dobrushin_t.min():.4f},{dobrushin_t.max():.4f}]")
print(f"  row_hetero: mean={row_hetero_t.mean():.4f}  range=[{row_hetero_t.min():.4f},{row_hetero_t.max():.4f}]")
print(f"  row_entropy:mean={row_entropy_t.mean():.4f}  range=[{row_entropy_t.min():.4f},{row_entropy_t.max():.4f}]")


# ---------------------------------------------------------------------------
# TASK 4: Volatility proxy and correlation analysis
# ---------------------------------------------------------------------------
print("\n=== TASK 4: Volatility proxy and correlations ===")

# 1-day log returns (using the same prices)
r_1day = compute_returns(prices, 1)   # length T_xt = 2368

# Rolling 21-day realized variance (std^2) at each timestep t
# RV_t = var(r_{t-20}, ..., r_t)  (21-point window)
WINDOW = 21
T_rv = len(r_1day)  # 2368
RV = np.full(T_rv, np.nan)
for t in range(WINDOW - 1, T_rv):
    RV[t] = float(np.std(r_1day[t - WINDOW + 1 : t + 1], ddof=1) ** 2)

# Diagnostics series covers t=0..T1-1 (T1=2367 = T_xt-1 = 2367)
# r_1day[t] is available for t=0..2367, so RV[t] for t=20..2367 is valid
# t in diag_df: 0..2366 (T1=2367 steps, indices 0..2366)
# Align: at diag index t, the raw return index is also t (same chronological t)
diag_t = diag_df["t"].values  # 0..2366

RV_aligned = RV[diag_t]  # same length as diag_df

# Test period for h=1 CK: same split as used for operator array
_, _, idx_test_h1 = make_splits(T_ck_h1)  # T_ck_h1 = 2367
# idx_test_h1 covers some range of t within 0..2366

# Identify valid (non-NaN) samples
valid_all  = np.isfinite(RV_aligned) & np.isfinite(dobrushin_t)
valid_test = valid_all & np.isin(diag_t, idx_test_h1)

print(f"  Valid full series: {valid_all.sum()} / {len(diag_t)}")
print(f"  Valid test period: {valid_test.sum()} / {len(idx_test_h1)}")

corr_rows = []
for metric_name, metric_vals in [
    ("dobrushin",   dobrushin_t),
    ("row_hetero",  row_hetero_t),
    ("row_entropy", row_entropy_t),
]:
    valid_a = np.isfinite(RV_aligned) & np.isfinite(metric_vals)
    valid_b = valid_a & np.isin(diag_t, idx_test_h1)

    # Full series
    r_full, p_full = pearsonr(metric_vals[valid_a], RV_aligned[valid_a])
    corr_rows.append({"metric": metric_name, "period": "full",
                      "pearson_r": float(r_full), "p_value": float(p_full)})

    # Test period
    r_test, p_test = pearsonr(metric_vals[valid_b], RV_aligned[valid_b])
    corr_rows.append({"metric": metric_name, "period": "test",
                      "pearson_r": float(r_test), "p_value": float(p_test)})

    print(f"  {metric_name:12s}  full: r={r_full:+.4f} p={p_full:.4e}  "
          f"test: r={r_test:+.4f} p={p_test:.4e}")

# VIX: try to load — dates are masked so skip alignment
vix_path = ROOT / "raw_data/PriceHistoryVIX.csv"
VIX_t = np.full(len(diag_t), np.nan)
vix_note = "Dates masked in VIX file — alignment skipped"
print(f"  VIX: {vix_note}")

# Save correlation CSV
corr_df = pd.DataFrame(corr_rows, columns=["metric", "period", "pearson_r", "p_value"])
corr_path = OUT_DIR / "volatility_correlation.csv"
corr_df.to_csv(corr_path, index=False)
print(f"  Saved {corr_path}")
print(corr_df.to_string(index=False))

# Save aligned timeseries CSV
vol_ts_df = pd.DataFrame({
    "t":          diag_t,
    "dobrushin":  dobrushin_t,
    "row_hetero": row_hetero_t,
    "RV_t":       RV_aligned,
    "VIX_t":      VIX_t,
})
vol_ts_path = OUT_DIR / "volatility_proxy_timeseries.csv"
vol_ts_df.to_csv(vol_ts_path, index=False)
print(f"  Saved {vol_ts_path}  shape={vol_ts_df.shape}")
print(f"  RV_t: mean={np.nanmean(RV_aligned):.6f}  range=[{np.nanmin(RV_aligned):.6f},{np.nanmax(RV_aligned):.6f}]")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print("\n=== ALL TASKS COMPLETE ===")
print(f"Output dir: {OUT_DIR}")
print(f"\nTask 1 — Direct operator arrays:")
for h in HORIZONS:
    for mtype in ["state_cond", "state_free"]:
        p = OUT_CACHE / f"A_t_direct_{mtype}_h{h}.npy"
        A = np.load(p)
        print(f"  {p.name}  shape={A.shape}  "
              f"min={A.min():.4f}  max={A.max():.4f}  "
              f"NaN={np.isnan(A).sum()}  "
              f"row_sums_ok={np.allclose(A.sum(2),1,atol=1e-4)}")

print(f"\nTask 2 — CK timeseries:  {ck_path.name}  rows={len(ck_df)}")
for h in HORIZONS:
    for mtype in ["state_cond", "state_free"]:
        sub = ck_df[(ck_df["h"]==h) & (ck_df["model"]==mtype)]
        print(f"  h={h} {mtype:12s}  KL mean={sub['KL_t'].mean():.4f}  TV mean={sub['TV_t'].mean():.4f}")

print(f"\nTask 3 — Diagnostics:  {diag_path.name}  rows={len(diag_df)}  NaN={diag_df.isnull().sum().sum()}")
print(f"\nTask 4 — Correlations saved.  VIX note: {vix_note}")
