#!/usr/bin/env python3
"""
Follow-up analysis Tasks 1–4.
No retraining. All inputs from results/paper_upgrade/2026-04-07/.
"""

import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, ttest_ind

ROOT    = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OUT_DIR   = ROOT / "results/paper_upgrade/2026-04-07"
CACHE_DIR = OUT_DIR / "cache"

from scripts.eval import (
    compute_dobrushin, compute_row_heterogeneity, compute_row_entropy,
)

EPS = 1e-10

# ============================================================
# HELPERS
# ============================================================

def parse_euro_float(s):
    """Parse European-decimal float: '17,44' -> 17.44"""
    if pd.isna(s):
        return np.nan
    s = str(s).replace('"', '').strip()
    s = s.replace(',', '.')
    try:
        return float(s)
    except ValueError:
        return np.nan


def pearson_safe(x, y):
    """Pearson r and p-value, masked to finite pairs."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan
    r, p = pearsonr(x[mask], y[mask])
    return float(r), float(p)


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ============================================================
# TASK 1: VIX alignment
# ============================================================
print_section("TASK 1: VIX alignment and correlations")

# ---- Load VIX and parse prices ----
vix_raw = pd.read_csv(ROOT / "raw_data/PriceHistoryVIX.csv")
vix_prices_rev = np.array([parse_euro_float(v) for v in vix_raw["Price"].values])
# File is newest-first → reverse to get oldest-first chronological order
vix_prices_chron = vix_prices_rev[::-1]

# ---- Load JPM Diagnostic prices for alignment ----
jpm_diag = pd.read_csv(ROOT / "raw_data/Diagnostic/PriceHistoryJPM.csv")
jpm_p_rev = jpm_diag["Price"].values  # newest-first
jpm_p_chron = jpm_p_rev[::-1]         # oldest-first

# ---- Load training prices ----
train_df = pd.read_csv(ROOT / "dataset/train_diagnostic.csv")
train_prices = train_df["Price"].values

# ---- Find alignment offset: where train_prices[0] appears in jpm_p_chron ----
# Verified offline: START = 4032 (training starts at JPM chron row 4032)
START = 4032
assert np.isclose(jpm_p_chron[START], train_prices[0], rtol=1e-3), \
    f"Price mismatch at START: {jpm_p_chron[START]} vs {train_prices[0]}"

N_TRAIN = len(train_prices)   # 2369
# VIX chron has same length as JPM chron (both 6501 rows)
assert len(vix_prices_chron) == len(jpm_p_chron), \
    f"VIX/JPM length mismatch: {len(vix_prices_chron)} vs {len(jpm_p_chron)}"

# Extract VIX window aligned to training period
vix_train_window = vix_prices_chron[START : START + N_TRAIN].copy()  # (2369,)
print(f"  VIX window shape: {vix_train_window.shape}")
print(f"  VIX range: [{np.nanmin(vix_train_window):.2f}, {np.nanmax(vix_train_window):.2f}]")
print(f"  VIX NaN: {np.isnan(vix_train_window).sum()}")
print(f"  Sanity — VIX first 3: {vix_train_window[:3]}")

# ---- Load existing diagnostics and volatility proxy ----
diag_df    = pd.read_csv(OUT_DIR / "operator_diagnostics_timeseries.csv")
vol_ts_df  = pd.read_csv(OUT_DIR / "volatility_proxy_timeseries.csv")
corr_df    = pd.read_csv(OUT_DIR / "volatility_correlation.csv")

T1 = len(diag_df)       # 2367 (t = 0..2366)
t_idx = diag_df["t"].values   # 0..2366

# VIX at t for the diagnostics window: vix_train_window[0..2366]
vix_aligned = vix_train_window[t_idx]   # shape (2367,)

# ---- Update volatility_proxy_timeseries.csv with VIX_t ----
vol_ts_df["VIX_t"] = vix_aligned
vol_ts_df.to_csv(OUT_DIR / "volatility_proxy_timeseries.csv", index=False)
print(f"\n  Updated volatility_proxy_timeseries.csv with VIX_t column")
print(f"  VIX_t: mean={np.nanmean(vix_aligned):.2f}  "
      f"range=[{np.nanmin(vix_aligned):.2f},{np.nanmax(vix_aligned):.2f}]")

# ---- Compute correlations: VIX vs diagnostics (full + test period) ----
from scripts.data import make_splits
T_ck_h1 = T1        # 2367
_, _, idx_test_h1 = make_splits(T_ck_h1)
test_mask = np.isin(t_idx, idx_test_h1)

dobrushin_t  = diag_df["dobrushin"].values
row_hetero_t = diag_df["row_hetero"].values
row_entropy_t= diag_df["row_entropy"].values

new_rows = []
for metric_name, metric_vals in [
    ("dobrushin",   dobrushin_t),
    ("row_hetero",  row_hetero_t),
    ("row_entropy", row_entropy_t),
]:
    valid_all  = np.isfinite(vix_aligned) & np.isfinite(metric_vals)
    valid_test = valid_all & test_mask

    r_full, p_full = pearson_safe(metric_vals, vix_aligned)
    r_test, p_test = pearson_safe(metric_vals[test_mask], vix_aligned[test_mask])

    new_rows.append({"metric": f"{metric_name}_vs_VIX", "period": "full",
                     "pearson_r": r_full, "p_value": p_full})
    new_rows.append({"metric": f"{metric_name}_vs_VIX", "period": "test",
                     "pearson_r": r_test, "p_value": p_test})

    print(f"  {metric_name:12s}  full: r={r_full:+.4f} p={p_full:.3e}  "
          f"test: r={r_test:+.4f} p={p_test:.3e}")

# Append to existing correlation CSV
new_corr_df = pd.concat([corr_df, pd.DataFrame(new_rows)], ignore_index=True)
new_corr_df.to_csv(OUT_DIR / "volatility_correlation.csv", index=False)
print(f"\n  Appended {len(new_rows)} rows to volatility_correlation.csv  "
      f"(total now {len(new_corr_df)} rows)")


# ============================================================
# TASK 2: CK h=5 focus — correlations with KL_t
# ============================================================
print_section("TASK 2: CK h=5 state_cond correlations")

ck_df = pd.read_csv(OUT_DIR / "ck_timeseries.csv")
rv_t  = vol_ts_df["RV_t"].values       # (2367,) aligned to t=0..2366

# Extract h=5, state_cond
ck5_sc = ck_df[(ck_df["h"] == 5) & (ck_df["model"] == "state_cond")].copy()
ck5_sc = ck5_sc.sort_values("t").reset_index(drop=True)

T5 = len(ck5_sc)   # T_ck for h=5 = 2363
print(f"  h=5 state_cond subset: {T5} rows, "
      f"KL range=[{ck5_sc['KL_t'].min():.4f},{ck5_sc['KL_t'].max():.4f}]")

# The h=5 operator covers t=0..2362 (T_ck = T_xt - 5 = 2368 - 5 = 2363)
# operator_diagnostics uses t=0..2366; we need matching t indices
t5 = ck5_sc["t"].values   # 0..2362
KL5 = ck5_sc["KL_t"].values

# Align diagnostic metrics to the same t indices
diag_t_set = dict(zip(t_idx, range(len(t_idx))))  # t -> position in diag_df
valid_diag = np.array([diag_t_set.get(ti, -1) for ti in t5])
has_diag   = valid_diag >= 0

row_entropy_h5 = row_entropy_t[valid_diag[has_diag]]
dobrushin_h5   = dobrushin_t[valid_diag[has_diag]]
rv_h5          = rv_t[valid_diag[has_diag]]
KL5_valid      = KL5[has_diag]

print(f"  Matched t indices: {has_diag.sum()} / {T5}")

ck_vol_rows = []
for pred_name, pred_vals in [
    ("row_entropy_t", row_entropy_h5),
    ("dobrushin_t",   dobrushin_h5),
    ("RV_t",          rv_h5),
]:
    r, p = pearson_safe(KL5_valid, pred_vals)
    ck_vol_rows.append({"predictor": pred_name, "pearson_r": r, "p_value": p})
    print(f"  KL_t(h=5,sc) vs {pred_name:16s}:  r={r:+.4f}  p={p:.3e}")

ck_vol_df = pd.DataFrame(ck_vol_rows, columns=["predictor", "pearson_r", "p_value"])
ck_vol_path = OUT_DIR / "ck_volatility_correlation.csv"
ck_vol_df.to_csv(ck_vol_path, index=False)
print(f"\n  Saved {ck_vol_path}")
print(f"  Shape={ck_vol_df.shape}  NaN={ck_vol_df.isnull().sum().sum()}")


# ============================================================
# TASK 3: Regime stratification table
# ============================================================
print_section("TASK 3: Regime stratification (RV_t top/bottom 20%)")

rv_all = vol_ts_df["RV_t"].values   # (2367,)
finite_mask = np.isfinite(rv_all)

rv_finite = rv_all[finite_mask]
p20 = np.percentile(rv_finite, 20)
p80 = np.percentile(rv_finite, 80)
print(f"  RV_t 20th pctile: {p20:.6f}   80th pctile: {p80:.6f}")

high_vol_mask  = finite_mask & (rv_all >= p80)   # top 20%
tranquil_mask  = finite_mask & (rv_all <= p20)   # bottom 20%
print(f"  High-vol timesteps: {high_vol_mask.sum()}   "
      f"Tranquil timesteps: {tranquil_mask.sum()}")

# KL_t for h=5, state_cond — need to map to the full t=0..2366 index
KL5_full = np.full(T1, np.nan)
for pos, ti in enumerate(t5):
    if ti < T1:
        KL5_full[ti] = KL5[pos]

metrics_dict = {
    "dobrushin":   dobrushin_t,
    "row_hetero":  row_hetero_t,
    "row_entropy": row_entropy_t,
    "KL_t_h5_sc":  KL5_full,
}

regime_rows = []
for metric_name, vals in metrics_dict.items():
    hv = vals[high_vol_mask & np.isfinite(vals)]
    tr = vals[tranquil_mask & np.isfinite(vals)]

    t_stat, t_pval = ttest_ind(hv, tr, equal_var=False)

    regime_rows.append({
        "metric":        metric_name,
        "high_vol_mean": float(np.mean(hv)),
        "high_vol_std":  float(np.std(hv, ddof=1)),
        "tranquil_mean": float(np.mean(tr)),
        "tranquil_std":  float(np.std(tr, ddof=1)),
        "t_stat":        float(t_stat),
        "p_value":       float(t_pval),
    })
    print(f"  {metric_name:14s}  HV: {np.mean(hv):.4f}±{np.std(hv,ddof=1):.4f}  "
          f"TR: {np.mean(tr):.4f}±{np.std(tr,ddof=1):.4f}  "
          f"t={t_stat:+.3f}  p={t_pval:.3e}")

regime_df   = pd.DataFrame(regime_rows)
regime_path = OUT_DIR / "regime_stratification_table.csv"
regime_df.to_csv(regime_path, index=False)
print(f"\n  Saved {regime_path}")
print(f"  Shape={regime_df.shape}  NaN={regime_df.isnull().sum().sum()}")


# ============================================================
# TASK 4: h=5 state_cond operator diagnostics vs h=1
# ============================================================
print_section("TASK 4: Operator diagnostics comparison (h=5 vs h=1)")

A_sc_h5 = np.load(CACHE_DIR / "A_t_direct_state_cond_h5.npy").astype(np.float64)
A_sf_h5 = np.load(CACHE_DIR / "A_t_direct_state_free_h5.npy").astype(np.float64)

print(f"  A_t_direct_state_cond_h5 shape: {A_sc_h5.shape}")
print(f"  A_t_direct_state_free_h5 shape: {A_sf_h5.shape}")

# Diagnostics for h=5 state_cond
print("  Computing diagnostics for h=5 state_cond ...")
dob_sc_h5   = compute_dobrushin(A_sc_h5)
het_sc_h5   = compute_row_heterogeneity(A_sc_h5)
ent_sc_h5   = compute_row_entropy(A_sc_h5)

# Row heterogeneity for h=5 state_free (sanity check — should be near 0)
print("  Computing row_hetero for h=5 state_free (sanity) ...")
het_sf_h5   = compute_row_heterogeneity(A_sf_h5)

# h=1 stats from existing CSV
dob_h1  = diag_df["dobrushin"].values
het_h1  = diag_df["row_hetero"].values
ent_h1  = diag_df["row_entropy"].values

def fmt(arr):
    return (f"mean={np.nanmean(arr):.4f}  "
            f"std={np.nanstd(arr,ddof=1):.4f}  "
            f"range=[{np.nanmin(arr):.4f},{np.nanmax(arr):.4f}]")

print()
print("  ┌─────────────────┬────────────────────────────────────────────┬────────────────────────────────────────────┐")
print("  │ metric          │ h=1  state_cond                            │ h=5  state_cond                            │")
print("  ├─────────────────┼────────────────────────────────────────────┼────────────────────────────────────────────┤")
for name, h1_arr, h5_arr in [
    ("dobrushin",   dob_h1, dob_sc_h5),
    ("row_hetero",  het_h1, het_sc_h5),
    ("row_entropy", ent_h1, ent_sc_h5),
]:
    print(f"  │ {name:15s} │ {fmt(h1_arr):42s} │ {fmt(h5_arr):42s} │")
print("  └─────────────────┴────────────────────────────────────────────┴────────────────────────────────────────────┘")

print()
print(f"  SANITY h=5 state_free row_hetero: {fmt(het_sf_h5)}")
max_theory = 0.5   # TV distance is in [0, 1], row_hetero ≤ 0.5
print(f"  (Expected: near 0 since state-free model ignores input state)")
print(f"  State-free h=5 max row_hetero: {np.nanmax(het_sf_h5):.6f}  — sanity {'PASS ✓' if np.nanmax(het_sf_h5) < 0.01 else 'UNEXPECTED'}")

# ── Summary check for h=5 genuine structure ──
print()
print("  h=5 state_cond genuine state-dependence check:")
print(f"    row_hetero h=5:  mean={np.nanmean(het_sc_h5):.4f}  (h=1 mean={np.nanmean(het_h1):.4f})")
print(f"    max A[t,i,j] h=5: {A_sc_h5.max():.4f}  (uniform=1/55≈0.018)")
print(f"    max A[t,i,j] h=1: {np.load(ROOT/'results/paper_upgrade/2026-03-07/cache/A_t_ck_state_cond_h1.npy').max():.4f}")


# ============================================================
# FINAL VERIFICATION SUMMARY
# ============================================================
print_section("VERIFICATION SUMMARY")

def check(path, df=None):
    p = Path(path)
    if df is None:
        df = pd.read_csv(p)
    nans = df.isnull().sum().sum()
    print(f"  {p.name:45s}  shape={str(df.shape):12s}  NaN={nans}")
    return df

print()
check(OUT_DIR / "volatility_proxy_timeseries.csv")
check(OUT_DIR / "volatility_correlation.csv")
check(OUT_DIR / "ck_volatility_correlation.csv")
check(OUT_DIR / "regime_stratification_table.csv")

print()
print("  VIX stats in proxy timeseries:")
vt = pd.read_csv(OUT_DIR / "volatility_proxy_timeseries.csv")
print(f"    VIX_t: mean={vt['VIX_t'].mean():.2f}  "
      f"std={vt['VIX_t'].std():.2f}  "
      f"range=[{vt['VIX_t'].min():.2f},{vt['VIX_t'].max():.2f}]  "
      f"NaN={vt['VIX_t'].isna().sum()}")
print(f"    RV_t:  mean={vt['RV_t'].mean():.6f}  NaN={vt['RV_t'].isna().sum()}")

print()
print("  Full correlation table:")
corr_final = pd.read_csv(OUT_DIR / "volatility_correlation.csv")
print(corr_final.to_string(index=False))

print()
print("  Regime stratification:")
print(pd.read_csv(OUT_DIR / "regime_stratification_table.csv").to_string(index=False))

print()
print("  CK volatility correlations (h=5):")
print(pd.read_csv(OUT_DIR / "ck_volatility_correlation.csv").to_string(index=False))

print()
print("ALL TASKS COMPLETE. Outputs in:", OUT_DIR)
