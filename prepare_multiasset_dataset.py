#!/usr/bin/env python3
"""
prepare_multiasset_dataset.py
=============================================================
Multi-asset dataset preparation pipeline for the DeepMarkov
paper extension.

Run with:
  /opt/miniconda3/envs/stat453/bin/python3 prepare_multiasset_dataset.py

Outputs to: dataset_multiasset/
"""

import json
import os
import random
import warnings

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "Extended Dataset", "deep_markov_expanded_universe.csv")
MS_PATH    = os.path.join(BASE_DIR, "Extended Dataset", "MS_updated.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "dataset_multiasset")

TARGET_STOCKS    = ["JPM", "C", "WFC", "GS", "MS", "PNC", "USB", "FITB", "MTB", "BAC"]
BENCHMARK_STOCKS = ["SPY", "XLF"]
ALL_TICKERS      = TARGET_STOCKS + BENCHMARK_STOCKS   # 12 tickers; order defines lag column order
DROP_TICKERS     = ["TFC", "KBE"]

DROP_COLS = [
    "earnings_surprise", "surprise_pct", "numest", "stdev",
    "inst_ownership_pct", "total_assets", "total_liabilities",
    "net_income", "common_equity", "eps", "revenue",
    "leverage_ratio", "roe", "cusip", "actual_eps",
]

MACRO_COLS  = ["DFF", "T10Y2Y", "BAA10Y", "VIXCLS", "DCOILWTICO", "USREC"]
LAG_DEPTHS  = [1, 2, 5, 21]
DATE_START  = "1999-05-05"
DATE_END    = "2023-12-29"
TRAIN_FRAC  = 0.70
VAL_FRAC    = 0.15


def sep(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def make_splits(n: int):
    """Chronological 70/15/15 — same logic as scripts/data.py:make_splits."""
    train_end = int(TRAIN_FRAC * n)
    val_end   = int((TRAIN_FRAC + VAL_FRAC) * n)
    return (
        list(range(0, train_end)),
        list(range(train_end, val_end)),
        list(range(val_end, n)),
    )


# ──────────────────────────────────────────────────────────
# STEP 0  Load raw CSV
# ──────────────────────────────────────────────────────────
sep("STEP 0 — Load raw data")

df = pd.read_csv(DATA_PATH)

# Parse dates; coerce all other columns except ticker to numeric
df["date"] = pd.to_datetime(df["date"])
for col in df.columns:
    if col not in ("date", "ticker"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")


# ──────────────────────────────────────────────────────────
# STEP 1  Inspect
# ──────────────────────────────────────────────────────────
sep("STEP 1 — Inspect")
print(f"Shape      : {df.shape}")
print(f"Date range : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"\nTicker counts:\n{df['ticker'].value_counts().to_string()}")


# ──────────────────────────────────────────────────────────
# STEP 1b  Replace contaminated MS rows with CRSP PERMNO 69032
# ──────────────────────────────────────────────────────────
# The ticker MS in CRSP was held by Milestone Scientific (PERMNO 82621)
# until Dec 2005; Morgan Stanley traded as MWD until Jan 2006 (PERMNO 69032).
# All MS rows in the original source before 2006-01-17 are wrong.
# We replace the entire MS series with the correct PERMNO 69032 data.
sep("STEP 1b — Replace MS series with CRSP PERMNO 69032 (Morgan Stanley)")

ms_crsp_raw = pd.read_csv(MS_PATH, low_memory=False)
ms_crsp_raw["date"] = pd.to_datetime(ms_crsp_raw["date"])

ms_correct = ms_crsp_raw[ms_crsp_raw["PERMNO"] == 69032].copy()
ms_correct["RET"]    = pd.to_numeric(ms_correct["RET"],    errors="coerce")
ms_correct["PRC"]    = pd.to_numeric(ms_correct["PRC"],    errors="coerce")
ms_correct["VOL"]    = pd.to_numeric(ms_correct["VOL"],    errors="coerce")
ms_correct["SHROUT"] = pd.to_numeric(ms_correct["SHROUT"], errors="coerce")

# Drop rows with missing price/return (e.g. 2012-10-29 Hurricane Sandy closure)
# and deduplicate exact duplicate rows present in the CRSP source file
n_before = len(ms_correct)
ms_correct = ms_correct.dropna(subset=["PRC", "RET"])
ms_correct = ms_correct.drop_duplicates(subset=["date"])
n_removed = n_before - len(ms_correct)
if n_removed:
    print(f"  Dropped {n_removed} NaN/duplicate rows from CRSP MS series")
if len(ms_correct) < n_before:
    print(f"  Dropped {n_before - len(ms_correct)} NaN rows from CRSP MS series "
          f"(e.g. exchange closure days)")

# Build replacement frame matching original df column structure
# Macro columns are date-level — join from JPM rows in the original df
macro_by_date = (
    df[df["ticker"] == "JPM"][["date"] + MACRO_COLS]
    .drop_duplicates("date")
    .set_index("date")
)

ms_new = pd.DataFrame({
    "permno"      : 69032,
    "ticker"      : "MS",
    "date"        : ms_correct["date"].values,
    "price"       : ms_correct["PRC"].abs().values,
    "daily_return": ms_correct["RET"].values,
    "volume"      : ms_correct["VOL"].values,
    "shares_out"  : ms_correct["SHROUT"].values,
    "market_cap"  : (ms_correct["PRC"].abs() * ms_correct["SHROUT"]).values,
})
ms_new = ms_new.merge(macro_by_date.reset_index(), on="date", how="left")

# Carry any remaining columns present in df but not yet in ms_new as NaN
# (they will be dropped in Step 3 anyway, but keeps DataFrame shapes aligned)
for col in df.columns:
    if col not in ms_new.columns:
        ms_new[col] = np.nan

ms_new = ms_new[df.columns]   # ensure identical column order

# ── Before/after comparison on the flagged dates
BAD_DATES   = ["2002-04-11", "2003-09-08", "2003-09-12", "2003-10-07"]
LEGIT_DATES = ["2008-10-13"]
check_dates = BAD_DATES + LEGIT_DATES

orig_ms = df[df["ticker"] == "MS"].set_index("date")["daily_return"]
new_ms  = ms_new.set_index("date")["daily_return"]

print(f"\n  {'Date':12s}  {'OLD return':>12s}  {'NEW return':>12s}  {'Note':}")
for d in check_dates:
    dt  = pd.Timestamp(d)
    old = orig_ms.get(dt, float("nan"))
    new = new_ms.get(dt, float("nan"))
    note = "GENUINE (GFC)" if d in LEGIT_DATES else ("FIXED" if abs(old) > 0.4 and abs(new) < 0.4 else "check")
    print(f"  {d:12s}  {old:>12.6f}  {new:>12.6f}  {note}")

# ── Swap MS rows in df
df = pd.concat([df[df["ticker"] != "MS"], ms_new], ignore_index=True)
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
print(f"\n  MS rows replaced. New MS row count: {(df['ticker'] == 'MS').sum():,}")


# ──────────────────────────────────────────────────────────
# STEP 2  Drop TFC and KBE
# ──────────────────────────────────────────────────────────
sep("STEP 2 — Drop TFC and KBE")
before = len(df)
df = df[~df["ticker"].isin(DROP_TICKERS)].copy()
print(f"Dropped {before - len(df):,} rows. Remaining tickers: {sorted(df['ticker'].unique())}")


# ──────────────────────────────────────────────────────────
# STEP 3  Drop unwanted columns
# ──────────────────────────────────────────────────────────
sep("STEP 3 — Drop unwanted columns")
present   = [c for c in DROP_COLS if c in df.columns]
absent    = [c for c in DROP_COLS if c not in df.columns]
df        = df.drop(columns=present)
print(f"Dropped {len(present)} columns : {present}")
if absent:
    print(f"Not found (already absent)  : {absent}")
print(f"Remaining columns: {list(df.columns)}")


# ──────────────────────────────────────────────────────────
# STEP 4  Filter date range
# ──────────────────────────────────────────────────────────
sep("STEP 4 — Filter date range")
before = len(df)
mask   = (df["date"] >= DATE_START) & (df["date"] <= DATE_END)
df     = df[mask].copy()
print(f"Kept {len(df):,} rows (dropped {before - len(df):,}). "
      f"Range: {df['date'].min().date()} → {df['date'].max().date()}")


# ──────────────────────────────────────────────────────────
# STEP 5  Verify and fix extreme returns (all tickers)
# ──────────────────────────────────────────────────────────
sep("STEP 5 — Verify and fix extreme returns (|return| > 0.4)")
df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

all_flagged = df[df["daily_return"].abs() > 0.4]
print(f"Flagged {len(all_flagged)} rows with |daily_return| > 0.4 across all tickers:\n")
if len(all_flagged):
    print(all_flagged[["ticker", "date", "daily_return", "price"]].to_string(index=False))

changes = []
for ticker, group in df.groupby("ticker"):
    grp  = group.sort_values("date").reset_index()   # 'index' col = original df index
    prices_series = grp.set_index("date")["price"]

    flagged_rows = grp[grp["daily_return"].abs() > 0.4]
    for _, row in flagged_rows.iterrows():
        orig_idx = row["index"]
        t        = row["date"]
        prior    = prices_series[prices_series.index < t]
        if prior.empty:
            print(f"  {ticker} {t.date()}: no prior price — skipping")
            continue
        p_prev = prior.iloc[-1]
        p_curr = prices_series.get(t, np.nan)
        if pd.isna(p_prev) or pd.isna(p_curr) or p_prev == 0:
            print(f"  {ticker} {t.date()}: missing price data — skipping")
            continue

        price_ret   = p_curr / p_prev - 1
        stored_ret  = row["daily_return"]

        # Both conditions must hold to overwrite
        if abs(stored_ret) > 0.4 and abs(price_ret - stored_ret) > 0.05:
            df.at[orig_idx, "daily_return"] = price_ret
            changes.append({
                "ticker"    : ticker,
                "date"      : str(t.date()),
                "old_return": round(stored_ret, 6),
                "new_return": round(price_ret,  6),
                "p_prev"    : round(p_prev, 4),
                "p_curr"    : round(p_curr, 4),
            })

print()
if changes:
    print(f"Changed {len(changes)} rows:")
    print(pd.DataFrame(changes).to_string(index=False))
else:
    print("No rows changed — all flagged returns are consistent with price-based returns.")


# ──────────────────────────────────────────────────────────
# STEP 6  Assert no duplicate (ticker, date) pairs
# ──────────────────────────────────────────────────────────
sep("STEP 6 — Assert no duplicate (ticker, date) pairs")
dupes = df[df.duplicated(["ticker", "date"], keep=False)]
if len(dupes) > 0:
    print("DUPLICATE (ticker, date) PAIRS FOUND:")
    print(dupes[["ticker", "date"]].to_string())
    raise ValueError(f"{len(dupes)} duplicate (ticker, date) rows found — aborting.")
print("PASS — no duplicate (ticker, date) pairs.")


# ──────────────────────────────────────────────────────────
# STEP 7  Master date index coverage
# ──────────────────────────────────────────────────────────
sep("STEP 7 — Master date index coverage")
master_dates = sorted(df["date"].unique())
master_set   = set(master_dates)
print(f"Master date index: {len(master_dates)} trading days "
      f"({master_dates[0].date()} → {master_dates[-1].date()})")

any_gaps = False
for ticker in sorted(df["ticker"].unique()):
    ticker_set = set(df[df["ticker"] == ticker]["date"].unique())
    gaps       = sorted(master_set - ticker_set)
    if gaps:
        any_gaps = True
        sample   = [str(d.date()) for d in gaps[:5]]
        print(f"  {ticker:6s}: {len(gaps)} missing dates — {sample}{'...' if len(gaps) > 5 else ''}")
    else:
        print(f"  {ticker:6s}: full coverage ✓")

if not any_gaps:
    print("All tickers have full coverage of master date index.")


# ──────────────────────────────────────────────────────────
# STEP 8  Build wide pivot tables (shared across all stocks)
# ──────────────────────────────────────────────────────────
sep("STEP 8 — Build wide pivot tables")

wide_returns = (df.pivot(index="date", columns="ticker", values="daily_return")
                  .sort_index()
                  .reindex(columns=ALL_TICKERS))   # consistent column order
wide_prices  = (df.pivot(index="date", columns="ticker", values="price")
                  .sort_index()
                  .reindex(columns=ALL_TICKERS))
wide_volumes = (df.pivot(index="date", columns="ticker", values="volume")
                  .sort_index()
                  .reindex(columns=ALL_TICKERS))

print(f"wide_returns shape: {wide_returns.shape}  "
      f"({wide_returns.index.min().date()} → {wide_returns.index.max().date()})")

# ── Forward-fill ticker-specific data gaps in wide pivot tables
# Handles two cases: (a) missing rows for a ticker on dates the market traded,
# (b) rows that exist but have NaN values (e.g. C volume on a few 2009 dates).
# Macro data (date-level) uses the stricter 5-day limit enforced separately.
print("\nForward-filling any NaN values in wide pivot tables:")
_any_filled = False
for wide_df, label in [(wide_returns, "returns"), (wide_prices, "prices"), (wide_volumes, "volumes")]:
    for ticker in ALL_TICKERS:
        if ticker not in wide_df.columns:
            continue
        col_nan = wide_df[ticker].isna()
        if not col_nan.any():
            continue
        n_nan = int(col_nan.sum())
        run, max_run = 0, 0
        for v in col_nan:
            run = run + 1 if v else 0
            max_run = max(max_run, run)
        print(f"  {label:8s}[{ticker:6s}]: {n_nan} NaN, max run={max_run} — forward-filling all")
        wide_df[ticker] = wide_df[ticker].ffill()
        _any_filled = True
if not _any_filled:
    print("  No NaN values found in any wide pivot — no filling needed.")

# ── Macro features (date-level constants — take first non-NaN per date)
macro_long = (df.sort_values(["date", "ticker"])
                .drop_duplicates("date", keep="first")
                .set_index("date")[MACRO_COLS]
                .sort_index()
                .reindex(wide_returns.index))

# Forward-fill macro gaps ≤ 5 days; warn if longer
print("\nMacro forward-fill check:")
for col in MACRO_COLS:
    col_series = macro_long[col]
    if col_series.isna().any():
        # Compute consecutive NaN run lengths
        is_nan   = col_series.isna()
        run_lens = []
        run = 0
        for v in is_nan:
            if v:
                run += 1
            else:
                if run:
                    run_lens.append(run)
                run = 0
        if run:
            run_lens.append(run)
        max_run = max(run_lens) if run_lens else 0
        if max_run > 5:
            warnings.warn(
                f"  Macro col '{col}' has a consecutive NaN run of {max_run} days — "
                f"only first 5 will be forward-filled; remainder left as NaN."
            )
            print(f"  WARNING: {col} — max gap = {max_run} days (> 5 threshold)")
        else:
            print(f"  {col}: max gap = {max_run} days — forward-filling")
        macro_long[col] = col_series.ffill(limit=5)
    else:
        print(f"  {col}: no missing values ✓")

# ── Lag features: shift(k) on sorted time series — computed once for all stocks
print("\nBuilding lag features for all 12 tickers...")
lag_data = {}
for ticker in ALL_TICKERS:
    if ticker not in wide_returns.columns or wide_returns[ticker].isna().all():
        print(f"  WARNING: {ticker} not found in pivot — lag features will be NaN")
        continue
    series = wide_returns[ticker]   # already sorted by date (index)
    for k in LAG_DEPTHS:
        lag_data[f"lag{k}_return_{ticker}"] = series.shift(k)

lag_df = pd.DataFrame(lag_data, index=wide_returns.index)

# ── Alignment verification: print first 10 rows for JPM lags
sep("STEP 8 — Lag alignment verification (JPM)")
jpm_lag_cols = [f"lag{k}_return_JPM" for k in LAG_DEPTHS]
verify_df = pd.DataFrame({
    "return_JPM"      : wide_returns["JPM"].values,
    "lag1_return_JPM" : lag_df["lag1_return_JPM"].values,
    "lag2_return_JPM" : lag_df["lag2_return_JPM"].values,
    "lag5_return_JPM" : lag_df["lag5_return_JPM"].values,
}, index=wide_returns.index)
print("First 10 rows (lag1 on row t should equal return_JPM on row t-1):")
print(verify_df.head(10).to_string())
print("\nConfirm: lag1_return_JPM[2] ==", round(verify_df["lag1_return_JPM"].iloc[2], 6),
      " return_JPM[1] ==", round(verify_df["return_JPM"].iloc[1], 6),
      " match:", abs(verify_df["lag1_return_JPM"].iloc[2] - verify_df["return_JPM"].iloc[1]) < 1e-10)


# ──────────────────────────────────────────────────────────
# STEPS 9–11  Per-stock: NaN handling → standardize → write
# ──────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

scaler_all      = {}   # {ticker: {col: {mean, std}}}
feature_cols_ref = None   # column names from first stock (for consistency check)
summary_lines   = []

for target in TARGET_STOCKS:
    sep(f"STEPS 9-11 — {target}")

    if target not in wide_returns.columns:
        print(f"  WARNING: {target} missing from pivot — skipping")
        continue

    # ── Contemporaneous cross-section: all other banks + SPY + XLF
    cs_tickers = [t for t in TARGET_STOCKS if t != target] + BENCHMARK_STOCKS
    cs_df      = wide_returns[cs_tickers].rename(columns=lambda c: f"return_{c}")

    # ── Volume for target
    vol_series    = wide_volumes[target].rename("volume")
    logvol_series = np.log1p(wide_volumes[target].clip(lower=0)).rename("log_volume")

    # ── Assemble feature matrix (column order: macro → cs → lags → volume)
    feat = pd.concat([macro_long, cs_df, lag_df, vol_series, logvol_series], axis=1)
    feat.index.name = "date"

    # ── STEP 9a: Drop rows where any lag feature is NaN (first ~21 rows)
    lag_cols  = [c for c in feat.columns if c.startswith("lag")]
    n_before  = len(feat)
    feat      = feat.dropna(subset=lag_cols)
    n_dropped = n_before - len(feat)
    eff_start = feat.index.min()
    print(f"  Dropped {n_dropped} lag-NaN rows. Effective start: {eff_start.date()}")

    # ── STEP 9b: Assert zero NaN remaining
    nan_counts = feat.isna().sum()
    bad_cols   = nan_counts[nan_counts > 0]
    if len(bad_cols) > 0:
        print("  NaN REMAINING after cleaning:")
        for col, cnt in bad_cols.items():
            bad_dates = feat.index[feat[col].isna()].tolist()
            print(f"    {col}: {cnt} NaN — first dates: {[str(d.date()) for d in bad_dates[:3]]}")
        raise ValueError(f"{target}: NaN remain after cleaning — aborting.")
    print(f"  NaN check: PASS (0 NaN in {len(feat)} rows × {len(feat.columns)} cols)")

    # ── STEP 10: Standardize using train-only statistics
    feat_cols = list(feat.columns)
    feat_arr  = feat.values.astype(np.float64)
    n         = len(feat_arr)
    i_train, i_val, i_test = make_splits(n)

    col_means  = feat_arr[i_train].mean(axis=0)
    col_stds   = feat_arr[i_train].std(axis=0)
    col_stds[col_stds < 1e-12] = 1.0   # guard constant columns

    feat_scaled = (feat_arr - col_means) / col_stds
    feat_scaled_df = pd.DataFrame(feat_scaled, index=feat.index, columns=feat_cols)

    scaler_all[target] = {
        col: {"mean": float(col_means[i]), "std": float(col_stds[i])}
        for i, col in enumerate(feat_cols)
    }

    # ── Verify consistent column structure across stocks
    if feature_cols_ref is None:
        feature_cols_ref = feat_cols
    else:
        # Non-cs columns (macro, lags, volume) must be identical and in same order
        non_cs_ref  = [c for c in feature_cols_ref if not c.startswith("return_")]
        non_cs_curr = [c for c in feat_cols        if not c.startswith("return_")]
        assert non_cs_ref == non_cs_curr, \
            f"{target}: non-cs feature columns differ from reference!\n{non_cs_ref}\nvs\n{non_cs_curr}"
        assert len(feat_cols) == len(feature_cols_ref), \
            f"{target}: feature column count {len(feat_cols)} ≠ reference {len(feature_cols_ref)}"

    # ── STEP 11: Write prices_{TICKER}.csv
    prices_out = pd.DataFrame({
        "date"        : wide_returns.index,
        "daily_return": wide_returns[target].values,
        "price"       : wide_prices[target].values,
    }).set_index("date").loc[feat.index].reset_index()
    prices_path = os.path.join(OUTPUT_DIR, f"prices_{target}.csv")
    prices_out.to_csv(prices_path, index=False)

    # ── Write features_{TICKER}.csv
    feat_out   = feat_scaled_df.reset_index()
    feat_path  = os.path.join(OUTPUT_DIR, f"features_{target}.csv")
    feat_out.to_csv(feat_path, index=False)

    train_end_date = feat.index[i_train[-1]]
    val_end_date   = feat.index[i_val[-1]]
    test_end_date  = feat.index[i_test[-1]]

    print(f"  Wrote: prices_{target}.csv  ({len(prices_out)} rows)")
    print(f"  Wrote: features_{target}.csv ({len(feat_out)} rows, {len(feat_cols)} features)")
    print(f"  Train : {len(i_train)} rows  (→ {train_end_date.date()})")
    print(f"  Val   : {len(i_val)} rows  (→ {val_end_date.date()})")
    print(f"  Test  : {len(i_test)} rows  (→ {test_end_date.date()})")

    summary_lines.append(
        f"{target:6s} | rows={len(feat)} | "
        f"{eff_start.date()} → {feat.index.max().date()} | "
        f"features={len(feat_cols)} | NaN=0 | "
        f"train={len(i_train)} val={len(i_val)} test={len(i_test)}"
    )


# ── Write shared files
with open(os.path.join(OUTPUT_DIR, "feature_names.json"), "w") as f:
    json.dump(feature_cols_ref, f, indent=2)

with open(os.path.join(OUTPUT_DIR, "feature_scaler.json"), "w") as f:
    json.dump(scaler_all, f, indent=2)

summary_text = "Multi-Asset Dataset Summary\n" + "="*60 + "\n"
summary_text += "\n".join(summary_lines) + "\n"
with open(os.path.join(OUTPUT_DIR, "dataset_summary.txt"), "w") as f:
    f.write(summary_text)

print(f"\nWrote: feature_names.json, feature_scaler.json, dataset_summary.txt")


# ──────────────────────────────────────────────────────────
# STEP 12  Final validation
# ──────────────────────────────────────────────────────────
sep("STEP 12 — Final validation")

price_dfs   = {
    t: pd.read_csv(os.path.join(OUTPUT_DIR, f"prices_{t}.csv"),   parse_dates=["date"])
    for t in TARGET_STOCKS
}
feature_dfs = {
    t: pd.read_csv(os.path.join(OUTPUT_DIR, f"features_{t}.csv"), parse_dates=["date"])
    for t in TARGET_STOCKS
}

with open(os.path.join(OUTPUT_DIR, "feature_scaler.json")) as f:
    scaler_loaded = json.load(f)

results = {}

# ── Check 1: identical date columns across all files
ref_dates = price_dfs[TARGET_STOCKS[0]]["date"].values
check1 = all(
    np.array_equal(price_dfs[t]["date"].values, ref_dates) and
    np.array_equal(feature_dfs[t]["date"].values, ref_dates)
    for t in TARGET_STOCKS
)
results["1. Identical date columns across all files"] = "PASS" if check1 else "FAIL"

# ── Check 2: zero NaN in any output file
nan_found = False
for t in TARGET_STOCKS:
    if price_dfs[t].isna().any().any():
        nan_found = True
        print(f"  NaN in prices_{t}.csv")
    if feature_dfs[t].isna().any().any():
        nan_found = True
        bad = feature_dfs[t].isna().sum()
        print(f"  NaN in features_{t}.csv: {bad[bad > 0].to_dict()}")
results["2. Zero NaN in all output files"] = "FAIL" if nan_found else "PASS"

# ── Check 3: same column count; non-cs columns in same order
feat_col_sets = [list(feature_dfs[t].columns) for t in TARGET_STOCKS]
non_cs_sets   = [
    [c for c in cols if not c.startswith("return_") and c != "date"]
    for cols in feat_col_sets
]
same_count = all(len(c) == len(feat_col_sets[0]) for c in feat_col_sets)
same_order = all(c == non_cs_sets[0] for c in non_cs_sets)
check3 = same_count and same_order
if not same_count:
    print(f"  Column count mismatch: {[len(c) for c in feat_col_sets]}")
if not same_order:
    for t, cols in zip(TARGET_STOCKS, non_cs_sets):
        if cols != non_cs_sets[0]:
            print(f"  {t} non-cs col order differs from {TARGET_STOCKS[0]}")
results["3. Consistent column count and order"] = "PASS" if check3 else "FAIL"

# ── Check 4: prices daily_return matches cleaned source
clean_ret_lookup = (
    df[df["ticker"].isin(TARGET_STOCKS)]
    .set_index(["ticker", "date"])["daily_return"]
)
check4 = True
for t in TARGET_STOCKS:
    pf = price_dfs[t].set_index("date")["daily_return"]
    for d, v in pf.items():
        try:
            src = clean_ret_lookup.loc[(t, d)]
        except KeyError:
            continue
        if abs(v - src) > 1e-7:
            print(f"  Mismatch: {t} {d.date()}: file={v:.6f} source={src:.6f}")
            check4 = False
results["4. prices_T daily_return matches cleaned source"] = "PASS" if check4 else "FAIL"

# ── Check 5: lag1 spot-check (10 random dates, JPM as reference stock)
random.seed(42)
ref_t       = TARGET_STOCKS[0]   # JPM
feat_jpm    = feature_dfs[ref_t].set_index("date")
price_jpm   = price_dfs[ref_t].set_index("date")
feat_dates  = list(feat_jpm.index)
sample_idx  = random.sample(range(2, len(feat_dates)), min(10, len(feat_dates) - 2))
sample_dates = sorted([feat_dates[i] for i in sample_idx])

lag1_col = "lag1_return_JPM"
mean_val  = scaler_loaded[ref_t][lag1_col]["mean"]
std_val   = scaler_loaded[ref_t][lag1_col]["std"]

print(f"\n  Lag1 spot-check ({lag1_col}): unscale → compare with JPM return on t-1")
print(f"  {'Date':12s}  {'lag1_scaled':>12s}  {'lag1_raw':>12s}  {'JPM_ret_t-1':>12s}  {'match':>6s}")

check5 = True
for d in sample_dates:
    lag1_scaled = feat_jpm.loc[d, lag1_col]
    lag1_raw    = lag1_scaled * std_val + mean_val   # inverse-transform
    d_idx       = feat_dates.index(d)
    d_prev      = feat_dates[d_idx - 1]
    if d_prev not in price_jpm.index:
        print(f"  {str(d.date()):12s}: prev date not in prices — skipping")
        continue
    jpm_prev    = price_jpm.loc[d_prev, "daily_return"]
    match       = abs(lag1_raw - jpm_prev) < 1e-5
    if not match:
        check5 = False
    print(f"  {str(d.date()):12s}  {lag1_scaled:>12.6f}  {lag1_raw:>12.6f}  {jpm_prev:>12.6f}  {'OK' if match else 'FAIL':>6s}")

results["5. lag1 spot-check (10 random dates, unscaled)"] = "PASS" if check5 else "FAIL"

# ── Check 6: train end date consistent across all stocks
train_end_dates = {}
for t in TARGET_STOCKS:
    n_t      = len(feature_dfs[t])
    i_tr, _, _ = make_splits(n_t)
    train_end_dates[t] = feature_dfs[t]["date"].iloc[i_tr[-1]]

unique_ends = set(str(d.date()) for d in train_end_dates.values())
check6 = len(unique_ends) == 1
if not check6:
    print("  Train end dates differ across stocks:")
    for t, d in train_end_dates.items():
        print(f"    {t}: {d.date()}")
results["6. Train end date consistent across all stocks"] = "PASS" if check6 else "FAIL"

# ── Print validation table
print("\n  ─── Validation Results ───")
all_pass = True
for k, v in results.items():
    mark = "✓" if v == "PASS" else "✗"
    print(f"  [{v}] {mark} {k}")
    if v != "PASS":
        all_pass = False

# ── Final summary
n_sample   = len(feature_dfs[TARGET_STOCKS[0]])
i_tr, i_vl, i_te = make_splits(n_sample)
n_feats    = len(feature_dfs[TARGET_STOCKS[0]].columns) - 1   # exclude date col
date_start = feature_dfs[TARGET_STOCKS[0]]["date"].iloc[0]
date_end   = feature_dfs[TARGET_STOCKS[0]]["date"].iloc[-1]
train_end  = feature_dfs[TARGET_STOCKS[0]]["date"].iloc[i_tr[-1]]

sep("FINAL SUMMARY")
print(f"  Stocks        : {len(TARGET_STOCKS)}  ({', '.join(TARGET_STOCKS)})")
print(f"  Features      : {n_feats}")
print(f"  Date range    : {date_start.date()} → {date_end.date()}")
print(f"  Total rows    : {n_sample}")
print(f"  Train rows    : {len(i_tr)}  (→ {train_end.date()})")
print(f"  Val rows      : {len(i_vl)}")
print(f"  Test rows     : {len(i_te)}")
print(f"  Output dir    : {OUTPUT_DIR}")
print(f"  All checks    : {'PASS ✓' if all_pass else 'SOME FAILED — see above'}")

print()
print(summary_text)
