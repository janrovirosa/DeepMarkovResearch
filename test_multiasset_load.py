#!/usr/bin/env python3
"""Smoke-test load_master_dataset(ticker=...) for all 10 target stocks."""

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from scripts.data import load_master_dataset

TICKERS = ["JPM", "C", "WFC", "GS", "MS", "PNC", "USB", "FITB", "MTB", "BAC"]

failures = []

print(f"{'Ticker':6s}  {'Rows':>6s}  {'Features':>8s}  {'First date':12s}  {'Last date':12s}  {'DateAlign':>9s}  {'NaN':>5s}")
print("-" * 80)

for ticker in TICKERS:
    try:
        prices, F_raw, feature_cols = load_master_dataset(ticker=ticker)

        # Re-load CSVs to verify date alignment independently
        prices_df   = pd.read_csv(f"dataset_multiasset/prices_{ticker}.csv",   parse_dates=["date"])
        features_df = pd.read_csv(f"dataset_multiasset/features_{ticker}.csv", parse_dates=["date"])

        dates_match = (prices_df["date"].values == features_df["date"].values).all()
        n_nan       = int(np.isnan(F_raw).sum())
        first_date  = str(prices_df["date"].iloc[0].date())
        last_date   = str(prices_df["date"].iloc[-1].date())

        status_align = "OK" if dates_match else "FAIL"
        status_nan   = "OK" if n_nan == 0 else f"FAIL({n_nan})"

        print(f"{ticker:6s}  {len(prices):>6,}  {len(feature_cols):>8d}  "
              f"{first_date:12s}  {last_date:12s}  {status_align:>9s}  {status_nan:>5s}")

        if not dates_match:
            failures.append(f"{ticker}: date misalignment")
        if n_nan > 0:
            failures.append(f"{ticker}: {n_nan} NaN values in F_raw")
        if len(prices) != F_raw.shape[0]:
            failures.append(f"{ticker}: prices length {len(prices)} != F_raw rows {F_raw.shape[0]}")

    except Exception as exc:
        print(f"{ticker:6s}  ERROR: {exc}")
        failures.append(f"{ticker}: exception — {exc}")

print()
if failures:
    print("FAILURES:")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
else:
    print("ALL TICKERS PASS")
