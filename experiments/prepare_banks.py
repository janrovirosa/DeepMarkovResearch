# builds data/banks (prices and standardized features for the 10 banks) from data/banks_raw
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/banks_raw"
OUT = ROOT / "data/banks"

BANKS = ["JPM", "C", "WFC", "GS", "MS", "PNC", "USB", "FITB", "MTB", "BAC"]
BENCHMARKS = ["SPY", "XLF"]
TICKERS = BANKS + BENCHMARKS
MACRO = ["DFF", "T10Y2Y", "BAA10Y", "VIXCLS", "DCOILWTICO", "USREC"]
LAGS = [1, 2, 5, 21]
DROP = ["earnings_surprise", "surprise_pct", "numest", "stdev", "inst_ownership_pct", "total_assets",
        "total_liabilities", "net_income", "common_equity", "eps", "revenue", "leverage_ratio", "roe", "cusip", "actual_eps"]

df = pd.read_csv(RAW / "deep_markov_expanded_universe.csv")
df["date"] = pd.to_datetime(df["date"])
for c in df.columns:
    if c not in ("date", "ticker"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

# ticker MS belonged to Milestone Scientific until 2006, swap in Morgan Stanley (CRSP PERMNO 69032)
ms = pd.read_csv(RAW / "MS_updated.csv", low_memory=False)
ms["date"] = pd.to_datetime(ms["date"])
ms = ms[ms["PERMNO"] == 69032].copy()
for c in ["RET", "PRC", "VOL", "SHROUT"]:
    ms[c] = pd.to_numeric(ms[c], errors="coerce")
ms = ms.dropna(subset=["PRC", "RET"]).drop_duplicates(subset=["date"])
macro_by_date = df[df["ticker"] == "JPM"][["date"] + MACRO].drop_duplicates("date")
ms = pd.DataFrame({"permno": 69032, "ticker": "MS", "date": ms["date"].values, "price": ms["PRC"].abs().values,
                   "daily_return": ms["RET"].values, "volume": ms["VOL"].values, "shares_out": ms["SHROUT"].values,
                   "market_cap": (ms["PRC"].abs() * ms["SHROUT"]).values}).merge(macro_by_date, on="date", how="left")
ms = ms.reindex(columns=df.columns)
df = pd.concat([df[df["ticker"] != "MS"], ms], ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)

df = df[~df["ticker"].isin(["TFC", "KBE"])].drop(columns=[c for c in DROP if c in df.columns])
df = df[(df["date"] >= "1999-05-05") & (df["date"] <= "2023-12-29")].sort_values(["ticker", "date"]).reset_index(drop=True)

# returns above 40% that disagree with the price change by more than 5 points are data errors, use the price change
for ticker, g in df.groupby("ticker"):
    p = g.set_index("date")["price"]
    for i, row in g[g["daily_return"].abs() > 0.4].iterrows():
        prev = p[p.index < row["date"]]
        if prev.empty or pd.isna(prev.iloc[-1]) or pd.isna(p[row["date"]]) or prev.iloc[-1] == 0:
            continue
        r = p[row["date"]] / prev.iloc[-1] - 1
        if abs(r - row["daily_return"]) > 0.05:
            print(f"{ticker} {row['date'].date()}: return {row['daily_return']:.4f} -> {r:.4f}")
            df.at[i, "daily_return"] = r
assert not df.duplicated(["ticker", "date"]).any()

wide = {c: df.pivot(index="date", columns="ticker", values=c).sort_index().reindex(columns=TICKERS).ffill()
        for c in ["daily_return", "price", "volume"]}
macro = df.sort_values(["date", "ticker"]).drop_duplicates("date").set_index("date")[MACRO].sort_index()
macro = macro.reindex(wide["daily_return"].index).ffill(limit=5)
lags = pd.DataFrame({f"lag{k}_return_{t}": wide["daily_return"][t].shift(k) for t in TICKERS for k in LAGS})

OUT.mkdir(parents=True, exist_ok=True)
scaler, names, summary = {}, None, []
for target in BANKS:
    # macro, same-day returns of the other banks and the ETFs, lagged returns of all 12, target volume
    cs = wide["daily_return"][[t for t in BANKS if t != target] + BENCHMARKS].rename(columns=lambda c: f"return_{c}")
    vol = wide["volume"][target]
    feat = pd.concat([macro, cs, lags, vol.rename("volume"), np.log1p(vol.clip(lower=0)).rename("log_volume")], axis=1)
    feat.index.name = "date"
    feat = feat.dropna(subset=[c for c in feat.columns if c.startswith("lag")])
    assert not feat.isna().any().any(), f"{target}: NaN left"

    # standardize with the first 70% of rows
    X = feat.values.astype(np.float64)
    n_train = int(0.70 * len(X))
    mean, std = X[:n_train].mean(axis=0), X[:n_train].std(axis=0)
    std[std < 1e-12] = 1.0
    scaled = pd.DataFrame((X - mean) / std, index=feat.index, columns=feat.columns)
    scaler[target] = {c: {"mean": float(m), "std": float(s)} for c, m, s in zip(feat.columns, mean, std)}
    names = names or list(feat.columns)

    prices = pd.DataFrame({"daily_return": wide["daily_return"][target], "price": wide["price"][target]}).loc[feat.index]
    prices.reset_index().to_csv(OUT / f"prices_{target}.csv", index=False)
    scaled.reset_index().to_csv(OUT / f"features_{target}.csv", index=False)
    n, n_val = len(feat), int((0.70 + 0.15) * len(feat)) - n_train
    summary.append(f"{target:6s} | rows={n} | {feat.index.min().date()} → {feat.index.max().date()} | features={len(names)} | "
                   f"NaN=0 | train={n_train} val={n_val} test={n - n_train - n_val}")
    print(summary[-1])

json.dump(names, open(OUT / "feature_names.json", "w"), indent=2)
json.dump(scaler, open(OUT / "feature_scaler.json", "w"), indent=2)
(OUT / "dataset_summary.txt").write_text("Multi-Asset Dataset Summary\n" + "=" * 60 + "\n" + "\n".join(summary) + "\n")
