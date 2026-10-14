import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TICKERS = ["JPM", "C", "WFC", "GS", "MS", "PNC", "USB", "FITB", "MTB", "BAC"]
MACRO = ["DFF", "T10Y2Y", "BAA10Y", "VIXCLS", "DCOILWTICO", "USREC"]
INDEXES = ["SPY", "XLF"]
EPS = 1e-8


def load_jpm(path=ROOT / "data/jpm/train_diagnostic.csv"):
    df = pd.read_csv(path).drop(columns=["Opinion"], errors="ignore")
    cols = [c for c in df.columns if c not in ("index", "Percent_change_forward", "Backward_Bin", "Price")]
    return df["Price"].values.astype(np.float64), df[cols].values.astype(np.float32), cols


def load_bank(ticker, folder=ROOT / "data/banks"):
    # features are already standardized by experiments/prepare_banks.py
    p = pd.read_csv(Path(folder) / f"prices_{ticker}.csv", parse_dates=["date"])
    f = pd.read_csv(Path(folder) / f"features_{ticker}.csv", parse_dates=["date"])
    assert (p["date"].values == f["date"].values).all(), f"{ticker}: dates misaligned"
    cols = [c for c in f.columns if c != "date"]
    return p["price"].values.astype(np.float64), f[cols].values.astype(np.float32), cols


def returns(prices, h):
    return (prices[h:] - prices[:-h]) / prices[:-h]


def standardize(F, idx_train):
    # z-score with train statistics, NaN/inf -> 0
    mean = F[idx_train].mean(axis=0, keepdims=True)
    std = F[idx_train].std(axis=0, keepdims=True) + EPS
    Z = ((F - mean) / std).astype(np.float32)
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)


def split(T):
    # chronological 70/15/15
    a, b = int(0.70 * T), int((0.70 + 0.15) * T)
    return np.arange(0, a), np.arange(a, b), np.arange(b, T)


def splits_by_horizon(n, horizons):
    # n = number of usable time steps before subtracting h
    out = {}
    for h in horizons:
        tr, va, te = split(n - h)
        out[h] = {"T": n - h, "idx_train": tr, "idx_val": va, "idx_test": te}
    return out


def quantile_edges(x_train, n):
    # n quantile bins fit on train, fewer if quantiles tie, open ended
    n = min(n, len(np.unique(np.quantile(x_train, np.linspace(0, 1, n + 1)))) - 1)
    edges = np.quantile(x_train, np.linspace(0, 1, n + 1))
    if len(np.unique(edges)) - 1 < n:
        n = len(np.unique(edges)) - 1
        edges = np.quantile(x_train, np.linspace(0, 1, n + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return n, edges


def to_bins(x, edges):
    return np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)


def label_sigma(edges, edges_anchor, sigma_anchor=1.0):
    # soft label width scaled so it covers the same return range as at the anchor N
    w = np.median(np.diff(edges[np.isfinite(edges)]))
    w_anchor = np.median(np.diff(edges_anchor[np.isfinite(edges_anchor)]))
    return sigma_anchor * w_anchor / w


def return_states(prices, n, train_end):
    # X_t = bin of the 1-day return, edges fit on the first train_end returns
    r = returns(prices, 1)
    n, edges = quantile_edges(r[:train_end], n)
    return to_bins(r, edges), n, edges


def vol_states(prices, window, n, train_end):
    # X_t = bin of rolling std of 1-day returns, warmup rows dropped
    r = returns(prices, 1)
    sigma = pd.Series(r).rolling(window).std(ddof=1).values[window - 1:]
    assert not np.isnan(sigma).any()
    n, edges = quantile_edges(sigma[:train_end], n)
    return to_bins(sigma, edges), sigma, n, edges, window - 1


def forward_labels(prices, splits, n_bins_list, sigma_anchor=1.0):
    # Y_t = bin of the h-day return from t+1 to t+1+h, so it never overlaps the return in X_t
    configs = {}
    for h, sp in splits.items():
        R = returns(prices[1:], h)
        R_train = R[sp["idx_train"]]
        _, anchor = quantile_edges(R_train, max(n_bins_list))
        for N in n_bins_list:
            n, edges = quantile_edges(R_train, N)
            configs[(h, N)] = {**sp, "R": R, "y_all": to_bins(R, edges), "edges": edges,
                               "sigma": label_sigma(edges, anchor, sigma_anchor), "N_actual": n}
    return configs


def feature_subset(F, cols, regime, ticker=None):
    if regime == "full":
        keep = list(cols)
    elif regime == "macro_only":
        keep = [c for c in cols if c in MACRO]
    elif regime == "own_only":
        own = re.compile(rf"^lag\d+_return_{re.escape(ticker)}$")
        keep = [c for c in cols if own.match(c) or c in ("volume", "log_volume")]
    elif regime == "other_banks_only":
        others = (set(TICKERS) | set(INDEXES)) - {ticker}
        name = re.compile(r"^(?:lag\d+_)?return_(.+)$")
        keep = [c for c in cols if name.match(c) and name.match(c).group(1) in others]
    else:
        raise ValueError(regime)
    assert keep, f"no columns for regime={regime} ticker={ticker}"
    return F[:, [cols.index(c) for c in keep]], keep
