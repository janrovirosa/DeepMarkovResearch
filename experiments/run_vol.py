# volatility-state experiments for paper 2: does conditioning on a realized volatility bin beat a return bin?
# models: state_free, state_cond_return, state_cond_vol_w{5,10,21}, state_cond_macro (return state, macro features only)
# writes results/volatility/<ticker>/<model>/: nll_metrics.csv, operator_ts_*.csv, ck_ts_*.csv, snapshots/
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from markov.data import TICKERS, load_bank, splits_by_horizon, return_states, vol_states, forward_labels, feature_subset
from markov.models import make_model
from markov.train import loaders, fit, operators, pick_device
from markov.metrics import evaluate, operator_ts, ck_ts, snapshot_times

HORIZONS = [1, 2, 3, 5]
BINS = [35, 45, 55]
WINDOWS = [5, 10, 21]
MODELS = ["state_free", "state_cond_return", "state_cond_vol_w5", "state_cond_vol_w10", "state_cond_vol_w21", "state_cond_macro"]
SEEDS = [42, 7, 123]
SIGMA = 1.0  # best sigma from the multi-asset sweep
WARMUP = max(WINDOWS) - 1  # every model drops the same 20 days so they see the same dates
OUT = ROOT / "results/volatility"
CACHE = OUT / "_cache"
DEVICE = pick_device()


def setup(ticker):
    prices, F, cols = load_bank(ticker)
    dates = pd.read_csv(ROOT / f"data/banks/prices_{ticker}.csv", parse_dates=["date"])["date"].values
    train_end = len(splits_by_horizon(len(prices) - 1, [1])[1]["idx_train"])

    # vol states on the full series, then trimmed so every window starts on the same day
    vols = {}
    for w in WINDOWS:
        X, sigma, *_ = vol_states(prices, w, 55, max(0, train_end - (w - 1)))
        vols[w] = (X[WARMUP - (w - 1):], sigma[WARMUP - (w - 1):])

    prices, F = prices[WARMUP:], F[WARMUP:]
    sp = splits_by_horizon(len(prices) - 1, HORIZONS)
    X_ret, n_x, _ = return_states(prices, 55, len(sp[1]["idx_train"]))
    # X_t is the return from t to t+1, so it is known on the date of price t+1
    dates = dates[WARMUP + 1:]

    ts = pd.DataFrame({"date": dates[:len(X_ret)]})
    for w in WINDOWS:
        ts[f"sigma_t_w{w}"], ts[f"X_t_vol_w{w}"] = vols[w][1][:len(ts)], vols[w][0][:len(ts)]
    OUT.mkdir(parents=True, exist_ok=True)
    ts.to_csv(OUT / f"vol_state_ts_{ticker}.csv", index=False)

    F_macro, _ = feature_subset(F, cols, "macro_only")
    states = {m: vols[int(m.split("_w")[1])][0] if "_vol_" in m else X_ret for m in MODELS}
    features = {m: F_macro if m == "state_cond_macro" else F for m in MODELS}
    return {"configs": forward_labels(prices, sp, BINS), "states": states, "features": features, "n_x": n_x, "dates": dates}


def model_for(td, m, h, N, seed):
    F, X, c = td["features"][m], td["states"][m], td["configs"][(h, N)]
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = make_model("state_free" if m == "state_free" else "state_cond", F.shape[1], td["n_x"], c["N_actual"])
    return fit(model, CACHE / f"{td['ticker']}_{m}_h{h}_N{N}_seed{seed}.pt", *loaders(F, X, c["y_all"], c)[:2],
               c["N_actual"], SIGMA, DEVICE, swa=True, swa_epochs=10)[0]


def run_one(td, m, h, N, seed):
    F, X, c = td["features"][m], td["states"][m], td["configs"][(h, N)]
    n = c["N_actual"]
    tr, va, te = loaders(F, X, c["y_all"], c)
    model = model_for(td, m, h, N, seed)
    nll = {s: -evaluate(model, l, n, DEVICE)["mean_ll"] for s, l in [("train", tr), ("val", va), ("test", te)]}
    # delta vs the uniform distribution log(N), which is the train marginal of quantile bins
    row = {"ticker": td["ticker"], "model": m, "vol_window": int(m.split("_w")[1]) if "_vol_" in m else None, "h": h, "N": N,
           "seed": seed, "nll_train": nll["train"], "nll_val": nll["val"], "nll_test": nll["test"], "delta_marginal": nll["test"] - np.log(n)}
    print(f"  {td['ticker']} {m:20s} h={h} N={N} seed={seed} nll_test={nll['test']:.4f} delta={row['delta_marginal']:+.4f}")

    out = OUT / td["ticker"] / m
    out.mkdir(parents=True, exist_ok=True)
    A = operators(model, F, c["idx_test"], td["n_x"], n, DEVICE, state_free=m == "state_free")
    operator_ts(A).assign(date=td["dates"][c["idx_test"]]).to_csv(out / f"operator_ts_h{h}_N{N}_seed{seed}.csv", index=False)
    if h == 1 and N == 55 and m != "state_free":
        (out / "snapshots").mkdir(exist_ok=True)
        for label, t in snapshot_times(A).items():
            np.save(out / f"snapshots/operator_snapshot_{label}_seed{seed}.npy", A[t].astype(np.float32))
    return row, A


def ck_pass(td, m, N, ops):
    # CK series for h > 1: direct h-step operator vs product of 1-step operators over the next h days
    # here both are forward-return operators (state -> return bin), not state -> state
    F = td["features"][m]
    for seed in SEEDS:
        model1 = model_for(td, m, 1, N, seed)
        for h in HORIZONS[1:]:
            te = td["configs"][(h, N)]["idx_test"]
            ext = np.arange(te[0], min(te[-1] + h, len(F)))
            A1 = operators(model1, F, ext, td["n_x"], td["configs"][(1, N)]["N_actual"], DEVICE, state_free=m == "state_free")
            ck_ts(A1, ops[(m, h, seed)], h).assign(date=lambda d: td["dates"][te[d.timestep]]).to_csv(
                OUT / td["ticker"] / m / f"ck_ts_h{h}_N{N}_seed{seed}.csv", index=False)


def run(tickers=TICKERS):
    CACHE.mkdir(parents=True, exist_ok=True)
    for ticker in tickers:
        print(f"\n{ticker}")
        td = {**setup(ticker), "ticker": ticker}
        rows = []
        for N in BINS:
            ops = {}
            for h in HORIZONS:
                for m in MODELS:
                    for seed in SEEDS:
                        row, ops[(m, h, seed)] = run_one(td, m, h, N, seed)
                        rows.append(row)
            if N == td["n_x"]:
                for m in MODELS:
                    ck_pass(td, m, N, ops)
        df = pd.DataFrame(rows)
        for m in MODELS:
            df[df.model == m].to_csv(OUT / ticker / m / "nll_metrics.csv", index=False)


if __name__ == "__main__":
    run(sys.argv[1:] or TICKERS)
