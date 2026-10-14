# multi-asset experiments on the 10 banks (return state)
# A: full grid h x N, no SWA       B: soft-label sigma sweep on JPM      C: grid again with SWA and the best sigma
# D: k-step state history (JPM)    E: feature subsets (macro only, own lags, other banks)
# F: cross-asset diagnostics from the C models
# writes results/multi_asset/<ticker>/<experiment>/ and results/multi_asset/cross_asset_diagnostics.csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from markov.data import TICKERS, load_bank, splits_by_horizon, return_states, forward_labels, feature_subset
from markov.models import make_model, HigherOrderNet
from markov.train import loaders, history_loaders, fit, operators, pick_device
from markov.metrics import evaluate, operator_ts, ck_ts, snapshot_times

HORIZONS = [1, 5, 10, 21, 63]
BINS = [4, 10, 20, 35, 55]
SEEDS = [42, 7, 123]
SIGMAS = [0.5, 1.0, 1.5, 2.0, 3.0]
K_LIST = [1, 2, 3, 5, 10]
D_HORIZONS = [1, 5, 21, 63]
E_HORIZONS = [1, 21]
REGIMES = ["full", "macro_only", "own_only", "other_banks_only"]
OUT = ROOT / "results/multi_asset"
CACHE = OUT / "_cache"
DEVICE = pick_device()


def setup(ticker, horizons=HORIZONS, bins=BINS, state_split=None):
    # state edges fit on the h=1 train window unless state_split says otherwise
    prices, F, cols = load_bank(ticker)
    dates = pd.read_csv(ROOT / f"data/banks/prices_{ticker}.csv", parse_dates=["date"])["date"].values
    sp = splits_by_horizon(len(prices) - 1, horizons)
    X, n_x, _ = return_states(prices, 55, len(sp[state_split or min(horizons)]["idx_train"]))
    # X_t is the return from t to t+1, known on the date of price t+1
    return {"ticker": ticker, "F": F, "cols": cols, "X": X, "n_x": n_x, "dates": dates[1:],
            "configs": forward_labels(prices, sp, bins)}


def train_eval(model, path, td, c, F, sigma, swa, history_k=None, train_nll=True):
    n = c["N_actual"]
    l = history_loaders(F, td["X"], c["y_all"], c, history_k) if history_k else loaders(F, td["X"], c["y_all"], c)
    model, _ = fit(model, path, l[0], l[1], n, sigma, DEVICE, swa=swa, swa_epochs=10)
    nll = {s: -evaluate(model, x, n, DEVICE)["mean_ll"] for s, x in zip(["train", "val", "test"], l) if s != "train" or train_nll}
    return model, {"nll_train": nll.get("train"), "nll_val": nll["val"], "nll_test": nll["test"], "delta_marginal": nll["test"] - np.log(n)}


def save_ts(df, td, idx, path):
    df.assign(date=td["dates"][idx[df.timestep]]).to_csv(path, index=False)


def grid(exp, prefix, sigma, swa):
    # experiments A and C: every ticker, horizon, bin count, model and seed
    for ticker in TICKERS:
        td = setup(ticker)
        out = OUT / ticker / exp
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for N in BINS:
            ops = {}
            for h in HORIZONS:
                c = td["configs"][(h, N)]
                for kind in ["state_cond", "state_free"]:
                    for seed in SEEDS:
                        torch.manual_seed(seed)
                        np.random.seed(seed)
                        model, r = train_eval(make_model(kind, td["F"].shape[1], td["n_x"], c["N_actual"]),
                                              CACHE / f"{prefix}_{ticker}_{kind}_h{h}_N{N}_seed{seed}.pt", td, c, td["F"], sigma, swa)
                        rows.append({"model": kind, "h": h, "N": N, "seed": seed, **r})
                        print(f"  [{exp}] {ticker} {kind} h={h} N={N} seed={seed} nll_test={r['nll_test']:.4f}")
                        A = operators(model, td["F"], c["idx_test"], td["n_x"], c["N_actual"], DEVICE, state_free=kind == "state_free")
                        save_ts(operator_ts(A), td, c["idx_test"], out / f"operator_ts_{kind}_h{h}_N{N}_seed{seed}.csv")
                        ops[(kind, h, seed)] = A
                        if h == 1 and N == 55 and kind == "state_cond":
                            (out / "snapshots").mkdir(exist_ok=True)
                            for label, t in snapshot_times(A).items():
                                np.save(out / f"snapshots/operator_snapshot_{label}_seed{seed}.npy", A[t].astype(np.float32))
            if N == td["n_x"]:
                # CK series: direct h-step forward-return operator vs product of 1-step ones over the next h days
                for kind in ["state_cond", "state_free"]:
                    for seed in SEEDS:
                        m1 = make_model(kind, td["F"].shape[1], td["n_x"], N)
                        m1.load_state_dict(torch.load(CACHE / f"{prefix}_{ticker}_{kind}_h1_N{N}_seed{seed}.pt", map_location="cpu"))
                        m1.to(DEVICE)
                        for h in HORIZONS[1:]:
                            te = td["configs"][(h, N)]["idx_test"]
                            A1 = operators(m1, td["F"], np.arange(te[0], min(te[-1] + h, len(td["F"]))), td["n_x"], N, DEVICE,
                                           state_free=kind == "state_free")
                            save_ts(ck_ts(A1, ops[(kind, h, seed)], h), td, te, out / f"ck_ts_{kind}_h{h}_N{N}_seed{seed}.csv")
        pd.DataFrame(rows).to_csv(out / "nll_metrics.csv", index=False)
        json.dump({"sigma": sigma, "use_swa": swa, "horizons": HORIZONS, "bins": BINS}, open(out / "config.json", "w"), indent=2)


def sigma_sweep():
    # experiment B: JPM, N=55, no SWA, pick the sigma with the lowest mean validation NLL at h=1
    out = OUT / "JPM/sigma_sweep"
    out.mkdir(parents=True, exist_ok=True)
    best = {}
    for h in [1, 21]:
        td = setup("JPM", [h], [55], state_split=h)
        c = td["configs"][(h, 55)]
        rows = []
        for sigma in SIGMAS:
            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                _, r = train_eval(make_model("state_cond", td["F"].shape[1], td["n_x"], c["N_actual"]),
                                  CACHE / f"sweep_sigma_ext/JPM_h{h}_N55_sigma{sigma}_seed{seed}.pt", td, c, td["F"], sigma, False, train_nll=False)
                rows.append({"sigma": sigma, "seed": seed, "nll_val": r["nll_val"], "nll_test": r["nll_test"]})
        df = pd.DataFrame(rows)
        df.to_csv(out / f"sweep_h{h}.csv", index=False)
        best[h] = float(df.groupby("sigma").nll_val.mean().idxmin())
        print(f"  [sigma sweep] h={h}: best sigma {best[h]}")
    return best[1]


def higher_order(sigma):
    # experiment D: condition on the last k states, JPM, N=55
    td = setup("JPM")
    out = OUT / "JPM/higher_order"
    out.mkdir(parents=True, exist_ok=True)
    rows, A_k1 = [], {}
    for k in K_LIST:
        for h in D_HORIZONS:
            c = td["configs"][(h, 55)]
            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model, r = train_eval(HigherOrderNet(td["F"].shape[1], td["n_x"], c["N_actual"], k=k),
                                      CACHE / f"Dext_JPM_ho_k{k}_h{h}_N55_seed{seed}.pt", td, c, td["F"], sigma, True,
                                      history_k=k, train_nll=False)
                rows.append({"model": f"ho_k{k}", "h": h, "N": 55, "seed": seed, **r})
                # rows vary the current state, the k-1 older states stay at their observed values
                hist = np.stack([td["X"][c["idx_test"] - lag] for lag in range(1, k)], axis=1) if k > 1 else None
                A = operators(model, td["F"], c["idx_test"], td["n_x"], c["N_actual"], DEVICE, history=hist)
                save_ts(operator_ts(A), td, c["idx_test"], out / f"operator_ts_ho_k{k}_h{h}_N55_seed{seed}.csv")
                if k == 1 and h == 1:
                    A_k1[seed] = model
                elif k == 1:
                    te = c["idx_test"]
                    A1 = operators(A_k1[seed], td["F"], np.arange(te[0], min(te[-1] + h, len(td["F"]))), td["n_x"], 55, DEVICE)
                    save_ts(ck_ts(A1, A, h), td, te, out / f"ck_ts_ho_k{k}_h{h}_N55_seed{seed}.csv")
    pd.DataFrame(rows).to_csv(out / "nll_metrics.csv", index=False)


def conditioning(sigma):
    # experiment E: same model on feature subsets, h=1 and 21, N=55
    for ticker in TICKERS:
        td = setup(ticker)
        out = OUT / ticker / "conditioning_regime"
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for regime in REGIMES:
            F, _ = feature_subset(td["F"], td["cols"], regime, ticker)
            models = {}
            for h in E_HORIZONS:
                c = td["configs"][(h, 55)]
                for seed in SEEDS:
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    model, r = train_eval(make_model("state_cond", F.shape[1], td["n_x"], c["N_actual"]),
                                          CACHE / f"Eext_{ticker}_regime{regime}_h{h}_N55_seed{seed}.pt", td, c, F, sigma, True, train_nll=False)
                    rows.append({"model": f"state_cond_{regime}", "h": h, "N": 55, "seed": seed, **r})
                    A = operators(model, F, c["idx_test"], td["n_x"], c["N_actual"], DEVICE)
                    save_ts(operator_ts(A), td, c["idx_test"], out / f"operator_ts_state_cond_{regime}_h{h}_N55_seed{seed}.csv")
                    models[(h, seed)] = (model, A)
            te = td["configs"][(21, 55)]["idx_test"]
            for seed in SEEDS:
                A1 = operators(models[(1, seed)][0], F, np.arange(te[0], min(te[-1] + 21, len(F))), td["n_x"], 55, DEVICE)
                save_ts(ck_ts(A1, models[(21, seed)][1], 21), td, te, out / f"ck_ts_state_cond_{regime}_h21_N55_seed{seed}.csv")
        pd.DataFrame(rows).to_csv(out / "nll_metrics.csv", index=False)


def cross_asset(sigma, h_ck=5):
    # experiment F: seed-averaged h=1 operator diagnostics and h=5 CK error per bank over the test period
    rows = []
    for ticker in TICKERS:
        td = setup(ticker)
        c1, ch = td["configs"][(1, 55)], td["configs"][(h_ck, 55)]
        As, cks = [], []
        for seed in SEEDS:
            m = {}
            for h in [1, h_ck]:
                m[h] = make_model("state_cond", td["F"].shape[1], td["n_x"], 55)
                m[h].load_state_dict(torch.load(CACHE / f"Cext_s{sigma}_{ticker}_state_cond_h{h}_N55_seed{seed}.pt", map_location="cpu"))
                m[h].to(DEVICE)
            As.append(operators(m[1], td["F"], c1["idx_test"], td["n_x"], 55, DEVICE))
            te = ch["idx_test"]
            A1 = operators(m[1], td["F"], np.arange(te[0], min(te[-1] + h_ck, len(td["F"]))), td["n_x"], 55, DEVICE)
            cks.append(ck_ts(A1, operators(m[h_ck], td["F"], te, td["n_x"], 55, DEVICE), h_ck).ck_kl.values)
        op = operator_ts(np.mean(As, axis=0))
        ck = pd.Series(np.mean(cks, axis=0), index=td["dates"][ch["idx_test"]])
        dates = td["dates"][c1["idx_test"]]
        rows.append(pd.DataFrame({"date": dates, "ticker": ticker, "row_heterogeneity": op.row_heterogeneity,
                                  "row_entropy": op.row_entropy, "dobrushin": op.dobrushin,
                                  f"ck_error_h{h_ck}": ck.reindex(dates).values}))
        print(f"  [cross asset] {ticker}: {len(op)} days")
    pd.concat(rows).to_csv(OUT / "cross_asset_diagnostics.csv", index=False)


def run():
    CACHE.mkdir(parents=True, exist_ok=True)
    grid("baseline", "Aext", 1.0, False)
    sigma = sigma_sweep()
    grid("swa_bestsigma", f"Cext_s{sigma}", sigma, True)
    higher_order(sigma)
    conditioning(sigma)
    cross_asset(sigma)


if __name__ == "__main__":
    steps = {"A": lambda: grid("baseline", "Aext", 1.0, False), "B": sigma_sweep, "C": lambda: grid("swa_bestsigma", "Cext_s1.0", 1.0, True),
             "D": lambda: higher_order(1.0), "E": lambda: conditioning(1.0), "F": lambda: cross_asset(1.0)}
    CACHE.mkdir(parents=True, exist_ok=True)
    for s in sys.argv[1:] or ["all"]:
        run() if s == "all" else steps[s]()
