import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

EPS = 1e-8


def evaluate(model, loader, n, device):
    # mean log-likelihood, accuracy, mean |E[bin] - true bin|, and per-sample log-likelihood
    model.eval()
    lps, preds, ys, exps = [], [], [], []
    bins = torch.arange(n, dtype=torch.float32, device=device)
    with torch.no_grad():
        for f, x, y in loader:
            f, x, y = f.to(device), x.to(device), y.to(device)
            logits = model(f, x)
            p = F.softmax(logits, dim=1)
            lps.append(torch.log(p[torch.arange(len(y), device=device), y] + EPS).cpu())
            preds.append(logits.argmax(1).cpu())
            ys.append(y.cpu())
            exps.append((p * bins).sum(1).cpu())
    lp, pred, y, ex = torch.cat(lps), torch.cat(preds), torch.cat(ys), torch.cat(exps)
    return {"mean_ll": lp.mean().item(), "accuracy": (pred == y).float().mean().item(),
            "severity": torch.abs(ex - y.float()).mean().item(), "loglik": lp.numpy()}


def predict(model, loader, device):
    model.eval()
    out = []
    with torch.no_grad():
        for f, x, _ in loader:
            out.append(F.softmax(model(f.to(device), x.to(device)), dim=1).cpu().numpy())
    return np.concatenate(out)


# count based baselines

def counts(s, y, n_x, n_y):
    C = np.zeros((n_x, n_y))
    np.add.at(C, (s, y), 1)
    return C


def marginal(y, n):
    c = np.bincount(y, minlength=n).astype(np.float64)
    return c / c.sum()


def additive(s, y, n_x, n_y, alpha):
    C = counts(s, y, n_x, n_y) + alpha
    return C / C.sum(axis=1, keepdims=True)


def backoff(s, y, n_x, n_y, alpha, tau, marg):
    # shrink each row toward the marginal, less so for states seen often
    C = counts(s, y, n_x, n_y)
    lam = C.sum(axis=1) / (C.sum(axis=1) + tau)
    return lam[:, None] * additive(s, y, n_x, n_y, alpha) + (1 - lam[:, None]) * marg[None, :]


def baselines(c, X, n_x, alphas, taus):
    # marginal, additive and backoff estimators, alpha / tau picked on validation
    y, n = c["y_all"], c["N_actual"]
    tr, va, te = c["idx_train"], c["idx_val"], c["idx_test"]
    marg = marginal(y[tr], n)
    ll = lambda P, idx: float(np.log(P[X[idx], y[idx]] + EPS).mean())

    best_add = max(alphas, key=lambda a: ll(additive(X[tr], y[tr], n_x, n, a), va))
    P_add = additive(X[tr], y[tr], n_x, n, best_add)
    best_bk = max(((a, t) for a in alphas for t in taus),
                  key=lambda at: ll(backoff(X[tr], y[tr], n_x, n, *at, marg), va))
    P_bk = backoff(X[tr], y[tr], n_x, n, *best_bk, marg)

    out = {"marginal": {"matrix": marg, "loglik": np.log(marg[y[te]] + EPS),
                        "val_ll": float(np.log(marg[y[va]] + EPS).mean()),
                        "accuracy": float((np.argmax(marg) == y[te]).mean()), "severity": np.nan}}
    for name, P, extra in [("additive", P_add, {"alpha": best_add}),
                           ("backoff", P_bk, {"alpha": best_bk[0], "tau": best_bk[1]})]:
        out[name] = {"matrix": P, "loglik": np.log(P[X[te], y[te]] + EPS), "val_ll": ll(P, va),
                     "accuracy": float((P[X[te]].argmax(1) == y[te]).mean()),
                     "severity": float(np.abs((P[X[te]] * np.arange(n)).sum(1) - y[te]).mean()), **extra}
    for r in out.values():
        r["test_ll"] = float(r["loglik"].mean())
    return out


# sparsity of empirical counts

def label_sparsity(y_train, n, thresholds=(5, 10)):
    c = np.bincount(y_train, minlength=n)
    return {"counts": c, "effective": int((c > 0).sum()),
            "frac_below": {k: float((c < k).mean()) for k in thresholds}}


def transition_sparsity(x, y, n_x, n_y):
    C = counts(x, y, n_x, n_y).astype(np.int64)
    rows = C.sum(axis=1)
    nz = (C > 0).sum(axis=1)
    P = C[rows > 0] / rows[rows > 0, None]
    ent = [-(p[p > 0] * np.log(p[p > 0])).sum() for p in P]
    return {"C": C, "row_sums": rows,
            "frac_cells_zero": float((C == 0).mean()), "frac_cells_lt5": float((C < 5).mean()),
            "frac_cells_lt10": float((C < 10).mean()), "frac_rows_lt5": float((rows < 5).mean()),
            "frac_rows_lt10": float((rows < 10).mean()),
            "median_nonzero_per_row": float(np.median(nz)), "p90_nonzero_per_row": float(np.percentile(nz, 90)),
            "median_row_entropy_empirical": float(np.median(ent)) if ent else np.nan,
            "median_row_maxprob_empirical": float(np.median(P.max(axis=1))) if len(P) else np.nan}


# operator diagnostics, A has shape (N, N) or (T, N, N)

def compose(A1, h):
    # A1[t] @ A1[t+1] @ ... @ A1[t+h-1]
    out = np.zeros((len(A1) - h + 1,) + A1.shape[1:])
    for t in range(len(out)):
        m = A1[t].astype(np.float64)
        for k in range(1, h):
            m = m @ A1[t + k].astype(np.float64)
        out[t] = m
    return out


def ck_errors(A_direct, A_composed, weights=None):
    # row-wise KL(direct || composed) and TV, averaged over rows
    P, Q = A_direct.astype(np.float64), A_composed.astype(np.float64)
    Pc, Qc = np.clip(P, EPS, None), np.clip(Q, EPS, None)
    kl = (Pc * np.log(Pc / Qc)).sum(axis=2)
    tv = 0.5 * np.abs(P - Q).sum(axis=2)
    if weights is None:
        mean_kl, mean_tv = float(kl.mean()), float(tv.mean())
    else:
        w = weights / (weights.sum() + EPS)
        mean_kl, mean_tv = float((kl * w).sum(1).mean()), float((tv * w).sum(1).mean())
    return {"mean_kl": mean_kl, "mean_tv": mean_tv,
            "frobenius": float(np.sqrt(((P - Q) ** 2).sum(axis=(1, 2))).mean()),
            "per_time_kl": kl.mean(axis=1), "per_time_tv": tv.mean(axis=1)}


def row_tv(A):
    # pairwise total variation between rows, shape (..., N, N)
    return 0.5 * np.abs(A[..., :, None, :] - A[..., None, :, :]).sum(-1)


def dobrushin(A):
    # largest TV distance between two rows
    if A.ndim == 2:
        return float(row_tv(A).max())
    return np.array([row_tv(a).max() for a in A])


def row_heterogeneity(A):
    # mean TV distance over row pairs i < j, 0 when the state is ignored
    iu = np.triu_indices(A.shape[-1], k=1)
    if A.ndim == 2:
        return float(row_tv(A)[iu].mean())
    return np.array([row_tv(a)[iu].mean() for a in A])


def row_entropy(A):
    Ac = np.clip(A, EPS, None)
    e = -(Ac * np.log(Ac)).sum(axis=-1).mean(axis=-1)
    return float(e) if A.ndim == 2 else e


def spectral_proxy(A, lazy=0.1, iters=200):
    # largest singular value of the lazy chain after removing its stationary part, lower = faster mixing
    def one(a):
        n = len(a)
        a = (1 - lazy) * a + lazy * np.ones((n, n)) / n
        pi = np.ones(n) / n
        for _ in range(iters):
            pi = a.T @ pi
            pi /= pi.sum()
        return float(np.linalg.svd(a - np.ones((n, 1)) @ pi[None, :], compute_uv=False)[0])
    return one(A) if A.ndim == 2 else np.array([one(a) for a in A])


def operator_ts(A):
    return pd.DataFrame({"timestep": np.arange(len(A)), "row_heterogeneity": row_heterogeneity(A),
                         "row_entropy": row_entropy(A), "dobrushin": dobrushin(A)})


def ck_ts(A1, Ah, h):
    comp = compose(A1, h)
    T = min(len(comp), len(Ah))
    e = ck_errors(Ah[:T], comp[:T])
    return pd.DataFrame({"timestep": np.arange(T), "ck_kl": e["per_time_kl"], "ck_tv": e["per_time_tv"]})


def snapshot_times(A):
    # calm = 10th percentile Dobrushin, bearish / bullish = lowest / highest mean predicted bin
    d = dobrushin(A)
    wmean = (A.astype(np.float64) * np.arange(A.shape[2])).sum(axis=2).mean(axis=1)
    calm = int(np.where(d == np.percentile(d, 10, method="nearest"))[0][0])
    return {"calm": calm, "bearish": int(np.argmin(wmean)), "bullish": int(np.argmax(wmean))}


# calibration

def pit(probs, y, seed=42):
    # randomized PIT for discrete forecasts, uniform if calibrated
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(probs, axis=1)
    out = np.zeros(len(y))
    for i, yi in enumerate(y):
        out[i] = rng.uniform(cdf[i, yi - 1] if yi > 0 else 0.0, cdf[i, yi])
    return out


def event_prob(probs, event):
    return probs[:, np.array([bool(event(b)) for b in range(probs.shape[1])])].sum(axis=1)


def ece(probs, y, event, n_bins=10):
    # expected calibration error for a binary event such as "negative return"
    p, hit = event_prob(probs, event), event(y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    conf, acc, cnt = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    for b in range(n_bins):
        m = (p >= edges[b]) & (p < edges[b + 1])
        if b == n_bins - 1:
            m |= p == 1.0
        cnt[b] = m.sum()
        if cnt[b] > 0:
            conf[b], acc[b] = p[m].mean(), hit[m].mean()
    return float(np.abs(conf - acc).dot(cnt) / len(y)), conf, acc


def brier(probs, y, event):
    return float(((event_prob(probs, event) - event(y).astype(float)) ** 2).mean())


# entropy floor and capacity

def entropy_floor(C, eps=1e-12):
    # expected row entropy of the empirical transition matrix on train
    C = C.astype(np.float64)
    rows = C.sum(axis=1)
    if rows.sum() == 0:
        return np.nan
    P = (C + eps) / (rows[:, None] + C.shape[1] * eps)
    return float((rows / rows.sum() * -(P * np.log(np.clip(P, eps, None))).sum(axis=1)).sum())


def mir(nll_model, h_irr):
    # share of the empirical entropy floor the model gets below, negative = worse than the floor
    return float((h_irr - nll_model) / h_irr) if h_irr and np.isfinite(h_irr) else np.nan


def spectral_norm_product(model):
    norms = [float(torch.linalg.svdvals(m.weight.detach().float())[0])
             for m in model.modules() if isinstance(m, torch.nn.Linear)]
    return float(np.prod(norms)) if norms else np.nan


def block_bootstrap(loglik, block=21, n_boot=500, alpha=0.05, seed=42):
    # circular block bootstrap CI for the mean log-likelihood
    rng = np.random.default_rng(seed)
    T = len(loglik)
    means = np.zeros(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, T, size=int(np.ceil(T / block)))
        idx = np.concatenate([(np.arange(block) + s) % T for s in starts])[:T]
        means[b] = loglik[idx].mean()
    return float(loglik.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))
