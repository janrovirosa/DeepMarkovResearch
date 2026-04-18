"""Hyperparameter sweep utilities for multi-asset experiments."""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from .bins import build_all_configs, compute_X_t
from .data import build_all_splits, load_master_dataset
from .eval import evaluate_model
from .models import StateConditionedNet
from .train import build_loaders, cache_model, is_cached, load_cached_model, train_one_run


def sweep_sigma(
    ticker: str,
    h: int,
    N: int,
    sigma_list: List[float],
    n_seeds: int,
    device: torch.device | None = None,
    max_epochs: int = 100,
    patience: int = 10,
    cache_dir: str | Path = "results_multiasset/sigma_sweep_cache",
) -> pd.DataFrame:
    """Train state_cond at each sigma × seed combination.

    Returns a DataFrame with columns: sigma, seed, nll_val, nll_test.
    Prints the sigma that minimises mean validation NLL.

    Parameters
    ----------
    ticker   : target stock ticker (pre-standardised features expected)
    h        : forecast horizon
    N        : number of output bins (requested; actual may differ)
    sigma_list : list of Gaussian soft-label bandwidths to sweep
    n_seeds  : number of seeds drawn from [42, 7, 123] (in that order)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    seeds = [42, 7, 123][:n_seeds]

    # ── Load data (once) ──
    prices, F_raw, _ = load_master_dataset(ticker=ticker)
    F_normed = F_raw  # already standardised

    splits = build_all_splits(prices, horizons=[h])
    train_end = int(splits[h]["idx_train"][-1]) + 1
    X_t_all, N_XT, edges_xt = compute_X_t(prices, N_XT_target=55, train_end=train_end)

    configs = build_all_configs(
        prices, F_normed, X_t_all,
        horizons=[h],
        n_bins_list=[N],
        N_XT=N_XT,
        edges_xt=edges_xt,
        splits=splits,
        sigma_anchor=1.0,
    )
    cfg_dict = configs[(h, N)]
    N_actual = cfg_dict["N_actual"]
    n_feat = F_normed.shape[1]

    train_loader, val_loader, test_loader = build_loaders(cfg_dict, F_normed, X_t_all)

    rows = []
    for sigma in sigma_list:
        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            cache_path = cache_dir / f"{ticker}_h{h}_N{N}_sigma{sigma}_seed{seed}.pt"
            model = StateConditionedNet(n_feat=n_feat, n_xt_states=N_XT, n_output=N_actual)

            if is_cached(cache_path):
                model = load_cached_model(model, cache_path)
            else:
                state_dict, _ = train_one_run(
                    model, train_loader, val_loader,
                    N=N_actual, sigma=sigma, device=device,
                    max_epochs=max_epochs, patience=patience,
                )
                cache_model(state_dict, cache_path)

            model = model.to(device)
            val_metrics = evaluate_model(model, val_loader, N=N_actual, device=device)
            test_metrics = evaluate_model(model, test_loader, N=N_actual, device=device)
            nll_val = -val_metrics["mean_ll"]
            nll_test = -test_metrics["mean_ll"]

            rows.append({"sigma": sigma, "seed": seed, "nll_val": nll_val, "nll_test": nll_test})
            print(f"  sweep  {ticker} h={h} N={N} sigma={sigma} seed={seed} "
                  f"→ nll_val={nll_val:.4f}  nll_test={nll_test:.4f}")

    df = pd.DataFrame(rows)
    best_sigma = df.groupby("sigma")["nll_val"].mean().idxmin()
    print(f"\n[sweep_sigma] Best sigma on {ticker} h={h} N={N}: {best_sigma}  "
          f"(mean nll_val={df.groupby('sigma')['nll_val'].mean()[best_sigma]:.4f})")
    return df
