#!/usr/bin/env python3
"""Multi-asset experiment runner.

Experiments:
  A — Baseline (10 tickers, h in {1,2,5,10}, state_cond + state_free, seed=42)
  B — Sigma sweep on JPM h=1 N=55 (5 sigma values × 3 seeds)
  C — SWA + best sigma (10 tickers, same grid as A)
  D — Higher-order k ablation (JPM, k in {1,2,3,5}, h=1, 3 seeds)
  E — Conditioning regime (JPM, regimes {own_only,macro_only,full}, h=1, 3 seeds)

All weights are cached.  Re-runs load from cache and skip training.
Failed experiments write a FAILED marker and continue.

Entry points
------------
  run_all_experiments(cfg)  — importable callable for notebook use
  __main__                  — calls run_all_experiments(make_config())
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.bins import build_all_configs, compute_X_t
from scripts.config import ExperimentConfig, make_config
from scripts.data import build_all_splits, build_feature_subset, load_master_dataset
from scripts.eval import evaluate_model
from scripts.models import StateConditionedNet, StateFreeNet
from scripts.models_extensions import (
    HigherOrderStateConditionedNet,
    build_loaders_higher_order,
)
from scripts.results_writer import save_results
from scripts.sweeps import sweep_sigma
from scripts.train import (
    MasterDataset,
    build_loaders,
    cache_model,
    is_cached,
    load_cached_model,
    train_one_run,
)

# ---------------------------------------------------------------------------
# Module-level constants (hardware / data scope — not in ExperimentConfig)
# ---------------------------------------------------------------------------

TARGET_STOCKS = ["JPM", "C", "WFC", "GS", "MS", "PNC", "USB", "FITB", "MTB", "BAC"]
N_BINS = 55
RESULTS_DIR = Path("results_multiasset")
CACHE_DIR = RESULTS_DIR / "_cache"
try:
    import torch_directml
    DEVICE = torch_directml.device()
    print(f"[device] Using AMD GPU via DirectML: {DEVICE}")
except ImportError:
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[device] DirectML not found, falling back to: {DEVICE}")


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------

def setup_ticker(ticker: str, cfg: ExperimentConfig) -> Dict:
    """Load and prepare one ticker; return a dict with all pipeline artefacts."""
    prices, F_raw, feature_cols = load_master_dataset(ticker=ticker)
    F_normed = F_raw  # already standardised in dataset_multiasset/

    splits = build_all_splits(prices, cfg.horizons)
    train_end = int(splits[min(cfg.horizons)]["idx_train"][-1]) + 1
    X_t_all, N_XT, edges_xt = compute_X_t(prices, N_XT_target=55, train_end=train_end)

    configs = build_all_configs(
        prices, F_normed, X_t_all,
        horizons=cfg.horizons,
        n_bins_list=[N_BINS],
        N_XT=N_XT,
        edges_xt=edges_xt,
        splits=splits,
        sigma_anchor=cfg.sigma_anchor,
    )
    return {
        "prices": prices,
        "F_normed": F_normed,
        "feature_cols": feature_cols,
        "n_feat": F_normed.shape[1],
        "X_t_all": X_t_all,
        "N_XT": N_XT,
        "splits": splits,
        "configs": configs,
    }


def _marginal_nll(N: int) -> float:
    """NLL of uniform marginal over N bins."""
    return float(np.log(N))


# ---------------------------------------------------------------------------
# Single (ticker, model_type, h, seed) training/eval
# ---------------------------------------------------------------------------

def run_one(
    ticker: str,
    model_type: str,
    h: int,
    seed: int,
    td: Dict,
    sigma: float,
    use_swa: bool,
    cfg: ExperimentConfig,
    experiment_prefix: str,
) -> Dict:
    """Train (or load from cache) one model; return metrics row dict."""
    cfg_dict = td["configs"][(h, N_BINS)]
    N_actual = cfg_dict["N_actual"]
    n_feat = td["n_feat"]
    N_XT = td["N_XT"]
    F_normed = td["F_normed"]
    X_t_all = td["X_t_all"]

    train_loader, val_loader, test_loader = build_loaders(cfg_dict, F_normed, X_t_all)

    torch.manual_seed(seed)
    np.random.seed(seed)

    if model_type == "state_cond":
        model = StateConditionedNet(n_feat, N_XT, N_actual)
    else:
        model = StateFreeNet(n_feat, N_actual)

    cache_key = f"{experiment_prefix}_{ticker}_{model_type}_h{h}_seed{seed}.pt"
    cache_path = CACHE_DIR / cache_key

    if is_cached(cache_path):
        model = load_cached_model(model, cache_path)
    else:
        state_dict, _ = train_one_run(
            model, train_loader, val_loader,
            N=N_actual, sigma=sigma, device=DEVICE,
            use_swa=use_swa, swa_epochs=cfg.swa_epochs,
            max_epochs=cfg.max_epochs, patience=cfg.patience,
        )
        cache_model(state_dict, cache_path)

    model = model.to(DEVICE)

    train_m = evaluate_model(model, train_loader, N_actual, DEVICE)
    val_m   = evaluate_model(model, val_loader,   N_actual, DEVICE)
    test_m  = evaluate_model(model, test_loader,  N_actual, DEVICE)

    nll_train = -train_m["mean_ll"]
    nll_val   = -val_m["mean_ll"]
    nll_test  = -test_m["mean_ll"]
    delta     = nll_test - _marginal_nll(N_actual)

    print(f"  [{experiment_prefix}] {ticker} {model_type:10s} h={h} seed={seed} "
          f"-> nll_test={nll_test:.4f}  delta={delta:+.4f}")

    return {
        "model": model_type, "h": h, "seed": seed,
        "nll_train": nll_train, "nll_val": nll_val,
        "nll_test": nll_test, "delta_marginal": delta,
        "_model_obj": model,
        "_cache_path": str(cache_path),
    }


# ---------------------------------------------------------------------------
# Experiment A — Baseline
# ---------------------------------------------------------------------------

def run_experiment_A(cfg: ExperimentConfig):
    print("\n" + "=" * 60)
    print("EXPERIMENT A — Baseline")
    print("=" * 60)

    for ticker in TARGET_STOCKS:
        try:
            td = setup_ticker(ticker, cfg)
            rows = []
            weights = {}
            for h in cfg.horizons:
                for model_type in ["state_cond", "state_free"]:
                    try:
                        row = run_one(ticker, model_type, h, seed=cfg.seed, td=td,
                                      sigma=cfg.sigma_anchor, use_swa=False,
                                      cfg=cfg, experiment_prefix="A")
                        model_obj = row.pop("_model_obj")
                        row.pop("_cache_path")
                        rows.append(row)
                        weights[f"{model_type}_h{h}_seed{cfg.seed}"] = {
                            k: v.cpu() for k, v in model_obj.state_dict().items()
                        }
                    except Exception:
                        print(f"  [A] FAILED {ticker} {model_type} h={h}")
                        traceback.print_exc()
                        rows.append({"model": model_type, "h": h, "seed": cfg.seed,
                                     "nll_train": None, "nll_val": None,
                                     "nll_test": None, "delta_marginal": None})

            save_results(
                RESULTS_DIR, ticker, "baseline",
                config_dict={"sigma": cfg.sigma_anchor, "use_swa": False,
                             "swa_epochs": cfg.swa_epochs, "N": N_BINS,
                             "horizons": cfg.horizons},
                metrics_dict={"rows": rows, "weights": weights},
            )
        except Exception:
            print(f"  [A] FAILED setup for {ticker}")
            traceback.print_exc()
            failed_dir = RESULTS_DIR / ticker / "baseline"
            failed_dir.mkdir(parents=True, exist_ok=True)
            (failed_dir / "FAILED").write_text(traceback.format_exc())


# ---------------------------------------------------------------------------
# Experiment B — Sigma sweep (JPM only) → returns BEST_SIGMA
# ---------------------------------------------------------------------------

def run_experiment_B(cfg: ExperimentConfig) -> float:
    print("\n" + "=" * 60)
    print("EXPERIMENT B — Sigma sweep on JPM")
    print("=" * 60)

    sigma_list = [0.5, 1.0, 1.5, 2.0, 3.0]
    sweep_cache = CACHE_DIR / "sweep_sigma"
    sweep_results_path = RESULTS_DIR / "JPM" / "sigma_sweep" / "sweep_results.csv"

    try:
        df = sweep_sigma(
            "JPM", h=1, N=N_BINS,
            sigma_list=sigma_list, n_seeds=len(cfg.seeds),
            device=DEVICE,
            max_epochs=cfg.max_epochs, patience=cfg.patience,
            cache_dir=sweep_cache,
        )
        sweep_results_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(sweep_results_path, index=False)
        best_sigma = float(df.groupby("sigma")["nll_val"].mean().idxmin())
        print(f"\n[B] BEST_SIGMA = {best_sigma}")
        return best_sigma
    except Exception:
        print("[B] FAILED — falling back to sigma_anchor from cfg")
        traceback.print_exc()
        return cfg.sigma_anchor


# ---------------------------------------------------------------------------
# Experiment C — SWA + best sigma
# ---------------------------------------------------------------------------

def run_experiment_C(cfg: ExperimentConfig, best_sigma: float):
    print("\n" + "=" * 60)
    print(f"EXPERIMENT C — SWA + best_sigma={best_sigma}")
    print("=" * 60)

    for ticker in TARGET_STOCKS:
        try:
            td = setup_ticker(ticker, cfg)
            rows = []
            weights = {}
            for h in cfg.horizons:
                for model_type in ["state_cond", "state_free"]:
                    try:
                        row = run_one(ticker, model_type, h, seed=cfg.seed, td=td,
                                      sigma=best_sigma, use_swa=True,
                                      cfg=cfg, experiment_prefix=f"C_s{best_sigma}")
                        model_obj = row.pop("_model_obj")
                        row.pop("_cache_path")
                        rows.append(row)
                        weights[f"{model_type}_h{h}_seed{cfg.seed}"] = {
                            k: v.cpu() for k, v in model_obj.state_dict().items()
                        }
                    except Exception:
                        print(f"  [C] FAILED {ticker} {model_type} h={h}")
                        traceback.print_exc()
                        rows.append({"model": model_type, "h": h, "seed": cfg.seed,
                                     "nll_train": None, "nll_val": None,
                                     "nll_test": None, "delta_marginal": None})

            save_results(
                RESULTS_DIR, ticker, "swa_bestsigma",
                config_dict={"sigma": best_sigma, "use_swa": True,
                             "swa_epochs": cfg.swa_epochs, "N": N_BINS,
                             "horizons": cfg.horizons},
                metrics_dict={"rows": rows, "weights": weights},
            )
        except Exception:
            print(f"  [C] FAILED setup for {ticker}")
            traceback.print_exc()


# ---------------------------------------------------------------------------
# Experiment D — Higher-order k ablation (JPM only)
# ---------------------------------------------------------------------------

def run_experiment_D(cfg: ExperimentConfig, best_sigma: float):
    print("\n" + "=" * 60)
    print("EXPERIMENT D — Higher-order k ablation (JPM)")
    print("=" * 60)

    ticker = "JPM"
    h = 1
    td = setup_ticker(ticker, ExperimentConfig(horizons=[h]))
    cfg_dict = td["configs"][(h, N_BINS)]
    N_actual = cfg_dict["N_actual"]
    n_feat = td["n_feat"]
    N_XT = td["N_XT"]
    F_normed = td["F_normed"]
    X_t_all = td["X_t_all"]

    for k in [1, 2, 3, 5]:
        for seed in cfg.seeds:
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)

                cache_path = CACHE_DIR / f"D_{ticker}_ho_k{k}_h{h}_seed{seed}.pt"
                train_lo, val_lo, test_lo = build_loaders_higher_order(
                    cfg_dict, F_normed, X_t_all, k=k)
                model = HigherOrderStateConditionedNet(n_feat, N_XT, N_actual, k=k)

                if is_cached(cache_path):
                    model = load_cached_model(model, cache_path)
                else:
                    state_dict, _ = train_one_run(
                        model, train_lo, val_lo,
                        N=N_actual, sigma=best_sigma, device=DEVICE,
                        use_swa=True, swa_epochs=cfg.swa_epochs,
                        max_epochs=cfg.max_epochs, patience=cfg.patience,
                    )
                    cache_model(state_dict, cache_path)

                model = model.to(DEVICE)
                val_m  = evaluate_model(model, val_lo,  N_actual, DEVICE)
                test_m = evaluate_model(model, test_lo, N_actual, DEVICE)
                nll_val  = -val_m["mean_ll"]
                nll_test = -test_m["mean_ll"]
                delta = nll_test - _marginal_nll(N_actual)
                print(f"  [D] k={k} seed={seed} -> nll_test={nll_test:.4f}  delta={delta:+.4f}")

                save_results(
                    RESULTS_DIR, ticker, "higher_order",
                    config_dict={"k": k, "sigma": best_sigma, "use_swa": True,
                                 "swa_epochs": cfg.swa_epochs, "h": h, "N": N_BINS},
                    metrics_dict={"rows": [{"model": f"ho_k{k}", "h": h, "seed": seed,
                                            "nll_train": None, "nll_val": nll_val,
                                            "nll_test": nll_test, "delta_marginal": delta}]},
                )
            except Exception:
                print(f"  [D] FAILED k={k} seed={seed}")
                traceback.print_exc()


# ---------------------------------------------------------------------------
# Experiment E — Conditioning regime (JPM only)
# ---------------------------------------------------------------------------

def run_experiment_E(cfg: ExperimentConfig, best_sigma: float):
    print("\n" + "=" * 60)
    print("EXPERIMENT E — Conditioning regime (JPM)")
    print("=" * 60)

    ticker = "JPM"
    h = 1
    td = setup_ticker(ticker, ExperimentConfig(horizons=[h]))
    cfg_dict_base = td["configs"][(h, N_BINS)]
    feature_cols = td["feature_cols"]
    F_normed_full = td["F_normed"]
    X_t_all = td["X_t_all"]
    N_XT = td["N_XT"]
    N_actual = cfg_dict_base["N_actual"]

    from torch.utils.data import DataLoader

    def _make_regime_loaders(F_sub):
        tr = MasterDataset(F_sub, X_t_all, cfg_dict_base["y_all"], cfg_dict_base["idx_train"])
        va = MasterDataset(F_sub, X_t_all, cfg_dict_base["y_all"], cfg_dict_base["idx_val"])
        te = MasterDataset(F_sub, X_t_all, cfg_dict_base["y_all"], cfg_dict_base["idx_test"])
        return (DataLoader(tr, 256, shuffle=True),
                DataLoader(va, 512, shuffle=False),
                DataLoader(te, 512, shuffle=False))

    for regime in ["own_only", "macro_only", "full"]:
        F_sub, cols_sub = build_feature_subset(
            F_normed_full, feature_cols, regime, target_ticker=ticker)
        n_feat_sub = F_sub.shape[1]
        print(f"\n  regime={regime}  n_feat={n_feat_sub}")

        train_lo, val_lo, test_lo = _make_regime_loaders(F_sub)

        for seed in cfg.seeds:
            try:
                torch.manual_seed(seed)
                np.random.seed(seed)

                cache_path = CACHE_DIR / f"E_{ticker}_regime{regime}_h{h}_seed{seed}.pt"
                model = StateConditionedNet(n_feat_sub, N_XT, N_actual)

                if is_cached(cache_path):
                    model = load_cached_model(model, cache_path)
                else:
                    state_dict, _ = train_one_run(
                        model, train_lo, val_lo,
                        N=N_actual, sigma=best_sigma, device=DEVICE,
                        use_swa=True, swa_epochs=cfg.swa_epochs,
                        max_epochs=cfg.max_epochs, patience=cfg.patience,
                    )
                    cache_model(state_dict, cache_path)

                model = model.to(DEVICE)
                val_m  = evaluate_model(model, val_lo,  N_actual, DEVICE)
                test_m = evaluate_model(model, test_lo, N_actual, DEVICE)
                nll_val  = -val_m["mean_ll"]
                nll_test = -test_m["mean_ll"]
                delta = nll_test - _marginal_nll(N_actual)
                print(f"  [E] regime={regime} seed={seed} -> nll_test={nll_test:.4f}  delta={delta:+.4f}")

                save_results(
                    RESULTS_DIR, ticker, "conditioning_regime",
                    config_dict={"regime": regime, "sigma": best_sigma,
                                 "use_swa": True, "swa_epochs": cfg.swa_epochs,
                                 "h": h, "N": N_BINS, "n_feat": n_feat_sub},
                    metrics_dict={"rows": [{"model": f"state_cond_{regime}", "h": h, "seed": seed,
                                            "nll_train": None, "nll_val": nll_val,
                                            "nll_test": nll_test, "delta_marginal": delta}]},
                )
            except Exception:
                print(f"  [E] FAILED regime={regime} seed={seed}")
                traceback.print_exc()


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary(best_sigma: float):
    import pandas as pd

    print("\n" + "=" * 60)
    print("Writing summary.md")
    print("=" * 60)

    lines = ["# Multi-Asset Experiment Summary\n"]

    lines.append("## Table 1 — Baseline NLL (state_cond vs state_free)\n")
    rows_A = []
    for ticker in TARGET_STOCKS:
        p = RESULTS_DIR / ticker / "baseline" / "nll_metrics.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["ticker"] = ticker
            rows_A.append(df)
    if rows_A:
        df_A = pd.concat(rows_A)
        tbl = df_A.groupby(["ticker", "model", "h"])[["nll_test", "delta_marginal"]].mean()
        lines.append(tbl.to_markdown())
    else:
        lines.append("_No results found._")
    lines.append("\n")

    lines.append("## Table 2 — Sigma Sweep on JPM (h=1, N=55)\n")
    p2 = RESULTS_DIR / "JPM" / "sigma_sweep" / "sweep_results.csv"
    if p2.exists():
        df2 = pd.read_csv(p2)
        tbl2 = df2.groupby("sigma")[["nll_val", "nll_test"]].mean()
        lines.append(tbl2.to_markdown())
    else:
        lines.append("_No sigma sweep results found._")
    lines.append("\n")

    lines.append(
        f"## Table 3 — SWA + best_sigma={best_sigma} Improvement over Baseline\n\n"
        f"> **Methodological note**: `BEST_SIGMA` ({best_sigma}) was selected via sweep on JPM "
        f"only (Experiment B) and applied uniformly to all 10 tickers in Experiment C. "
        f"This is intentional — per-ticker sigma selection on validation data would risk "
        f"overfitting the bandwidth to individual stocks.\n"
    )
    rows_C = []
    for ticker in TARGET_STOCKS:
        pA = RESULTS_DIR / ticker / "baseline" / "nll_metrics.csv"
        pC = RESULTS_DIR / ticker / "swa_bestsigma" / "nll_metrics.csv"
        if pA.exists() and pC.exists():
            dfA = pd.read_csv(pA)
            dfC = pd.read_csv(pC)
            for _, rowA in dfA.iterrows():
                match = dfC[(dfC.model == rowA.model) & (dfC.h == rowA.h)]
                if not match.empty:
                    rows_C.append({
                        "ticker": ticker, "model": rowA.model, "h": rowA.h,
                        "nll_test_A": rowA.nll_test,
                        "nll_test_C": match.iloc[0].nll_test,
                        "delta": match.iloc[0].nll_test - rowA.nll_test,
                    })
    if rows_C:
        df_C = pd.DataFrame(rows_C)
        lines.append(df_C.groupby(["ticker", "model"])["delta"].mean().to_markdown())
    else:
        lines.append("_No SWA results found._")
    lines.append("\n")

    lines.append("## Table 4 — Higher-order k Ablation (JPM, h=1)\n")
    p4 = RESULTS_DIR / "JPM" / "higher_order" / "nll_metrics.csv"
    if p4.exists():
        df4 = pd.read_csv(p4)
        lines.append(df4.groupby("model")[["nll_val", "nll_test", "delta_marginal"]].mean().to_markdown())
    else:
        lines.append("_No higher-order results found._")
    lines.append("\n")

    lines.append("## Table 5 — Conditioning Regime Comparison (JPM, h=1)\n")
    p5 = RESULTS_DIR / "JPM" / "conditioning_regime" / "nll_metrics.csv"
    if p5.exists():
        df5 = pd.read_csv(p5)
        lines.append(df5.groupby("model")[["nll_val", "nll_test", "delta_marginal"]].mean().to_markdown())
    else:
        lines.append("_No regime results found._")
    lines.append("\n")

    lines.append("## Summary\n")
    lines.append(
        "- **Baseline**: State-conditioned models trained on all 10 bank stocks with sigma=1.0.\n"
        f"- **Sigma sweep**: Performed on JPM only; best sigma = {best_sigma}.\n"
        "- **SWA**: Rolling average of last 10 checkpoints applied with best sigma.\n"
        "- **Higher-order Markov**: k-step history ablation on JPM.\n"
        "- **Regime conditioning**: Compares full features vs macro-only vs own-stock lags.\n"
    )

    summary_path = RESULTS_DIR / "summary.md"
    summary_path.write_text("\n".join(lines))
    print(f"  Written: {summary_path}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_all_experiments(cfg: ExperimentConfig):
    """Run the full multi-asset experiment suite (A → E) and write summary.md.

    Parameters
    ----------
    cfg : ExperimentConfig
        Controls horizons, seeds, swa_epochs, sigma_anchor, max_epochs, patience.
    """
    print(f"Results directory: {RESULTS_DIR.resolve()}")
    print(f"Device: {DEVICE}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    run_experiment_A(cfg)
    best_sigma = run_experiment_B(cfg)
    run_experiment_C(cfg, best_sigma)
    run_experiment_D(cfg, best_sigma)
    run_experiment_E(cfg, best_sigma)
    write_summary(best_sigma)

    print("\nAll experiments complete.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all_experiments(make_config())
