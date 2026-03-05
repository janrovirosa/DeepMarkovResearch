"""Plotting utilities for MathFrameworkExperiments.

Every public function:
  - accepts data + out_dir
  - saves .png and .pdf
  - returns the matplotlib Figure

Standard style: white background, labeled axes, title, legend.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def save_fig(fig: plt.Figure, path_stem: str | Path, dpi: int = 150) -> None:
    """Save figure as both .png and .pdf."""
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path_stem) + ".png", dpi=dpi, bbox_inches="tight")
    fig.savefig(str(path_stem) + ".pdf", bbox_inches="tight")


def _set_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


# ---------------------------------------------------------------------------
# Section A: Degeneracy
# ---------------------------------------------------------------------------

def plot_sparsity_vs_N(
    sparsity_data: Dict,   # {(h, N): {"frac_cells_lt5": float, ...}}
    horizons: list,
    n_bins_list: list,
    threshold: int,
    out_dir: str | Path,
) -> plt.Figure:
    """Fraction of zero/sparse cells vs N, one curve per horizon.

    Uses frac_cells_lt5 (cell count < 5) as the primary cell-level sparsity metric.
    """
    _set_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    markers = ["o", "s", "^", "D"]
    for i, h in enumerate(horizons):
        fracs = [sparsity_data.get((h, N), {}).get("frac_cells_lt5", np.nan)
                 for N in n_bins_list]
        ax.plot(n_bins_list, fracs, marker=markers[i % len(markers)], label=f"h={h}")

    ax.set_xlabel("N (number of output bins)", fontsize=12)
    ax.set_ylabel("Fraction of cells C[i,j] < 5", fontsize=12)
    ax.set_title("Transition-Cell Sparsity vs N\n(fraction of cells with < 5 observations)", fontsize=12)
    ax.legend(title="Horizon h")
    ax.set_ylim(0, 1)
    ax.set_xticks(n_bins_list)
    fig.tight_layout()
    save_fig(fig, Path(out_dir) / "sparsity_vs_N")
    return fig


def plot_transition_sparsity_table(
    df_sparsity: pd.DataFrame,
    threshold: int,
    out_dir: str | Path,
) -> plt.Figure:
    """Heatmap of cell-level sparsity (frac_cells_lt5) for each config.

    df_sparsity must have columns: config_type, h, N, frac_cells_lt5
    """
    _set_style()
    col = "frac_cells_lt5"
    cum_df = df_sparsity[df_sparsity["config_type"] == "cumulative"].copy()
    ck_df  = df_sparsity[df_sparsity["config_type"] == "ck"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [4, 1]})

    # Cumulative heatmap
    if len(cum_df) > 0 and col in cum_df.columns:
        pivot = cum_df.pivot(index="h", columns="N", values=col)
        sns.heatmap(pivot, ax=axes[0], annot=True, fmt=".2f", cmap="Reds",
                    vmin=0, vmax=1, cbar_kws={"label": "Frac cells < 5"})
        axes[0].set_title("Cumulative Configs: Fraction of cells C[i,j] < 5", fontsize=11)
        axes[0].set_xlabel("N (output bins)")
        axes[0].set_ylabel("Horizon h")

    # CK heatmap (N=55 fixed, vary h)
    if len(ck_df) > 0 and col in ck_df.columns:
        ck_pivot = ck_df.set_index("h")[col].to_frame().T
        sns.heatmap(ck_pivot, ax=axes[1], annot=True, fmt=".2f", cmap="Reds",
                    vmin=0, vmax=1, cbar=False)
        axes[1].set_title("CK (N=55)\nFrac cells < 5", fontsize=11)
        axes[1].set_xlabel("Horizon h")
        axes[1].set_ylabel("")
        axes[1].set_yticks([])

    fig.suptitle("Transition Cell Sparsity (fraction of cells with < 5 observations)", fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, Path(out_dir) / "transition_sparsity_heatmap")
    return fig


# ---------------------------------------------------------------------------
# Section B: CK
# ---------------------------------------------------------------------------

def plot_ck_error_summary(
    ck_df: pd.DataFrame,   # columns: model, h, mean_kl, mean_tv, frobenius
    out_dir: str | Path,
) -> plt.Figure:
    """Bar chart of mean KL and mean TV CK errors by model and horizon."""
    _set_style()
    models = ck_df["model"].unique()
    horizons = sorted(ck_df["h"].unique())
    n_h = len(horizons)
    n_m = len(models)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    x = np.arange(n_h)
    width = 0.8 / n_m

    for ax, metric, label in [
        (axes[0], "mean_kl", "Mean KL (CK error)"),
        (axes[1], "mean_tv", "Mean TV (CK error)"),
    ]:
        for i, model in enumerate(models):
            sub = ck_df[ck_df["model"] == model]
            vals = [sub[sub["h"] == h][metric].values[0] if len(sub[sub["h"] == h]) > 0 else np.nan
                    for h in horizons]
            ax.bar(x + i * width - (n_m - 1) * width / 2, vals, width, label=model)
        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in horizons])
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=8)

    fig.suptitle("Chapman–Kolmogorov Error by Model and Horizon\n"
                 "(direct A^(h) vs composed Π A^(1)_{t+k})", fontsize=12)
    fig.tight_layout()
    save_fig(fig, Path(out_dir) / "ck_error_summary")
    return fig


def plot_ck_time_series(
    ck_time_dict: Dict,  # {(model, h): np.ndarray per_time_kl}
    out_dir: str | Path,
    regime_windows: Optional[List[Tuple]] = None,
) -> plt.Figure:
    """Line plot of CK KL error over test period for each model × horizon."""
    _set_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    axes = axes.flatten()
    horizons = sorted(set(h for (_, h) in ck_time_dict.keys()))

    for ax_idx, h in enumerate(horizons[:4]):
        ax = axes[ax_idx]
        for (model, mh), series in ck_time_dict.items():
            if mh != h:
                continue
            label = model
            if "state_free" in model:
                label += " (degenerate rows)"
            ax.plot(series, label=label, alpha=0.8)
        if regime_windows:
            for (start, end, name) in regime_windows:
                ax.axvspan(start, end, alpha=0.15, color="red", label=name)
        ax.set_title(f"CK KL Error over Time  (h={h})")
        ax.set_xlabel("Test time index")
        ax.set_ylabel("Mean KL per time step")
        ax.legend(fontsize=8)

    for ax in axes[len(horizons):]:
        ax.set_visible(False)

    fig.suptitle("Time-Inhomogeneous CK Error over Test Period", fontsize=13)
    fig.tight_layout()
    save_fig(fig, Path(out_dir) / "ck_error_time_series")
    return fig


# ---------------------------------------------------------------------------
# Section C: Operator diagnostics
# ---------------------------------------------------------------------------

def _plot_diagnostic_series(
    series_dict: Dict[str, np.ndarray],
    ylabel: str,
    title: str,
    out_path: Path,
    regime_windows: Optional[List[Tuple]] = None,
) -> plt.Figure:
    _set_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    for model, series in series_dict.items():
        label = model
        if "state_free" in model:
            label += " (≈0 expected)"
        finite_mask = np.isfinite(series)
        if finite_mask.any():
            ax.plot(np.where(finite_mask)[0], series[finite_mask], label=label, alpha=0.8)
    if regime_windows:
        for (start, end, name) in regime_windows:
            ax.axvspan(start, end, alpha=0.15, color="red", label=f"Regime: {name}")
    ax.set_xlabel("Time index (full series)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_fig(fig, out_path)
    return fig


def plot_dobrushin_over_time(
    series_dict: Dict[str, np.ndarray],
    out_dir: str | Path,
    regime_windows=None,
) -> plt.Figure:
    return _plot_diagnostic_series(
        series_dict,
        ylabel="Dobrushin coefficient δ(A_t)",
        title="Dobrushin Coefficient over Time\n(δ=0.5·max‖A[i,:]−A[j,:]‖₁; higher = more contraction)",
        out_path=Path(out_dir) / "dobrushin_over_time",
        regime_windows=regime_windows,
    )


def plot_row_heterogeneity_over_time(
    series_dict: Dict[str, np.ndarray],
    out_dir: str | Path,
    regime_windows=None,
) -> plt.Figure:
    return _plot_diagnostic_series(
        series_dict,
        ylabel="Average pairwise TV between rows",
        title="Row Heterogeneity over Time\n(avg pairwise TV; higher = stronger state dependence)",
        out_path=Path(out_dir) / "row_heterogeneity_over_time",
        regime_windows=regime_windows,
    )


def plot_entropy_over_time(
    series_dict: Dict[str, np.ndarray],
    out_dir: str | Path,
    regime_windows=None,
) -> plt.Figure:
    return _plot_diagnostic_series(
        series_dict,
        ylabel="Mean row entropy H(A_t)",
        title="Mean Row Entropy over Time\n(higher = more uniform / less predictive transitions)",
        out_path=Path(out_dir) / "entropy_over_time",
        regime_windows=regime_windows,
    )


def plot_spectral_proxy_over_time(
    series_dict: Dict[str, np.ndarray],
    out_dir: str | Path,
    regime_windows=None,
) -> Optional[plt.Figure]:
    """Only plots if ≥50% of values are finite for at least one model."""
    for series in series_dict.values():
        if np.isfinite(series).mean() >= 0.5:
            break
    else:
        return None  # skip if all series have too many NaNs

    return _plot_diagnostic_series(
        series_dict,
        ylabel="σ_max(M)  (spectral mixing proxy)",
        title="Spectral Mixing Proxy over Time\n"
              "(σ_max of lazy deflated operator; lower = faster mixing)",
        out_path=Path(out_dir) / "spectral_proxy_over_time",
        regime_windows=regime_windows,
    )


def plot_At_heatmap_snapshot(
    A: np.ndarray,
    t_label: str,
    out_dir: str | Path,
    model_name: str = "",
) -> plt.Figure:
    """Heatmap of a single 55×55 A_t transition matrix."""
    _set_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(A, aspect="auto", cmap="Blues", vmin=0, vmax=A.max())
    plt.colorbar(im, ax=ax, label="P(next state | current state, F_t)")
    ax.set_title(f"A_t Transition Matrix — {t_label}" +
                 (f"\n(model: {model_name})" if model_name else ""))
    ax.set_xlabel("Output state X_{t+h}")
    ax.set_ylabel("Input state X_t")
    # Only label every 10th tick for readability
    ticks = np.arange(0, A.shape[0], 10)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    fig.tight_layout()
    safe_label = t_label.replace("/", "-").replace(" ", "_")
    save_fig(fig, Path(out_dir) / f"At_heatmap_{safe_label}")
    return fig


def plot_regime_panel(
    diagnostics_df: pd.DataFrame,
    regime_windows: List[Tuple],
    models: List[str],
    out_dir: str | Path,
) -> plt.Figure:
    """Multi-panel figure: 3 diagnostic time series + regime shading.

    diagnostics_df: columns = [time_idx, model, dobrushin, row_heterogeneity, row_entropy]
    regime_windows: list of (start_idx, end_idx, label)
    """
    _set_style()
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    metrics = [
        ("dobrushin",         "Dobrushin δ(A_t)"),
        ("row_heterogeneity", "Row Heterogeneity"),
        ("row_entropy",       "Row Entropy H(A_t)"),
    ]

    colors = plt.cm.tab10.colors
    for ax, (col, ylabel) in zip(axes, metrics):
        if col not in diagnostics_df.columns:
            ax.set_ylabel(ylabel)
            continue
        for i, model in enumerate(models):
            sub = diagnostics_df[diagnostics_df["model"] == model]
            label = model + (" (degenerate)" if "state_free" in model else "")
            ax.plot(sub["time_idx"], sub[col], label=label,
                    color=colors[i % len(colors)], alpha=0.8)
        for (start, end, name) in regime_windows:
            ax.axvspan(start, end, alpha=0.18, color="salmon", label=f"Regime: {name}")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Time index (full series)")
    axes[0].set_title("Operator Diagnostics over Time with Regime Windows", fontsize=13)
    fig.tight_layout()
    save_fig(fig, Path(out_dir) / "regime_diagnostic_panel")
    return fig


# ---------------------------------------------------------------------------
# Section D: Calibration
# ---------------------------------------------------------------------------

def plot_pit_histogram(
    pit_values: np.ndarray,
    model_name: str,
    out_dir: str | Path,
    n_bins: int = 20,
) -> plt.Figure:
    """PIT histogram — should be ~uniform if calibrated."""
    _set_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pit_values, bins=n_bins, range=(0, 1), density=True,
            color="steelblue", edgecolor="white", alpha=0.85)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Uniform (ideal)")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("Density")
    ax.set_title(f"PIT Histogram — {model_name}\n(uniform ≈ well-calibrated)")
    ax.legend()
    ax.set_xlim(0, 1)
    fig.tight_layout()
    safe = model_name.replace(" ", "_")
    save_fig(fig, Path(out_dir) / f"pit_histogram_{safe}")
    return fig


def plot_reliability_curve(
    conf_bins: np.ndarray,
    acc_bins: np.ndarray,
    ece: float,
    event_name: str,
    out_dir: str | Path,
    model_labels: Optional[List[str]] = None,
) -> plt.Figure:
    """Reliability diagram for a binary event.

    Can accept multiple models if conf_bins and acc_bins are 2D (n_models, n_bins).
    """
    _set_style()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    if conf_bins.ndim == 1:
        ax.plot(conf_bins, acc_bins, "o-", label=f"ECE={ece:.3f}")
    else:
        for i, (c, a) in enumerate(zip(conf_bins, acc_bins)):
            lbl = model_labels[i] if model_labels else f"Model {i}"
            ax.plot(c, a, "o-", label=f"{lbl} (ECE={ece:.3f})" if i == 0 else lbl)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical frequency")
    ax.set_title(f"Reliability Curve — {event_name}")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    safe = event_name.replace(" ", "_").replace("<", "lt").replace(">", "gt")
    save_fig(fig, Path(out_dir) / f"reliability_{safe}")
    return fig


# ---------------------------------------------------------------------------
# Section G: Learning curves
# ---------------------------------------------------------------------------

def plot_learning_curve(
    df: pd.DataFrame,
    h: int,
    N: int,
    out_dir: str | Path,
    metric: str = "nll_test",
) -> plt.Figure:
    """NLL vs train_frac for each model, one subplot per (h, N).

    df must have columns: model, h, N, train_frac, seed, nll_val, nll_test
    """
    _set_style()
    sub = df[(df["h"] == h) & (df["N"] == N)].copy()
    if len(sub) == 0:
        fig, ax = plt.subplots()
        ax.set_title(f"No data for h={h} N={N}")
        return fig

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"state_cond": "#1f77b4", "state_free": "#ff7f0e", "backoff": "#2ca02c"}
    markers = {"state_cond": "o", "state_free": "s", "backoff": "^"}

    for model_type, grp in sub.groupby("model"):
        # Average over seeds; plot with shaded std if multiple seeds
        agg = grp.groupby("train_frac")[metric].agg(["mean", "std"]).reset_index()
        color = colors.get(model_type, None)
        marker = markers.get(model_type, "D")
        ax.plot(agg["train_frac"], agg["mean"], marker=marker, label=model_type,
                color=color, linewidth=1.8)
        if agg["std"].notna().any() and (agg["std"] > 0).any():
            ax.fill_between(
                agg["train_frac"],
                agg["mean"] - agg["std"],
                agg["mean"] + agg["std"],
                alpha=0.15, color=color,
            )

    ax.set_xlabel("Train fraction", fontsize=12)
    ax.set_ylabel(f"NLL ({metric})", fontsize=12)
    ax.set_title(f"Prefix Learning Curves — h={h}, N={N}", fontsize=12)
    ax.legend(title="Model")
    ax.set_xticks([0.25, 0.5, 0.75, 1.0])
    fig.tight_layout()
    save_fig(fig, Path(out_dir) / f"learning_curve_nll_{h}_{N}")
    return fig


# ---------------------------------------------------------------------------
# Section I: Depth ablation
# ---------------------------------------------------------------------------

def plot_depth_ablation_bar(
    df: pd.DataFrame,
    h: int,
    N: int,
    out_dir: str | Path,
) -> plt.Figure:
    """Grouped bar chart comparing shallow vs deep NLL for each model type.

    df must have columns: model, arch, h, N, seed, nll_test
    """
    _set_style()
    sub = df[(df["h"] == h) & (df["N"] == N)].copy()
    if len(sub) == 0:
        fig, ax = plt.subplots()
        ax.set_title(f"No data for h={h} N={N}")
        return fig

    agg = sub.groupby(["model", "arch"])["nll_test"].agg(["mean", "std"]).reset_index()
    models = agg["model"].unique()
    archs = ["shallow", "deep"]
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    palette = {"shallow": "#aec7e8", "deep": "#1f77b4"}

    for i, arch in enumerate(archs):
        vals = []
        errs = []
        for m in models:
            row = agg[(agg["model"] == m) & (agg["arch"] == arch)]
            vals.append(float(row["mean"].values[0]) if len(row) else np.nan)
            errs.append(float(row["std"].values[0]) if len(row) else 0.0)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=arch,
                      color=palette.get(arch, None),
                      yerr=[np.nan_to_num(e) for e in errs],
                      capsize=4, error_kw={"elinewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("NLL (test, nats)", fontsize=12)
    ax.set_title(f"Depth Ablation — h={h}, N={N}", fontsize=12)
    ax.legend(title="Architecture")
    fig.tight_layout()
    save_fig(fig, Path(out_dir) / f"depth_ablation_bar_{h}_{N}")
    return fig


# ---------------------------------------------------------------------------
# Section J: Generalization gap + spectral norm
# ---------------------------------------------------------------------------

def plot_gen_gap(
    history: dict,
    out_dir: str | Path,
    prefix: str = "gengap",
) -> List[plt.Figure]:
    """Two plots: (1) gap vs epoch, (2) NLL curves (train+val) vs epoch.

    history must have keys: train_nll, val_nll (lists of per-epoch values).
    Returns list of two figures.
    """
    _set_style()
    figs = []
    epochs = np.arange(1, len(history["train_nll"]) + 1)
    train_nll = np.array(history["train_nll"])
    val_nll = np.array(history["val_nll"])
    gap = val_nll - train_nll

    # Plot 1: NLL curves
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, train_nll, label="Train NLL", color="#1f77b4")
    ax1.plot(epochs, val_nll, label="Val NLL", color="#ff7f0e")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("NLL (nats)")
    ax1.set_title("Train vs Val NLL per Epoch")
    ax1.legend()
    fig1.tight_layout()
    save_fig(fig1, Path(out_dir) / f"{prefix}_nll_curves")
    figs.append(fig1)

    # Plot 2: generalization gap
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(epochs, gap, color="#d62728")
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val NLL − Train NLL (gap)")
    ax2.set_title("Generalization Gap per Epoch")
    fig2.tight_layout()
    save_fig(fig2, Path(out_dir) / f"{prefix}_gap")
    figs.append(fig2)

    return figs


# ---------------------------------------------------------------------------
# Section K: Feature-dimension ablation
# ---------------------------------------------------------------------------

def plot_feature_ablation(
    df: pd.DataFrame,
    h: int,
    N: int,
    out_dir: str | Path,
) -> plt.Figure:
    """Three-panel figure: test_nll / gen_gap / spectral_product vs n_features.

    df must have columns:
        n_features, train_nll, val_nll, test_nll, gen_gap, spectral_product
    Only rows matching h, N are used.
    """
    _set_style()
    sub = df[(df["horizon"] == h) & (df["N_bins"] == N)].copy()
    if len(sub) == 0:
        fig, ax = plt.subplots()
        ax.set_title(f"No data for h={h} N={N}")
        return fig

    sub = sub.sort_values("n_features")
    x = sub["n_features"].values

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # Panel 1: test NLL
    axes[0].plot(x, sub["test_nll"].values, "o-", color="#1f77b4", linewidth=1.8)
    if "val_nll" in sub.columns:
        axes[0].plot(x, sub["val_nll"].values, "s--", color="#aec7e8",
                     linewidth=1.2, label="val NLL")
        axes[0].legend(fontsize=9)
    axes[0].set_xlabel("Number of input features", fontsize=11)
    axes[0].set_ylabel("NLL (nats)", fontsize=11)
    axes[0].set_title("Test NLL vs Feature Dim", fontsize=11)
    axes[0].set_xticks(x)

    # Panel 2: generalization gap
    axes[1].plot(x, sub["gen_gap"].values, "o-", color="#d62728", linewidth=1.8)
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Number of input features", fontsize=11)
    axes[1].set_ylabel("Test NLL − Train NLL", fontsize=11)
    axes[1].set_title("Generalization Gap vs Feature Dim", fontsize=11)
    axes[1].set_xticks(x)

    # Panel 3: spectral norm product
    if sub["spectral_product"].notna().any():
        axes[2].plot(x, sub["spectral_product"].values, "o-", color="#2ca02c", linewidth=1.8)
        axes[2].set_xlabel("Number of input features", fontsize=11)
        axes[2].set_ylabel("Spectral norm product", fontsize=11)
        axes[2].set_title("Spectral Norm Product vs Feature Dim", fontsize=11)
        axes[2].set_xticks(x)
    else:
        axes[2].set_visible(False)

    fig.suptitle(f"Feature-Dimension Ablation — h={h}, N={N}", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, Path(out_dir) / f"feature_ablation_h{h}_N{N}")
    return fig
