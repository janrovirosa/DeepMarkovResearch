"""Figure generation script for paper.

Run from project root:
    python -m scripts.make_figures

Figures are saved to figures/paper/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Make sure project root is on path when run directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data import load_master_dataset, compute_returns, make_splits
from scripts.bins import get_edges, assign_bins

OUT_DIR = ROOT / "figures" / "paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Figure 1 — Transition-Cell Sparsity vs N
# ---------------------------------------------------------------------------

def fig1_sparsity_vs_N():
    """
    For each N in n_bins_list, compute the empirical NxN transition matrix
    from 1-day returns (train split only) and report the fraction of cells
    with < k observations for k in {1, 5, 10}.

    Produces a line plot: x = N, y = fraction of cells below threshold.
    """
    prices, _, _ = load_master_dataset(ROOT / "dataset")
    r1 = compute_returns(prices, 1)
    train_end = int(0.70 * len(r1))
    r_train = r1[:train_end]

    n_bins_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    thresholds = [1, 5, 10]
    colors = ["#2166ac", "#d6604d", "#4dac26"]

    fracs = {k: [] for k in thresholds}

    for N in n_bins_list:
        N_actual, edges = get_edges(r_train, N)
        bins_all = assign_bins(r1, edges)

        # Build transition matrix on train window only (t -> t+1)
        counts = np.zeros((N_actual, N_actual), dtype=np.int64)
        for t in range(train_end - 1):
            i, j = bins_all[t], bins_all[t + 1]
            counts[i, j] += 1

        total_cells = N_actual * N_actual
        for k in thresholds:
            fracs[k].append((counts < k).sum() / total_cells)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    for k, color in zip(thresholds, colors):
        label = f"< {k} obs."
        ax.plot(n_bins_list, [f * 100 for f in fracs[k]],
                marker="o", markersize=4, linewidth=1.8,
                color=color, label=label)

    # Annotate N=55 < 5 obs value
    idx_55 = n_bins_list.index(55)
    val_55 = fracs[5][idx_55] * 100
    ax.annotate(
        f"{val_55:.1f}%",
        xy=(55, val_55),
        xytext=(49, val_55 - 6),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )

    ax.set_xlabel("Number of bins $N$")
    ax.set_ylabel("Fraction of transition cells (%)")
    ax.set_xlim(n_bins_list[0] - 1, n_bins_list[-1] + 1)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g%%"))
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    path = OUT_DIR / "fig1_sparsity_vs_N.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 3 — State-Conditioned A_t heatmap snapshots (3-panel)
# ---------------------------------------------------------------------------

def fig3_At_snapshots():
    """
    Load the cached A_t matrices for the state-conditioned model (h=1)
    and plot 3 side-by-side snapshots at maximally-separated time points.

    Each heatmap panel: row = current state X_t (input),
    column = predicted next state Y_{t+1} (output).

    The column-average distribution (plotted below each heatmap) shows how the
    model's overall output bias shifts — from below-center (bearish regime) to
    above-center (bullish regime) — reflecting time-varying market dynamics.
    """
    CACHE = ROOT / "results" / "paper_upgrade" / "2026-03-07" / "cache"
    At_path = CACHE / "A_t_ck_state_cond_h1.npy"
    if not At_path.exists():
        print(f"Cache not found: {At_path}\nSkipping fig3.")
        return

    A_full = np.load(At_path)   # shape (T, N, N)
    N = A_full.shape[1]
    bins = np.arange(N)

    # Pick 3 time points spanning the full range of column-weighted mean:
    #   t=1042: wmean=20.45 (bearish — output concentrated left of center)
    #   t=511:  wmean=27.14 (neutral — near center)
    #   t=2170: wmean=27.92 (bullish — output concentrated right of center)
    snap_times = [1042, 511, 2170]
    period_labels = [
        "Bearish period\n($\\bar{y}=20.5$, bin 27 = neutral)",
        "Neutral period\n($\\bar{y}=27.1$)",
        "Bullish period\n($\\bar{y}=27.9$)",
    ]
    center_bin = N / 2  # 27.5 for N=55

    matrices = [A_full[t] for t in snap_times]

    # Clip colorscale at 99th percentile across all three matrices to avoid
    # the (0,0) spike at t=1042 saturating the scale.
    all_vals = np.concatenate([A.ravel() for A in matrices])
    vmax = float(np.percentile(all_vals, 99.5))

    fig, axes = plt.subplots(
        2, 3,
        figsize=(11, 6.5),
        gridspec_kw={"height_ratios": [3.5, 1.2], "hspace": 0.45, "wspace": 0.12},
    )

    for col, (A, label, t) in enumerate(zip(matrices, period_labels, snap_times)):
        ax_heat = axes[0, col]
        ax_dist = axes[1, col]

        # ── Heatmap
        im = ax_heat.imshow(
            A,
            origin="upper",
            aspect="equal",
            cmap="Blues",
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
        )

        # Column-weighted mean as a vertical line on the heatmap
        mean_dist = A.mean(axis=0)
        wmean = float(np.dot(mean_dist, bins))
        ax_heat.axvline(wmean, color="#d62728", linestyle="--", linewidth=1.6, alpha=0.9)
        ax_heat.axvline(center_bin, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)

        ax_heat.set_xlabel("Output state $Y_{t+1}$", fontsize=8.5)
        ax_heat.tick_params(labelsize=7.5)

        # ── Column-average distribution
        ax_dist.fill_between(bins, mean_dist, alpha=0.35, color="#1f77b4")
        ax_dist.plot(bins, mean_dist, color="#1f77b4", linewidth=1.2)
        ax_dist.axvline(wmean, color="#d62728", linestyle="--", linewidth=1.4, alpha=0.9)
        ax_dist.axvline(center_bin, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        ax_dist.set_xlim(0, N - 1)
        ax_dist.set_xlabel("Output bin", fontsize=8)
        ax_dist.tick_params(labelsize=7)
        ax_dist.set_yticks([])

    axes[0, 0].set_ylabel("Input state $X_t$", fontsize=9)
    for ax in axes[0, 1:]:
        ax.set_yticks([])

    axes[1, 0].set_ylabel("Avg. prob.", fontsize=8)

    # Shared colourbar (attached to heatmap row)
    cbar = fig.colorbar(im, ax=axes[0, :], shrink=0.75, pad=0.02, aspect=25)
    cbar.set_label("Predicted probability\n(clipped at 99.5th pct.)", fontsize=8)
    cbar.ax.tick_params(labelsize=7.5)

    # Legend annotation
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], color="#d62728", linestyle="--", linewidth=1.5, label="Weighted mean output bin"),
        Line2D([0], [0], color="gray",    linestyle=":",  linewidth=1.2, label="Neutral center (bin 27.5)"),
    ]
    axes[1, 1].legend(handles=legend_els, loc="upper center",
                      bbox_to_anchor=(0.5, -0.55), fontsize=7.5,
                      ncol=2, frameon=False)

    path = OUT_DIR / "fig3_At_snapshots_state_cond.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 4 — Regime-tracking metrics over time
# ---------------------------------------------------------------------------

def fig4_regime_metrics():
    """
    Two metrics derived from the full A_t sequence (state-conditioned, h=1)
    plotted over all 2367 time steps, with JPM price as a reference:

      State Dependence  — std across rows of each row's weighted-mean output bin.
                          High = different starting states predict very different
                          outcomes (current state matters a lot).

      Market Uncertainty — mean Shannon entropy of each row's predicted distribution.
                           High = model is uncertain / diffuse; low = model is
                           concentrated / confident about the next state.

    Both series are shown raw (thin) + 21-day rolling mean (thick).
    Vertical dashed lines mark train / val / test boundaries.
    """
    from scipy.stats import entropy as scipy_entropy
    from scipy.ndimage import uniform_filter1d

    CACHE = ROOT / "results" / "paper_upgrade" / "2026-03-07" / "cache"
    At_path = CACHE / "A_t_ck_state_cond_h1.npy"
    if not At_path.exists():
        print(f"Cache not found: {At_path}\nSkipping fig4.")
        return

    A_full = np.load(At_path)
    T, N, _ = A_full.shape
    bins = np.arange(N)
    t_axis = np.arange(T)

    print("  Computing state dependence...")
    state_dep = np.array([A_full[t].dot(bins).std() for t in range(T)])

    print("  Computing row entropy...")
    row_entropy = np.array([
        np.mean([scipy_entropy(A_full[t, i, :] + 1e-12) for i in range(N)])
        for t in range(T)
    ])

    # 21-day smoothed versions
    sd_smooth  = uniform_filter1d(state_dep,  size=21)
    ent_smooth = uniform_filter1d(row_entropy, size=21)

    # JPM price aligned to A_t (A_t has T = len(prices)-2 due to h=1 lag)
    prices, _, _ = load_master_dataset(ROOT / "dataset")
    price = prices[:T]

    # Split boundaries (h=1)
    train_end = int(0.70 * T)
    val_end   = int(0.85 * T)

    # ── Layout: 3 rows (price, state-dep, entropy), shared x-axis
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.5, 1.5], "hspace": 0.10},
    )

    split_kw  = dict(color="#555", linestyle="--", linewidth=0.9, alpha=0.7)
    shade_kw  = dict(alpha=0.07, color="gray")

    # ── Panel 0: JPM price
    ax0 = axes[0]
    ax0.plot(t_axis, price, color="#333", linewidth=0.7)
    ax0.fill_between(t_axis, price, price.min(), alpha=0.15, color="#1f77b4")
    ax0.set_ylabel("JPM price ($)", fontsize=9)
    ax0.tick_params(labelsize=8)
    ax0.grid(axis="y", linestyle=":", alpha=0.4)

    # ── Panel 1: State Dependence
    ax1 = axes[1]
    ax1.plot(t_axis, state_dep, color="#9ecae1", linewidth=0.5, alpha=0.6)
    ax1.plot(t_axis, sd_smooth, color="#2166ac", linewidth=1.6,
             label="State dependence (21-day avg)")
    ax1.set_ylabel("Std of row\nweighted means", fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    ax1.legend(fontsize=8, loc="upper right")

    # ── Panel 2: Market Uncertainty (entropy)
    ax2 = axes[2]
    ax2.plot(t_axis, row_entropy, color="#fdae6b", linewidth=0.5, alpha=0.6)
    ax2.plot(t_axis, ent_smooth, color="#d6604d", linewidth=1.6,
             label="Market uncertainty — avg row entropy (21-day avg)")
    ax2.set_ylabel("Mean row entropy\n(nats)", fontsize=9)
    ax2.set_xlabel("Time step $t$", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    ax2.legend(fontsize=8, loc="upper right")

    # Shade regions + labels on top panel
    for ax in axes:
        ax.axvline(train_end, **split_kw)
        ax.axvline(val_end,   **split_kw)

    label_y = ax0.get_ylim()[0] + 0.05 * (ax0.get_ylim()[1] - ax0.get_ylim()[0])
    for pos, txt in [(train_end / 2, "Train"), ((train_end + val_end) / 2, "Val"),
                     ((val_end + T) / 2, "Test")]:
        ax0.text(pos, label_y, txt, ha="center", fontsize=8,
                 color="#555", style="italic")

    path = OUT_DIR / "fig4_regime_metrics.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 5 — Time-resolved Chapman-Kolmogorov discrepancy
# ---------------------------------------------------------------------------

def fig5_ck_discrepancy():
    """
    Chapman-Kolmogorov discrepancy for h=5 over all time steps.

    CK equation: A_t^(5) = A_t^(1) @ A_{t+1}^(1) @ ... @ A_{t+4}^(1)

    Discrepancy(t) = ||A_t^(5)  -  product_{k=0}^{4} A_{t+k}^(1)||_F

    A spike means the model's direct 5-day prediction cannot be explained
    by chaining 5 independent 1-day predictions — i.e., the market's
    "first-order memory" assumption breaks down at that moment.

    Both state-conditioned and state-free models are shown for comparison.
    """
    from scipy.ndimage import uniform_filter1d

    CACHE = ROOT / "results" / "paper_upgrade" / "2026-03-07" / "cache"

    # Try to load precomputed discrepancies; recompute if missing
    disc_sc_path = CACHE / "ck_disc_state_cond_h5.npy"
    disc_sf_path = CACHE / "ck_disc_state_free_h5.npy"

    if disc_sc_path.exists() and disc_sf_path.exists():
        disc_sc = np.load(disc_sc_path)
        disc_sf = np.load(disc_sf_path)
    else:
        A1sc = np.load(CACHE / "A_t_ck_state_cond_h1.npy")
        A5sc = np.load(CACHE / "A_t_ck_state_cond_h5.npy")
        A1sf = np.load(CACHE / "A_t_ck_state_free_h1.npy")
        A5sf = np.load(CACHE / "A_t_ck_state_free_h5.npy")
        T5 = A5sc.shape[0]
        disc_sc = np.zeros(T5)
        disc_sf = np.zeros(T5)
        for t in range(T5):
            p_sc = A1sc[t]
            p_sf = A1sf[t]
            for k in range(1, 5):
                p_sc = p_sc @ A1sc[t + k]
                p_sf = p_sf @ A1sf[t + k]
            disc_sc[t] = np.linalg.norm(A5sc[t] - p_sc, "fro")
            disc_sf[t] = np.linalg.norm(A5sf[t] - p_sf, "fro")
        np.save(disc_sc_path, disc_sc)
        np.save(disc_sf_path, disc_sf)

    T = len(disc_sc)
    t_axis = np.arange(T)

    sc_smooth = uniform_filter1d(disc_sc, size=21)
    sf_smooth = uniform_filter1d(disc_sf, size=21)

    prices, _, _ = load_master_dataset(ROOT / "dataset")
    price = prices[:T]

    train_end = int(0.70 * T)
    val_end   = int(0.85 * T)

    # Top-3 spike locations (state-cond)
    top3_idx = np.argsort(disc_sc)[-3:][::-1]

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2.2], "hspace": 0.08},
    )

    split_kw = dict(color="#555", linestyle="--", linewidth=0.9, alpha=0.7)

    # ── Panel 0: JPM price (reference)
    ax0 = axes[0]
    ax0.plot(t_axis, price, color="#333", linewidth=0.7)
    ax0.fill_between(t_axis, price, price.min(), alpha=0.12, color="#1f77b4")
    ax0.set_ylabel("JPM price ($)", fontsize=9)
    ax0.tick_params(labelsize=8)
    ax0.grid(axis="y", linestyle=":", alpha=0.4)

    label_y = ax0.get_ylim()[0] + 0.08 * (ax0.get_ylim()[1] - ax0.get_ylim()[0])
    for pos, txt in [(train_end / 2, "Train"), ((train_end + val_end) / 2, "Val"),
                     ((val_end + T) / 2, "Test")]:
        ax0.text(pos, label_y, txt, ha="center", fontsize=8,
                 color="#555", style="italic")

    # ── Panel 1: CK discrepancy
    ax1 = axes[1]

    # State-free (background)
    ax1.fill_between(t_axis, disc_sf, alpha=0.18, color="#4dac26")
    ax1.plot(t_axis, sf_smooth, color="#4dac26", linewidth=1.4, alpha=0.85,
             label="State-free (21-day avg)")

    # State-conditioned (foreground)
    ax1.fill_between(t_axis, disc_sc, alpha=0.18, color="#d62728")
    ax1.plot(t_axis, sc_smooth, color="#d62728", linewidth=1.8,
             label="State-conditioned (21-day avg)")

    # Annotate the top-3 spikes
    for rank, t_spike in enumerate(top3_idx):
        ax1.annotate(
            f"$t={t_spike}$",
            xy=(t_spike, disc_sc[t_spike]),
            xytext=(t_spike + 40, disc_sc[t_spike] * (0.92 - rank * 0.06)),
            fontsize=7.5, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=0.8),
        )

    ax1.set_ylabel("CK discrepancy\n$\\|A_t^{(5)} - \\prod A_{t+k}^{(1)}\\|_F$", fontsize=9)
    ax1.set_xlabel("Time step $t$", fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.grid(axis="y", linestyle=":", alpha=0.4)
    ax1.legend(fontsize=8.5, loc="upper left")

    for ax in axes:
        ax.axvline(train_end, **split_kw)
        ax.axvline(val_end,   **split_kw)

    path = OUT_DIR / "fig5_ck_discrepancy.pdf"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

FIGURES = {
    "fig1": fig1_sparsity_vs_N,
    "fig3": fig3_At_snapshots,
    "fig4": fig4_regime_metrics,
    "fig5": fig5_ck_discrepancy,
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument(
        "figs",
        nargs="*",
        help="Figure keys to generate (e.g. fig1). Omit to generate all.",
    )
    args = parser.parse_args()

    keys = args.figs if args.figs else list(FIGURES.keys())
    for key in keys:
        if key not in FIGURES:
            print(f"Unknown figure: {key}. Available: {list(FIGURES.keys())}")
            continue
        print(f"Generating {key}...")
        FIGURES[key]()

    print("Done.")


if __name__ == "__main__":
    main()
