from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3})


def save(fig, stem, dpi=150):
    # png for viewing, pdf for the paper
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")


def over_time(series, ylabel, stem, windows=(), figsize=(6, 7), title=None):
    # one line per model, shaded regime windows (start, end, name)
    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, s) in enumerate(series.items()):
        ok = np.isfinite(s)
        label = name + (" (degenerate)" if "state_free" in name else "")
        ax.plot(np.where(ok)[0], s[ok], label=label, color=plt.cm.tab10.colors[i], alpha=0.8)
    for start, end, name in windows:
        ax.axvspan(start, end, alpha=0.18, color="salmon", label=f"Regime: {name}")
    ax.set_xlabel("Time index (full series)")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, stem)
    return fig


def heatmap(A, title, stem, vmin=0.0, vmax=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(A, aspect="auto", cmap="Blues", vmin=vmin, vmax=A.max() if vmax is None else vmax)
    plt.colorbar(im, ax=ax, label="P(next state | current state, F_t)")
    ax.set_title(title)
    ax.set_xlabel("Output state X_{t+h}")
    ax.set_ylabel("Input state X_t")
    ax.set_xticks(np.arange(0, A.shape[0], 10))
    ax.set_yticks(np.arange(0, A.shape[0], 10))
    fig.tight_layout()
    save(fig, stem)
    return fig
