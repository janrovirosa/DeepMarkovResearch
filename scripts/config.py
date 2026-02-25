"""Experiment configuration dataclass with YAML i/o and git hash capture."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import List, Tuple

import yaml


@dataclass
class ExperimentConfig:
    # Reproducibility
    seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [42, 7, 123])

    # Experiment grid
    horizons: List[int] = field(default_factory=lambda: [1, 2, 5, 10])
    n_bins_list: List[int] = field(default_factory=lambda: [10, 20, 35, 55])
    n_xt_target: int = 55
    ck_horizons: List[int] = field(default_factory=lambda: [1, 2, 5, 10])

    # Model architecture
    hidden_dims: Tuple[int, ...] = (64, 128, 256, 128, 64)
    dropout: float = 0.2

    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_train: int = 256
    batch_eval: int = 512
    max_epochs: int = 100
    patience: int = 10
    grad_clip: float = 1.0
    sigma_anchor: float = 1.0

    # Baseline hyperparameter grids
    alpha_grid: List[float] = field(
        default_factory=lambda: [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 5.0, 10.0]
    )
    tau_grid: List[float] = field(
        default_factory=lambda: [10, 50, 100, 200, 500, 1000]
    )

    # Degeneracy analysis
    sparsity_thresh: List[int] = field(default_factory=lambda: [5, 10])

    # Bootstrap CI (only key configs)
    n_boot: int = 500
    boot_block_size: int = 21
    bootstrap_key_configs: List[Tuple[int, int]] = field(
        default_factory=lambda: [(1, 55), (10, 55)]
    )

    # Runtime metadata (filled by make_config())
    date_stamp: str = ""
    git_hash: str = ""
    output_dir: str = ""


def _get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def make_config(**overrides) -> ExperimentConfig:
    """Create a config with today's date stamp and git hash filled in."""
    cfg = ExperimentConfig(**overrides)
    cfg.date_stamp = date.today().isoformat()
    cfg.git_hash = _get_git_hash()
    cfg.output_dir = str(
        Path(__file__).resolve().parent.parent
        / "results" / "paper_upgrade" / cfg.date_stamp
    )
    return cfg


def save_config(cfg: ExperimentConfig, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(cfg)
    # Convert tuples to lists for YAML serialibility
    d["hidden_dims"] = list(d["hidden_dims"])
    with open(path, "w") as f:
        yaml.dump(d, f, default_flow_style=False)


def load_config(path: Path | str) -> ExperimentConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    d["hidden_dims"] = tuple(d["hidden_dims"])
    # bootstrap_key_configs stored as list-of-lists → convert back to list-of-tuples
    if "bootstrap_key_configs" in d:
        d["bootstrap_key_configs"] = [tuple(x) for x in d["bootstrap_key_configs"]]
    return ExperimentConfig(**d)
