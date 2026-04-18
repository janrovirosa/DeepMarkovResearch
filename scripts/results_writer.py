"""Unified results writer for multi-asset experiments."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch


def save_results(
    results_dir: str | Path,
    ticker: str,
    experiment_tag: str,
    config_dict: Dict[str, Any],
    metrics_dict: Dict[str, Any],
    diagnostics_dict: Dict[str, Any] | None = None,
) -> Path:
    """Write per-experiment outputs; return the output directory path.

    Directory structure
    -------------------
    {results_dir}/{ticker}/{experiment_tag}/
        config.json
        nll_metrics.csv          (appended incrementally)
        operator_diagnostics.csv (written if diagnostics_dict["operator"] present)
        ck_errors.csv            (written if diagnostics_dict["ck_errors"] present)
        weights/
            {model_type}_h{h}_seed{seed}.pt
        FAILED                   (written if any metric value is None or NaN)

    Parameters
    ----------
    metrics_dict
        Expected to have key "rows": list of dicts with keys
        model, h, seed, nll_train, nll_val, nll_test, delta_marginal.
        Optional key "weights": dict mapping name → state_dict.
    diagnostics_dict
        Optional. Keys:
        - "operator": dict {metric_name: float}  (row_heterogeneity, row_entropy, dobrushin_mean)
        - "ck_errors": list of dicts [{h, mean_kl, mean_tv}, ...]
    """
    out_dir = Path(results_dir) / ticker / experiment_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── config.json ──
    config_path = out_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=_json_default)

    # ── nll_metrics.csv ──
    rows = metrics_dict.get("rows", [])
    if rows:
        new_df = pd.DataFrame(rows)
        metrics_path = out_dir / "nll_metrics.csv"
        if metrics_path.exists():
            existing = pd.read_csv(metrics_path)
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df.to_csv(metrics_path, index=False)

    # ── weights/ ──
    weights = metrics_dict.get("weights", {})
    if weights:
        weights_dir = out_dir / "weights"
        weights_dir.mkdir(exist_ok=True)
        for name, state_dict in weights.items():
            torch.save(state_dict, weights_dir / f"{name}.pt")

    # ── FAILED marker ──
    failed = _has_bad_metrics(rows)
    if failed:
        (out_dir / "FAILED").write_text("One or more metric values are None or NaN.\n")

    if diagnostics_dict is None:
        return out_dir

    # ── operator_diagnostics.csv ──
    operator = diagnostics_dict.get("operator")
    if operator:
        op_rows = [{"metric": k, "value": v} for k, v in operator.items()]
        pd.DataFrame(op_rows).to_csv(out_dir / "operator_diagnostics.csv", index=False)

    # ── ck_errors.csv ──
    ck_errors = diagnostics_dict.get("ck_errors")
    if ck_errors:
        pd.DataFrame(ck_errors).to_csv(out_dir / "ck_errors.csv", index=False)

    return out_dir


def _has_bad_metrics(rows: list) -> bool:
    for row in rows:
        for v in row.values():
            if v is None:
                return True
            try:
                if math.isnan(float(v)):
                    return True
            except (TypeError, ValueError):
                pass
    return False


def _json_default(obj):
    """Fallback serialiser for non-JSON-native types."""
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)
