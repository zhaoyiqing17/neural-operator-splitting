"""Metric helpers and result serialization for dictionary-reduction runs."""

from __future__ import annotations

import csv
import json
import os

import numpy as np


def degradation_pct(reduced, baseline):
    """Relative degradation of a reduced-dictionary NRMSE vs the full baseline."""
    if baseline is None or baseline <= 0:
        return float("nan")
    return (float(reduced) - float(baseline)) / float(baseline) * 100.0


def candidate_count(beam_width, n_operators, max_operators):
    """Upper-bound candidate evaluations per test sample: B * N' * M."""
    return int(beam_width) * int(n_operators) * int(max_operators)


def param_mae(compositions, reduced_meta, true_c, true_D):
    """Mean absolute error of the composed physical parameters."""
    composed_c = [
        sum(float(reduced_meta[o].get("advection_speed", 0.0)) for o in comp)
        for comp in compositions
    ]
    composed_D = [
        sum(float(reduced_meta[o].get("diffusion", 0.0)) for o in comp)
        for comp in compositions
    ]
    mae_c = float(np.mean(np.abs(np.asarray(composed_c, dtype=float)
                                 - np.asarray(true_c, dtype=float))))
    mae_D = float(np.mean(np.abs(np.asarray(composed_D, dtype=float)
                                 - np.asarray(true_D, dtype=float))))
    return {"c": mae_c, "D": mae_D}


def random_summary(seed_rows):
    """Aggregate per-seed rows into mean + standard deviation."""
    errs = np.asarray([r["nrmse"] for r in seed_rows], dtype=float)
    maes_c = np.asarray([r["param_mae"]["c"] for r in seed_rows], dtype=float)
    maes_D = np.asarray([r["param_mae"]["D"] for r in seed_rows], dtype=float)
    return {
        "nrmse": float(errs.mean()),
        "nrmse_std": float(errs.std(ddof=1)) if len(errs) > 1 else 0.0,
        "param_mae": {"c": float(maes_c.mean()), "D": float(maes_D.mean())},
    }


def append_history_row(path, row):
    """Append one row to a CSV, writing the header on first use."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            writer.writeheader()
        writer.writerow(row)


def write_summary_json(path, summary):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
