"""Run dictionary-reduction experiments for NOS test-time beam search.

Primary benchmark: 1D advection-diffusion (checkpoint
``advection-diffusion/last.ckpt``), tasks ``E_AD_v`` (advection extrapolation,
c in [1,3]), ``E_AD_D`` (diffusion extrapolation, D in [1,3]), and ``E_AD_ALL``
(composed advection+diffusion in [0.01,1.0], used to reproduce paper Appendix
C.5 / Table 7).

Example:
    python test_time_compute/reduce_dictionary.py \\
        --model_path /model/nos_disco/advection-diffusion/last.ckpt \\
        --tasks E_AD_v E_AD_D \\
        --strategies random frequency cluster \\
        --n_operators 16 32 64 128 --random_seeds 0 1 2 \\
        --beam_width 4 --max_operators 5 \\
        --output_dir /data/nos_work/results/dict_reduce
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from test_time_compute import metrics
from test_time_compute.reduction import (
    cluster_reduce,
    frequency_reduce,
    random_reduce,
)
from test_time_compute.ttc_methods import (
    beam_search_operator_selection_batch,
    get_state_labels,
)
from test_time_compute.ttc_utils import DEVICE
from train.train import DISCOLitModule, TemporalBatchDatasetFly


AD_EXPERIMENTS = {
    "E_AD_ALL": {"v_range": (0.01, 1.0), "D_range": (0.01, 1.0), "axis": "both"},
    "E_AD_v": {"v_range": (1.0, 3.0), "D_range": (0.0, 0.0), "axis": "c"},
    "E_AD_D": {"v_range": (0.0, 0.0), "D_range": (1.0, 3.0), "axis": "D"},
}

TRAIN_V_RANGE = (0.01, 1.0)
TRAIN_D_RANGE = (0.001, 1.0)


def _to_float(v):
    if hasattr(v, "item"):
        return float(v.item())
    return float(v)


def _temporal_ds(split, n_batches, v_range, D_range, seed):
    return TemporalBatchDatasetFly(
        n_batches=n_batches,
        batch_size=64,
        sub_x=1,
        sub_t=1,
        split=split,
        input_frames=16,
        output_frames=34,
        L=16.0,
        nx=256,
        nt=100,
        T=10.0,
        v_range=v_range,
        D_range=D_range,
        fractal_degree=256,
        fractal_power_range=3,
        seed=seed,
        mixed_physics=False,
        mixed_physics_ratio=0.0,
    )


def build_dictionary(model, n_batches=4):
    """Encode training trajectories into a dictionary of latent operators."""
    ds = _temporal_ds("train", n_batches, TRAIN_V_RANGE, TRAIN_D_RANGE, 42)
    latents, meta = [], []
    model.eval()
    with torch.no_grad():
        for batch in ds:
            inp = batch["input"].to(DEVICE)
            state_labels = get_state_labels(inp)
            z, _ = model.encode_theta_latent(inp, state_labels)
            latents.append(z.cpu())
            for k in range(inp.size(0)):
                meta.append({
                    "operator_id": len(meta),
                    "advection_speed": _to_float(batch["advection_speed"][k]),
                    "diffusion": _to_float(batch["diffusion"][k]),
                })
    return torch.cat(latents, 0), meta


def run_beam(model, latent, test_ds, beam_width, max_operators, dt, beam_batch_size):
    """Run beam search over the whole test set, returning per-sample results."""
    latent_dev = latent.detach().to(DEVICE)
    errors, compositions, true_c, true_D = [], [], [], []
    t0 = time.time()
    model.eval()
    with torch.no_grad():
        for batch in test_ds:
            inp = batch["input"].to(DEVICE)
            tgt = batch["target"].to(DEVICE)
            for i in range(inp.size(0)):
                comp, err, _pred = beam_search_operator_selection_batch(
                    model, latent_dev, inp[i:i + 1], tgt[i:i + 1],
                    beam_width=beam_width, max_operators=max_operators,
                    dt=dt, batch_size=beam_batch_size)
                errors.append(float(err))
                compositions.append([int(x) for x in comp])
                true_c.append(_to_float(batch["advection_speed"][i]))
                true_D.append(_to_float(batch["diffusion"][i]))
    wall = time.time() - t0
    return errors, compositions, true_c, true_D, wall


def apply_reduce(strategy, model, lat, meta, n_operators, args, seed=None):
    if strategy == "random":
        return random_reduce(lat, meta, n_operators, seed=seed)
    if strategy == "frequency":
        heldout = _temporal_ds("train", args.heldout_batches,
                               TRAIN_V_RANGE, TRAIN_D_RANGE, args.heldout_seed)
        return frequency_reduce(model, lat, meta, heldout, n_operators,
                                beam_width=args.beam_width,
                                max_operators=args.max_operators,
                                dt=args.dt, beam_batch_size=args.beam_batch_size,
                                device=DEVICE)
    if strategy == "cluster":
        heldout = _temporal_ds("train", args.heldout_batches,
                               TRAIN_V_RANGE, TRAIN_D_RANGE, args.heldout_seed)
        return cluster_reduce(model, lat, meta, n_operators, val_loader=heldout,
                              pca_dim=args.pca_dim, dt=args.dt, dim=1,
                              device=DEVICE, weighting="error",
                              max_val_samples=64)
    raise ValueError(f"unknown strategy: {strategy}")


def main():
    p = argparse.ArgumentParser(description="NOS dictionary-reduction experiments")
    p.add_argument("--model_path", required=True)
    p.add_argument("--tasks", nargs="+", default=["E_AD_v"], choices=list(AD_EXPERIMENTS))
    p.add_argument("--strategies", nargs="+",
                   default=["full", "random", "frequency", "cluster"],
                   choices=["full", "random", "frequency", "cluster"])
    p.add_argument("--n_operators", type=int, nargs="+", default=[16, 32, 64, 128])
    p.add_argument("--random_seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="random retention seeds; seed 42 reproduces paper C.5")
    p.add_argument("--beam_width", type=int, default=4)
    p.add_argument("--max_operators", type=int, default=5)
    p.add_argument("--improvement_threshold", type=float, default=5.0)
    p.add_argument("--dt", type=float, default=10.0 / 100)
    p.add_argument("--beam_batch_size", type=int, default=32)
    p.add_argument("--n_dict_batches", type=int, default=4,
                   help="full dictionary size = n_dict_batches * 64 (default 256)")
    p.add_argument("--heldout_batches", type=int, default=2,
                   help="held-out training trajectories = heldout_batches * 64 (default 128)")
    p.add_argument("--heldout_seed", type=int, default=7)
    p.add_argument("--pca_dim", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=512)
    p.add_argument("--output_dir", default="./results/dict_reduce")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    lit = DISCOLitModule.load_from_checkpoint(args.model_path, map_location=DEVICE)
    lit.eval()
    model = lit.model.to(DEVICE).eval()
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} params")

    if args.check_only:
        lat, meta = build_dictionary(model, 1)
        test_ds = _temporal_ds("test", 1,
                               AD_EXPERIMENTS["E_AD_v"]["v_range"],
                               AD_EXPERIMENTS["E_AD_v"]["D_range"], 124)
        errors, comps, tc, td, wall = run_beam(
            model, lat, test_ds, args.beam_width, 1, args.dt, args.beam_batch_size)
        print("CHECK OK")
        print(f"dict: {tuple(lat.shape)} | samples: {len(errors)} | "
              f"mean nrmse: {float(np.mean(errors)):.4f} | wall: {wall:.1f}s")
        return

    summary = {}
    history_path = os.path.join(args.output_dir, "history.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")

    for task in args.tasks:
        cfg = AD_EXPERIMENTS[task]
        test_ds = _temporal_ds("test", max(1, math.ceil(args.num_samples / 64)),
                               cfg["v_range"], cfg["D_range"], 124)
        lat, meta = build_dictionary(model, args.n_dict_batches)
        print(f"[{task}] full dictionary: {tuple(lat.shape)}")

        errs, comps, tc, td, wall = run_beam(
            model, lat, test_ds, args.beam_width, args.max_operators,
            args.dt, args.beam_batch_size)
        baseline = float(np.mean(errs))
        baseline_mae = metrics.param_mae(comps, meta, tc, td)
        print(f"[{task}] baseline NRMSE: {baseline:.4f} "
              f"(MAE c={baseline_mae['c']:.3f}, D={baseline_mae['D']:.3f}, {wall:.1f}s)")

        summary[task] = {
            "baseline_nrmse": baseline,
            "baseline_param_mae": baseline_mae,
            "baseline_wall_time_s": wall,
            "configs": [],
        }
        metrics.append_history_row(history_path, {
            "task": task, "strategy": "full", "n_operators": lat.shape[0],
            "seed": "", "nrmse": baseline,
            "degradation_pct": 0.0, "candidates": metrics.candidate_count(
                args.beam_width, lat.shape[0], args.max_operators),
            "wall_time_s": wall, "param_mae_c": baseline_mae["c"],
            "param_mae_D": baseline_mae["D"], "traceable": True,
        })

        for strategy in args.strategies:
            if strategy == "full":
                continue
            for n_operators in args.n_operators:
                n_op = min(n_operators, lat.shape[0])
                seeds = args.random_seeds if strategy == "random" else [None]
                seed_rows = []
                for seed in seeds:
                    tag = "" if seed is None else str(seed)
                    reduced_lat, reduced_meta, trace = apply_reduce(
                        strategy, model, lat, meta, n_operators, args, seed)
                    errs, comps, tc, td, wall = run_beam(
                        model, reduced_lat, test_ds, args.beam_width,
                        args.max_operators, args.dt, args.beam_batch_size)
                    nrmse = float(np.mean(errs))
                    mae = metrics.param_mae(comps, reduced_meta, tc, td)
                    row = {
                        "task": task, "strategy": strategy, "n_operators": n_op,
                        "seed": tag, "nrmse": nrmse,
                        "degradation_pct": metrics.degradation_pct(nrmse, baseline),
                        "candidates": metrics.candidate_count(
                            args.beam_width, n_op, args.max_operators),
                        "wall_time_s": wall, "param_mae_c": mae["c"],
                        "param_mae_D": mae["D"],
                        "traceable": trace["type"] != "cluster",
                    }
                    seed_rows.append(row)
                    metrics.append_history_row(history_path, row)
                    print(f"[{task}] {strategy} N'={n_op}{' seed=' + tag if tag else ''} "
                          f"NRMSE={nrmse:.4f} deg={row['degradation_pct']:+.1f}% "
                          f"MAE c={mae['c']:.3f} D={mae['D']:.3f} {wall:.1f}s")

                agg = metrics.random_summary(seed_rows) if strategy == "random" \
                    else {
                        "nrmse": seed_rows[0]["nrmse"],
                        "nrmse_std": 0.0,
                        "param_mae": seed_rows[0]["param_mae"],
                    }
                summary[task]["configs"].append({
                    "strategy": strategy,
                    "n_operators": n_op,
                    "nrmse": agg["nrmse"],
                    "nrmse_std": agg["nrmse_std"],
                    "degradation_pct": metrics.degradation_pct(agg["nrmse"], baseline),
                    "candidates": seed_rows[0]["candidates"],
                    "wall_time_s": sum(r["wall_time_s"] for r in seed_rows),
                    "param_mae": agg["param_mae"],
                    "traceable": seed_rows[0]["traceable"],
                    "seeds": {r["seed"]: r["nrmse"] for r in seed_rows},
                })

    metrics.write_summary_json(summary_path, summary)
    print(f"Done. history -> {history_path}")
    print(f"Done. summary -> {summary_path}")


if __name__ == "__main__":
    main()
