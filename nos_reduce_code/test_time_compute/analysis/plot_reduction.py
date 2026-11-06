"""Plot N' vs degradation and N' vs search cost from summary.json."""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", required=True, help="path to summary.json")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    with open(args.summary) as f:
        data = json.load(f)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.summary))
    os.makedirs(out_dir, exist_ok=True)

    for task, tdata in data.items():
        base = float(tdata["baseline_nrmse"])
        by_strategy = {}
        for c in tdata.get("configs", []):
            by_strategy.setdefault(c["strategy"], []).append(c)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for strat, configs in by_strategy.items():
            configs = sorted(configs, key=lambda c: c["n_operators"])
            xs = [c["n_operators"] for c in configs]
            ys = [(c["nrmse"] - base) / base * 100.0 if base > 0 else float("nan")
                  for c in configs]
            ax1.plot(xs, ys, marker="o", label=strat)
            ax2.plot(xs, [c["candidates"] for c in configs], marker="o", label=strat)

        ax1.axhline(0.0, color="gray", lw=0.8)
        ax1.axhline(5.0, color="red", lw=0.8, ls="--", label="5% degradation line")
        ax1.set_xlabel("N'")
        ax1.set_ylabel("degradation % vs full dictionary")
        ax1.set_title(f"{task} — accuracy degradation")
        ax1.legend()

        ax2.set_xlabel("N'")
        ax2.set_ylabel("candidate evaluations B*N'*M")
        ax2.set_title(f"{task} — search cost")
        ax2.legend()

        fig.tight_layout()
        path = os.path.join(out_dir, f"{task}_reduction_curves.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print("saved", path)


if __name__ == "__main__":
    main()
