"""Multi-seed benchmark with mean +/- std reporting.

Runs every (case, algorithm) pair across N seeds and prints/saves
aggregated mean and std for each metric.

Usage:  python -m scripts.run_benchmark_multiseed --seeds 30
"""
from __future__ import annotations
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.simulator import run
from src.scenarios import CASES
from src.algorithms import ALGORITHMS
from src.metrics import metrics
from src.plotting import setup, COLOR

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(exist_ok=True)

def main(n_seeds: int = 30):
    rows = []
    for case_name, profile in CASES.items():
        print(f"\n=== {case_name}: {profile.name} ===")
        for algo_name, AlgoClass in ALGORITHMS.items():
            etas, ts, ripples, energies = [], [], [], []
            for seed in range(n_seeds):
                # P&O is deterministic; only stochastic algos accept seed.
                try:
                    algo = AlgoClass(seed=seed)
                except TypeError:
                    algo = AlgoClass()
                df = run(profile, algo)
                m = metrics(df)
                etas.append(m["eta_pct"])
                ts.append(m["t_track_s"])
                ripples.append(m["ripple_W"])
                energies.append(m["energy_pct"])
            rows.append(dict(
                case=case_name, algo=algo_name,
                eta_mean=np.mean(etas), eta_std=np.std(etas),
                t_mean=np.nanmean(ts), t_std=np.nanstd(ts),
                ripple_mean=np.mean(ripples), ripple_std=np.std(ripples),
                E_mean=np.mean(energies), E_std=np.std(energies),
            ))
            print(f"  {algo_name:<8s}  eta={np.mean(etas):6.2f}+/-{np.std(etas):.2f}%   "
                  f"E={np.mean(energies):6.2f}+/-{np.std(energies):.2f}%   "
                  f"ripple={np.mean(ripples):6.2f}+/-{np.std(ripples):.2f}W")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "summary_multiseed.csv", index=False)

    # Mean+/-std bar chart of efficiency
    setup()
    pivot_m = summary.pivot(index="case", columns="algo", values="eta_mean")
    pivot_s = summary.pivot(index="case", columns="algo", values="eta_std")
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot_m.plot(kind="bar", yerr=pivot_s, ax=ax, capsize=3,
                 color=[COLOR.get(a) for a in pivot_m.columns])
    ax.set_ylabel("Tracking Efficiency η (%)")
    ax.set_ylim(40, 102)
    ax.set_title(f"Algorithm comparison ({n_seeds} seeds, mean +/- std)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "summary_multiseed.png", bbox_inches="tight")
    print("\nSaved", OUT_DIR / "summary_multiseed.csv",
          "and summary_multiseed.png")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    args = p.parse_args()
    main(args.seeds)
