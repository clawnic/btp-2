"""Run all algorithms on all PSC scenarios and produce comparison plots/CSVs.

Usage:
    python -m scripts.run_benchmark
"""
from __future__ import annotations
import os, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
from src.simulator import run
from src.scenarios import CASES
from src.algorithms import ALGORITHMS
from src.metrics import metrics
from src.plotting import plot_run, plot_summary

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(exist_ok=True)

def main():
    summary = []
    for case_name, profile in CASES.items():
        print(f"\n=== {case_name}: {profile.name} ===")
        runs = {}
        for algo_name, AlgoClass in ALGORITHMS.items():
            algo = AlgoClass()
            df = run(profile, algo)
            df.to_csv(OUT_DIR / f"{case_name}_{algo_name.replace('&','and')}.csv",
                      index=False)
            m = metrics(df)
            print(f"  {algo_name:<8s}  eta={m['eta_pct']:6.2f}%  "
                  f"t_track={m['t_track_s']:.3f}s  "
                  f"ripple={m['ripple_W']:6.2f}W  "
                  f"E={m['energy_pct']:5.2f}%")
            summary.append({"case": case_name, "algo": algo_name, **m})
            runs[algo_name] = df
        fig = plot_run(runs, profile.name)
        fig.savefig(OUT_DIR / f"{case_name}_traces.png", bbox_inches="tight")
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    fig = plot_summary(summary_df)
    fig.savefig(OUT_DIR / "summary_efficiency.png", bbox_inches="tight")
    print("\nSaved results to", OUT_DIR)

if __name__ == "__main__":
    main()
