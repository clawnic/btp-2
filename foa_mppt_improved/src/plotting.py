"""Plotting helpers for thesis-quality figures."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLT_RC = {"font.family": "serif", "font.size": 11,
         "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130}

def setup():
    plt.rcParams.update(PLT_RC)

COLOR = {"P&O":"#94a3b8", "PSO":"#60a5fa", "FOA":"#0d9488", "AH-FOA":"#dc2626"}

def plot_pv_curve(V, I, P, title="P-V / I-V"):
    setup()
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(V, P, color="#0d9488", lw=2, label="P (W)")
    ax1.set_xlabel("V (V)"); ax1.set_ylabel("Power (W)", color="#0d9488")
    ax2 = ax1.twinx()
    ax2.plot(V, I, color="#dc2626", lw=1.4, ls="--", label="I (A)")
    ax2.set_ylabel("Current (A)", color="#dc2626")
    ax1.set_title(title)
    return fig

def plot_run(runs: dict, case_name: str):
    """runs : {algo_name: dataframe} for one case."""
    setup()
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    for name, df in runs.items():
        c = COLOR.get(name, None)
        axes[0].plot(df["t"], df["P"], label=name, lw=1.3, color=c)
        axes[1].plot(df["t"], df["V"], label=name, lw=1.0, color=c)
        axes[2].plot(df["t"], df["D"], label=name, lw=1.0, color=c)
    # GMPP reference
    df0 = next(iter(runs.values()))
    axes[0].plot(df0["t"], df0["P_gmpp"], "k--", lw=1, label="GMPP")
    axes[0].set_ylabel("Power (W)"); axes[0].legend(loc="lower right")
    axes[1].set_ylabel("V_pv (V)")
    axes[2].set_ylabel("Duty D"); axes[2].set_xlabel("Time (s)")
    fig.suptitle(case_name)
    fig.tight_layout()
    return fig

def plot_summary(summary: pd.DataFrame):
    """Bar chart of efficiency across cases & algorithms."""
    setup()
    pivot = summary.pivot(index="case", columns="algo", values="eta_pct")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot.plot(kind="bar", ax=ax,
               color=[COLOR.get(a) for a in pivot.columns])
    ax.set_ylabel("Tracking Efficiency η (%)")
    ax.set_ylim(60, 101)
    ax.set_title("Algorithm comparison across PSC cases")
    ax.legend(title="Algorithm")
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h):
            ax.text(p.get_x() + p.get_width()/2, h + 0.2,
                    f"{h:.1f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig
