"""Plot the static P-V / I-V characteristic of the 3-module string for
each PSC case. Useful for the introduction figure of your report.

Usage:  python -m scripts.plot_pv_curves
"""
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from src.pv_model import string_pv_curve, gmpp
from src.scenarios import CASES
from src.plotting import setup

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT_DIR.mkdir(exist_ok=True)

def main():
    setup()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    items = [(k, v) for k, v in CASES.items() if not k.startswith("Case5")]
    for ax, (name, prof) in zip(axes.flat, items):
        G = prof.segments[0][1]
        V, I, P = string_pv_curve(G)
        Vg, Ig, Pg = gmpp(G)
        ax.plot(V, P, color="#0d9488", lw=2)
        ax.plot(Vg, Pg, "ro", ms=8, label=f"GMPP={Pg:.1f}W")
        ax.set_title(f"{prof.name}\nG={G} W/m^2")
        ax.set_xlabel("V (V)"); ax.set_ylabel("P (W)")
        ax.legend()
    fig.tight_layout()
    out = OUT_DIR / "pv_curves.png"
    fig.savefig(out, bbox_inches="tight")
    print("Saved", out)

if __name__ == "__main__":
    main()
