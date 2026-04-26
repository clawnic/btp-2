"""Closed-loop simulation harness."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .converter import OperatingPoint
from .pv_model  import gmpp
from .scenarios import IrradianceProfile
from .algorithms.base import MPPT

def run(profile: IrradianceProfile, algo: MPPT,
        sample_dt: float = 0.01, settle_tau: float = 0.005) -> pd.DataFrame:
    op = OperatingPoint(settle_tau=settle_tau, sample_dt=sample_dt)
    op.set_irradiance(profile.segments[0][1])
    algo.reset()

    n = int(profile.duration_s / sample_dt) + 1
    rows = []
    D = 0.5
    last_G = profile.segments[0][1]
    Vm, Im, Pm = op.step(D)
    P_gmpp = gmpp(last_G)[2]

    for k in range(n):
        t = k * sample_dt
        G = profile.at(t)
        if G != last_G:
            op.set_irradiance(G)
            last_G = G
            P_gmpp = gmpp(G)[2]
        Vm, Im, Pm = op.step(D)
        D = float(algo.step(Vm, Im))
        rows.append((t, Vm, Im, Pm, D, P_gmpp,
                     G[0], G[1], G[2]))

    return pd.DataFrame(rows, columns=[
        "t", "V", "I", "P", "D", "P_gmpp", "G1", "G2", "G3"])
