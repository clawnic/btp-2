"""Tracking-performance metrics (Alshareef 2022 Eqs. 15-18)."""
from __future__ import annotations
import numpy as np
import pandas as pd

def metrics(df: pd.DataFrame, settle_window: float = 0.1,
            eta_thr: float = 0.95, v_tol: float = 0.02) -> dict:
    """Compute tracking metrics from a simulation log.

    settle_window : last fraction of the run used for steady-state stats.
    """
    n_ss = max(int(len(df) * settle_window), 5)
    ss = df.iloc[-n_ss:]
    P_ss = float(ss["P"].mean())
    V_ss = float(ss["V"].mean())
    P_gmpp = float(df["P_gmpp"].iloc[-1])
    eta = 100.0 * P_ss / max(P_gmpp, 1e-9)

    ripple_W = float(ss["P"].std())
    P_target = 0.98 * P_ss
    cross = df.index[df["P"] >= P_target]
    t_track = float(df["t"].iloc[cross[0]]) if len(cross) else float("nan")

    # Energy yield (np.trapezoid in NumPy >= 2.0; fall back to trapz)
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    energy_Ws = float(_trap(df["P"], df["t"]))
    ideal_Ws  = float(_trap(df["P_gmpp"], df["t"]))
    energy_pct = 100.0 * energy_Ws / max(ideal_Ws, 1e-9)

    success = (eta >= 100*eta_thr)
    return dict(eta_pct=eta, P_ss_W=P_ss, V_ss_V=V_ss, P_gmpp_W=P_gmpp,
                ripple_W=ripple_W, t_track_s=t_track,
                energy_pct=energy_pct, success=bool(success))
