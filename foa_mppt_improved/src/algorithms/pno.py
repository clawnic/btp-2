"""Conventional Perturb & Observe."""
from __future__ import annotations
from .base import MPPT

class PnO(MPPT):
    name = "P&O"

    def __init__(self, step: float = 0.005, D0: float = 0.5):
        self.dD = step
        self.D = D0
        self.V_old = 0.0
        self.P_old = 0.0
        self.dir   = +1

    def reset(self):
        self.V_old = 0.0; self.P_old = 0.0; self.dir = +1

    def step(self, V: float, I: float) -> float:
        P = V * I
        dP = P - self.P_old
        dV = V - self.V_old
        if dP > 0:
            self.dir = +1 if dV >= 0 else -1
        elif dP < 0:
            self.dir = -1 if dV >= 0 else +1
        # Increasing D on a boost converter -> lower V_pv
        self.D = max(self.D_min, min(self.D_max, self.D - self.dir * self.dD))
        self.V_old, self.P_old = V, P
        return self.D
