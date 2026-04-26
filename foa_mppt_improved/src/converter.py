"""Quasi-static converter / operating-point map.

For algorithm benchmarking we treat the boost converter + load as an
*operating-point selector*: the duty cycle D in [D_min, D_max] selects a
voltage on the array's P-V curve. We add a first-order lag to mimic the
physical converter settling time so that the algorithms must wait between
samples (matches Alshareef's 10 ms MPPT sample time).
"""
from __future__ import annotations
import numpy as np
from .pv_model import string_pv_curve, ModuleParams, DEFAULT_MODULE

class OperatingPoint:
    """Map duty cycle D -> (V_pv, I_pv, P_pv) on the current PV curve."""

    def __init__(self, mp: ModuleParams = DEFAULT_MODULE,
                 D_min: float = 0.10, D_max: float = 0.90,
                 settle_tau: float = 0.005,   # seconds
                 sample_dt: float = 0.01):    # seconds
        self.mp = mp
        self.D_min, self.D_max = D_min, D_max
        self.tau = settle_tau
        self.dt = sample_dt
        self._V_curve = None
        self._I_curve = None
        self._Voc = None
        self._V_state = None  # filtered terminal voltage

    def set_irradiance(self, G_list, T_C: float = 25.0):
        V, I, _ = string_pv_curve(G_list, T_C, self.mp)
        self._V_curve, self._I_curve = V, I
        self._Voc = float(V.max())
        if self._V_state is None:
            self._V_state = self._Voc * 0.5

    def _D_to_V_target(self, D: float) -> float:
        D = float(np.clip(D, self.D_min, self.D_max))
        # Affine map: D=D_min -> V=Voc, D=D_max -> V=0.05*Voc
        a = (D - self.D_min) / (self.D_max - self.D_min)
        return (1 - a) * self._Voc + a * 0.05 * self._Voc

    def step(self, D: float):
        """Advance one MPPT sample: apply D, return measured (V,I,P)."""
        V_target = self._D_to_V_target(D)
        # First-order lag toward V_target
        alpha = 1.0 - np.exp(-self.dt / max(self.tau, 1e-6))
        self._V_state += alpha * (V_target - self._V_state)
        V = float(np.clip(self._V_state, self._V_curve.min(),
                          self._V_curve.max()))
        I = float(np.interp(V, self._V_curve, self._I_curve))
        return V, I, V * I
