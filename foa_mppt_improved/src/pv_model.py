"""3-module series PV array with bypass diodes (single-diode model).

Equations follow Alshareef 2022 (IEEE Access vol.10), Eqs. (1)-(4):

    I = I_ph - I_o [exp((V+I*Rs)/(a*Vt*Ns)) - 1] - (V+I*Rs)/Rsh
    I_ph(G,T) = (I_ph_STC + ki*(T-T_STC)) * G/G_STC

For a string with N modules in series and an ideal bypass diode in
anti-parallel across each module, we solve per-module V(I) and clamp
the module voltage at -V_d (~ -0.7 V) when the bypass diode conducts.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq

K_BOLTZ = 1.380649e-23
Q_ELEC  = 1.602176634e-19
T_STC   = 25.0 + 273.15        # K
G_STC   = 1000.0               # W/m^2

@dataclass
class ModuleParams:
    """Sharp NT-180/4U-like single-diode parameters (one module = 50 cells)."""
    Ns: int = 50               # cells in series per module
    Np: int = 1                # cells in parallel per module
    Iph_stc: float = 5.40      # A
    Io_stc:  float = 1.5e-9    # A
    a:       float = 1.30      # ideality factor
    Rs:      float = 0.40      # ohm
    Rsh:    float = 600.0      # ohm
    ki:      float = 0.0024    # A/K
    kv:      float = -0.144    # V/K (not used here)
    Voc_stc: float = 30.5      # V
    Isc_stc: float = 5.40      # A

DEFAULT_MODULE = ModuleParams()
V_BYPASS = -0.7                # forward drop of bypass diode

def _vt(T_kelvin: float) -> float:
    return K_BOLTZ * T_kelvin / Q_ELEC

def module_iv_current(V: float, G: float, T_C: float = 25.0,
                      mp: ModuleParams = DEFAULT_MODULE) -> float:
    """Return module current I (A) at terminal voltage V (V).

    If V <= V_BYPASS the bypass diode is conducting and the module passes
    through whatever string current is forced (handled at string level).
    """
    if V <= V_BYPASS:
        return mp.Iph_stc * G / G_STC                # nominal short-cct
    Tk = T_C + 273.15
    Iph = (mp.Iph_stc + mp.ki * (Tk - T_STC)) * G / max(G_STC, 1e-6)
    Io = mp.Io_stc * (Tk / T_STC) ** 3 * np.exp(
        (mp.Voc_stc / (mp.a * _vt(T_STC) * mp.Ns)) *
        (1 - T_STC / Tk)
    )
    Vt = _vt(Tk) * mp.a * mp.Ns
    # Implicit equation: I = Iph - Io*(exp((V+I*Rs)/Vt)-1) - (V+I*Rs)/Rsh
    def f(I):
        return Iph - Io*(np.exp(np.clip((V + I*mp.Rs)/Vt, -50, 50)) - 1) \
                   - (V + I*mp.Rs)/mp.Rsh - I
    try:
        return brentq(f, -2.0, mp.Iph_stc * 1.5, xtol=1e-6)
    except ValueError:
        return 0.0

def module_voltage_at_current(I_string: float, G: float, T_C: float = 25.0,
                              mp: ModuleParams = DEFAULT_MODULE) -> float:
    """Invert: given the string current I_string forced through a module
    (irradiance G), return the module's terminal voltage V_mod.

    If I_string exceeds the module's photocurrent, the bypass diode kicks
    in and V_mod clamps to V_BYPASS.
    """
    Iph_eff = (mp.Iph_stc + mp.ki * (T_C + 273.15 - T_STC)) * G / G_STC
    if I_string >= Iph_eff - 1e-3:
        return V_BYPASS
    # bracket V in [V_BYPASS, Voc]
    Voc_eff = mp.Voc_stc * (1.0 if G < 1 else 1.0)  # simple, ignore Voc(G)
    def f(V):
        return module_iv_current(V, G, T_C, mp) - I_string
    try:
        return brentq(f, V_BYPASS + 1e-3, Voc_eff, xtol=1e-5)
    except ValueError:
        return V_BYPASS

def string_pv_curve(G_list, T_C: float = 25.0,
                    mp: ModuleParams = DEFAULT_MODULE,
                    n_points: int = 400):
    """Compute the (V_string, I_string, P_string) curve for a series
    string whose modules have irradiances G_list (one per module)."""
    Iph_max = max(
        (mp.Iph_stc + mp.ki * (T_C + 273.15 - T_STC)) * g / G_STC
        for g in G_list
    )
    I_grid = np.linspace(0.0, max(Iph_max, 0.1) * 1.05, n_points)
    V_string = np.zeros_like(I_grid)
    for i, Iv in enumerate(I_grid):
        V_string[i] = sum(module_voltage_at_current(Iv, g, T_C, mp)
                          for g in G_list)
    P_string = V_string * I_grid
    # Ensure monotone-ascending V for downstream interp
    order = np.argsort(V_string)
    return V_string[order], I_grid[order], P_string[order]

def gmpp(G_list, T_C: float = 25.0,
         mp: ModuleParams = DEFAULT_MODULE) -> tuple[float, float, float]:
    """Return (V_gmpp, I_gmpp, P_gmpp) by exhaustive curve search."""
    V, I, P = string_pv_curve(G_list, T_C, mp)
    k = int(np.argmax(P))
    return float(V[k]), float(I[k]), float(P[k])
