"""Partial-shading scenarios mirroring Alshareef 2022 (3 modules)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class IrradianceProfile:
    """Step-wise irradiance changes:  list of (t_start_s, [G1, G2, G3])."""
    name: str
    duration_s: float
    segments: List[Tuple[float, list]] = field(default_factory=list)

    def at(self, t: float) -> list:
        active = self.segments[0][1]
        for ts, g in self.segments:
            if t >= ts:
                active = g
        return active

CASES = {
    "Case1_Uniform":  IrradianceProfile(
        "Case 1 - Uniform 1000 W/m^2", 1.0,
        [(0.0, [1000, 1000, 1000])]),
    "Case2_Light":    IrradianceProfile(
        "Case 2 - Light PSC",          1.0,
        [(0.0, [1000, 700, 500])]),
    "Case3_Heavy":    IrradianceProfile(
        "Case 3 - Heavy PSC",          1.0,
        [(0.0, [1000, 700, 300])]),
    "Case4_Severe":   IrradianceProfile(
        "Case 4 - Severe PSC",         1.0,
        [(0.0, [1000, 300, 200])]),
    "Case5_Dynamic":  IrradianceProfile(
        "Case 5 - Dynamic step changes", 3.0,
        [(0.0,  [1000, 1000, 1000]),     # uniform
         (0.6,  [1000,  700,  300]),     # heavy PSC
         (1.2,  [ 800,  500,  200]),     # severe PSC, GMPP shifts
         (1.8,  [ 900,  900,  400]),     # light PSC, GMPP shifts again
         (2.4,  [ 600,  600,  600])]),   # back to uniform low irradiance
}
