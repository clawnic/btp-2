"""Vanilla Falcon Optimization Algorithm MPPT (faithful to Alshareef 2022).

Equations referenced are from M.J. Alshareef, "An Effective Falcon
Optimization Algorithm Based MPPT Under Partial Shaded Photovoltaic
Systems", IEEE Access, vol. 10, pp. 131345-131360, 2022.

  - Eq. (5)  uniform deterministic init  d^k_0 = k/(ss+1)
  - Eq. (6,7) v_max = 0.1*ub, v_min = -v_max
  - Eq. (8)  fitness f(d) = max P_pv(d)
  - Eq. (9)  search-phase update (PSO-like)
  - Eq. (10) logarithmic-spiral approach
  - Eq. (11) dive
  - Eq. (12) stay near best
"""
from __future__ import annotations
import numpy as np
from .base import MPPT

class FOA(MPPT):
    name = "FOA"

    def __init__(self, n_falcons: int = 4, max_iter: int = 8,
                 cc: float = 1.5, sc: float = 1.5, fc: float = 0.5,
                 AP: float = 0.5, DP: float = 0.5,
                 settle_steps: int = 2, reinit_jump: float = 0.10,
                 b_spiral: float = 1.0, seed: int = 0):
        self.NP = n_falcons
        self.K = max_iter
        self.cc, self.sc, self.fc = cc, sc, fc
        self.AP, self.DP = AP, DP
        self.settle = settle_steps
        self.reinit_jump = reinit_jump
        self.b = b_spiral
        self.rng = np.random.default_rng(seed)
        self._init()

    def _init(self):
        # Eq. (5): deterministic uniform initialisation
        ks = np.arange(1, self.NP + 1)
        self.X = self.D_min + (self.D_max - self.D_min) * ks / (self.NP + 1)
        v_max = 0.1 * self.D_max
        self.V = self.rng.uniform(-v_max, v_max, self.NP)
        self.v_max = v_max
        self.Xbest = self.X.copy()
        self.XbestF = np.full(self.NP, -np.inf)
        self.gbest = float(self.X[0])
        self.gbestF = -np.inf
        self.idx = 0
        self.settle_cnt = 0
        self.iter = 0
        self.converged = False

    def reset(self):
        self._init()

    def _update_position(self, i: int):
        """Generate new position for falcon i using Eqs. (9-12)."""
        rAP = self.rng.random()
        rDP = self.rng.random()
        r   = self.rng.random()
        # randomly chosen prey from personal-best memory
        Xchosen = float(self.Xbest[self.rng.integers(0, self.NP)])

        if rAP < self.AP:                                          # Eq. (9)
            X_new = self.X[i] + self.V[i] \
                    + self.cc * r * (self.Xbest[i] - self.X[i]) \
                    + self.sc * r * (self.gbest    - self.X[i])
        elif self.DP < rDP:                                        # Eq. (10)
            t = 2 * self.rng.random() - 1
            X_new = self.X[i] + abs(Xchosen - self.X[i]) \
                                * np.exp(self.b * t) * np.cos(2 * np.pi * t)
        elif self.XbestF[i] < self.gbestF:                         # Eq. (11)
            X_new = self.X[i] + self.V[i] \
                    + self.fc * r * (Xchosen - self.X[i])
        else:                                                       # Eq. (12)
            X_new = self.X[i] + self.V[i] \
                    + self.cc * r * (self.gbest - self.X[i])

        self.V[i] = float(np.clip(X_new - self.X[i], -self.v_max, self.v_max))
        self.X[i] = float(np.clip(X_new, self.D_min, self.D_max))

    def step(self, V: float, I: float) -> float:
        P = V * I
        # Irradiance-change re-init (full reset; AH-FOA does this smarter)
        if self.converged and abs(P - self.gbestF)/max(self.gbestF, 1e-3) > self.reinit_jump:
            self._init()

        self.settle_cnt += 1
        if self.settle_cnt < self.settle:
            return float(self.X[self.idx])
        self.settle_cnt = 0

        # Evaluate (Eq. 8)
        if P > self.XbestF[self.idx]:
            self.XbestF[self.idx] = P
            self.Xbest[self.idx]  = self.X[self.idx]
        if P > self.gbestF:
            self.gbestF = P
            self.gbest  = float(self.X[self.idx])

        # Advance to next falcon and update its position
        self.idx = (self.idx + 1) % self.NP
        if self.idx == 0:
            self.iter += 1
        self._update_position(self.idx)

        if self.iter >= self.K:
            self.converged = True
            return float(self.gbest)
        return float(self.X[self.idx])
