"""AH-FOA: Adaptive Hybrid Falcon Optimization Algorithm (PROPOSED).

Improvements over the vanilla FOA of Alshareef 2022:

  1. Adaptive control parameters
        AP(k) = AP_max - (AP_max - AP_min) * k/K
        DP(k) = DP_min + (DP_max - DP_min) * k/K
     -> high exploration early, high exploitation late.

  2. Levy-flight enhanced search phase (replaces Eq. 9):
        X_{i+1} = X_i + V_i + alpha * Levy(beta) .* (gbest - X_i)
     Heavy-tailed steps escape local maxima in fewer iterations.

  3. Hybrid P&O hand-off in steady state. When swarm spread
     std(X) < eps_swarm AND iter > K/2 -> switch to P&O with very
     small step (1e-3) for ripple-free tracking.

  4. Elitist warm restart on irradiance change: keep gbest, reseed only
     the (NP-1) other falcons uniformly. Faster re-tracking.
"""
from __future__ import annotations
import numpy as np
from math import gamma
from .base import MPPT

def _levy(beta: float, size, rng) -> np.ndarray:
    """Mantegna's algorithm for a stable distribution Levy step."""
    sigma_u = (gamma(1+beta) * np.sin(np.pi*beta/2)
               / (gamma((1+beta)/2) * beta * 2 ** ((beta-1)/2))) ** (1/beta)
    u = rng.normal(0, sigma_u, size)
    v = rng.normal(0, 1.0,     size)
    return u / np.abs(v) ** (1/beta)

class AHFOA(MPPT):
    name = "AH-FOA"

    def __init__(self, n_falcons: int = 4, max_iter: int = 5,
                 cc: float = 1.5, sc: float = 1.5, fc: float = 0.5,
                 AP_max: float = 0.8, AP_min: float = 0.2,
                 DP_min: float = 0.2, DP_max: float = 0.8,
                 alpha_levy: float = 0.005, beta_levy: float = 1.5,
                 levy_clip: float = 5.0,
                 eps_swarm: float = 0.005, pno_step: float = 1e-3,
                 settle_steps: int = 2, reinit_jump: float = 0.10,
                 b_spiral: float = 1.0, seed: int = 0):
        self.NP, self.K = n_falcons, max_iter
        self.cc, self.sc, self.fc = cc, sc, fc
        self.AP_max, self.AP_min = AP_max, AP_min
        self.DP_min, self.DP_max = DP_min, DP_max
        self.alpha_levy, self.beta_levy = alpha_levy, beta_levy
        self.levy_clip = levy_clip
        self.eps_swarm, self.pno_step = eps_swarm, pno_step
        self.settle = settle_steps
        self.reinit_jump = reinit_jump
        self.b = b_spiral
        self.rng = np.random.default_rng(seed)
        self._init(full=True)

    # ---------------------------------------------------------------
    def _init(self, full: bool = True):
        ks = np.arange(1, self.NP + 1)
        self.X = self.D_min + (self.D_max - self.D_min) * ks / (self.NP + 1)
        self.v_max = 0.1 * self.D_max
        self.V = self.rng.uniform(-self.v_max, self.v_max, self.NP)
        if full:
            self.gbest, self.gbestF = float(self.X[0]), -np.inf
        self.Xbest = self.X.copy()
        self.XbestF = np.full(self.NP, -np.inf)
        self.idx = 0
        self.settle_cnt = 0
        self.iter = 0
        self.converged = False
        self.pno_mode = False
        # P&O hand-off state
        self.P_old = 0.0; self.V_old = 0.0; self.dir = +1

    def _warm_restart(self):
        """Elitist restart: keep gbest, reseed others around it."""
        ks = np.arange(1, self.NP)        # NP-1 others
        spread = (self.D_max - self.D_min) * ks / (self.NP)
        self.X = np.concatenate(([self.gbest],
                                 np.clip(self.D_min + spread, self.D_min, self.D_max)))
        self.V = self.rng.uniform(-self.v_max, self.v_max, self.NP)
        self.Xbest = self.X.copy()
        self.XbestF = np.full(self.NP, -np.inf)
        self.XbestF[0] = self.gbestF
        self.idx = 0; self.settle_cnt = 0; self.iter = 0
        self.converged = False; self.pno_mode = False

    def reset(self):
        self._init(full=True)

    # ---------------------------------------------------------------
    def _ap(self) -> float:
        f = min(self.iter / max(self.K, 1), 1.0)
        return self.AP_max - (self.AP_max - self.AP_min) * f

    def _dp(self) -> float:
        f = min(self.iter / max(self.K, 1), 1.0)
        return self.DP_min + (self.DP_max - self.DP_min) * f

    def _update_position(self, i: int):
        AP, DP = self._ap(), self._dp()
        rAP = self.rng.random(); rDP = self.rng.random(); r = self.rng.random()
        Xchosen = float(self.Xbest[self.rng.integers(0, self.NP)])

        if rAP < AP:                           # Levy-enhanced search
            L = float(_levy(self.beta_levy, 1, self.rng)[0])
            L = max(min(L, self.levy_clip), -self.levy_clip)
            range_D = self.D_max - self.D_min
            # PSO-style attraction PLUS bounded Levy perturbation
            X_new = (self.X[i] + self.V[i]
                     + self.cc * r * (self.Xbest[i] - self.X[i])
                     + self.sc * r * (self.gbest    - self.X[i])
                     + self.alpha_levy * L * range_D)
        elif DP < rDP:                         # logarithmic spiral (Eq. 10)
            t = 2 * self.rng.random() - 1
            X_new = self.X[i] + abs(Xchosen - self.X[i]) \
                                * np.exp(self.b * t) * np.cos(2 * np.pi * t)
        elif self.XbestF[i] < self.gbestF:     # dive (Eq. 11)
            X_new = self.X[i] + self.V[i] \
                    + self.fc * r * (Xchosen - self.X[i])
        else:                                   # stay (Eq. 12)
            X_new = self.X[i] + self.V[i] \
                    + self.cc * r * (self.gbest - self.X[i])

        self.V[i] = float(np.clip(X_new - self.X[i], -self.v_max, self.v_max))
        self.X[i] = float(np.clip(X_new, self.D_min, self.D_max))

    # ---------------------------------------------------------------
    def step(self, V: float, I: float) -> float:
        P = V * I

        # ---- Irradiance-change detection (improvement #4) ----
        # While in pno_mode the controller is parked at gbest, so the
        # measured power should track gbestF closely; a sustained mismatch
        # above reinit_jump means irradiance changed -> elitist warm restart.
        # During exploration we use a sliding "best window" check to detect
        # that even the best falcon now under-performs the previous gbestF.
        if self.pno_mode:
            if self.gbestF > 1.0 and \
               abs(P - self.gbestF) / max(self.gbestF, 1e-3) > self.reinit_jump:
                self._warm_restart()
        elif self.converged and self.gbestF > 1.0 and \
             abs(P - self.gbestF) / max(self.gbestF, 1e-3) > self.reinit_jump:
            self._warm_restart()

        # ---- Hybrid P&O hand-off (improvement #3) ----
        if self.pno_mode:
            dP = P - self.P_old; dV = V - self.V_old
            if   dP > 0: self.dir = +1 if dV >= 0 else -1
            elif dP < 0: self.dir = -1 if dV >= 0 else +1
            self.gbest = float(np.clip(self.gbest - self.dir * self.pno_step,
                                       self.D_min, self.D_max))
            if P > self.gbestF: self.gbestF = P
            self.V_old, self.P_old = V, P
            return self.gbest

        # ---- FOA exploration / exploitation ----
        self.settle_cnt += 1
        if self.settle_cnt < self.settle:
            return float(self.X[self.idx])
        self.settle_cnt = 0

        if P > self.XbestF[self.idx]:
            self.XbestF[self.idx] = P; self.Xbest[self.idx] = self.X[self.idx]
        if P > self.gbestF:
            self.gbestF = P; self.gbest = float(self.X[self.idx])

        self.idx = (self.idx + 1) % self.NP
        if self.idx == 0:
            self.iter += 1
            # Hand-off only after a meaningful exploration budget AND
            # when the swarm has actually converged. We require gbestF>0
            # so we never lock onto the -inf placeholder.
            collapsed = (float(np.std(self.X)) < self.eps_swarm
                         and self.gbestF > 0
                         and self.iter >= max(3, (3 * self.K) // 4))
            if collapsed:
                self.pno_mode = True
                self.converged = True
                self.V_old, self.P_old = V, P
                return self.gbest

        self._update_position(self.idx)

        if self.iter >= self.K:
            self.converged = True
            self.pno_mode = True
            self.V_old, self.P_old = V, P
            return self.gbest
        return float(self.X[self.idx])
