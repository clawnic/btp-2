"""Particle Swarm Optimization MPPT (baseline metaheuristic)."""
from __future__ import annotations
import numpy as np
from .base import MPPT

class PSO(MPPT):
    name = "PSO"

    def __init__(self, n_particles: int = 4, w: float = 0.4,
                 c1: float = 1.2, c2: float = 1.6,
                 settle_steps: int = 2, max_iter: int = 8,
                 reinit_jump: float = 0.10, seed: int = 0):
        self.N = n_particles
        self.w, self.c1, self.c2 = w, c1, c2
        self.settle = settle_steps
        self.K = max_iter
        self.reinit_jump = reinit_jump
        self.seed = seed
        self._init()

    def _init(self):
        rng = np.random.default_rng(self.seed)
        self.rng = rng
        self.X = rng.uniform(self.D_min, self.D_max, self.N)
        self.V = np.zeros(self.N)
        self.pbest_X = self.X.copy()
        self.pbest_F = np.full(self.N, -np.inf)
        self.gbest_X = self.X[0]
        self.gbest_F = -np.inf
        self.idx = 0
        self.settle_cnt = 0
        self.iter = 0
        self.converged = False
        self.last_P = 0.0

    def reset(self):
        self._init()

    def step(self, V: float, I: float) -> float:
        P = V * I
        # Detect irradiance change
        if self.converged and abs(P - self.gbest_F)/max(self.gbest_F, 1e-3) > self.reinit_jump:
            self._init()

        # Wait for converter to settle on the current candidate
        self.settle_cnt += 1
        if self.settle_cnt < self.settle:
            return float(self.X[self.idx])
        self.settle_cnt = 0

        # Evaluate
        if P > self.pbest_F[self.idx]:
            self.pbest_F[self.idx] = P
            self.pbest_X[self.idx] = self.X[self.idx]
        if P > self.gbest_F:
            self.gbest_F = P
            self.gbest_X = self.X[self.idx]

        # Move to next particle, update its velocity/position
        self.idx = (self.idx + 1) % self.N
        if self.idx == 0:
            self.iter += 1
            r1 = self.rng.random(self.N); r2 = self.rng.random(self.N)
            self.V = (self.w * self.V
                      + self.c1 * r1 * (self.pbest_X - self.X)
                      + self.c2 * r2 * (self.gbest_X - self.X))
            self.X = np.clip(self.X + self.V, self.D_min, self.D_max)
        if self.iter >= self.K:
            self.converged = True
            return float(self.gbest_X)
        return float(self.X[self.idx])
