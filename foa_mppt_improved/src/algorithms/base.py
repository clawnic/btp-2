"""MPPT algorithm interface."""
from __future__ import annotations
from abc import ABC, abstractmethod

class MPPT(ABC):
    name: str = "MPPT"
    D_min: float = 0.10
    D_max: float = 0.90

    @abstractmethod
    def step(self, V: float, I: float) -> float:
        """Given the latest PV measurement, return the next duty cycle."""

    def reset(self) -> None:
        pass
