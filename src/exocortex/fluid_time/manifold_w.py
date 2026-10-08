"""Complex time manifold W = t + iψ and related utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ComplexTime:
    """Canonical representation of the complex time coordinate W = t + iψ."""

    t: float
    psi: float

    @property
    def w(self) -> complex:
        return complex(self.t, self.psi)

    def __complex__(self) -> complex:
        return self.w


class FluidTimeManifold:
    """Support a fluid-like complex time coordinate for smoothing and modulation."""

    def __init__(self, base_time: float = 0.0, psi: float = 0.0) -> None:
        self.base_time = float(base_time)
        self.psi = float(psi)

    def point(self, time: float | None = None, psi: float | None = None) -> ComplexTime:
        return ComplexTime(float(time if time is not None else self.base_time), float(psi if psi is not None else self.psi))

    @staticmethod
    def from_sequence(values: Sequence[float]) -> np.ndarray:
        return np.asarray(values, dtype=np.float64)

    def modulate(self, signal: Iterable[float], phase_shift: float = 0.0) -> np.ndarray:
        values = np.asarray(list(signal), dtype=np.float64)
        return values * np.exp(1j * phase_shift)


ManifoldW = FluidTimeManifold
