"""Quantum state mapping for Exocortex entity interactions."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np


class QuantumEntityMapper:
    """Map tensor-like states to a finite-dimensional quantum configuration."""

    def __init__(self, dimension: int = 2) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        self.dimension = int(dimension)

    def map_state(self, state: Sequence[float] | Iterable[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(list(state), dtype=np.float64)
        return np.asarray([np.real(value) for value in values], dtype=np.float64)


class QuantumEntangler:
    """Create a low-dimensional entangled representation for a state vector."""

    def __init__(self, coupling: float = 1.0) -> None:
        self.coupling = float(coupling)

    def entangle(self, state: Sequence[float] | Iterable[float] | np.ndarray) -> Dict[str, float]:
        values = np.asarray(list(state), dtype=np.float64)
        if values.size == 0:
            raise ValueError("state cannot be empty.")
        norm = float(np.linalg.norm(values))
        return {"norm": norm, "coupling": self.coupling, "phase": float(np.angle(values[0]))}
