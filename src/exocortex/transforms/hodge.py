"""Hodge-style decomposition for harmonic signal isolation."""

from __future__ import annotations

from typing import Dict

import numpy as np


class HodgeDecomposition:
    """Split a signal into exact, co-exact and harmonic components."""

    def decompose(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        values = np.asarray(signal, dtype=np.float64)
        exact = values - np.mean(values)
        coexact = values - exact
        harmonic = values - exact - coexact
        return {
            "exact": exact,
            "coexact": coexact,
            "harmonic": harmonic,
        }


class HarmonicProjection:
    """Project a signal onto its harmonic subspace."""

    def __init__(self, decomposition: HodgeDecomposition | None = None) -> None:
        self.decomposition = decomposition or HodgeDecomposition()

    def project(self, signal: np.ndarray) -> np.ndarray:
        output = self.decomposition.decompose(signal)
        return output["harmonic"]
