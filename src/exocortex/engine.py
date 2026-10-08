"""High-level orchestration for the Exocortex library."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .core.entropy import ZeroEntropyVerifier
from .core.substrate import KahlerianSubstrate
from .transforms.amador import AmadorTransform


class ExocortexEngine:
    """Coordinate the Exocortex core computations in a small, deterministic pipeline."""

    def __init__(self, alpha: float = 0.5, tau: float = 1.0) -> None:
        self.substrate = KahlerianSubstrate(dimension=3)
        self.transform = AmadorTransform(alpha=alpha, tau=tau)
        self.verifier = ZeroEntropyVerifier(tolerance=1e-15)

    def process(self, sample: Sequence[float], reference: Sequence[float]) -> np.ndarray:
        state = np.asarray(sample, dtype=np.float64)
        value = np.asarray(reference, dtype=np.float64)
        if state.shape != value.shape:
            raise ValueError("sample and reference must match in shape.")
        stabilized = self.transform.apply(state, value)
        self.verifier.assert_zero(state, stabilized)
        return stabilized

    def zero_entropy(self, previous: Sequence[float], current: Sequence[float]) -> float:
        return float(np.linalg.norm(np.asarray(current, dtype=np.float64) - np.asarray(previous, dtype=np.float64)))
