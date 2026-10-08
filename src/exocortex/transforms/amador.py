"""Amador transform used to filter stochastic noise while retaining harmonic content."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


class AmadorTransform:
    """Apply the Amador transform to a state-value pair.

    The transform blends the previous state with the new measurement and removes
    any residual deviation below a threshold tau while preserving the invariant
    harmonic component.
    """

    def __init__(self, alpha: float = 0.5, tau: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.tau = float(tau)
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in the closed interval [0, 1].")
        if self.tau < 0.0:
            raise ValueError("tau must be non-negative.")

    def apply(
        self,
        state: Sequence[float] | Iterable[float] | np.ndarray,
        value: Sequence[float] | Iterable[float] | np.ndarray,
    ) -> np.ndarray:
        state_array = np.asarray(list(state), dtype=np.float64)
        value_array = np.asarray(list(value), dtype=np.float64)
        if state_array.shape != value_array.shape:
            raise ValueError("state and value must have the same shape.")
        residual = np.abs(value_array - state_array)
        correction = np.where(residual <= self.tau, residual, 0.0)
        return self.alpha * state_array + (1.0 - self.alpha) * value_array - correction

    __call__ = apply


TransformadaAmador = AmadorTransform
