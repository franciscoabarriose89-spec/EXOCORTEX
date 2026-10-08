"""Holomorphic modulation and Ricci-type smoothing utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class HolomorphicPlate:
    """Apply a phase-based holomorphic smoothing pass to a metric tensor."""

    def __init__(self, lambda_value: float = 1.0) -> None:
        self.lambda_value = float(lambda_value)

    def smooth(self, metric: Sequence[float] | np.ndarray, phase_shift: float = 0.0) -> np.ndarray:
        values = np.asarray(list(metric), dtype=np.float64)
        phase = np.exp(1j * phase_shift)
        return values * phase + self.lambda_value * np.ones_like(values, dtype=np.float64)

    def ricci_like_smoothing(self, metric: Sequence[float] | np.ndarray, damping: float = 0.1) -> np.ndarray:
        values = np.asarray(list(metric), dtype=np.float64)
        return values - damping * (values - np.mean(values))


class PhaseModulator:
    """Directly modulate a signal using a destructive phase flip."""

    def modulate(self, signal: Sequence[float] | np.ndarray, phase_shift: float = np.pi) -> np.ndarray:
        values = np.asarray(list(signal), dtype=np.float64)
        return values * np.exp(1j * phase_shift)
