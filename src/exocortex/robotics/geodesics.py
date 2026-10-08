"""Deterministic geodesic navigation and control primitives."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class GeodesicNavigator:
    """Compute a deterministic geodesic step between start and target vectors."""

    def __init__(self, max_speed: float = 1.0) -> None:
        self.max_speed = float(max_speed)

    def navigate(
        self,
        start: Sequence[float],
        target: Sequence[float],
        dt: float = 1.0,
    ) -> np.ndarray:
        start_array = np.asarray(start, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)
        if start_array.shape != target_array.shape:
            raise ValueError("start and target must share the same shape.")
        displacement = target_array - start_array
        distance = float(np.linalg.norm(displacement))
        if distance == 0.0:
            return start_array.copy()

        if 0.0 <= dt <= 1.0:
            fraction = float(dt)
        else:
            fraction = min(float(self.max_speed * dt / max(distance, 1e-12)), 1.0)

        return start_array + fraction * displacement


class GeodesicController:
    """Expose a deterministic controller whose latency is effectively zero."""

    def __init__(self, max_speed: float = 1.0) -> None:
        self.navigator = GeodesicNavigator(max_speed=max_speed)

    @staticmethod
    def deterministic_latency() -> float:
        return 0.0

    def step(self, start: Sequence[float], target: Sequence[float], dt: float = 1.0) -> np.ndarray:
        return self.navigator.navigate(start, target, dt=dt)
