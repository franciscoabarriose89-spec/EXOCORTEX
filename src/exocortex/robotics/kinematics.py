"""Robotic kinematics for industrial and geodesic actuation."""

from __future__ import annotations

from typing import Sequence

import numpy as np


class IndustrialKinematicsController:
    """A simple deterministic actuator controller for a robotic mechanism."""

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = float(gain)

    def command(self, current: Sequence[float], target: Sequence[float]) -> np.ndarray:
        current_array = np.asarray(current, dtype=np.float64)
        target_array = np.asarray(target, dtype=np.float64)
        if current_array.shape != target_array.shape:
            raise ValueError("current and target must share the same shape.")
        return current_array + self.gain * (target_array - current_array)


class RobotKinematicsController(IndustrialKinematicsController):
    """Compatibility alias for the canonical industrial kinematics controller."""

    pass
