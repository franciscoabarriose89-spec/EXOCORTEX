"""Projected complex-space geometry and Fubini-Study metric utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ComplexProjectiveSubstrate:
    """Represent a point in projective space as a normalized homogeneous vector.

    The state is stored as a complex vector and normalized to unit Hermitian norm.
    The Fubini-Study geodesic distance between two points is computed as the
    principal angle in the complex projective space CP^N.
    """

    vector: np.ndarray

    def __post_init__(self) -> None:
        vector = np.asarray(self.vector, dtype=np.complex128)
        if vector.size == 0:
            raise ValueError("projective states must contain at least one coordinate.")
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("projective state must be finite and non-zero.")
        object.__setattr__(self, "vector", vector / norm)

    @property
    def dimension(self) -> int:
        return int(self.vector.size)

    def to_real(self) -> np.ndarray:
        return np.real(np.asarray(self.vector, dtype=np.complex128))

    def fubini_study_metric(self, other: "ComplexProjectiveSubstrate") -> float:
        """Return the Fubini-Study distance between two projective points."""
        left = np.asarray(self.vector, dtype=np.complex128)
        right = np.asarray(other.vector, dtype=np.complex128)
        numerator = abs(np.vdot(left, right)) ** 2
        denominator = np.linalg.norm(left) ** 2 * np.linalg.norm(right) ** 2
        cosine = float(np.clip(numerator / denominator, 0.0, 1.0))
        return float(np.arccos(np.sqrt(cosine)))

    def __matmul__(self, other: "ComplexProjectiveSubstrate") -> complex:
        return float(np.vdot(self.vector, other.vector))


class KahlerianSubstrate:
    """Practical wrapper around complex projective states with zero-entropy checks."""

    def __init__(self, dimension: int = 3, scale: float = 1.0) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        self.dimension = int(dimension)
        self.scale = float(scale)

    def make_state(self, values: Sequence[complex]) -> ComplexProjectiveSubstrate:
        vector = self.scale * np.asarray(list(values), dtype=np.complex128)
        return ComplexProjectiveSubstrate(vector)

    @staticmethod
    def fubini_study_potential(vector: Iterable[complex]) -> float:
        values = np.asarray(list(vector), dtype=np.complex128)
        return float(np.log(1.0 + np.sum(np.abs(values) ** 2)))

    @staticmethod
    def entropy_delta(previous: Sequence[complex], current: Sequence[complex]) -> float:
        prev = np.asarray(list(previous), dtype=np.float64)
        curr = np.asarray(list(current), dtype=np.float64)
        return float(np.linalg.norm(curr - prev))


class FubiniStudyMetric:
    """Deterministic helper returning the canonical CP^N projective metric."""

    @staticmethod
    def metric(vector: Sequence[complex]) -> float:
        values = np.asarray(list(vector), dtype=np.complex128)
        return float(np.log(1.0 + np.sum(np.abs(values) ** 2)))


SustratoKahlariano = KahlerianSubstrate
