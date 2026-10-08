"""Zero-entropy verification utilities."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def zero_entropy_delta(
    previous: Sequence[complex] | Iterable[float] | np.ndarray,
    current: Sequence[complex] | Iterable[float] | np.ndarray,
) -> float:
    """Return the Euclidean norm of the state delta for a zero-entropy check.

    The canonical Exocortex requirement is that the entropy variation stays below
    machine precision, which is represented by the tolerance parameter here.
    """
    prev = np.asarray(list(previous), dtype=np.float64)
    curr = np.asarray(list(current), dtype=np.float64)
    if prev.shape != curr.shape:
        raise ValueError("previous and current states must have matching shape.")
    return float(np.linalg.norm(curr - prev))


class ZeroEntropyVerifier:
    """Validate that a signal does not drift beyond the numerical zero tolerance."""

    def __init__(self, tolerance: float = 1e-15) -> None:
        if tolerance <= 0.0:
            raise ValueError("tolerance must be strictly positive.")
        self.tolerance = float(tolerance)

    def check(
        self,
        previous: Sequence[complex] | Iterable[float] | np.ndarray,
        current: Sequence[complex] | Iterable[float] | np.ndarray,
    ) -> bool:
        return zero_entropy_delta(previous, current) <= self.tolerance

    def assert_zero(
        self,
        previous: Sequence[complex] | Iterable[float] | np.ndarray,
        current: Sequence[complex] | Iterable[float] | np.ndarray,
    ) -> None:
        if not self.check(previous, current):
            delta = zero_entropy_delta(previous, current)
            raise ValueError(
                "Entropy drift exceeds tolerance: "
                f"delta={delta:.3e} > tolerance={self.tolerance:.3e}"
            )


def assert_zero_entropy(
    previous: Sequence[complex] | Iterable[float] | np.ndarray,
    current: Sequence[complex] | Iterable[float] | np.ndarray,
    tolerance: float = 1e-15,
) -> None:
    """Raise if the variation is above the zero-entropy tolerance."""
    ZeroEntropyVerifier(tolerance).assert_zero(previous, current)
