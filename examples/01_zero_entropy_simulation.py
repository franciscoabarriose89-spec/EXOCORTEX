"""Minimal zero-entropy simulation."""

import numpy as np

from exocortex.core.entropy import ZeroEntropyVerifier


if __name__ == "__main__":
    previous = np.array([1.0, 2.0, 3.0], dtype=float)
    current = np.array([1.0, 2.0, 3.0], dtype=float)
    verifier = ZeroEntropyVerifier(tolerance=1e-15)
    print(verifier.check(previous, current))
