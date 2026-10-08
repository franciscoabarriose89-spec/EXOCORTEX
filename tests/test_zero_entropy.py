import numpy as np

from exocortex.core.entropy import ZeroEntropyVerifier, zero_entropy_delta


def test_zero_entropy_delta_is_zero_for_equal_states():
    previous = np.array([1.0, 2.0, 3.0], dtype=float)
    current = np.array([1.0, 2.0, 3.0], dtype=float)
    assert zero_entropy_delta(previous, current) == 0.0
    assert ZeroEntropyVerifier(tolerance=1e-15).check(previous, current)


def test_zero_entropy_delta_detects_change():
    previous = np.array([1.0, 2.0, 3.0], dtype=float)
    current = np.array([1.0 + 1e-14, 2.0, 3.0], dtype=float)
    delta = zero_entropy_delta(previous, current)
    assert delta > 0.0
    assert not ZeroEntropyVerifier(tolerance=1e-15).check(previous, current)
