import numpy as np

from exocortex.transforms.amador import AmadorTransform


def test_amador_transform_preserves_invariant_signal():
    transform = AmadorTransform(alpha=0.5, tau=0.2)
    state = np.array([1.0, 2.0, 3.0], dtype=float)
    value = np.array([1.0, 2.1, 3.0], dtype=float)
    result = transform.apply(state, value)
    expected = np.array([1.0, 2.0, 3.0], dtype=float) * 0.5 + np.array([1.0, 2.1, 3.0], dtype=float) * 0.5 - np.array([0.0, 0.1, 0.0], dtype=float)
    np.testing.assert_allclose(result, expected)


def test_amador_transform_reduces_large_residual():
    transform = AmadorTransform(alpha=0.25, tau=0.5)
    state = np.array([0.0, 0.0], dtype=float)
    value = np.array([10.0, 10.0], dtype=float)
    result = transform.apply(state, value)
    expected = 0.25 * state + 0.75 * value - np.array([0.0, 0.0], dtype=float)
    np.testing.assert_allclose(result, expected)
