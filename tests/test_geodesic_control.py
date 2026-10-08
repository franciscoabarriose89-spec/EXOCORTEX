import numpy as np

from exocortex.robotics.geodesics import GeodesicController


def test_geodesic_controller_is_deterministic():
    controller = GeodesicController(max_speed=5.0)
    assert controller.deterministic_latency() == 0.0

    start = np.array([0.0, 0.0], dtype=float)
    target = np.array([10.0, 0.0], dtype=float)
    path = controller.step(start, target, dt=0.5)
    assert path.shape == start.shape
    assert np.isclose(np.linalg.norm(path - target), 5.0, atol=1e-12)
