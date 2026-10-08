"""Real-time geodesic robotics demo."""

from exocortex.robotics.geodesics import GeodesicController


if __name__ == "__main__":
    controller = GeodesicController(max_speed=3.0)
    print(controller.step([0.0, 0.0], [6.0, 0.0], dt=1.0))
    print(controller.deterministic_latency())
