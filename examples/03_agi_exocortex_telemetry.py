"""Telemetry bridge example for AGI integration."""

from exocortex.interfaces.agi_bridge import AGIBridge


if __name__ == "__main__":
    bridge = AGIBridge(protocol="exocortex-v1")
    message = bridge.encode({"state": [1.0, 2.0, 3.0]})
    print(message)
    print(bridge.decode(message))
