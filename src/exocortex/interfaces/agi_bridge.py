"""AGI communication protocols for Exocortex."""

from __future__ import annotations

from typing import Any, Dict


class AGIBridge:
    """Serialize and deserialize AGI messages through an Exocortex bridge."""

    def __init__(self, protocol: str = "exocortex-v1") -> None:
        self.protocol = protocol

    def encode(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"protocol": self.protocol, "payload": payload}

    def decode(self, message: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(message, dict) or "payload" not in message:
            raise ValueError("message must contain a payload field.")
        return dict(message["payload"])


class ExocortexBridge(AGIBridge):
    """Convenience alias for the canonical bridge implementation."""

    pass
