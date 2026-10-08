"""Interfaces for AGI and quantum interactions."""

from .agi_bridge import AGIBridge, ExocortexBridge
from .quantum_ent import QuantumEntityMapper, QuantumEntangler

__all__ = [
    "AGIBridge",
    "ExocortexBridge",
    "QuantumEntangler",
    "QuantumEntityMapper",
]
