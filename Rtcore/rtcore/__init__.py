"""Modular tensor and mesh processing utilities for RT-Core."""

import torch


def _select_device() -> torch.device:
    """Select the best available PyTorch device in CUDA/MPS/CPU order."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _select_device()

from .engine import ExocortexEngine
from .ingestion import IngestorBiometrico
from .math import TransformadaAmador
from .render import MotorFresnel
from .topology import BovedaTopologica, SustratoKahlariano

__all__ = [
    "DEVICE",
    "BovedaTopologica",
    "ExocortexEngine",
    "IngestorBiometrico",
    "MotorFresnel",
    "SustratoKahlariano",
    "TransformadaAmador",
]
