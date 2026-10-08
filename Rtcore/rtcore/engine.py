"""Orchestration of mesh ingestion, stabilization, memory folding, and shading."""

import os
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch

from . import DEVICE
from .ingestion import IngestorBiometrico
from .math import TransformadaAmador
from .render import MotorFresnel
from .topology import SustratoKahlariano


class ExocortexEngine:
    """Run the deterministic mesh-to-frame RT-Core processing pipeline."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        """Initialize pipeline components on one selected execution device."""
        self.device = torch.device(device) if device is not None else DEVICE
        self.ingestor = IngestorBiometrico(self.device)
        self.stabilizer = TransformadaAmador(device=self.device)
        self.renderer = MotorFresnel(self.device)
        self.memory: Optional[SustratoKahlariano] = None
        self._previous_state: Optional[torch.Tensor] = None

    def process_mesh(
        self,
        path: Union[str, os.PathLike],
        resolution: int = 256,
        alpha: float = 0.5,
        tau: float = 1.0,
        phi: float = 0.0,
        material: Optional[Dict[str, object]] = None,
        view_direction: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> torch.Tensor:
        """Process an OBJ mesh and return an RGB frame tensor on the selected device."""
        vertices, _faces = self.ingestor.load_mesh(path)
        return self.process_vertices(
            vertices,
            resolution=resolution,
            alpha=alpha,
            tau=tau,
            phi=phi,
            material=material,
            view_direction=view_direction,
        )

    def process_vertices(
        self,
        vertices: torch.Tensor,
        resolution: int = 256,
        alpha: float = 0.5,
        tau: float = 1.0,
        phi: float = 0.0,
        material: Optional[Dict[str, object]] = None,
        view_direction: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> torch.Tensor:
        """Run the processing stages on preloaded vertices and return an RGB frame."""
        density = self.ingestor.rasterize(vertices, resolution)
        if self._previous_state is None or self._previous_state.shape != density.shape:
            self._previous_state = torch.zeros_like(density)
            self.memory = SustratoKahlariano(density.shape, device=self.device)

        stabilized = self.stabilizer.apply(
            self._previous_state, density, alpha=alpha, tau=tau
        )
        self._previous_state = stabilized
        if self.memory is None:
            raise RuntimeError("Memory substrate was not initialized.")
        memory_state = self.memory.fold(stabilized, phi=phi)
        return self.renderer.shade(
            memory_state.real, material=material, view_direction=view_direction
        )

    @staticmethod
    def to_opencv(frame: torch.Tensor) -> np.ndarray:
        """Transfer an RGB frame to CPU and return an 8-bit BGR OpenCV image."""
        frame_tensor = torch.as_tensor(frame)
        if frame_tensor.ndim != 3 or frame_tensor.shape[-1] != 3:
            raise ValueError("frame must have shape (height, width, 3).")
        rgb = frame_tensor.detach().clamp(0.0, 1.0).mul(255.0).round()
        return rgb.to(device="cpu", dtype=torch.uint8).numpy()[..., ::-1].copy()
