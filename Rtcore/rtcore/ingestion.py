"""Mesh ingestion, deterministic normalization, and tensor rasterization."""

import os
from typing import Optional, Tuple, Union

import numpy as np
import torch
import trimesh

from . import DEVICE


class IngestorBiometrico:
    """Load, normalize, and project triangle meshes into 2D density tensors."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        """Set the device used for returned mesh and raster tensors."""
        self.device = torch.device(device) if device is not None else DEVICE

    def load_mesh(
        self, path: Union[str, os.PathLike]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a mesh and return centered, max-extent-normalized vertices and faces."""
        mesh = trimesh.load(path, force="mesh", process=False)
        if not isinstance(mesh, trimesh.Trimesh) or mesh.vertices.size == 0:
            raise ValueError(f"No mesh vertices were found in {path}.")

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        center = vertices.mean(axis=0, dtype=np.float64).astype(np.float32)
        centered = vertices - center
        extent = float(np.max(np.abs(centered)))
        if not np.isfinite(extent) or extent == 0.0:
            raise ValueError("Cannot normalize a mesh with zero or invalid extent.")

        normalized = centered / extent
        faces = np.asarray(mesh.faces, dtype=np.int64)
        return (
            torch.as_tensor(normalized, dtype=torch.float32, device=self.device),
            torch.as_tensor(faces, dtype=torch.int64, device=self.device),
        )

    def rasterize(
        self,
        vertices: torch.Tensor,
        resolution: int = 256,
    ) -> torch.Tensor:
        """Accumulate projected vertex density into a square 2D tensor."""
        if not isinstance(resolution, int) or resolution < 2:
            raise ValueError("resolution must be an integer greater than or equal to 2.")

        vertex_tensor = torch.as_tensor(
            vertices, dtype=torch.float32, device=self.device
        )
        if vertex_tensor.ndim != 2 or vertex_tensor.shape[1] < 2:
            raise ValueError("vertices must have shape (N, 2) or (N, 3).")
        if vertex_tensor.shape[0] == 0:
            raise ValueError("vertices must contain at least one point.")
        if not torch.isfinite(vertex_tensor).all():
            raise ValueError("vertices must contain only finite values.")

        xy = vertex_tensor[:, :2].clamp(-1.0, 1.0)
        pixels = torch.round((xy + 1.0) * ((resolution - 1) / 2.0)).to(
            dtype=torch.int64
        )
        flat_indices = pixels[:, 1] * resolution + pixels[:, 0]
        density = torch.zeros(
            resolution * resolution, dtype=torch.float32, device=self.device
        )
        density.scatter_add_(
            0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float32)
        )
        return density.reshape(resolution, resolution)
