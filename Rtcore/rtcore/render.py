"""Tensor-based surface normals and Schlick Fresnel shading."""

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as functional

from . import DEVICE


class MotorFresnel:
    """Shade a 2D height field using gradient normals and Schlick reflectance."""

    def __init__(self, device: Optional[torch.device] = None) -> None:
        """Set the device used for gradient, normal, and shading operations."""
        self.device = torch.device(device) if device is not None else DEVICE

    def calculate_normals(self, height: torch.Tensor) -> torch.Tensor:
        """Derive normalized ``(nx, ny, nz)`` surface normals from a height field."""
        height_tensor = torch.as_tensor(
            height, dtype=torch.float32, device=self.device
        )
        if height_tensor.ndim != 2 or min(height_tensor.shape) < 2:
            raise ValueError("height must be a 2D tensor with both dimensions >= 2.")
        gradient_y, gradient_x = torch.gradient(height_tensor)
        normals = torch.stack(
            (-gradient_x, -gradient_y, torch.ones_like(height_tensor)), dim=-1
        )
        return functional.normalize(normals, p=2, dim=-1)

    def shade(
        self,
        height: torch.Tensor,
        material: Optional[Dict[str, object]] = None,
        view_direction: Sequence[float] = (0.0, 0.0, 1.0),
    ) -> torch.Tensor:
        """Return an RGB float tensor shaded with Schlick reflectance."""
        settings = {
            "color_base": (0.72, 0.62, 0.55),
            "color_reflejo": (1.0, 1.0, 1.0),
            "rugosidad": 0.35,
            "r0": 0.04,
        }
        if material is not None:
            settings.update(material)

        roughness = float(settings["rugosidad"])
        r0 = float(settings["r0"])
        if not 0.0 <= roughness <= 1.0:
            raise ValueError("rugosidad must be in [0, 1].")
        if not 0.0 <= r0 <= 1.0:
            raise ValueError("r0 must be in [0, 1].")

        normals = self.calculate_normals(height)
        view = torch.as_tensor(
            view_direction, dtype=normals.dtype, device=self.device
        )
        if view.shape != (3,):
            raise ValueError("view_direction must contain exactly three values.")
        view = functional.normalize(view, p=2, dim=0)
        cosine = torch.sum(normals * view, dim=-1).clamp(0.0, 1.0)
        fresnel = r0 + (1.0 - r0) * (1.0 - cosine).pow(5)
        fresnel = fresnel * (1.0 - roughness)

        base_color = torch.as_tensor(
            settings["color_base"], dtype=normals.dtype, device=self.device
        )
        reflection_color = torch.as_tensor(
            settings["color_reflejo"], dtype=normals.dtype, device=self.device
        )
        if base_color.shape != (3,) or reflection_color.shape != (3,):
            raise ValueError("material colors must each contain three channels.")
        return (
            (1.0 - fresnel.unsqueeze(-1)) * base_color
            + fresnel.unsqueeze(-1) * reflection_color
        ).clamp(0.0, 1.0)
