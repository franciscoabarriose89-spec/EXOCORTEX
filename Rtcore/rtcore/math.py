"""Deterministic tensor operators used by the RT-Core pipeline."""

import math
from typing import Optional

import torch

from . import DEVICE


class TransformadaAmador:
    """Apply the element-wise Amador stabilization transform to tensors."""

    def __init__(
        self,
        alpha: float = 0.5,
        tau: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        """Configure default blend, friction threshold, and execution device."""
        self.device = torch.device(device) if device is not None else DEVICE
        self._validate_parameters(alpha, tau)
        self.alpha = alpha
        self.tau = tau

    @staticmethod
    def _validate_parameters(alpha: float, tau: float) -> None:
        """Reject blend and threshold values outside their defined domains."""
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be a finite value in [0, 1].")
        if not math.isfinite(tau) or tau < 0.0:
            raise ValueError("tau must be a finite non-negative value.")

    def apply(
        self,
        state: torch.Tensor,
        value: torch.Tensor,
        alpha: Optional[float] = None,
        tau: Optional[float] = None,
    ) -> torch.Tensor:
        """Blend state and input, then subtract the thresholded absolute difference."""
        blend = self.alpha if alpha is None else alpha
        threshold = self.tau if tau is None else tau
        self._validate_parameters(blend, threshold)

        state_tensor = torch.as_tensor(state, device=self.device)
        value_tensor = torch.as_tensor(value, device=self.device)
        dtype = torch.promote_types(state_tensor.dtype, value_tensor.dtype)
        if not dtype.is_floating_point:
            dtype = torch.float32
        state_tensor = state_tensor.to(dtype=dtype)
        value_tensor = value_tensor.to(dtype=dtype)
        state_tensor, value_tensor = torch.broadcast_tensors(
            state_tensor, value_tensor
        )
        friction = torch.abs(value_tensor - state_tensor)
        correction = torch.where(
            friction <= threshold, friction, torch.zeros_like(friction)
        )
        return blend * state_tensor + (1.0 - blend) * value_tensor - correction

    __call__ = apply
