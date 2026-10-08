"""Complex-valued memory folding and atomic tensor persistence."""

import os
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Union

import torch

from . import DEVICE


class SustratoKahlariano:
    """Maintain complex memory and fold each input through a phase rotation."""

    def __init__(
        self,
        shape: Sequence[int],
        device: Optional[torch.device] = None,
    ) -> None:
        """Allocate zero-initialized complex memory with the requested shape."""
        self.device = torch.device(device) if device is not None else DEVICE
        self.state = torch.zeros(
            tuple(shape), dtype=torch.complex64, device=self.device
        )

    def fold(self, value: torch.Tensor, phi: float = 0.0) -> torch.Tensor:
        """Add the phase-rotated input to the complex memory state."""
        phase = torch.as_tensor(phi, dtype=torch.float32, device=self.device)
        rotation = torch.exp(torch.complex(torch.zeros_like(phase), -phase))
        value_tensor = torch.as_tensor(value, device=self.device)
        value_tensor = value_tensor.to(dtype=torch.complex64)
        self.state = self.state + value_tensor * rotation
        return self.state

    def reset(self) -> None:
        """Clear the memory state without introducing random values."""
        self.state.zero_()


class BovedaTopologica:
    """Persist tensors using same-directory temporary files and atomic replacement."""

    @staticmethod
    def save(
        tensor: torch.Tensor,
        path: Union[str, os.PathLike],
    ) -> Path:
        """Atomically save a tensor to a PyTorch ``.pt`` file."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
            torch.save(tensor, temporary_path)
            os.replace(temporary_path, destination)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise
        return destination

    @staticmethod
    def load(
        path: Union[str, os.PathLike],
        map_location: Optional[Union[str, torch.device]] = None,
    ) -> torch.Tensor:
        """Load a saved tensor onto the requested device, defaulting to RT-Core's."""
        source = Path(path)
        location = DEVICE if map_location is None else map_location
        loaded = torch.load(source, map_location=location, weights_only=True)
        if not isinstance(loaded, torch.Tensor):
            raise TypeError(f"Expected a tensor in {source}, got {type(loaded).__name__}.")
        return loaded
