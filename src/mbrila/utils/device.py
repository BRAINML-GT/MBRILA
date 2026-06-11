"""Device & dtype defaults for mbrila.

The library defaults to CUDA when available and silently falls back to CPU.
Users may override per-model with an explicit ``device=`` argument.
"""

from __future__ import annotations

import torch

DEFAULT_DTYPE: torch.dtype = torch.float64
DEFAULT_COMPLEX_DTYPE: torch.dtype = torch.complex128


def default_device() -> torch.device:
    """Return the preferred default device.

    CUDA is selected if available; otherwise CPU. This is queried lazily so
    users who set ``CUDA_VISIBLE_DEVICES`` after import still get the right
    answer.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Normalise a user-supplied device argument."""
    if device is None:
        return default_device()
    if isinstance(device, torch.device):
        return device
    return torch.device(device)
