"""Delay parameterisation contract.

Two flavours coexist in v1:

- :class:`FixedDelay` (DLAG / mDLAG): one scalar delay per ``(region, latent)``
  pair, optimised by EM. Subclasses bound the delay with a tanh
  parameterisation so it stays in ``(-D_max, +D_max)`` bins.
- :class:`TimeVaryingDelay` (ADM): a learnable trajectory of shape
  ``(T, n_regions - 1)`` smoothed by a 1D Gaussian kernel; the first region
  is the reference and is fixed at zero.

Both share the abstract :class:`Delay` interface which exposes a uniform way
of (a) materialising the delay tensor and (b) producing the frequency-domain
phase shift used by frequency-EM and Kalman engines alike.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn


class Delay(nn.Module, ABC):
    """Per-region per-latent delay model.

    Concrete subclasses register their parameters as ``nn.Parameter``s and
    implement :meth:`as_tensor`. Reference (zero) region is conventionally
    region 0; this is enforced by subclasses returning zero for that index.
    """

    n_regions: int
    n_latent: int

    def __init__(self, n_regions: int, n_latent: int) -> None:
        super().__init__()
        if n_regions < 1:
            raise ValueError(f"n_regions must be >= 1; got {n_regions}")
        if n_latent < 1:
            raise ValueError(f"n_latent must be >= 1; got {n_latent}")
        self.n_regions = n_regions
        self.n_latent = n_latent

    @property
    @abstractmethod
    def is_time_varying(self) -> bool:
        """Whether the delay depends on time."""

    @abstractmethod
    def as_tensor(self, T: int | None = None) -> Tensor:
        """Materialise the delay tensor in *bin* units.

        Parameters
        ----------
        T:
            Trial length. Required for time-varying delays; ignored (and
            may be ``None``) for fixed delays.

        Returns
        -------
        Tensor of shape ``(n_regions, n_latent)`` for fixed delays or
        ``(T, n_regions, n_latent)`` for time-varying delays. The
        ``r=0`` slice is always zero (reference region).
        """

    def phase_shift(self, freqs: Tensor, T: int | None = None) -> Tensor:
        """Frequency-domain phase shift ``exp(-i * 2 pi * f * D)``.

        Parameters
        ----------
        freqs:
            Real frequencies (cycles per bin), shape ``(F,)``.
        T:
            Trial length, forwarded to :meth:`as_tensor`.

        Returns
        -------
        Complex tensor of shape ``(F, n_regions, n_latent)`` (fixed) or
        ``(T, F, n_regions, n_latent)`` (time-varying). Convention: a
        *positive* delay shifts the latent *backward* in time at the target
        region, so the phase factor is ``exp(-i * 2 pi * f * D)``.
        """
        D = self.as_tensor(T)
        # Build the broadcast: (..., F, R, L) where ... is empty for fixed
        # delays and is a leading T axis for time-varying delays.
        f = freqs.to(dtype=D.dtype)
        # Use complex64/complex128 matching the delay tensor's real dtype.
        complex_dtype = torch.complex64 if D.dtype == torch.float32 else torch.complex128
        # phase = -i * 2 pi * f * D
        if D.ndim == 2:  # (R, L)
            phase = -2j * torch.pi * f.view(-1, 1, 1) * D.unsqueeze(0)
        else:  # (T, R, L)
            phase = -2j * torch.pi * f.view(1, -1, 1, 1) * D.unsqueeze(1)
        return torch.exp(phase.to(dtype=complex_dtype))
