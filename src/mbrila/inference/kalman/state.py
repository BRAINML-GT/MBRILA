"""Gaussian state container used by the Kalman filter / smoother.

Convention
----------
- ``mean`` has shape ``(..., D)``. We do *not* carry a trailing singleton
  dimension; matrix products use einsum or transient ``unsqueeze(-1)``.
  This matches the rest of mbrila where latent tensors are stored as
  ``(n_trials, T, D)``.
- ``covariance`` has shape ``(..., D, D)``.
- ``precision`` is optional and has shape ``(..., D, D)`` when present.

Leading dimensions are arbitrary — typically ``(n_trials,)`` or
``(n_trials, T)`` depending on the use site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Self

import torch
from torch import Tensor


@dataclass(slots=True)
class GaussianState:
    """A multivariate Gaussian carried through the Kalman recursion.

    Attributes
    ----------
    mean:
        Shape ``(..., D)``.
    covariance:
        Shape ``(..., D, D)``; symmetric positive (semi-)definite.
    precision:
        Optional inverse covariance, shape ``(..., D, D)``. Cached when an
        update step pre-computes it for re-use.
    """

    mean: Tensor
    covariance: Tensor
    precision: Tensor | None = None

    def __post_init__(self) -> None:
        if self.mean.shape[-1:] != self.covariance.shape[-2:-1]:
            raise ValueError(
                f"mean trailing dim {self.mean.shape[-1]} does not match covariance "
                f"trailing dim {self.covariance.shape[-2]}"
            )
        if self.covariance.shape[-1] != self.covariance.shape[-2]:
            raise ValueError(
                f"covariance must be square in its last two dims; got {tuple(self.covariance.shape[-2:])}"
            )
        if self.precision is not None and self.precision.shape != self.covariance.shape:
            raise ValueError(
                f"precision shape {tuple(self.precision.shape)} does not match "
                f"covariance shape {tuple(self.covariance.shape)}"
            )

    @property
    def state_dim(self) -> int:
        return int(self.mean.shape[-1])

    @property
    def device(self) -> torch.device:
        return self.mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mean.dtype

    def clone(self) -> Self:
        return replace(
            self,
            mean=self.mean.clone(),
            covariance=self.covariance.clone(),
            precision=None if self.precision is None else self.precision.clone(),
        )

    def to(self, *, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> Self:
        if device is None and dtype is None:
            return self
        return replace(
            self,
            mean=self.mean.to(device=device, dtype=dtype),
            covariance=self.covariance.to(device=device, dtype=dtype),
            precision=None if self.precision is None else self.precision.to(device=device, dtype=dtype),
        )

    def log_density(self, x: Tensor) -> Tensor:
        """Log-density ``log N(x | mean, covariance)``.

        Parameters
        ----------
        x:
            Same shape as :attr:`mean` (``(..., D)``).

        Returns
        -------
        Tensor of shape ``(...,)`` (the leading dims of ``mean`` minus the
        last).
        """
        diff = x - self.mean
        # Cholesky-based: ½ ‖L^{-1} diff‖² + log|L| + (D/2) log(2π)
        L = torch.linalg.cholesky(self.covariance)
        # Solve L z = diff for z, then quadratic = ‖z‖²
        # diff has shape (..., D); upcast to (..., D, 1) for triangular_solve.
        z = torch.linalg.solve_triangular(L, diff.unsqueeze(-1), upper=False).squeeze(-1)
        quad = (z * z).sum(dim=-1)
        log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
        d = float(self.state_dim)
        result: Tensor = -0.5 * (quad + log_det + d * math.log(2.0 * math.pi))
        return result
