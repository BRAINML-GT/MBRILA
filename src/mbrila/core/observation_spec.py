"""Observation model contract.

Every method in v1 maps the latent state to neural recordings via a
linear-Gaussian observation::

    y_r = C_r x_r^{eff} + d_r + eps_r,   eps_r ~ N(0, diag(R_r))

where ``x_r^{eff}`` is region ``r``'s view of the latent state (the relevant
across-region columns delayed by the per-region delay, concatenated with the
region's within-region columns). The block-diagonal layout of ``C`` is
assembled by the dynamics layer; this module just declares the per-region
parameters.

Concrete subclasses (Phase 2+) live in :mod:`mbrila.observations`. The ARD
variant adds a column-wise precision parameter and lives there as well to
avoid pulling Phase-4 machinery into ``core``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Observation(nn.Module, ABC):
    """Multi-region linear-Gaussian emission base class.

    Parameters
    ----------
    y_dims:
        Per-region neuron counts.
    n_latent_total:
        Total latent dimensionality (across + within across all regions).
    """

    y_dims: tuple[int, ...]
    n_latent_total: int

    def __init__(self, y_dims: tuple[int, ...], n_latent_total: int) -> None:
        super().__init__()
        if not y_dims or any(d <= 0 for d in y_dims):
            raise ValueError(f"y_dims must be a non-empty tuple of positive ints; got {y_dims}")
        if n_latent_total < 1:
            raise ValueError(f"n_latent_total must be >= 1; got {n_latent_total}")
        self.y_dims = tuple(y_dims)
        self.n_latent_total = int(n_latent_total)

    @property
    def n_regions(self) -> int:
        return len(self.y_dims)

    @property
    def n_neurons(self) -> int:
        return int(sum(self.y_dims))

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Predict noiseless ``E[y | x]``.

        Parameters
        ----------
        x:
            Latent tensor of shape ``(n_trials, T, n_latent_total)``.

        Returns
        -------
        Tensor of shape ``(n_trials, T, sum(y_dims))``.
        """

    @abstractmethod
    def block_diag_C(self) -> Tensor:
        """Return the assembled block-diagonal emission matrix.

        Shape ``(sum(y_dims), n_latent_total)``. Used by EM closed-form
        updates and by frequency-domain engines that need the dense matrix
        rather than per-region forwards.
        """

    @abstractmethod
    def diag_R(self) -> Tensor:
        """Diagonal observation noise variances, shape ``(sum(y_dims),)``."""

    @abstractmethod
    def offset(self) -> Tensor:
        """Per-neuron mean offset ``d``, shape ``(sum(y_dims),)``."""
