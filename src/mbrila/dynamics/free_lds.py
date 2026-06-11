"""Free linear dynamical system (naive SSM, no GP / kernel prior).

The simplest "no kernel" instantiation. The latent state evolves as

    x_{t+1} = A · x_t + ε,    ε ~ N(0, Q)

with ``A`` and ``Q`` learned directly from data — no Markovian-GP lift,
no spectral structure, no temporal smoothness prior beyond what ``A``
implicitly encodes. This is the canonical "fit an LDS to neural data"
baseline (Kalman/Roweis/Ghahramani 1999) lifted into the mbrila 4-axis
framework: it slots in wherever a kernel-based dynamics would, exposed
through the same :class:`KalmanEMEngine`.

Parameterisation
----------------
- ``A``: stored as a dense ``(n_latent, n_latent)`` :class:`nn.Parameter`.
  No stability constraint is enforced; users seeking a stable LDS prior
  should initialise with a contractive ``init_A`` (default
  ``0.95 · I``).
- ``Q``: parameterised via its Cholesky factor ``L`` (lower triangular).
  ``L`` has positive diagonal (stored as ``log_diag`` so optimisation is
  unconstrained) and free strict-lower entries. ``Q = L · Lᵀ`` is PSD
  by construction. Mirrors the trick :class:`MultiRegionLinearObservation`
  uses for its noise covariance.

Multi-region observation
------------------------
``FreeLDSLatent`` represents a *single shared* latent state. To play in
the multi-region framework, ``H_select`` is constructed to replicate the
state across regions: slot ``(r, k)`` of the per-region observable
vector points back to state component ``k``. Each region's emission
``C_r`` is then a dense ``(y_dim_r, n_latent)`` matrix acting on the
*same* ``x_t``. This is the standard "shared-latent factor analysis"
formulation, equivalent to mDLAG-without-delays-without-ARD or
single-region GPFA when ``n_regions = 1``.

Sampling and inference interface
--------------------------------
Exposes the same surface :class:`KalmanEMEngine` consumes from
:class:`~mbrila.dynamics.markov_gp.BlockDiagonalDynamics`:

- ``CAPABILITIES = frozenset({"to_lds"})``
- ``forward() -> (A_t, Q_t)`` of shape ``(T, n_latent, n_latent)``
  (single ``(A, Q)`` broadcast over the time axis — the LDS is
  stationary in v1)
- ``H_select`` buffer of shape ``(n_regions · n_latent, n_latent)``
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class FreeLDSLatent(nn.Module):
    """Naive linear dynamical system with learnable ``(A, Q)`` and replicate-across-regions ``H_select``.

    Parameters
    ----------
    n_latent:
        Dimensionality of the shared latent state ``x_t``.
    n_regions:
        Number of regions sharing the latent. Each region's emission
        sees the full ``x_t`` via the replicated ``H_select``.
    T:
        Trial length. Used only to broadcast the stationary ``(A, Q)``
        over the time axis so the output contract matches
        :class:`BlockDiagonalDynamics`.
    init_A:
        Optional explicit initial ``A``. Defaults to ``0.95 · I`` —
        contractive enough to keep the Cholesky in the Kalman filter
        well-conditioned on a freshly-built model.
    init_Q_diag:
        Initial scalar value for the diagonal of ``Q`` (``Q = q · I``
        initially). Must be positive.
    dtype:
        Floating point dtype for parameters. ``torch.float64`` by default
        (matches mbrila's library-wide convention).
    """

    CAPABILITIES = frozenset({"to_lds"})

    # Class-level annotations so mypy can see ``H_select`` and the parameters
    # without going through ``nn.Module.__getattr__``.
    H_select: Tensor
    n_latent: int
    n_regions: int
    T: int

    def __init__(
        self,
        n_latent: int,
        n_regions: int,
        T: int,
        *,
        init_A: Tensor | None = None,
        init_Q_diag: float = 0.1,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if n_latent < 1:
            raise ValueError(f"n_latent must be >= 1; got {n_latent}")
        if n_regions < 1:
            raise ValueError(f"n_regions must be >= 1; got {n_regions}")
        if T < 1:
            raise ValueError(f"T must be >= 1; got {T}")
        if init_Q_diag <= 0:
            raise ValueError(f"init_Q_diag must be positive; got {init_Q_diag}")
        self.n_latent = int(n_latent)
        self.n_regions = int(n_regions)
        self.T = int(T)

        # --- A ---
        if init_A is None:
            A_init = 0.95 * torch.eye(self.n_latent, dtype=dtype)
        else:
            if init_A.shape != (self.n_latent, self.n_latent):
                raise ValueError(
                    f"init_A must have shape ({self.n_latent}, {self.n_latent}); got {tuple(init_A.shape)}"
                )
            A_init = init_A.to(dtype=dtype)
        self.A_param = nn.Parameter(A_init)

        # --- Q via Cholesky: Q = L·Lᵀ with L lower-triangular,
        # L_diag > 0 (parameterised as exp(log_diag)).
        # Initial Q = init_Q_diag · I  =>  L = sqrt(init_Q_diag) · I.
        log_diag_init = 0.5 * math.log(init_Q_diag)
        self.L_log_diag = nn.Parameter(torch.full((self.n_latent,), log_diag_init, dtype=dtype))
        n_off = self.n_latent * (self.n_latent - 1) // 2
        self.L_off_diag = nn.Parameter(torch.zeros(n_off, dtype=dtype))

        # --- H_select: replicate state across regions ---
        # Slot (r, k) of the per-region observable picks state component k.
        # Observable layout matches the (n_regions, n_obs_per_region)
        # convention used by MultiRegionLinearObservation and ADM/GPFA.
        H_select = torch.zeros(self.n_regions * self.n_latent, self.n_latent, dtype=dtype)
        # The double loop is over O(R · K) constants at construction time —
        # not a hot path and not subject to the trial-loop ban.
        for r in range(self.n_regions):
            for k in range(self.n_latent):
                H_select[r * self.n_latent + k, k] = 1.0
        self.register_buffer("H_select", H_select)

    @property
    def total_state_dim(self) -> int:
        """Size of the latent state (no lifting — equals ``n_latent``)."""
        return self.n_latent

    def _build_L(self) -> Tensor:
        """Assemble the lower-triangular Cholesky factor of ``Q``."""
        device = self.A_param.device
        dtype = self.A_param.dtype
        n = self.n_latent
        L = torch.zeros(n, n, dtype=dtype, device=device)
        if n > 1:
            tril_idx = torch.tril_indices(n, n, offset=-1, device=device)
            L = L.index_put((tril_idx[0], tril_idx[1]), self.L_off_diag)
        diag = torch.exp(self.L_log_diag)
        # Add the diagonal in-place via scatter — avoids losing the
        # off-diagonal entries we just set.
        diag_idx = torch.arange(n, device=device)
        L = L.index_put((diag_idx, diag_idx), diag)
        return L

    def Q(self) -> Tensor:
        """Return the time-invariant innovation covariance ``Q = L·Lᵀ``."""
        L = self._build_L()
        return L @ L.transpose(-1, -2)

    def forward(self) -> tuple[Tensor, Tensor]:
        """Return ``(A_t, Q_t)`` with shape ``(T, n_latent, n_latent)``.

        The LDS is stationary in v1: one ``(A, Q)`` pair broadcast over
        the time axis. Output shape matches
        :meth:`BlockDiagonalDynamics.forward` so :class:`KalmanEMEngine`
        consumes both uniformly.
        """
        A_one = self.A_param
        Q_one = self.Q()
        A = A_one.unsqueeze(0).expand(self.T, -1, -1).contiguous()
        Q = Q_one.unsqueeze(0).expand(self.T, -1, -1).contiguous()
        return A, Q
