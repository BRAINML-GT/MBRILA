"""Single-block Markovian-GP latent dynamics.

A :class:`MarkovianGPLatent` instance owns one MOSE kernel and an optional
per-region delay, lifts the kernel into a lag-``P`` LDS via
:func:`mbrila.dynamics.kernel_to_sde.kernel_to_lds`, and exposes the
result as ``(A_t, Q_t)``. Multiple instances are composed
block-diagonally by :class:`BlockDiagonalDynamics`.

In ADM-speak: each across-region latent maps to one block with
``num_regions = num_brain_regions`` and a time-varying delay (the "MOSE"
case); each within-region latent maps to one block with
``num_regions = 1`` and no delay (the "SOSE" case).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mbrila.core.delay_spec import Delay
from mbrila.delays.none import NoDelay
from mbrila.dynamics.kernel_to_sde import (
    LaggedCovKernel,
    kernel_to_lds,
    lagged_cov_grid,
)


class MarkovianGPLatent(nn.Module):
    """One block of Markovian-GP dynamics: kernel + (optional) delay → (A, Q).

    Parameters
    ----------
    kernel:
        MOSE kernel for this block. ``kernel.num_regions`` determines
        ``num_dim`` of the lifted state; per-block ``num_regions`` may
        differ (e.g. 1 for within-region, ``R`` for across-region).
    lag:
        Markov order ``P``. The lifted state is
        ``s_t = [x_t, x_{t-1}, …, x_{t-P+1}]`` of size
        ``P · kernel.num_regions``.
    delay:
        Per-region delay. One of:

        - ``None`` or :class:`~mbrila.delays.none.NoDelay` — no delay; the
          block is time-invariant (single shared ``(A, Q)`` broadcast over
          the time axis). GPFA-via-Markov-lift uses this.
        - :class:`~mbrila.delays.fixed.FixedDelay` — constant per-region
          delay; ``(A, Q)`` is still time-invariant (computed once on the
          ``(R,)`` static delay vector) and broadcast over the time axis.
          DLAG-SSM / mDLAG-SSM use this.
        - :class:`~mbrila.delays.time_varying.TimeVaryingDelay` — fully
          time-varying delay; ``(A, Q)`` has a leading ``T`` axis. ADM
          uses this.

        Any non-``None`` ``delay`` must have ``n_regions ==
        kernel.num_regions`` and ``n_latent == 1`` (each block represents
        one latent factor).
    T:
        Trial length. Used to broadcast time-invariant ``(A, Q)`` to the
        time axis (and forwarded to time-varying delays via
        :meth:`Delay.as_tensor`).
    cov_jitter, lds_jitter:
        Numerical stability knobs forwarded to
        :func:`kernel_to_lds`.
    """

    delay: Delay | None

    def __init__(
        self,
        kernel: LaggedCovKernel,
        lag: int,
        *,
        T: int,
        delay: Delay | None = None,
        num_dim: int | None = None,
        cov_jitter: float = 1e-4,
    ) -> None:
        super().__init__()
        if lag < 1:
            raise ValueError(f"lag must be >= 1; got {lag}")
        if T < 1:
            raise ValueError(f"T must be >= 1; got {T}")
        # ``num_dim`` (= per-time output dimensionality = number of
        # regions sharing this latent block) is passed explicitly to
        # support both MOSE (which exposes ``num_regions``) and generic
        # scalar kernels (Matérn, custom) that don't. When the kernel
        # exposes ``num_regions`` we fall back to it for backward
        # compatibility.
        kernel_R = getattr(kernel, "num_regions", None)
        if num_dim is None:
            if kernel_R is None:
                raise ValueError(
                    "MarkovianGPLatent: must pass num_dim explicitly when the "
                    "kernel does not expose a num_regions attribute (e.g. Matérn / custom)."
                )
            num_dim = int(kernel_R)
        elif kernel_R is not None and int(kernel_R) != num_dim:
            raise ValueError(f"num_dim ({num_dim}) disagrees with kernel.num_regions ({kernel_R})")
        if num_dim < 1:
            raise ValueError(f"num_dim must be >= 1; got {num_dim}")
        if delay is not None:
            if delay.n_regions != num_dim:
                raise ValueError(f"delay.n_regions ({delay.n_regions}) must equal num_dim ({num_dim})")
            if delay.n_latent != 1:
                raise ValueError(
                    f"MarkovianGPLatent represents a single latent factor; "
                    f"got delay.n_latent={delay.n_latent}"
                )
            # A time-varying delay is anchored to a specific T at construction
            # time (the Gaussian smoothing kernel is sized for it). Validate.
            delay_T = getattr(delay, "T", None)
            if delay_T is not None and delay_T != T:
                raise ValueError(f"delay.T ({delay_T}) must equal T ({T})")
        # ``kernel`` is structurally typed as ``LaggedCovKernel`` for
        # the SSM bridge; nn.Module attribute assignment requires
        # ``nn.Module``. All concrete kernels in mbrila subclass
        # ``nn.Module`` (via ``BaseKernel``) so this is safe at runtime.
        if not isinstance(kernel, nn.Module):
            raise TypeError("MarkovianGPLatent.kernel must be an nn.Module subclass (BaseKernel).")
        self.kernel = kernel
        self.delay = delay
        self.lag = lag
        self.T = T
        self._num_dim = int(num_dim)
        self.cov_jitter = cov_jitter

    @property
    def num_dim(self) -> int:
        """Per-time output dimensionality (= number of regions sharing this block)."""
        return self._num_dim

    @property
    def state_dim(self) -> int:
        """Size of the lifted state ``s_t``."""
        return self.lag * self.num_dim

    def _kernel_dtype_device(self) -> tuple[torch.dtype, torch.device]:
        """Probe a kernel parameter for the dtype/device the lag grid should use.

        Avoids hard-coding ``log_sigma`` (MOSE) — works for any kernel that
        is an ``nn.Module`` with at least one ``Parameter`` (all of our
        kernels register their hyperparameters as Parameters).
        """
        param = next(self.kernel.parameters(), None)
        if param is None:
            # Pathological: a kernel with no learnable parameters. Fall
            # back to mbrila's default precision (float64 on CPU).
            return torch.float64, torch.device("cpu")
        return param.dtype, param.device

    def forward(self) -> tuple[Tensor, Tensor]:
        """Return ``(A_t, Q_t)`` with shape ``(T, state_dim, state_dim)``.

        Three dispatch paths share the same output contract:

        - **No delay** (``delay is None`` or :class:`NoDelay`): one
          ``(A, Q)`` from a zero-delay kernel, broadcast over the time
          axis.
        - **Static delay** (e.g. :class:`FixedDelay`): one ``(A, Q)``
          from the static per-region delay vector, broadcast over the
          time axis. Avoids materialising T identical copies of the
          lagged covariance.
        - **Time-varying delay**: full ``(T, lag+1, lag+1, R, R)``
          lagged covariance → ``(T, ...)`` lifted coefficients.
        """
        sigma_dtype, device = self._kernel_dtype_device()

        # Resolve the delay tensor into one of the three shapes
        # :func:`lagged_cov_grid` accepts: ``None`` for no-delay,
        # ``(R,)`` for static per-region delay, ``(T, R)`` for time-varying.
        delays_arg: Tensor | None
        broadcast_over_T: bool
        if self.delay is None or isinstance(self.delay, NoDelay):
            delays_arg = None
            broadcast_over_T = True
        elif not self.delay.is_time_varying:
            delays_static = self.delay.as_tensor()  # (R, 1)
            if delays_static.ndim != 2 or delays_static.shape[-1] != 1:
                raise RuntimeError(
                    f"expected static delay tensor of shape (R, 1); got {tuple(delays_static.shape)}"
                )
            delays_arg = delays_static[..., 0]  # (R,)
            broadcast_over_T = True
        else:
            delays_full = self.delay.as_tensor(self.T)  # (T, R, 1)
            if delays_full.ndim != 3 or delays_full.shape[-1] != 1:
                raise RuntimeError(
                    f"expected time-varying delay tensor of shape (T, R, 1); got {tuple(delays_full.shape)}"
                )
            delays_arg = delays_full[..., 0]  # (T, R)
            broadcast_over_T = False

        # Generic Markovian-GP → LDS bridge: only requires ``kernel.cov``.
        K_lag = lagged_cov_grid(
            self.kernel,
            self.lag,
            num_dim=self.num_dim,
            delays=delays_arg,
            dtype=sigma_dtype,
            device=device,
        )
        A_out, Q_out = kernel_to_lds(
            K_lag,
            lag=self.lag,
            num_dim=self.num_dim,
            cov_jitter=self.cov_jitter,
        )
        if broadcast_over_T:
            # Static or no-delay: single ``(state_dim, state_dim)`` pair
            # broadcast over T so the output contract is uniform with the
            # time-varying path.
            A = A_out.unsqueeze(0).expand(self.T, -1, -1).contiguous()
            Q = Q_out.unsqueeze(0).expand(self.T, -1, -1).contiguous()
            return A, Q
        return A_out, Q_out


class BlockDiagonalDynamics(nn.Module):
    """Stack of :class:`MarkovianGPLatent` blocks block-diagonally.

    Holds the per-block list, plus a fixed selection matrix ``H_select``
    that pulls the "current time" output of each latent out of the
    lifted state. The combined ``H_select`` has shape
    ``(n_observable, total_state)`` where ``n_observable`` is the number
    of region-by-latent pairs the model treats as observable.

    Capabilities advertised: ``"to_lds"`` (the engine knows it can ask
    for ``(A_t, Q_t)`` and ``H_select``).
    """

    CAPABILITIES = frozenset({"to_lds"})

    # Type annotations for the registered buffer / submodule list. mypy
    # otherwise widens accesses through ``nn.Module.__getattr__`` to
    # ``Tensor | Module``, which forces every call site to assert the
    # narrower type. Declaring them here keeps the rest of the codebase
    # free of redundant casts.
    H_select: Tensor
    blocks: nn.ModuleList
    T: int

    def __init__(
        self,
        blocks: list[MarkovianGPLatent],
        *,
        n_observable: int,
        observable_to_state_indices: list[tuple[int, int]],
    ) -> None:
        """``observable_to_state_indices`` is a list of ``(obs_idx, state_idx)``
        pairs; the resulting selection matrix has a 1 at each pair.
        """
        super().__init__()
        if not blocks:
            raise ValueError("blocks must be non-empty")
        self.blocks = nn.ModuleList(blocks)
        self.T = blocks[0].T
        for b in blocks:
            if b.T != self.T:
                raise ValueError(f"all blocks must share T; got {b.T} vs {self.T}")

        total_state = sum(b.state_dim for b in blocks)
        H_select = torch.zeros(n_observable, total_state)
        for obs_idx, state_idx in observable_to_state_indices:
            if not (0 <= obs_idx < n_observable):
                raise ValueError(f"obs_idx {obs_idx} out of range [0, {n_observable})")
            if not (0 <= state_idx < total_state):
                raise ValueError(f"state_idx {state_idx} out of range [0, {total_state})")
            H_select[obs_idx, state_idx] = 1.0
        self.register_buffer("H_select", H_select)
        self._block_offsets = self._compute_block_offsets()

    def _compute_block_offsets(self) -> list[int]:
        offsets: list[int] = [0]
        for b in self.blocks:
            assert isinstance(b, MarkovianGPLatent)
            offsets.append(offsets[-1] + b.state_dim)
        return offsets

    @property
    def total_state_dim(self) -> int:
        return self._block_offsets[-1]

    def forward(self) -> tuple[Tensor, Tensor]:
        """Assemble the block-diagonal ``(A, Q)`` over all latent blocks.

        Returns shapes ``(T, total_state, total_state)``.
        """
        A_list: list[Tensor] = []
        Q_list: list[Tensor] = []
        for block in self.blocks:
            assert isinstance(block, MarkovianGPLatent)
            A_b, Q_b = block.forward()
            A_list.append(A_b)
            Q_list.append(Q_b)
        from mbrila.dynamics.ssm_base import block_diag_time

        A = block_diag_time(A_list)
        Q = block_diag_time(Q_list)
        return A, Q
