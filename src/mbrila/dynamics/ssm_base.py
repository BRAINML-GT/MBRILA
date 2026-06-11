"""Building blocks for lag-``P`` linear-state-space dynamics.

A lag-``P`` LDS represents a ``num_dim``-output stationary Gaussian
process with state ``s_t = [x_t, x_{t-1}, …, x_{t-P+1}]``. The state
evolution is
::

    s_{t+1} = F_t s_t + ε_t,    ε_t ~ N(0, Q_t)

with the deterministic-shift structure

::

    F_t = ⎡ A_0(t)  A_1(t)  …  A_{P-1}(t) ⎤
          ⎢   I        0      …     0      ⎥
          ⎢   0        I      …     0      ⎥
          ⎣   …                            ⎦

where each ``A_k(t)`` is ``num_dim × num_dim``. The first ``num_dim``
rows of ``s_{t+1}`` carry the GP's innovation covariance ``Q_full(t)``;
the remaining lag rows are deterministic shifts of the previous state.
See :mod:`mbrila.dynamics.kernel_to_sde` for the construction of
``(F_t, Q_t)`` from a stationary kernel.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def identity_shift_block(lag: int, num_dim: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Return the ``((lag - 1) * num_dim, lag * num_dim)`` identity-shift block.

    Used as the lower rows of the lag-``P`` transition matrix to copy
    the previous state's first ``(lag - 1) * num_dim`` entries forward
    without modification.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1; got {lag}")
    eye_block = torch.eye((lag - 1) * num_dim, dtype=dtype, device=device)
    zero_block = torch.zeros((lag - 1) * num_dim, num_dim, dtype=dtype, device=device)
    return torch.cat([eye_block, zero_block], dim=1)


def block_diag_time(blocks: Sequence[Tensor]) -> Tensor:
    """Assemble per-block-diagonal time-varying matrices.

    Each ``blocks[i]`` has shape ``(T, d_i, d_i)``. The returned tensor
    has shape ``(T, sum d_i, sum d_i)`` with ``blocks[i]`` on the
    ``i``-th diagonal block at every time step.
    """
    if not blocks:
        raise ValueError("blocks must be non-empty")
    T = int(blocks[0].shape[0])
    dtype = blocks[0].dtype
    device = blocks[0].device
    sizes = [int(b.shape[1]) for b in blocks]
    total = sum(sizes)
    for b, d in zip(blocks, sizes, strict=True):
        if b.ndim != 3 or b.shape[0] != T or b.shape[1] != d or b.shape[2] != d:
            raise ValueError(f"every block must have shape (T={T}, d_i, d_i); got {tuple(b.shape)}")
    out = torch.zeros(T, total, total, dtype=dtype, device=device)
    offset = 0
    for b, d in zip(blocks, sizes, strict=True):
        out[:, offset : offset + d, offset : offset + d] = b
        offset += d
    return out


def lifted_state_dim(lag: int, num_dim: int) -> int:
    """Total dimensionality of the lifted state ``s_t = [x_t, …, x_{t-P+1}]``."""
    return lag * num_dim
