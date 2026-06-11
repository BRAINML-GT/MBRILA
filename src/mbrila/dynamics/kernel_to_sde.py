"""Lift a stationary kernel into a lag-``P`` linear-state-space model.

Given the joint covariance of ``(x_{t-P}, x_{t-P+1}, …, x_t)`` for a
``num_dim``-output stationary GP — assembled by the kernel layer at the
``(P+1, P+1)`` lag grid — this module returns the AR coefficients
``A_t`` and innovation covariance ``Q_t`` for the lifted state
``s_t = [x_t, x_{t-1}, …, x_{t-P+1}]``.

Algorithm (Schur / Cholesky-based one-step AR fit)
--------------------------------------------------
1. Reshape the lagged kernel covariance from
   ``(P+1, P+1, num_dim, num_dim)`` (or ``(T, P+1, P+1, num_dim, num_dim)``
   for time-varying kernels) to ``((P+1)·num_dim, (P+1)·num_dim)`` so
   rows index ``(time-lag, region)``.
2. Cholesky-factor the resulting big covariance ``K = L Lᵀ``. Setting
   ``R := Lᵀ`` and partitioning into a ``(P·num_dim, num_dim)`` block,

   ::

       R = ⎡ R₁₁  R₁₂ ⎤
           ⎣  0   R₂₂ ⎦

   the standard one-step-ahead AR coefficients are
   ``B = L₂₁ L₁₁⁻¹ = (R₁₁⁻¹ R₁₂)ᵀ`` (shape ``num_dim × P·num_dim``)
   and the conditional covariance is ``Q_full = R₂₂ᵀ R₂₂``.
3. Reverse the ``P`` column blocks of ``B`` so the columns line up with
   the state convention ``s_t = [x_t, x_{t-1}, …, x_{t-P+1}]`` (the
   kernel naturally orders them oldest-first).
4. Build the lifted transition matrix
   ``F_t = stack( [B_reversed; identity-shift] )`` and the lifted
   noise covariance ``Q_t = block_diag( Q_full, Q_full, …, Q_full )``
   with ``Q_full = R₂₂ᵀ R₂₂ / 2`` (see :func:`kernel_to_lds` for the
   ``/2`` constant).

Why every lagged block uses the same ``Q_full``
-----------------------------------------------
Mathematically the lagged copies in ``s_t`` are deterministic shifts of
the previous state — their innovation blocks "should" be zero. But a
zero lag block makes ``Q`` singular, and the Cholesky step downstream
fails. Filling those blocks with ``Q_full`` keeps ``Q`` strictly
positive definite; gradient descent absorbs the resulting bias into
``C`` / ``R`` / ``μ₀`` so downstream training is robust to it.

API
---
The conversion function accepts either a time-invariant lag covariance
``(P+1, P+1, num_dim, num_dim)`` or a time-varying batch
``(T, P+1, P+1, num_dim, num_dim)``. The returned ``(A, Q)`` mirror
the leading shape: ``(P·num_dim, P·num_dim)`` or
``(T, P·num_dim, P·num_dim)``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from mbrila.dynamics.ssm_base import identity_shift_block


@runtime_checkable
class LaggedCovKernel(Protocol):
    """Minimal kernel contract consumed by :func:`lagged_cov_grid`.

    Any object exposing ``cov(tau) -> Tensor`` that evaluates a scalar
    stationary kernel pointwise on a real-valued lag tensor of arbitrary
    shape is acceptable. The dynamics layer handles the multi-region
    structure (broadcasting over ``(r1, r2)``) and the delay shifts; the
    kernel itself is not delay-aware.

    Defining this Protocol *here*, rather than in
    :mod:`mbrila.core.kernel_spec`, keeps the SSM-bridge dependency
    tight: the bridge needs only ``cov``, not the broader (and partly
    optional) ``Kernel`` interface. Concrete kernel classes —
    :class:`MOSEKernel`, Matérn variants, user-defined kernels —
    satisfy this structurally without subclassing anything.
    """

    def cov(self, tau: Tensor) -> Tensor: ...


# Every lag-shift diagonal block of ``Q`` is set to the same ``Q_full``
# (the innovation covariance of the first block). A zero lag block
# would make ``Q`` singular and break the Cholesky step inside the
# Kalman filter, so we keep ``Q`` strictly positive definite by reusing
# ``Q_full`` everywhere; downstream training (gradient descent on
# ``C`` / ``R`` / ``μ_0``) is robust to the resulting prior bias on
# the shifted state.


def _arrange_big_cov(K: Tensor, num_dim: int) -> Tensor:
    """Reshape the lagged kernel cov to a square joint covariance.

    Input ``K`` has shape ``(..., P+1, P+1, num_dim, num_dim)`` where
    ``...`` is an arbitrary leading shape (empty or a time axis).
    Returns shape ``(..., (P+1)·num_dim, (P+1)·num_dim)`` with rows
    grouped as ``(time-lag, region)``.
    """
    *leading, lp1_a, lp1_b, d_a, d_b = K.shape
    if lp1_a != lp1_b:
        raise ValueError(f"first two trailing axes must agree; got {lp1_a} vs {lp1_b}")
    if d_a != num_dim or d_b != num_dim:
        raise ValueError(f"trailing region axes must equal num_dim={num_dim}; got {d_a}, {d_b}")
    # permute (..., P+1, P+1, num_dim, num_dim) → (..., P+1, num_dim, P+1, num_dim)
    perm = list(range(K.ndim))
    last4 = list(range(K.ndim - 4, K.ndim))
    perm[K.ndim - 4 :] = [last4[0], last4[2], last4[1], last4[3]]
    K_perm = K.permute(*perm).contiguous()
    new_shape = (*leading, lp1_a * num_dim, lp1_a * num_dim)
    return K_perm.reshape(*new_shape)


def kernel_to_lds(
    K_lagged: Tensor,
    *,
    lag: int,
    num_dim: int,
    cov_jitter: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Convert a lagged kernel covariance to lag-``P`` LDS coefficients.

    Parameters
    ----------
    K_lagged:
        Either ``(P+1, P+1, num_dim, num_dim)`` (time-invariant) or
        ``(T, P+1, P+1, num_dim, num_dim)`` (time-varying). The
        ``(i, j, :, :)`` entry stores the cross-region covariance
        ``Cov(x_i, x_j)``.
    lag:
        Markov order ``P`` (= ``K_lagged.shape[-3] - 1``). Passed
        explicitly for clarity / validation.
    num_dim:
        Number of regions / outputs.
    cov_jitter:
        Diagonal jitter added to the joint covariance before Cholesky
        factorisation, to keep it strictly positive definite.

    Note on the ``Q_full = R22^T R22 / 2`` halving: the AR-form
    innovation covariance carries a free positive constant (the
    temperature of the latent prior in the joint LL); changing it
    rescales ``Q^{-1}`` in the LL, which trades off how much the
    filter "trusts" the dynamics vs the observations. The factor
    ``1/2`` is the choice that empirically gives the strongest
    gradient signal on the kernel / delay parameters.

    Returns
    -------
    F:
        Lifted state-transition matrix, shape
        ``(P·num_dim, P·num_dim)`` or ``(T, P·num_dim, P·num_dim)``.
    Q:
        Lifted state-noise covariance, shape matching ``F``.
    """
    if K_lagged.ndim not in (4, 5):
        raise ValueError(
            f"K_lagged must have 4 or 5 dims (time-invariant or time-varying); got {K_lagged.ndim}"
        )
    expected_lp1 = lag + 1
    if K_lagged.shape[-3] != expected_lp1 or K_lagged.shape[-4] != expected_lp1:
        raise ValueError(
            f"K_lagged must have lag+1={expected_lp1} on the lag axes; got {tuple(K_lagged.shape[-4:-2])}"
        )

    big = _arrange_big_cov(K_lagged, num_dim)  # (..., (P+1)*d, (P+1)*d)
    big_size = big.shape[-1]
    eye_big = torch.eye(big_size, dtype=big.dtype, device=big.device)
    L = torch.linalg.cholesky(big + cov_jitter * eye_big)
    R = L.transpose(-1, -2)  # upper triangular

    n_p = lag * num_dim
    R11 = R[..., :n_p, :n_p]  # upper triangular (..., n_p, n_p)
    R12 = R[..., :n_p, n_p:]  # (..., n_p, num_dim)
    R22 = R[..., n_p:, n_p:]  # (..., num_dim, num_dim)

    # AR coefficients: B = L21 L11^{-1} = (R11^{-1} R12)^T  → shape (..., d, n_p)
    eye_p = torch.eye(n_p, dtype=big.dtype, device=big.device)
    eye_p_b = eye_p.expand(*R11.shape[:-2], n_p, n_p) if R11.ndim > 2 else eye_p
    R11_inv = torch.linalg.solve_triangular(R11, eye_p_b, upper=True)
    B = (R11_inv @ R12).transpose(-1, -2)  # (..., d, n_p)

    # Innovation covariance: Q_full = R22^T R22 / 2. See the docstring
    # for the /2 constant — it is the free temperature scale in the
    # AR-form derivation, picked empirically for optimisation.
    Q_full = (R22.transpose(-1, -2) @ R22) * 0.5  # (..., d, d)

    # Reverse the column blocks of B so columns align with the state convention
    # s_t = [x_t, x_{t-1}, …, x_{t-P+1}]: kernel ordering is oldest-first, our
    # state ordering is newest-first.
    # B has shape (..., d, P*d); reshape to (..., d, P, d), reverse the P axis,
    # then flatten back.
    B_view = B.reshape(*B.shape[:-1], lag, num_dim)
    B_rev = torch.flip(B_view, dims=(-2,))  # reverse the P axis
    A_first_rows = B_rev.reshape(*B.shape[:-1], lag * num_dim)  # (..., d, n_p)

    shift = identity_shift_block(lag, num_dim, dtype=big.dtype, device=big.device)
    if A_first_rows.ndim > 2:
        shift_b = shift.expand(*A_first_rows.shape[:-2], (lag - 1) * num_dim, lag * num_dim)
    else:
        shift_b = shift
    F = torch.cat([A_first_rows, shift_b], dim=-2)  # (..., n_p, n_p)

    # Q = block_diag(Q_full, Q_full, …, Q_full) on every leading slice.
    # See the module-level note for the reasoning behind reusing
    # Q_full on every lag block.
    Q = torch.zeros_like(F)
    Q[..., :num_dim, :num_dim] = Q_full
    if lag > 1:
        for k in range(1, lag):  # trial-loop: ok  (loop over lag, not trials)
            start = k * num_dim
            Q[..., start : start + num_dim, start : start + num_dim] = Q_full

    return F, Q


def lag_pair_grid(lag: int, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Return the ``(lag+1, lag+1)`` lag-difference grid ``τ_{ij} = i - j``."""
    t = torch.arange(0, lag + 1, dtype=dtype, device=device)
    return t.unsqueeze(1) - t.unsqueeze(0)


def lagged_cov_grid(
    kernel: LaggedCovKernel,
    lag: int,
    *,
    num_dim: int,
    delays: Tensor | None = None,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Build the lagged kernel covariance grid for the AR(P) lift.

    This is the generic bridge from any scalar stationary kernel to the
    ``(P+1, P+1, num_dim, num_dim)`` covariance tensor consumed by
    :func:`kernel_to_lds`. The kernel only needs to expose
    :meth:`LaggedCovKernel.cov` — multi-region structure and per-region
    delays are layered on here, not inside the kernel.

    Three input shapes for ``delays`` are accepted, mirroring the three
    :class:`~mbrila.core.delay_spec.Delay` flavours used by
    :class:`~mbrila.dynamics.markov_gp.MarkovianGPLatent`:

    - ``None`` — no delay; output ``(lag+1, lag+1, num_dim, num_dim)``
      with every ``(r1, r2)`` slot equal to ``k(τ_{ij})``.
    - ``(num_dim,)`` — static per-region delay; output
      ``(lag+1, lag+1, num_dim, num_dim)`` with
      ``effective_τ[i, j, r1, r2] = τ_{ij} + δ_{r2} - δ_{r1}``.
    - ``(T, num_dim)`` — time-varying per-region delay; output
      ``(T, lag+1, lag+1, num_dim, num_dim)`` with
      ``effective_τ[t, i, j, r1, r2] = τ_{ij} + δ_{r2}(t) - δ_{r1}(t)``.

    Parameters
    ----------
    kernel:
        Any object satisfying :class:`LaggedCovKernel` (i.e. exposing
        a pointwise ``cov(tau)`` method).
    lag:
        Markov order ``P``. The lag grid has ``P+1`` points.
    num_dim:
        Number of regions / outputs ``R``. The dynamics layer treats the
        multi-region structure as living outside the kernel.
    delays:
        See above. Reference region (``r = 0``) is expected to carry a
        zero entry per the library's convention; this is the caller's
        responsibility (the :class:`~mbrila.core.delay_spec.Delay`
        subclasses enforce it).
    dtype, device:
        Used to construct the lag grid. The kernel's ``cov`` is expected
        to return a tensor matching its own parameter dtype/device; the
        caller normally passes ``kernel``'s native dtype/device here so
        intermediate ``tau`` tensors match.
    """
    if lag < 1:
        raise ValueError(f"lag must be >= 1; got {lag}")
    if num_dim < 1:
        raise ValueError(f"num_dim must be >= 1; got {num_dim}")
    tau = lag_pair_grid(lag, dtype=dtype, device=device)  # (L, L)

    if delays is None:
        K_tau = kernel.cov(tau)  # (L, L)
        ones_rr = K_tau.new_ones(num_dim, num_dim)
        return K_tau.unsqueeze(-1).unsqueeze(-1) * ones_rr

    if delays.ndim == 1:
        if delays.shape[0] != num_dim:
            raise ValueError(f"1-D delays must have shape ({num_dim},); got {tuple(delays.shape)}")
        # effective_τ[i, j, r1, r2] = τ[i, j] + δ[r2] - δ[r1]
        tau_b = tau.unsqueeze(-1).unsqueeze(-1)  # (L, L, 1, 1)
        delays = delays.to(dtype=dtype, device=device)
        d_i = delays.view(1, 1, num_dim, 1)
        d_j = delays.view(1, 1, 1, num_dim)
        effective = tau_b + d_j - d_i  # (L, L, R, R)
        return kernel.cov(effective)

    if delays.ndim != 2 or delays.shape[-1] != num_dim:
        raise ValueError(f"2-D delays must have shape (T, {num_dim}); got {tuple(delays.shape)}")
    T_dim = delays.shape[0]
    tau_b = tau.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # (1, L, L, 1, 1)
    delays = delays.to(dtype=dtype, device=device)
    d_i = delays.view(T_dim, 1, 1, num_dim, 1)
    d_j = delays.view(T_dim, 1, 1, 1, num_dim)
    effective = tau_b + d_j - d_i  # (T, L, L, R, R)
    return kernel.cov(effective)
