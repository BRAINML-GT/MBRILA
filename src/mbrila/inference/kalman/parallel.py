"""Parallel-scan Kalman filter + RTS smoother (Sarkka & Garcia-Fernandez, 2021).

Implements the associative-scan formulation of both the Kalman filter
and the RTS smoother so the forward / backward sweeps become
:math:`O(\\log T)` work-depth instead of :math:`O(T)`. Each routine
defines a per-time-step element and an associative combine operator and
scans them with a Blelloch-style recursion implemented in this module
(:func:`associative_scan`).

Conventions
-----------
- Same as :mod:`mbrila.inference.kalman.sequential`: ``y`` is ``(B, T, N)``,
  dynamics are ``(T, D, D)`` (or ``(D, D)`` constant), prior ``(m0, P0)``
  lives at ``t = -1``.
- Outputs match the sequential routines to within floating-point
  tolerance (verified by ``tests/unit/test_kalman_parallel.py``).

Why a custom ``associative_scan`` and not ``torch._higher_order_ops``?
---------------------------------------------------------------------
PyTorch 2.11 ships ``torch._higher_order_ops.associative_scan``, but it is
not a drop-in replacement for what mbrila needs:

1. **No autograd support.** The official scan is a prototype HOP that
   integrates with ``torch.compile`` and explicitly does *not* support
   autograd. The Kalman filter here is on the autograd path of the
   training objective (marginal log-likelihood backpropagated through
   the filter to update kernel σ and delays), so the filter must use
   the custom scan regardless of device. This is a hard constraint —
   not a stylistic choice.

2. **CUDA-only.** The official scan requires runtime codegen (CUDA only).
   mbrila's tests, recovery checks, and CI all run on CPU; the custom
   scan works everywhere.

3. **No measurable speed-up on relevant problem sizes.** Even on the
   smoother (which runs under ``torch.no_grad`` and *could* in principle
   use the official scan), benchmarking on a single GPU shows the
   custom Blelloch implementation is uniformly **1.2-1.8× faster** for
   problem sizes from ``T=50`` to ``T=5000`` and ``D=6`` to ``D=30``.
   The official ``combine_mode="generic"`` path adds per-call dispatch
   overhead that is not amortised over our combine's already-large
   matrix operations (matmul / Cholesky solve on small ``D``).

   Indicative numbers on one CUDA GPU (float64, median of 10 repeats
   after 3 warmups; sequential is provided as a baseline)::

         B    T    D    sequential   custom_par     official
         8   50    6      29.6 ms       3.9 ms        6.7 ms
         8  200    6     119.4 ms       5.0 ms        8.9 ms
         8 1000    6     607.8 ms       6.9 ms       11.6 ms
         4 5000    6    3084.7 ms       9.9 ms       15.6 ms
         4   50   30      31.6 ms       4.0 ms        6.7 ms
         4 1000   30     675.4 ms      13.3 ms       15.9 ms

   Custom and official agree to ``1e-12`` (float64 round-off).

If a future PyTorch release adds (a) autograd support and (b) a CPU
backend to ``associative_scan``, **and** the per-call overhead becomes
competitive on D≈10 matrix combines, switching is a one-line change in
:func:`kalman_filter_parallel` and :func:`rts_smoother_parallel`.
Re-run ``scratch/bench_smoother_backends.py`` before that change.

Notes on the first scan element (filter)
----------------------------------------
The published Jax reference (Corenflos / Sarkka 2021,
``parallel_kalman_jax.ipynb``) uses ``A_0 = 0`` but populates ``J_0``
and ``eta_0`` from the "general element" formulas (``H Q_0 H^T + R``).
This port sets all three to zero. Both choices yield identical
filtered ``b`` and ``C`` outputs because ``J_0`` and ``eta_0`` are only
read as the *left*-hand element of subsequent associative combines
inside the scan, and the combine's ``b``/``C`` outputs depend solely
on the right-hand element's ``J``/``eta`` — never on the left's. We
zero them out for clarity (and to save a Cholesky solve on the first
element); the unit tests verify exact agreement with the sequential
filter so the equivalence is checked numerically.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# Generic Blelloch-style associative scan
# ---------------------------------------------------------------------------


_Tup = tuple[Tensor, ...]


def _combine_pairs(operator: Callable[[_Tup, _Tup], _Tup], left: _Tup, right: _Tup) -> _Tup:
    return operator(left, right)


def associative_scan(
    operator: Callable[[_Tup, _Tup], _Tup],
    elems: _Tup,
    *,
    dim: int = 0,
    reverse: bool = False,
) -> _Tup:
    """Inclusive associative scan along ``dim``.

    The scan returns a tuple of tensors with the same shapes as ``elems``,
    where output index ``t`` along ``dim`` is the combine of inputs
    ``[0, 1, …, t]``.

    Implemented via Blelloch's recursive halving so the work depth is
    :math:`O(\\log T)` and the work is :math:`O(T)`. The operator is called
    with batched element-pair tuples (each tensor has ``T_pair`` items
    along ``dim``), so make sure the operator's tensor ops broadcast
    correctly along that axis.
    """
    if not elems:
        raise ValueError("associative_scan: elems must be non-empty")

    if reverse:
        elems = tuple(torch.flip(e, dims=(dim,)) for e in elems)

    out = _scan_recursive(operator, elems, dim)

    if reverse:
        out = tuple(torch.flip(e, dims=(dim,)) for e in out)
    return out


def _slice_along(t: Tensor, dim: int, start: int, end: int | None, step: int = 1) -> Tensor:
    """Return ``t[..., start:end:step, ...]`` along ``dim``."""
    idx = [slice(None)] * t.ndim
    idx[dim] = slice(start, end, step)
    return t[tuple(idx)]


def _scan_recursive(
    operator: Callable[[_Tup, _Tup], _Tup],
    elems: _Tup,
    dim: int,
) -> _Tup:
    n = elems[0].shape[dim]
    if n < 2:
        return elems

    # Combine adjacent pairs: result has ⌊n/2⌋ entries.
    even = tuple(_slice_along(e, dim, 0, None, 2) for e in elems)
    odd = tuple(_slice_along(e, dim, 1, None, 2) for e in elems)
    # If n is odd, drop the lone trailing 'even' (it has no partner).
    if n % 2 == 1:
        even = tuple(_slice_along(e, dim, 0, n // 2) for e in even)
    paired = _combine_pairs(operator, even, odd)

    # Recurse on the reduced sequence.
    scanned_paired = _scan_recursive(operator, paired, dim)

    # Build the final output of length n. At even output indices > 0 we
    # combine the previous "scanned_paired[k-1]" with the original element at
    # 2k. At odd output indices we use scanned_paired[k] directly.
    # Output index 0 is just elems[0].
    even_out_left = scanned_paired
    even_out_right = tuple(_slice_along(e, dim, 2, None, 2) for e in elems)
    if n % 2 == 0:
        # We need n/2 - 1 even-index outputs (indices 2, 4, …, n-2).
        # scanned_paired has n//2 items; we use [:-1].
        even_out_left = tuple(_slice_along(e, dim, 0, -1) for e in scanned_paired)
    even_out_combined = _combine_pairs(operator, even_out_left, even_out_right)

    # Prepend output index 0 (the original first element) to even outputs.
    first = tuple(_slice_along(e, dim, 0, 1) for e in elems)
    even_combined_full = tuple(
        torch.cat([f, c], dim=dim) for f, c in zip(first, even_out_combined, strict=True)
    )

    # Interleave even_combined_full (indices 0, 2, 4, …) with scanned_paired (1, 3, …).
    return tuple(_interleave(a, b, dim) for a, b in zip(even_combined_full, scanned_paired, strict=True))


def _interleave(a: Tensor, b: Tensor, dim: int) -> Tensor:
    """Interleave ``a`` (even positions) and ``b`` (odd positions) along ``dim``.

    Supports the case where ``a.shape[dim] == b.shape[dim] + 1`` (the final
    even position has no odd partner): we pad ``b`` with a zero slice and
    then drop the trailing position after stacking.
    """
    a_n = a.shape[dim]
    b_n = b.shape[dim]
    if a_n != b_n and a_n != b_n + 1:
        raise ValueError(f"interleave: incompatible lengths {a_n} vs {b_n} along dim {dim}")
    pad_b = a_n == b_n + 1
    if pad_b:
        pad_shape = list(b.shape)
        pad_shape[dim] = 1
        zero = torch.zeros(pad_shape, dtype=b.dtype, device=b.device)
        b = torch.cat([b, zero], dim=dim)

    stacked = torch.stack([a, b], dim=dim + 1)
    flat_shape = list(a.shape)
    flat_shape[dim] = a_n + b.shape[dim]
    interleaved = stacked.flatten(start_dim=dim, end_dim=dim + 1)
    if pad_b:
        interleaved = _slice_along(interleaved, dim, 0, a_n + b_n)
    return interleaved


# ---------------------------------------------------------------------------
# Kalman scan element + combine operator
# ---------------------------------------------------------------------------


def _kalman_combine(left: _Tup, right: _Tup) -> _Tup:
    """Sarkka–Garcia-Fernandez 2021, Eq. 25-26 combine operator.

    Each element is a tuple ``(A, b, C, J, eta)`` whose tensors share an
    arbitrary leading shape (the scan axis plus optional batch axes).
    ``A``, ``C``, ``J`` carry an extra ``(D, D)`` matrix tail and ``b``,
    ``eta`` carry a ``(D,)`` vector tail.
    """
    A1, b1, C1, J1, eta1 = left
    A2, b2, C2, J2, eta2 = right

    D = A1.shape[-1]
    eye = torch.eye(D, dtype=A1.dtype, device=A1.device)

    # temp_b = A2 (I + C1 J2)^{-1}                    (used for b, C, A combine)
    M_bc = eye + C1 @ J2
    # solve M_bc^T X = A2^T → X = M_bc^{-T} A2^T → X^T = A2 M_bc^{-1}
    temp_b = torch.linalg.solve(M_bc.transpose(-1, -2), A2.transpose(-1, -2)).transpose(-1, -2)

    # temp_e = A1^T (I + J2 C1)^{-1}                  (used for eta, J combine)
    M_ej = eye + J2 @ C1
    temp_e = torch.linalg.solve(M_ej.transpose(-1, -2), A1).transpose(-1, -2)

    A_out = temp_b @ A1
    C_out = temp_b @ C1 @ A2.transpose(-1, -2) + C2

    # b_out = A2 (I + C1 J2)^{-1} (b1 + C1 eta2) + b2
    b1u = b1.unsqueeze(-1)  # (..., D, 1)
    eta2u = eta2.unsqueeze(-1)
    b_inner = b1u + C1 @ eta2u
    b_out = (temp_b @ b_inner).squeeze(-1) + b2

    # eta_out = A1^T (I + J2 C1)^{-1} (eta2 - J2 b1) + eta1
    eta_inner = eta2u - J2 @ b1u
    eta_out = (temp_e @ eta_inner).squeeze(-1) + eta1

    J_out = temp_e @ J2 @ A1 + J1

    return (A_out, b_out, C_out, J_out, eta_out)


# ---------------------------------------------------------------------------
# Public filter
# ---------------------------------------------------------------------------


def kalman_filter_parallel(
    y: Tensor,
    F: Tensor,
    Q: Tensor,
    H: Tensor,
    R: Tensor,
    m0: Tensor,
    P0: Tensor,
) -> tuple[Tensor, Tensor]:
    """Parallel-scan Kalman filter.

    Same signature as :func:`mbrila.inference.kalman.sequential.kalman_filter`
    minus the log-marginal-likelihood return: the parallel filter does not
    track per-step prediction densities (compute the marginal in the
    sequential filter or via a separate batched pass when needed).

    Returns
    -------
    filtered_means:
        ``(B, T, D)``.
    filtered_covs:
        ``(B, T, D, D)``.
    """
    if y.ndim != 3:
        raise ValueError(f"y must have shape (B, T, N); got {tuple(y.shape)}")
    B, T, N = y.shape
    if H.shape[0] != N:
        raise ValueError(f"H rows ({H.shape[0]}) must match y last dim ({N})")
    D = H.shape[1]
    if R.shape != (N, N):
        raise ValueError(f"R must have shape ({N}, {N}); got {tuple(R.shape)}")

    # Broadcast dynamics to (T, D, D).
    if F.ndim == 2:
        F = F.unsqueeze(0).expand(T, D, D)
    if Q.ndim == 2:
        Q = Q.unsqueeze(0).expand(T, D, D)
    if F.shape != (T, D, D):
        raise ValueError(f"F must have shape (T={T}, D={D}, D); got {tuple(F.shape)}")
    if Q.shape != (T, D, D):
        raise ValueError(f"Q must have shape (T={T}, D={D}, D); got {tuple(Q.shape)}")

    # Broadcast prior to (B, D), (B, D, D).
    if m0.ndim == 1:
        m0 = m0.unsqueeze(0).expand(B, D)
    if P0.ndim == 2:
        P0 = P0.unsqueeze(0).expand(B, D, D)

    eye_D = torch.eye(D, dtype=y.dtype, device=y.device)

    # ------------------------------------------------------------------
    # General-element quantities for *every* time step (will overwrite t=0).
    #
    # S_t       = H Q_t H^T + R                              (T, N, N)
    # K_t       = Q_t H^T S_t^{-1}                          (T, D, N)
    # I_KH_t    = I - K_t H                                   (T, D, D)
    # A_t       = I_KH_t F_t                                  (T, D, D)
    # C_t       = I_KH_t Q_t I_KH_t^T + K_t R K_t^T           (T, D, D)
    # J_t       = F_t^T H^T S_t^{-1} H F_t                    (T, D, D)
    # b_t       = K_t y_t                                     (B, T, D)
    # eta_t     = F_t^T H^T S_t^{-1} y_t                      (B, T, D)
    # ------------------------------------------------------------------
    HQ = torch.einsum("ij,tjk->tik", H, Q)  # (T, N, D)
    S = torch.einsum("tij,kj->tik", HQ, H) + R  # (T, N, N)
    L_S = torch.linalg.cholesky(S)
    # K = Q H^T S^{-1} = (HQ)^T S^{-T} = (HQ)^T S^{-1} since S is symmetric.
    # Compute via cholesky_solve(HQ): solve S X = HQ → X = (T, N, D); K = X^T = (T, D, N).
    K = torch.cholesky_solve(HQ, L_S).transpose(-1, -2)  # (T, D, N)

    I_KH = eye_D - torch.einsum("tij,jk->tik", K, H)  # (T, D, D)
    A_general = torch.einsum("tij,tjk->tik", I_KH, F)  # (T, D, D)
    KQ = torch.einsum("tij,tjk->tik", I_KH, Q)  # (T, D, D)
    C_general = torch.einsum("tij,tlj->til", KQ, I_KH) + torch.einsum("tij,jk,tlk->til", K, R, K)  # (T, D, D)

    HF = torch.einsum("ij,tjk->tik", H, F)  # (T, N, D)
    S_inv_HF = torch.cholesky_solve(HF, L_S)  # (T, N, D)
    J_general = torch.einsum("tji,tjk->tik", HF, S_inv_HF)  # (T, D, D)

    # b and eta need both time and trial axes.
    # b_t = K_t y_t: y has (B, T, N); K has (T, D, N) → (B, T, D)
    b_general = torch.einsum("tij,btj->bti", K, y)  # (B, T, D)
    # eta_t = HF^T S^{-1} y_t = (HF)^T (S^{-1} y_t)
    # First solve S z = y per time bin. y has (B, T, N) and S has (T, N, N).
    # We solve along the N dim; treat each (B,) trial as a separate RHS vector.
    y_for_solve = y.transpose(0, 1).unsqueeze(-1)  # (T, B, N, 1)
    S_inv_y = torch.cholesky_solve(y_for_solve, L_S.unsqueeze(1)).squeeze(-1)  # (T, B, N)
    eta_general = torch.einsum("tji,tbj->tbi", HF, S_inv_y).transpose(0, 1)  # (B, T, D)

    # ------------------------------------------------------------------
    # First element (t = 0): prior absorption. A_0 = J_0 = eta_0 = 0; b_0 and C_0 are
    # the standard one-step filter posterior given (m0, P0).
    # ------------------------------------------------------------------
    F0 = F[0]
    Q0 = Q[0]
    m_pred_0 = torch.einsum("ij,bj->bi", F0, m0)  # (B, D)
    P_pred_0 = torch.einsum("ij,bjk,lk->bil", F0, P0, F0) + Q0  # (B, D, D)
    HP_pred_0 = torch.einsum("ij,bjk->bik", H, P_pred_0)  # (B, N, D)
    S_full_0 = torch.einsum("bij,kj->bik", HP_pred_0, H) + R  # (B, N, N)
    L_full_0 = torch.linalg.cholesky(S_full_0)
    K_0 = torch.cholesky_solve(HP_pred_0, L_full_0).transpose(-1, -2)  # (B, D, N)
    innovation_0 = y[:, 0] - torch.einsum("ij,bj->bi", H, m_pred_0)  # (B, N)
    b_0 = m_pred_0 + torch.einsum("bij,bj->bi", K_0, innovation_0)  # (B, D)
    I_K0H = eye_D - torch.einsum("bij,jk->bik", K_0, H)  # (B, D, D)
    C_0 = torch.einsum("bij,bjk,blk->bil", I_K0H, P_pred_0, I_K0H) + torch.einsum(
        "bij,jk,blk->bil", K_0, R, K_0
    )

    # Assemble per-time elements with shape (T, B, *).
    # For trial-shared matrices (A, C, J) we expand the batch dim.
    A_all = A_general.unsqueeze(1).expand(T, B, D, D).contiguous()
    C_all = C_general.unsqueeze(1).expand(T, B, D, D).contiguous()
    J_all = J_general.unsqueeze(1).expand(T, B, D, D).contiguous()
    b_all = b_general.transpose(0, 1).contiguous()  # (T, B, D)
    eta_all = eta_general.transpose(0, 1).contiguous()  # (T, B, D)

    A_all[0] = torch.zeros_like(A_all[0])
    J_all[0] = torch.zeros_like(J_all[0])
    eta_all[0] = torch.zeros_like(eta_all[0])
    b_all[0] = b_0
    C_all[0] = C_0

    # Run the scan along the time axis.
    _A_scan, b_scan, C_scan, _J_scan, _eta_scan = associative_scan(
        _kalman_combine, (A_all, b_all, C_all, J_all, eta_all), dim=0
    )

    filtered_means = b_scan.transpose(0, 1).contiguous()  # (B, T, D)
    filtered_covs = C_scan.transpose(0, 1).contiguous()  # (B, T, D, D)
    return filtered_means, filtered_covs


# ---------------------------------------------------------------------------
# Parallel RTS smoother (Sarkka & Garcia-Fernandez 2021, smoother section)
# ---------------------------------------------------------------------------


def _smoother_combine(left: _Tup, right: _Tup) -> _Tup:
    """Reverse-direction associative combine for the parallel RTS smoother.

    Each element is a triple ``(E, g, L)`` representing the linear
    recurrence ``m_s_t = E_t m_s_{t+1} + g_t``,
    ``P_s_t = L_t + E_t P_s_{t+1} E_t^T``. Composition (applied
    right-to-left) is

    ::

        E_combined = E_right @ E_left
        g_combined = E_right @ g_left + g_right
        L_combined = E_right @ L_left @ E_right^T + L_right

    This matches the Jax reference (``smoothing_operator`` in
    ``parallel_kalman_jax.ipynb``).
    """
    E1, g1, L1 = left
    E2, g2, L2 = right
    E_out = E2 @ E1
    g_out = (E2 @ g1.unsqueeze(-1)).squeeze(-1) + g2
    L_out = E2 @ L1 @ E2.transpose(-1, -2) + L2
    return (E_out, g_out, L_out)


def rts_smoother_parallel(
    filtered_means: Tensor,
    filtered_covs: Tensor,
    F: Tensor,
    Q: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Parallel-scan Rauch–Tung–Striebel smoother.

    Reproduces the sequential smoother in
    :func:`mbrila.inference.kalman.sequential.rts_smoother` to
    floating-point tolerance, but with :math:`O(\\log T)` work-depth via
    a reverse associative scan. The construction follows
    Särkkä & Garcia-Fernández (2021); the elements ``(E_t, g_t, L_t)``
    encode the smoother recurrence
    ``m_s_t = E_t m_s_{t+1} + g_t`` so that combining them with the
    associative operator above yields the smoothed posterior in one
    parallel sweep.

    Parameters
    ----------
    filtered_means:
        Output of :func:`kalman_filter_parallel`, shape ``(B, T, D)``.
    filtered_covs:
        Output of :func:`kalman_filter_parallel`, shape ``(B, T, D, D)``.
    F, Q:
        Same dynamics matrices as the filter; ``(T, D, D)`` (or ``(D, D)``
        constant). ``F[t]`` propagates ``x_{t-1} → x_t``.

    Returns
    -------
    smoothed_means:
        ``(B, T, D)``.
    smoothed_covs:
        ``(B, T, D, D)``.
    pairwise_covs:
        ``(B, T-1, D, D)`` storing the centred cross-time covariance
        ``Cov(x_t, x_{t+1} | y_{0:T-1}) = G_t P_s_{t+1}`` where ``G_t``
        is the smoother gain. Computed as a small post-processing step
        after the scan.
    """
    if filtered_means.ndim != 3:
        raise ValueError(f"filtered_means must have shape (B, T, D); got {tuple(filtered_means.shape)}")
    B, T, D = filtered_means.shape
    if filtered_covs.shape != (B, T, D, D):
        raise ValueError(
            f"filtered_covs must have shape ({B}, {T}, {D}, {D}); got {tuple(filtered_covs.shape)}"
        )

    # Broadcast dynamics to (T, D, D).
    if F.ndim == 2:
        F = F.unsqueeze(0).expand(T, D, D)
    if Q.ndim == 2:
        Q = Q.unsqueeze(0).expand(T, D, D)
    if F.shape != (T, D, D):
        raise ValueError(f"F must have shape (T={T}, D={D}, D); got {tuple(F.shape)}")
    if Q.shape != (T, D, D):
        raise ValueError(f"Q must have shape (T={T}, D={D}, D); got {tuple(Q.shape)}")

    # ------------------------------------------------------------------
    # Build per-time smoother elements (E_t, g_t, L_t) for t = 0..T-1.
    # The element at t uses F[t+1] / Q[t+1] (the dynamics that propagate
    # x_t → x_{t+1}); the last element (t = T-1) is the terminal element
    # with E = 0, g = m_f_{T-1}, L = P_f_{T-1}.
    # ------------------------------------------------------------------
    F_next = F[1:]  # (T-1, D, D)
    Q_next = Q[1:]  # (T-1, D, D)

    f_means_prefix = filtered_means[:, :-1]  # (B, T-1, D)
    f_covs_prefix = filtered_covs[:, :-1]  # (B, T-1, D, D)

    # Predicted dynamics at each prefix step (per trial).
    # m_pred_{t+1} = F_{t+1} m_f_t,  shape (B, T-1, D)
    # P_pred_{t+1} = F_{t+1} P_f_t F_{t+1}^T + Q_{t+1}, shape (B, T-1, D, D)
    m_pred = torch.einsum("tij,btj->bti", F_next, f_means_prefix)
    P_f_FT = torch.einsum("btij,tkj->btik", f_covs_prefix, F_next)  # P_f_t F_{t+1}^T
    P_pred = torch.einsum("tij,btjk->btik", F_next, P_f_FT) + Q_next  # (B, T-1, D, D)

    # Symmetrize P_pred: F P_f F^T + Q is exactly symmetric in math but
    # einsum + add accumulates tiny asymmetry that intermittently trips
    # cuSOLVER's batched Cholesky check. Cheap, no semantic change in fp64.
    P_pred = 0.5 * (P_pred + P_pred.transpose(-1, -2))

    # Smoother gain G_t = P_f_t F_{t+1}^T (P_pred)^{-1}, shape (B, T-1, D, D).
    # Adaptive jitter retry: occasionally P_pred is genuinely on the edge
    # of PSD (lifted-state lag blocks can have very small leading
    # eigenvalues). Try clean Cholesky first; on failure, add scaled
    # diagonal jitter and retry.
    try:
        L_pred = torch.linalg.cholesky(P_pred)
    except RuntimeError:
        diag_eye = torch.eye(D, dtype=P_pred.dtype, device=P_pred.device)
        for jit in (1e-10, 1e-8, 1e-6):
            try:
                L_pred = torch.linalg.cholesky(P_pred + jit * diag_eye)
                break
            except RuntimeError:
                continue
        else:
            raise
    # solve P_pred X = (P_f F^T)^T → X^T = G
    G = torch.cholesky_solve(P_f_FT.transpose(-1, -2), L_pred).transpose(-1, -2)

    # Element triple for the prefix (t = 0..T-2).
    E_prefix = G  # (B, T-1, D, D)
    g_prefix = f_means_prefix - torch.einsum("btij,btj->bti", G, m_pred)
    # L_t = P_f_t - G_t P_pred_{t+1} G_t^T
    L_prefix = f_covs_prefix - torch.einsum("btij,btjk,btlk->btil", G, P_pred, G)

    # Terminal element (t = T-1): E = 0, g = m_f_{T-1}, L = P_f_{T-1}.
    last_E = torch.zeros(B, 1, D, D, dtype=filtered_means.dtype, device=filtered_means.device)
    last_g = filtered_means[:, -1:].clone()
    last_L = filtered_covs[:, -1:].clone()

    E_all = torch.cat([E_prefix, last_E], dim=1).transpose(0, 1).contiguous()  # (T, B, D, D)
    g_all = torch.cat([g_prefix, last_g], dim=1).transpose(0, 1).contiguous()  # (T, B, D)
    L_all = torch.cat([L_prefix, last_L], dim=1).transpose(0, 1).contiguous()  # (T, B, D, D)

    # Reverse associative scan (combine right-to-left in time).
    _E_scan, g_scan, L_scan = associative_scan(_smoother_combine, (E_all, g_all, L_all), dim=0, reverse=True)

    smoothed_means = g_scan.transpose(0, 1).contiguous()  # (B, T, D)
    smoothed_covs = L_scan.transpose(0, 1).contiguous()  # (B, T, D, D)

    # Pairwise centred covariance: Cov(x_t, x_{t+1}) = G_t P_s_{t+1}.
    # G has shape (B, T-1, D, D); P_s_{t+1} for t = 0..T-2 lives in
    # smoothed_covs[:, 1:].
    pairwise_covs = torch.einsum("btij,btjk->btik", G, smoothed_covs[:, 1:])

    return smoothed_means, smoothed_covs, pairwise_covs
