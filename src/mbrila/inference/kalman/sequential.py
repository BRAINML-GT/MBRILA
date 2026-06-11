"""Sequential (loop-over-time) Kalman filter and RTS smoother.

The implementation is fully batched over the trial dimension: every operation
inside the time loop processes the entire ``(n_trials, ...)`` batch at once,
so the only Python loop is over time. The trial loop is forbidden by mbrila's
no-trial-loop contract.

Conventions
-----------
- Latent state ``x_t`` lives in ``R^D``.
- Observation ``y_t`` lives in ``R^N``.
- Dynamics ``x_t = F_t x_{t-1} + N(0, Q_t)`` with the convention that the
  initial state ``(m0, P0)`` is the **prior at t = -1**, so that the first
  predict step ``F_0 m0`` produces the prior at t = 0 *before* observing
  ``y_0``. This matches the ADM parallel-filter convention.
- Observation ``y_t = H x_t + N(0, R)``; ``H`` and ``R`` are time-invariant
  in v1.
- Tensor layout: ``y`` has shape ``(B, T, N)``; outputs have shape
  ``(B, T, D)`` for means, ``(B, T, D, D)`` for covariances.

Numerical care
--------------
- The covariance update uses the Joseph form
  ``P_new = (I - K H) P_pred (I - K H)^T + K R K^T`` to remain symmetric
  positive-definite under finite precision.
- Cholesky factors of the innovation covariance ``S_t`` are reused for
  computing the Kalman gain (via :func:`torch.cholesky_solve`) *and* the
  log-marginal-likelihood (via the diagonal of the factor).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from mbrila.inference.kalman.state import GaussianState


def _broadcast_priors(
    m0: Tensor,
    P0: Tensor,
    n_trials: int,
    D: int,
) -> tuple[Tensor, Tensor]:
    """Broadcast ``(D,) / (B, D)`` and ``(D, D) / (B, D, D)`` priors to ``(B, ...)``."""
    if m0.ndim == 1:
        m0 = m0.unsqueeze(0).expand(n_trials, D)
    if P0.ndim == 2:
        P0 = P0.unsqueeze(0).expand(n_trials, D, D)
    if m0.shape != (n_trials, D):
        raise ValueError(f"m0 must have shape ({n_trials}, {D}); got {tuple(m0.shape)}")
    if P0.shape != (n_trials, D, D):
        raise ValueError(f"P0 must have shape ({n_trials}, {D}, {D}); got {tuple(P0.shape)}")
    return m0.contiguous(), P0.contiguous()


def _broadcast_dynamics(F: Tensor, Q: Tensor, T: int, D: int) -> tuple[Tensor, Tensor]:
    """Broadcast ``(D, D)`` constant or ``(T, D, D)`` time-varying to ``(T, D, D)``."""
    if F.ndim == 2:
        F = F.unsqueeze(0).expand(T, D, D)
    if Q.ndim == 2:
        Q = Q.unsqueeze(0).expand(T, D, D)
    if F.shape != (T, D, D):
        raise ValueError(f"F must have shape (T={T}, D={D}, D); got {tuple(F.shape)}")
    if Q.shape != (T, D, D):
        raise ValueError(f"Q must have shape (T={T}, D={D}, D); got {tuple(Q.shape)}")
    return F, Q


def kalman_filter(
    y: Tensor,
    F: Tensor,
    Q: Tensor,
    H: Tensor,
    R: Tensor,
    m0: Tensor,
    P0: Tensor,
    *,
    return_log_marginal: bool = True,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run a forward Kalman filter over ``T`` time bins.

    Parameters
    ----------
    y:
        Observations of shape ``(B, T, N)``.
    F, Q:
        Dynamics matrices of shape ``(T, D, D)`` (or ``(D, D)`` if constant).
    H:
        Emission matrix of shape ``(N, D)``.
    R:
        Observation noise covariance of shape ``(N, N)``, symmetric PD.
    m0, P0:
        Prior on ``x_{-1}``: ``m0`` is ``(D,)`` or ``(B, D)``, ``P0`` is
        ``(D, D)`` or ``(B, D, D)``.
    return_log_marginal:
        If ``True``, also return the per-trial log marginal likelihood
        ``log p(y_{0:T-1})``. Set to ``False`` to skip the small extra
        compute when only the filtered means / covs are needed.

    Returns
    -------
    filtered_means:
        ``(B, T, D)``.
    filtered_covs:
        ``(B, T, D, D)``.
    log_marginal:
        ``(B,)`` if ``return_log_marginal`` else an empty tensor of shape
        ``(0,)`` (kept in the signature to make the return type stable for
        callers that need both modes).
    """
    if y.ndim != 3:
        raise ValueError(f"y must have shape (B, T, N); got {tuple(y.shape)}")
    B, T, N = y.shape
    if H.shape[0] != N:
        raise ValueError(f"H rows ({H.shape[0]}) must match y last dim ({N})")
    D = H.shape[1]
    if R.shape != (N, N):
        raise ValueError(f"R must have shape ({N}, {N}); got {tuple(R.shape)}")

    F, Q = _broadcast_dynamics(F, Q, T, D)
    m_prev, P_prev = _broadcast_priors(m0, P0, B, D)

    eye_D = torch.eye(D, dtype=y.dtype, device=y.device)
    log_two_pi = math.log(2.0 * math.pi)

    means_per_t: list[Tensor] = []
    covs_per_t: list[Tensor] = []
    log_ml = torch.zeros(B, dtype=y.dtype, device=y.device) if return_log_marginal else None

    for t in range(T):
        F_t = F[t]  # (D, D)
        Q_t = Q[t]  # (D, D)

        # --- predict: m_pred = F m_prev,  P_pred = F P_prev F^T + Q
        m_pred = torch.einsum("ij,bj->bi", F_t, m_prev)
        P_pred = torch.einsum("ij,bjk,lk->bil", F_t, P_prev, F_t) + Q_t

        # --- innovation
        y_t = y[:, t]  # (B, N)
        innovation = y_t - torch.einsum("ij,bj->bi", H, m_pred)  # (B, N)
        # S = H P_pred H^T + R, shape (B, N, N)
        HP = torch.einsum("ij,bjk->bik", H, P_pred)  # (B, N, D)
        S = torch.einsum("bij,kj->bik", HP, H) + R  # (B, N, N)

        # Cholesky of S — reused for the gain and the log-marginal.
        L = torch.linalg.cholesky(S)

        if log_ml is not None:
            # log|S| = 2 * sum(log(diag(L)))
            log_det_S = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)  # (B,)
            # Mahalanobis: ‖L^{-1} innovation‖²
            z = torch.linalg.solve_triangular(L, innovation.unsqueeze(-1), upper=False).squeeze(-1)
            quad = (z * z).sum(dim=-1)  # (B,)
            log_ml = log_ml + (-0.5) * (quad + log_det_S + N * log_two_pi)

        # --- gain: K = P_pred H^T S^{-1}, shape (B, D, N)
        # Compute via Cholesky: solve S X = (H P_pred), then K = X^T.
        # HP has shape (B, N, D); we want X = S^{-1} HP, so K = HP^T S^{-T} = HP^T S^{-1}.
        S_inv_HP = torch.cholesky_solve(HP, L)  # (B, N, D)
        K = S_inv_HP.transpose(-1, -2)  # (B, D, N)

        # --- update (Joseph form)
        m_new = m_pred + torch.einsum("bij,bj->bi", K, innovation)
        I_KH = eye_D - torch.einsum("bij,jk->bik", K, H)  # (B, D, D)
        P_joseph = torch.einsum("bij,bjk,blk->bil", I_KH, P_pred, I_KH)
        # K R K^T term
        KR = torch.einsum("bij,jk->bik", K, R)  # (B, D, N)
        P_new = P_joseph + torch.einsum("bij,blj->bil", KR, K)

        means_per_t.append(m_new)
        covs_per_t.append(P_new)
        m_prev, P_prev = m_new, P_new

    filtered_means = torch.stack(means_per_t, dim=1)  # (B, T, D)
    filtered_covs = torch.stack(covs_per_t, dim=1)  # (B, T, D, D)
    out_log_ml = log_ml if log_ml is not None else torch.empty(0, dtype=y.dtype, device=y.device)
    return filtered_means, filtered_covs, out_log_ml


def rts_smoother(
    filtered_means: Tensor,
    filtered_covs: Tensor,
    F: Tensor,
    Q: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Rauch–Tung–Striebel backward smoother.

    Implements the standard sequential recursion
    ::

        m_s_T = m_f_T,                P_s_T = P_f_T
        for t = T-2, …, 0:
            m_pred  = F_{t+1} m_f_t
            P_pred  = F_{t+1} P_f_t F_{t+1}^T + Q_{t+1}
            G_t     = P_f_t F_{t+1}^T P_pred^{-1}
            m_s_t   = m_f_t + G_t (m_s_{t+1} - m_pred)
            P_s_t   = P_f_t + G_t (P_s_{t+1} - P_pred) G_t^T

    Parameters
    ----------
    filtered_means:
        Output of :func:`kalman_filter`, shape ``(B, T, D)``.
    filtered_covs:
        Output of :func:`kalman_filter`, shape ``(B, T, D, D)``.
    F, Q:
        Dynamics matrices of shape ``(T, D, D)`` (or ``(D, D)`` if constant).
        ``F[t]`` is the transition from ``x_{t-1}`` to ``x_t`` — same
        convention as :func:`kalman_filter`.

    Returns
    -------
    smoothed_means:
        ``(B, T, D)``.
    smoothed_covs:
        ``(B, T, D, D)``.
    pairwise_covs:
        ``(B, T-1, D, D)`` storing ``Cov(x_t, x_{t+1} | y_{0:T-1})`` —
        i.e., the cross-time *centred* covariance, *not* the raw
        ``E[x_t x_{t+1}^T]`` second moment. Add
        ``smoothed_means[:, :-1, :, None] * smoothed_means[:, 1:, None, :]`` to
        recover the second moment when needed by the M-step.

    Note
    ----
    we run the recursion backward in time rather than vectorising it
    incorrectly.
    """
    if filtered_means.ndim != 3:
        raise ValueError(f"filtered_means must have shape (B, T, D); got {tuple(filtered_means.shape)}")
    B, T, D = filtered_means.shape
    if filtered_covs.shape != (B, T, D, D):
        raise ValueError(
            f"filtered_covs must have shape ({B}, {T}, {D}, {D}); got {tuple(filtered_covs.shape)}"
        )
    F, Q = _broadcast_dynamics(F, Q, T, D)

    # Build the smoothed trajectories backwards in time as Python lists, then
    # stack at the end. This keeps every operation differentiable through
    # autograd without relying on indexed in-place assignment.
    smoothed_means_rev: list[Tensor] = [filtered_means[:, -1]]
    smoothed_covs_rev: list[Tensor] = [filtered_covs[:, -1]]
    pairwise_covs_rev: list[Tensor] = []

    m_s_next = filtered_means[:, -1]
    P_s_next = filtered_covs[:, -1]

    for t in range(T - 2, -1, -1):
        F_next = F[t + 1]  # transition x_t -> x_{t+1}
        Q_next = Q[t + 1]
        m_f = filtered_means[:, t]  # (B, D)
        P_f = filtered_covs[:, t]  # (B, D, D)

        # m_pred = F_{t+1} m_f, P_pred = F_{t+1} P_f F_{t+1}^T + Q_{t+1}
        m_pred = torch.einsum("ij,bj->bi", F_next, m_f)
        P_pred = torch.einsum("ij,bjk,lk->bil", F_next, P_f, F_next) + Q_next

        # G_t = P_f F_{t+1}^T P_pred^{-1}, shape (B, D, D)
        # Compute via Cholesky for stability:
        # solve P_pred X = (P_f F^T)^T  ⇒  X^T = G  (P_pred is symmetric so the transpose is harmless).
        L_pred = torch.linalg.cholesky(P_pred)
        Pf_FT = torch.einsum("bij,kj->bik", P_f, F_next)  # P_f F_{t+1}^T
        G = torch.cholesky_solve(Pf_FT.transpose(-1, -2), L_pred).transpose(-1, -2)

        m_s = m_f + torch.einsum("bij,bj->bi", G, m_s_next - m_pred)
        cov_diff = P_s_next - P_pred
        P_s = P_f + torch.einsum("bij,bjk,blk->bil", G, cov_diff, G)

        # Cov(x_t, x_{t+1}) = G_t P_s_{t+1} (centred cross-time covariance)
        pair = torch.einsum("bij,bjk->bik", G, P_s_next)

        smoothed_means_rev.append(m_s)
        smoothed_covs_rev.append(P_s)
        pairwise_covs_rev.append(pair)

        m_s_next, P_s_next = m_s, P_s

    smoothed_means = torch.stack(list(reversed(smoothed_means_rev)), dim=1)
    smoothed_covs = torch.stack(list(reversed(smoothed_covs_rev)), dim=1)
    pairwise_covs = (
        torch.stack(list(reversed(pairwise_covs_rev)), dim=1)
        if pairwise_covs_rev
        else (torch.empty(B, 0, D, D, dtype=filtered_means.dtype, device=filtered_means.device))
    )
    return smoothed_means, smoothed_covs, pairwise_covs


def filter_state(
    y: Tensor,
    F: Tensor,
    Q: Tensor,
    H: Tensor,
    R: Tensor,
    m0: Tensor,
    P0: Tensor,
) -> GaussianState:
    """Convenience wrapper returning the filtered trajectory as a :class:`GaussianState`.

    The state's ``mean`` has shape ``(B, T, D)`` and ``covariance`` has
    shape ``(B, T, D, D)``.
    """
    means, covs, _ = kalman_filter(y, F, Q, H, R, m0, P0, return_log_marginal=False)
    return GaussianState(mean=means, covariance=covs)
