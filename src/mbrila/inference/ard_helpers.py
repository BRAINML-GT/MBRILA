"""Engine-agnostic helpers for ARD-observation inference.

The variational mean-field updates for ``q(C, α, φ, d)`` in
:class:`~mbrila.observations.ard.ARDObservation` and the
sufficient-statistic aggregation that drives them depend only on:

- the ARDObservation's current variational moments (``C_means``,
  ``C_moments``, ``phi_mean``, ``d_mean``);
- the latent posterior moments ``x_hat`` (mean, shape ``(B, T, M)``) and
  ``P_per_time`` (per-time-block cov, shape ``(T, M, M)``) produced by
  *some* E-step.

They do **not** depend on whether that E-step is a dense GP Cholesky
(time-domain :class:`VEMARDEngine`), a circulant frequency approximation
(:class:`VEMARDFreqEngine`), or a Kalman filter/smoother in a lifted SSM
(:class:`VEMKalmanARDEngine`, the mDLAG-SSM hybrid engine).

This module is the engine-agnostic surface. All three ARD-using
engines consume the same helpers, so the ARD machinery is implemented
once and the cross-engine "is the ARD logic the same?" question
reduces to "do they all call into ``ard_helpers``?". Yes — by
construction.

What stays engine-specific
--------------------------
Each engine still owns:

- the latent E-step (dense Cholesky vs FFT vs Kalman filter);
- the GP / kernel / delay M-step (LBFGS on K_big vs autograd on lifted
  LDS hyperparameters);
- the latent-side ELBO terms (``log|K_big|`` vs ``log|Σ_X|`` from the
  filter).

Only the ARD-observation pieces are shared.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from mbrila.core.data import MultiRegionData
    from mbrila.observations.ard import ARDObservation


# ---------------------------------------------------------------------------
# CPhi / CPhiC: variational expected emission moments
# ---------------------------------------------------------------------------


def compute_CPhi(obs: ARDObservation) -> Tensor:
    """``CPhi = ⟨C⟩ᵀ · diag(φ_mean)`` — the linear projection of data onto the latent.

    Shape ``(M, n_y)`` where ``M = R · k`` is the total latent slot count.
    Used by the latent E-step to map centred observations into the
    latent precision space.

    This is a thin wrapper that uses the current ARD moments — by
    contrast, a point-emission engine would just use ``C.T @ diag(R⁻¹)``.
    """
    C_mean = obs.block_diag_C()  # (n_y, M)
    phi_mean = obs.phi_mean  # (n_y,)
    return C_mean.transpose(0, 1) * phi_mean.unsqueeze(0)


def compute_CPhiC_block(obs: ARDObservation) -> Tensor:
    """Block-diagonal ``⟨Cᵀ diag(φ) C⟩`` of shape ``(M, M)``.

    Per-region: ``CPhiC[r] = Σ_i φ_mean_i · ⟨C_r[i] C_r[i]ᵀ⟩``. The
    bracketed second moment is
    ``⟨C_r[i] C_r[i]ᵀ⟩ = C_cov[r, i] + outer(C_mean[r, i])`` — the
    full variational second moment, not the point-estimate
    ``outer(C_mean)`` alone. Dropping the ``C_cov`` term collapses the
    variational uncertainty on ``C`` and can drive the latent
    posterior into a degenerate basin.

    Returns a dense block-diagonal ``(M, M)`` tensor. The caller decides
    how to use it — in the dense engine it gets Kron'd with ``I_T``; in
    the upcoming hybrid Kalman engine it lands directly inside the
    per-time observation precision.
    """
    phi_mean = obs.phi_mean
    blocks: list[Tensor] = []
    cum = 0
    for r, y_r in enumerate(obs.y_dims):
        phi_r = phi_mean[cum : cum + y_r]  # (y_r,)
        moment_r = obs.C_moments[r]  # (y_r, k, k)
        block_r = (phi_r.view(-1, 1, 1) * moment_r).sum(dim=0)
        # Symmetrise (numerical hygiene — the rank-1 weighted sum is
        # symmetric on paper but accumulates float64 imbalance).
        block_r = 0.5 * (block_r + block_r.transpose(-2, -1))
        blocks.append(block_r)
        cum += y_r
    out: Tensor = torch.block_diag(*blocks)  # type: ignore[no-untyped-call]
    return out


# ---------------------------------------------------------------------------
# Sufficient-statistic aggregation for the emission M-step
# ---------------------------------------------------------------------------


def aggregate_emission_stats(
    data: MultiRegionData,
    x_hat: Tensor,  # (B, T, M)
    P_per_time: Tensor,  # (T, M, M)
    y_dims: tuple[int, ...],
    k: int,
) -> dict[str, Tensor]:
    """Build the sufficient stats the four ``ARDObservation.update_*`` methods consume.

    Engine-agnostic: any latent E-step that produces a per-trial mean
    ``x_hat`` of shape ``(B, T, M)`` and a (per-trial-shared) per-time
    covariance ``P_per_time`` of shape ``(T, M, M)`` can feed this
    function.

    Returns
    -------
    Dict with keys ``sum_y``, ``sum_y2``, ``sum_x_per_region``,
    ``XX_per_region``, ``XY_per_region``. The last is a list of ``(k,
    y_r)`` tensors per region (region neuron counts vary).
    """
    B, T, M = x_hat.shape
    R = len(y_dims)
    if R * k != M:
        raise ValueError(f"R·k={R * k} must equal M={M}")

    sum_y = data.y.sum(dim=(0, 1))  # (n_y,)
    sum_y2 = (data.y * data.y).sum(dim=(0, 1))  # (n_y,)

    x_per_region = x_hat.view(B, T, R, k)  # (B, T, R, k)
    # Sum over (b, t): (R, k)
    sum_x_per_region = x_per_region.sum(dim=(0, 1))

    # XX per region:
    #   Σ_{b,t} ⟨x_{b,t,r} x_{b,t,r}ᵀ⟩
    # = Σ_{b,t} x_hat[b,t,r] x_hat[b,t,r]ᵀ + B · Σ_t cov_t[r, r]
    outer_sum = torch.einsum("btrk,btrl->rkl", x_per_region, x_per_region)
    P_reshaped = P_per_time.view(T, R, k, R, k)
    cov_diag_per_region = P_reshaped.diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2)
    # cov_diag_per_region: (T, R, k, k). Sum over T then scale by B.
    cov_sum = B * cov_diag_per_region.sum(dim=0)  # (R, k, k)
    XX_per_region = outer_sum + cov_sum

    # XY per region — list of (k, y_r) tensors (region neuron counts vary).
    XY_per_region: list[Tensor] = []
    cum = 0
    for r, y_r in enumerate(y_dims):
        y_r_data = data.y[:, :, cum : cum + y_r]  # (B, T, y_r)
        XY_r = torch.einsum("btk,bti->ki", x_per_region[:, :, r, :], y_r_data)
        XY_per_region.append(XY_r)
        cum += y_r

    return {
        "sum_y": sum_y,
        "sum_y2": sum_y2,
        "sum_x_per_region": sum_x_per_region,
        "XX_per_region": XX_per_region,
        "XY_per_region": XY_per_region,  # type: ignore[dict-item]
    }


# ---------------------------------------------------------------------------
# Once-per-fit ARD setup
# ---------------------------------------------------------------------------


def setup_ard_posteriors(
    obs: ARDObservation,
    data: MultiRegionData,
    *,
    learn_emission: bool,
) -> None:
    """One-time per-fit initialisation of the ARD posterior bookkeeping.

    Three things must happen once per ``fit()`` call before the first
    iteration:

    1. ``φ``'s shape parameter is ``a = a_prior + NT/2`` and stays fixed
       — :meth:`ARDObservation.set_phi_shape_from_NT` computes it once.
    2. Per-neuron variance floor seeded from sample variance (matches
       fast-mDLAG MATLAB :func:`em_mdlag.m`).
    3. If ``learn_emission`` is on, prime ``α_mean`` from the current
       ``⟨C C^T⟩`` so the ARD prune gate in the very first GP M-step
       has data-driven ``α`` to test against. With ``learn_emission`` off,
       ``α`` is frozen and we must not touch it.

    Engine-agnostic: shape arithmetic only depends on ``data``.
    """
    B, _, n_y = data.y.shape
    T = data.y.shape[1]
    NT = B * T
    obs.set_phi_shape_from_NT(NT)
    y_var = data.y.reshape(-1, n_y).var(dim=0, unbiased=False)
    obs.set_variance_floor(y_var)
    if learn_emission:
        obs.update_alpha()


# ---------------------------------------------------------------------------
# Emission M-step (canonical d → C → α → φ order)
# ---------------------------------------------------------------------------


def build_variational_kalman_inputs(
    obs: ARDObservation,
    H_select: Tensor,
    y: Tensor,
    *,
    jitter: float = 1e-10,
) -> dict[str, Tensor]:
    """Convert ARD variational moments to standard-Kalman inputs.

    Maps the variational E-step's "effective observation" — defined by
    the precision contribution ``H_selectᵀ · ⟨CᵀΦC⟩ · H_select`` and the
    info-vector ``H_selectᵀ · ⟨C⟩ᵀ · diag(φ) · (y - d)`` — to a synthetic
    standard-Kalman triple ``(y_pseudo, H_eff, R_eff)``. Running
    :func:`mbrila.inference.kalman.sequential.kalman_filter` (or the
    parallel-scan variant) on these reproduces the variational latent
    posterior **without** modifying the Kalman filter implementation.

    The trick
    ---------
    With ``M = R · k`` latent observable slots, let

        L_A · L_Aᵀ = CPhiC_block        (Cholesky in M-space)

    where ``CPhiC_block = compute_CPhiC_block(obs)``. Setting

        H_eff    = L_Aᵀ · H_select               (M, D)
        R_eff    = I_M                           (M, M)
        y_pseudo = L_A⁻¹ · CPhi · (y - d_mean)   (B, T, M)

    makes the standard-Kalman observation contributions per time bin

        Hᵀ · R⁻¹ · H  =  H_selectᵀ · L_A · L_Aᵀ · H_select
                       =  H_selectᵀ · CPhiC_block · H_select
        Hᵀ · R⁻¹ · y  =  H_selectᵀ · L_A · L_A⁻¹ · CPhi · (y - d)
                       =  H_selectᵀ · CPhi · (y - d)

    — i.e. exactly the variational precision and info-vector
    contributions to the latent posterior. The Kalman filter then runs
    unchanged; the variational structure is fully absorbed into the
    synthetic ``(y_pseudo, H_eff)``.

    Caveats
    -------
    - ``CPhiC_block`` is PSD by construction (sum of weighted second
      moments). For Cholesky we add ``jitter · I`` to keep it strictly PD
      in the rare zero-row case (e.g. a region whose ``φ_mean`` collapsed
      during an early iteration).
    - The standard-Kalman ``log p(y)`` returned alongside the filter
      reflects the *synthetic* pseudo-observations, not the real data.
      The downstream ELBO computation must therefore assemble its own
      data term — the filter is only used to produce the latent
      posterior ``q(X) = N(μ_X, Σ_X)``; that part is exact.

    Parameters
    ----------
    obs:
        Current :class:`ARDObservation` instance. Its variational moments
        (``C_means``, ``C_moments``, ``phi_mean``, ``d_mean``) are
        consumed read-only.
    H_select:
        ``(M, D)`` selector from the full lifted state to the per-region
        per-latent observable slots. Same convention as
        :class:`~mbrila.dynamics.markov_gp.BlockDiagonalDynamics.H_select`.
    y:
        Raw observations of shape ``(B, T, n_y)``.
    jitter:
        Diagonal jitter added to ``CPhiC_block`` before Cholesky.

    Returns
    -------
    Dict with keys:

    - ``H_eff``    : ``(M, D)``
    - ``R_eff``    : ``(M, M)``
    - ``y_pseudo`` : ``(B, T, M)``
    """
    if H_select.ndim != 2:
        raise ValueError(f"H_select must be 2-D; got shape {tuple(H_select.shape)}")
    if y.ndim != 3:
        raise ValueError(f"y must be (B, T, n_y); got shape {tuple(y.shape)}")
    M = H_select.shape[0]
    if M < 1:
        raise ValueError(f"H_select must have at least one row; got shape {tuple(H_select.shape)}")

    CPhiC_block = compute_CPhiC_block(obs)  # (M, M)
    CPhi = compute_CPhi(obs)  # (M, n_y)
    if CPhiC_block.shape != (M, M):
        raise ValueError(f"CPhiC_block shape {tuple(CPhiC_block.shape)} disagrees with H_select rows ({M})")

    dtype = CPhiC_block.dtype
    device = CPhiC_block.device
    eye_M = torch.eye(M, dtype=dtype, device=device)
    L_A = torch.linalg.cholesky(CPhiC_block + jitter * eye_M)

    H_eff = L_A.transpose(-1, -2) @ H_select  # (M, D)
    R_eff = eye_M  # standard Kalman expects observation noise; identity is the
    # natural choice once L_A absorbs the variational precision.

    y_centred = y - obs.d_mean  # (B, T, n_y)
    # CPhi·(y - d) per time bin, all batches: result (B, T, M).
    info_M = torch.einsum("mi,bti->btm", CPhi, y_centred)
    # y_pseudo = L_A⁻¹ · info_M. Solve L_A · y_pseudo = info_M.
    info_M_flat = info_M.reshape(-1, M)  # (B·T, M)
    y_pseudo_flat = torch.linalg.solve_triangular(L_A, info_M_flat.transpose(0, 1), upper=False).transpose(
        0, 1
    )
    y_pseudo = y_pseudo_flat.reshape(*info_M.shape)  # (B, T, M)

    return {"H_eff": H_eff, "R_eff": R_eff, "y_pseudo": y_pseudo}


def run_emission_m_step(
    obs: ARDObservation,
    stats: dict[str, Tensor],
    *,
    NT: int,
) -> None:
    """Update ``q(d, C, α, φ)`` in the canonical fast-mDLAG order.

    The order matches ``em_mdlag.m`` lines 432–510: ``d → C → α → φ``.
    Re-ordering changes the numerical trajectory (and in degenerate
    cases, the fixed point) — do not reshuffle without a recovery test
    in hand.

    ``stats`` is the dict returned by :func:`aggregate_emission_stats`.
    """
    obs.update_d(
        sum_y=stats["sum_y"],
        sum_x_per_region=stats["sum_x_per_region"],
        NT=NT,
    )
    obs.update_C(
        XX=stats["XX_per_region"],
        XY=stats["XY_per_region"],  # type: ignore[arg-type]
        sum_x_per_region=stats["sum_x_per_region"],
    )
    obs.update_alpha()
    obs.update_phi(
        NT=NT,
        sum_y=stats["sum_y"],
        sum_y2=stats["sum_y2"],
        XX=stats["XX_per_region"],
        XY=stats["XY_per_region"],  # type: ignore[arg-type]
        sum_x_per_region=stats["sum_x_per_region"],
    )
