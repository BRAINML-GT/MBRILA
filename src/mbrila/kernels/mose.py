"""Multi-Output Squared-Exponential (MOSE) kernel.

The default kernel used by every Markovian-GP method in mbrila
(DLAG / mDLAG / ADM). MOSE describes the cross-time, cross-region
covariance of a single across-region latent factor::

    k(t, t', r, r') = exp(-½ σ (t - t' + δ_{r'}(t') - δ_r(t))²)

where ``σ > 0`` controls the timescale (larger σ → faster decay) and
``δ_r(t)`` is the per-region delay at time ``t``. With
``num_regions = 1`` and zero delay, MOSE collapses to the plain scalar
squared-exponential kernel used by within-region latents (the ``SOSE``
case in ADM).

The kernel is constructed at the *lag pairs* ``τ_{ij} = i - j`` for
``i, j ∈ {0, …, lag}`` so the dynamics layer can lift it into a lag-``P``
linear state-space model via :func:`mbrila.dynamics.kernel_to_sde.kernel_to_lds`.

Parameterisation
----------------
``σ`` is accepted as either a scalar (one shared timescale across all
regions) or a per-region vector of shape ``(num_regions,)``. Per-time-step
``σ`` is intentionally not supported — it would prevent the kernel from
being lifted into a lag-``P`` LDS, which is the construction every
SSM-GP engine in mbrila relies on.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mbrila.kernels.base import BaseKernel


class MOSEKernel(BaseKernel):
    """Multi-output SE (= RBF) kernel with optional per-region time delays.

    Subclasses :class:`BaseKernel` so it participates in the unified
    kernel hierarchy. Note that ``is_markovian`` stays ``False``: RBF has
    no exact finite-dimensional SDE form. The Kalman engine consumes
    MOSE through the AR(``P``) approximation bridge
    (:func:`mbrila.dynamics.kernel_to_sde.lagged_cov_grid` + ``kernel_to_lds``).

    Parameters
    ----------
    num_regions:
        Number of regions whose delays this kernel block can multiplex.
        Consumed by the legacy :meth:`lagged_cov` method (which
        validates the delay shape against ``num_regions``). The SSM
        bridge consumes :meth:`cov` instead and is independent of this
        value. ``num_regions = 1`` recovers the plain scalar SE kernel.
    init_sigma:
        Initial value for ``σ`` (the inverse-squared timescale). Stored
        internally as ``log σ`` so optimisation is unconstrained.
    """

    is_markovian = False

    def __init__(self, num_regions: int = 1, *, init_sigma: float = 0.05) -> None:
        super().__init__()
        if num_regions < 1:
            raise ValueError(f"num_regions must be >= 1; got {num_regions}")
        if init_sigma <= 0:
            raise ValueError(f"init_sigma must be positive; got {init_sigma}")
        self.num_regions = num_regions
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(init_sigma, dtype=torch.float64)))

    @property
    def sigma(self) -> Tensor:
        return torch.exp(self.log_sigma)

    def spectral_density(self, omega: Tensor) -> Tensor:
        """Kernel-only PSD ``S(ω) = sqrt(2π/σ) · exp(-ω²/(2σ))``.

        Returns the spectral density of the kernel itself, **without** any
        observation-noise floor. The dynamics layer (``ExactGPLatent``)
        applies the ``(1 - ε) · S(ω) + ε`` blend with its own ``ε`` since
        the noise floor is a dynamics-level decision, not a kernel property.
        """
        sigma = self.sigma
        omega_c = omega.to(sigma.dtype)
        return torch.sqrt(2.0 * torch.pi / sigma) * torch.exp(-0.5 * omega_c.square() / sigma)

    def cov(self, tau: Tensor) -> Tensor:
        """Pointwise stationary covariance ``k(tau) = exp(-σ/2 · tau²)``.

        This is the scalar primitive the generic Markovian-GP→LDS bridge
        (:func:`mbrila.dynamics.kernel_to_sde.lagged_cov_grid`) consumes. The
        multi-region structure and per-region delays are *not* the kernel's
        concern: the dynamics layer pre-shifts ``tau`` per ``(r1, r2)`` and
        broadcasts the result. So this method just evaluates the scalar
        kernel pointwise on any real-valued ``tau`` and returns a tensor of
        the same shape.

        Parameters
        ----------
        tau:
            Real lag tensor of arbitrary shape.

        Returns
        -------
        Tensor of the same shape as ``tau`` with values
        ``exp(-σ/2 · tau²)``.
        """
        return torch.exp(-0.5 * self.sigma * tau.to(self.sigma.dtype).square())

    def lagged_cov(self, tau: Tensor, delays: Tensor | None = None) -> Tensor:
        """Evaluate the kernel on a lag-pair grid, with optional time-varying delays.

        Parameters
        ----------
        tau:
            Pairwise lag tensor of shape ``(lag+1, lag+1)`` whose
            ``(i, j)`` entry is ``i - j``. Real-valued.
        delays:
            One of:

            - ``None`` — time-invariant kernel, equivalent to zero delays
              everywhere;
            - shape ``(num_regions,)`` — time-invariant per-region delays
              (DLAG / mDLAG / GPFA-via-Markov-lift);
            - shape ``(T, num_regions)`` — time-varying per-region delays
              (ADM).

        Returns
        -------
        Tensor with shape

        - ``(lag+1, lag+1, num_regions, num_regions)`` if ``delays`` is
          ``None`` **or** a 1-D ``(num_regions,)`` tensor (both
          time-invariant);
        - ``(T, lag+1, lag+1, num_regions, num_regions)`` if ``delays``
          is a 2-D ``(T, num_regions)`` tensor (time-varying).

        Following the convention ``δ_r → 0`` on the reference region,
        callers usually pin region 0's delay column to zero before
        passing it in (see :class:`mbrila.delays.time_varying.TimeVaryingDelay`).
        """
        if tau.ndim != 2 or tau.shape[0] != tau.shape[1]:
            raise ValueError(f"tau must be a square matrix (lag+1, lag+1); got {tuple(tau.shape)}")
        sigma = self.sigma
        R = self.num_regions

        if delays is None:
            # Time-invariant: K[i, j, r1, r2] = exp(-σ/2 * tau[i, j]^2) — same for every (r1, r2).
            tau_sq = tau.square().to(sigma.dtype)
            base = torch.exp(-0.5 * sigma * tau_sq)  # (lag+1, lag+1)
            # Broadcast to (lag+1, lag+1, R, R) via outer product with a constant 1.
            ones_RR = base.new_ones(R, R)
            return base.unsqueeze(-1).unsqueeze(-1) * ones_RR

        if delays.ndim == 1:
            if delays.shape[0] != R:
                raise ValueError(f"1-D delays must have shape ({R},); got {tuple(delays.shape)}")
            # Static per-region delay: δt[i, j, r1, r2] = tau[i, j] + δ_{r2} - δ_{r1}.
            # Broadcast: tau (lag+1, lag+1) → (lag+1, lag+1, 1, 1)
            #            δ_{r1} (R,) → (1, 1, R, 1)
            #            δ_{r2} (R,) → (1, 1, 1, R)
            tau_b = tau.unsqueeze(-1).unsqueeze(-1).to(sigma.dtype)
            d_i = delays.view(1, 1, R, 1).to(sigma.dtype)
            d_j = delays.view(1, 1, 1, R).to(sigma.dtype)
            delta = tau_b + d_j - d_i
            return torch.exp(-0.5 * sigma * delta.square())

        if delays.ndim != 2 or delays.shape[-1] != R:
            raise ValueError(f"2-D delays must have shape (T, {R}); got {tuple(delays.shape)}")
        T_dim = delays.shape[0]
        # δt[t, i, j, r1, r2] = tau[i, j] + δ_{r2}(t) - δ_{r1}(t)
        # Broadcast: tau (lag+1, lag+1) → (1, lag+1, lag+1, 1, 1)
        #            δ_{r1} (T, R) → (T, 1, 1, R, 1)
        #            δ_{r2} (T, R) → (T, 1, 1, 1, R)
        tau_b = tau.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(sigma.dtype)
        d_i = delays.view(T_dim, 1, 1, R, 1).to(sigma.dtype)
        d_j = delays.view(T_dim, 1, 1, 1, R).to(sigma.dtype)
        delta = tau_b + d_j - d_i
        return torch.exp(-0.5 * sigma * delta.square())


def rbf_kernel_with_eps(
    delta_t: Tensor,
    log_gamma: Tensor,
    eps: Tensor,
    *,
    diag_mask: Tensor | None = None,
) -> Tensor:
    """Evaluate ``K = (1 - ε) · exp(-γ/2 · Δt²) + ε · I`` for the RBF kernel.

    This is the per-latent covariance used by the exact-GP DLAG/mDLAG
    construction. The white-noise floor ``ε`` is applied only where the
    optional ``diag_mask`` is non-zero (Kronecker delta on the latent's
    structural identity — e.g. region & time both equal). ``ε`` is
    interpreted as ``DLAG.eps`` rather than per-element additive jitter:
    on the structural diagonal the kernel value is exactly ``1`` (not
    ``1 - ε + ε``) because the construction is ``(1-ε)·k + ε·δ``.

    Parameters
    ----------
    delta_t:
        Lag tensor ``Δt`` of arbitrary shape. The kernel acts pointwise.
    log_gamma:
        ``log γ`` (unconstrained). Scalar or broadcastable to ``delta_t``.
    eps:
        White-noise floor ``ε ∈ [0, 1)``. Scalar or broadcastable.
    diag_mask:
        Boolean/0-1 tensor with the same shape as ``delta_t``. Where it
        is non-zero, ``ε`` is added on top of ``(1 - ε) · k(0) = 1 - ε``,
        recovering kernel value ``1``. If ``None`` the diagonal injection
        is skipped — caller must add ``ε`` themselves if needed.

    Returns
    -------
    Tensor with the same shape as ``delta_t``.
    """
    gamma = torch.exp(log_gamma)
    temp = torch.exp(-0.5 * gamma * delta_t.square())
    base = (1.0 - eps) * temp
    if diag_mask is None:
        return base
    return base + eps * diag_mask


def rbf_grad_log_gamma(
    delta_t: Tensor,
    log_gamma: Tensor,
    eps: Tensor,
) -> Tensor:
    """Return ``∂K / ∂(log γ)`` for the RBF kernel above.

    Pure analytical form, used by the DLAG / mDLAG M-step so we never
    have to autograd through a ``(M·T, M·T)`` Cholesky factorisation:

        ∂K / ∂(log γ) = -½ · γ · Δt² · (1 - ε) · exp(-γ/2 · Δt²)
                     = -½ · γ · Δt² · base

    Same shape as ``delta_t``. The white-noise floor contributes zero
    because it is constant in ``γ``.
    """
    gamma = torch.exp(log_gamma)
    temp = torch.exp(-0.5 * gamma * delta_t.square())
    return -0.5 * gamma * delta_t.square() * (1.0 - eps) * temp


def rbf_grad_delta_t(
    delta_t: Tensor,
    log_gamma: Tensor,
    eps: Tensor,
) -> Tensor:
    """Return ``∂K / ∂(Δt)`` for the RBF kernel above.

    Used to chain into delay parameters: ``∂K / ∂δ_r`` for region ``r``
    is the row/column slice of this pointwise derivative against the
    sign of ``∂(Δt) / ∂δ_r`` (``-1`` when the perturbed region is the
    "from" side, ``+1`` when it is the "to" side; zero on the
    same-region diagonal where ``Δt`` does not depend on ``δ_r``).

        ∂K / ∂(Δt) = -γ · Δt · (1 - ε) · exp(-γ/2 · Δt²)

    Same shape as ``delta_t``.
    """
    gamma = torch.exp(log_gamma)
    temp = torch.exp(-0.5 * gamma * delta_t.square())
    return -gamma * delta_t * (1.0 - eps) * temp


# ---------------------------------------------------------------------
# Frequency-domain RBF: spectral density + analytical gradient.
# ---------------------------------------------------------------------
#
# The RBF kernel ``(1 - ε) · exp(-γτ²/2) + ε · δ_τ`` has analytical
# spectral density
#
#     S(ω) = (1 - ε) · sqrt(2π / γ) · exp(-ω² / (2γ))  +  ε
#
# where ``ω = 2π · f`` is angular frequency. This is the same form as
# ``make_S_mdlag.m`` (case 'rbf') and is the diagonal element of the
# circulant-approximation eigendecomposition of ``K_big``: at large
# ``T`` the unitary FFT diagonalises a stationary RBF covariance to
# ``diag(S(ω))``. The ``ε`` floor contributes a flat additive white-
# noise term in frequency.


def rbf_psd(
    omega: Tensor,
    log_gamma: Tensor,
    eps: Tensor,
) -> Tensor:
    """Analytical PSD of the RBF kernel with noise floor ``ε``.

    Computes

        S(ω) = (1 - ε) · sqrt(2π / γ) · exp(-ω² / (2γ)) + ε

    with broadcast semantics matching :func:`rbf_kernel_with_eps`. The
    result is real and positive (under valid inputs ``γ > 0``,
    ``ε ∈ [0, 1)``).

    Parameters
    ----------
    omega:
        Angular frequency tensor ``ω = 2π · f``. Any shape.
    log_gamma:
        ``log γ`` (unconstrained). Scalar or broadcastable to ``omega``.
    eps:
        White-noise floor ``ε``. Scalar or broadcastable.

    Returns
    -------
    Tensor of the same broadcast shape as ``omega``.
    """
    gamma = torch.exp(log_gamma)
    sqexp = torch.exp(-0.5 * omega.square() / gamma)
    return (1.0 - eps) * torch.sqrt(2.0 * torch.pi / gamma) * sqexp + eps


def rbf_psd_grad_log_gamma(
    omega: Tensor,
    log_gamma: Tensor,
    eps: Tensor,
) -> Tensor:
    """Return ``∂S(ω) / ∂(log γ)`` for the RBF PSD above.

    Derivation:

        ∂S/∂γ = (1 - ε) · sqrt(π/2) · γ^(-5/2) · (ω² - γ) · exp(-ω²/(2γ))

    (matches ``grad_GPparams_rbf_freq.m`` line 45-47); chain rule
    converts to ``∂S/∂(log γ) = γ · ∂S/∂γ``:

        ∂S/∂(log γ) = (1 - ε) · sqrt(π/2) · γ^(-3/2) · (ω² - γ) · exp(-ω²/(2γ))

    Same shape as ``omega``. The ``ε`` term contributes zero since it
    is constant in ``γ``.
    """
    gamma = torch.exp(log_gamma)
    sqexp = torch.exp(-0.5 * omega.square() / gamma)
    prefactor = (1.0 - eps) * torch.sqrt(torch.tensor(torch.pi / 2.0, dtype=gamma.dtype, device=gamma.device))
    return prefactor * gamma.pow(-1.5) * (omega.square() - gamma) * sqexp
