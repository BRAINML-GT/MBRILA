"""Post-fit scale anchoring for per-block across-latent magnitudes.

GP-prior latent variable models are non-identifiable under the
rescaling ``(C, g) → (αC, g/α)`` — the predicted observation
``y = C·g`` is invariant, so any ``α > 0`` gives the same likelihood
on data. Adam-based optimisers can drift along this invariant
direction across many iterations, leaving the fitted posterior ``g``
at a non-canonical magnitude even when the fit on ``y`` is fine. The
visible symptom is "latent trace looks like wrong amplitude" while
``y`` reconstruction stays accurate.

This module exposes :func:`normalize_latent_scales`, a post-fit
calibration that pins each across block's posterior ``g_k`` to the
block's own GP prior marginal variance ``Var[g_k(t)] = K_k(0)``. The
rescaling is done in-place on the emission matrix, so the model can
be re-inferred immediately and the latent trace is at the
identifiability-canonical scale.

The helper is kernel-aware: it queries each block's
:meth:`mbrila.kernels.base.BaseKernel.cov` at ``τ = 0`` rather than
assuming ``K(0) = 1``. This matters for kernels that carry an
explicit amplitude parameter (e.g. ``K(τ) = σ²·exp(-τ²/(2ℓ²))`` or
``K(τ) = σ²/(1+(τ/ℓ)²)``) where ``K(0)`` varies with the optimiser
update. Structurally-unit kernels (MOSE, Matérn family, periodic)
yield ``K(0) = 1`` and the helper degrades to the historical
behaviour bit-identically.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from mbrila.core.base_model import BaseModel
    from mbrila.core.data import MultiRegionData


def normalize_latent_scales(model: BaseModel, data: MultiRegionData) -> dict[str, list[float]]:
    """Anchor per-block across-latent posterior magnitudes to ``K(0)``.

    For each across block ``k = 0, …, n_across-1``:

    1. Run the model's smoother to get ``g_k(t)`` (observable view
       at region 0, where the delay is zero).
    2. Query the block's kernel for ``K_k(0)`` — the prior marginal
       variance of ``g_k(t)``. For ``MOSE`` / Matérn this is
       structurally ``1``; for kernels with an explicit amplitude
       (``Cauchy``, amplitude-bearing ``RBF``, user-defined) it varies.
    3. Compute the empirical posterior rms ``r_k = √mean(g_k²)``.
    4. Rescale ``C[:, k] *= r_k / √K_k(0)`` on every region's emission.

    ``y = C·g`` is invariant in the moment (we don't touch ``g``); the
    next smoother call will yield ``g_k`` with rms ≈ ``√K_k(0)``
    matching the prior.

    Requirements
    ------------
    - ``model.inference``: :class:`mbrila.inference.kalman_em.KalmanEMEngine`
      (anything exposing ``_smoother_posterior(model, data) -> {"means": …}``).
    - ``model.dynamics``: :class:`mbrila.dynamics.markov_gp.BlockDiagonalDynamics`
      (per-block kernels accessible via ``dynamics.blocks[k].kernel``).
    - ``model.observation``: :class:`mbrila.observations.multi_region.MultiRegionLinearObservation`
      (exposes the per-region emission list ``Cs``). The ARD
      observation path (:class:`mbrila.observations.ard.ARDObservation`,
      used by mDLAG) is **not** supported — its variational ``α``
      already manages per-column scale and an anchor would
      double-correct.

    Parameters
    ----------
    model:
        Fitted model.
    data:
        Training data used to compute the smoother posterior.

    Returns
    -------
    dict with:
        - ``"alphas"``: list of length ``n_across`` — the rescaling
          factor applied to ``C[:, k]`` on every region.
        - ``"k0_values"``: list of length ``n_across`` — the per-block
          ``K_k(0)`` queried from the kernel.

    Raises
    ------
    TypeError:
        If ``model.observation`` is :class:`ARDObservation` or any
        other type lacking a ``Cs`` parameter list.
    """
    # Local imports to keep the public ``mbrila.init`` import light —
    # ``KalmanEMEngine`` pulls in the whole inference module.
    from mbrila.dynamics.markov_gp import BlockDiagonalDynamics, MarkovianGPLatent
    from mbrila.inference.kalman_em import KalmanEMEngine
    from mbrila.observations.ard import ARDObservation
    from mbrila.observations.multi_region import MultiRegionLinearObservation

    engine = model.inference
    if not isinstance(engine, KalmanEMEngine):
        raise TypeError(f"normalize_latent_scales requires a KalmanEMEngine; got {type(engine).__name__}")

    dyn = model.dynamics
    if not isinstance(dyn, BlockDiagonalDynamics):
        raise TypeError(f"normalize_latent_scales requires BlockDiagonalDynamics; got {type(dyn).__name__}")

    obs = model.observation
    if isinstance(obs, ARDObservation):
        raise TypeError(
            "normalize_latent_scales is not supported for ARDObservation models "
            "(mDLAG). ARD's variational α already manages per-column emission "
            "scale and an anchor would double-correct."
        )
    if not isinstance(obs, MultiRegionLinearObservation):
        raise TypeError(
            f"normalize_latent_scales requires MultiRegionLinearObservation; got {type(obs).__name__}"
        )

    n_across = model.latent_spec.n_across
    if n_across == 0:
        return {"alphas": [], "k0_values": []}

    with torch.no_grad():
        info = engine._smoother_posterior(model, data)
        s_means: Tensor = info["means"]  # (B, T, D_lifted)
        H_select: Tensor = dyn.H_select  # (n_obs_total, D_lifted)
        # ``observable[:, :, k]`` is the region-0 view of latent k
        # (across blocks have the region-0 delay anchored to zero).
        observable = torch.einsum("ij,btj->bti", H_select, s_means)

        # One scalar τ=0 tensor in the kernel's dtype/device — each
        # block's kernel may carry its own parameter dtype/device.
        sample = s_means
        zero_tau = sample.new_zeros(1)

        alphas: list[float] = []
        k0_values: list[float] = []
        for k in range(n_across):
            block = dyn.blocks[k]
            assert isinstance(block, MarkovianGPLatent)
            # ``K_k(0)`` — prior marginal variance for block k. For
            # MOSE / Matérn / periodic this is structurally 1; for
            # amplitude-bearing kernels (Cauchy, RBF with σ²) it
            # tracks the kernel's current σ.
            k0 = float(block.kernel.cov(zero_tau).item())
            expected_rms = math.sqrt(max(k0, 1e-12))
            k0_values.append(k0)

            g_k = observable[:, :, k]
            rms_k = float(g_k.pow(2).mean().sqrt().item())
            alpha_k = rms_k / expected_rms
            alphas.append(alpha_k)

            if alpha_k < 1e-8:
                # ``g_k`` is essentially dead — leave C alone to avoid
                # multiplying by ~0 (would erase the column).
                continue
            for r in range(obs.n_regions):
                obs.Cs[r][:, k] *= alpha_k

    return {"alphas": alphas, "k0_values": k0_values}
