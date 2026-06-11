"""Variational mean-field EM engine for mDLAG (time-domain).

The engine pairs an :class:`mbrila.dynamics.exact_gp.ExactGPLatent`
prior with an :class:`mbrila.observations.ard.ARDObservation` emission
and runs the fast-mDLAG VEM algorithm
(``em_mdlag.m`` in ``fast-mDLAG/mDLAG/core_mdlag``).

E-step
------
Given the current variational posteriors ``q(C), q(α), q(φ), q(d)``,
the optimal Gaussian ``q(X) = N(μ_X, Σ_X)`` over the stacked latent
``X ∈ ℝ^{M·T}`` (with ``M = R · k`` slots per time bin) has precision

::

    Λ = K_big⁻¹ + I_T ⊗ blkdiag(⟨Cᵀ diag(φ) C⟩_r)
    ⟨Cᵀ diag(φ) C⟩_r = Σ_i ⟨φ_i⟩ · ⟨C_r[i] C_r[i]ᵀ⟩
                     = Σ_i φ_mean_i · (Σ_C[r, i] + C_mean[r, i] · C_mean[r, i]ᵀ)

Note the **second moment** of ``C_r`` (the bracketed term) — not
``C_mean^T C_mean`` — distinguishes the mDLAG E-step from DLAG's
point-estimate one. The mean follows the Woodbury form

::

    μ_X_b = K_big · (I - blkCPhiC · Σ_X) · term1_b
    term1_b = blkdiag(C_mean_rᵀ · diag(φ_mean_r)) · (y_b - d_mean).vec()

so the per-trial mean costs one ``(M·T)`` matmul on top of the shared
``(M·T)³`` Cholesky used for both ``Λ`` and ``K_big``.

M-step
------
Five blocks, in the order used by ``em_mdlag.m``:

1. **GP** — LBFGS on ``log γ_a, log γ_w, β`` minimising the negative
   Q-function ``½ [B · log|K_big| + tr(K_big⁻¹ S)]`` where
   ``S = Σ_b μ_X_b μ_X_bᵀ + B · Σ_X`` is the GP-prior sufficient stat.
   Identical to DLAG's GP M-step.
2. ``d``, ``C``, ``α``, ``φ`` — closed-form via the matching methods
   on :class:`ARDObservation` (see ``observations/ard.py``).

Sufficient statistics for the emission updates are aggregated from
``μ_X`` and ``Σ_X`` once per iteration; the per-region ``XX``, ``XY``,
``sum_x_per_region`` are passed through to the four update methods.

ELBO
----
Computed at the *start* of each iteration (after a fresh E-step,
before the M-step), so the value reflects the model state at the end
of the previous iteration. The decomposition is

::

    ELBO = elbo_emission(NT) + 0.5 · B · M · T
                              + lb_gp + 0.5 · B · log|Σ_X|
    lb_gp = -0.5 · B · log|K_big| - 0.5 · tr(K_big⁻¹ S)

with ``elbo_emission`` supplied by :class:`ARDObservation`. The two
GP-side ``logdet`` terms come from the Gaussian-entropy / cross-entropy
of ``q(X) || p(X)``.

Trial-batched
-------------
No Python loops over trials. The E-step batches trial means via a
single ``(B, M·T)`` matmul. Sufficient-stat aggregation is a sum over
``(0, 1)`` axes (batch, time) inside einsums. Loops over regions are
allowed (see ``check_no_trial_loops.py``).

Float64
-------
The ``(M·T)`` Cholesky is the dominant numerics-sensitive step; per
``mbrila`` policy we run float64 by default.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mbrila.core.inference_engine import FitResult, InferenceEngine, Posterior
from mbrila.inference.ard_helpers import (
    aggregate_emission_stats,
    compute_CPhi,
    compute_CPhiC_block,
    run_emission_m_step,
    setup_ard_posteriors,
)

if TYPE_CHECKING:
    from mbrila.core.base_model import BaseModel
    from mbrila.core.data import MultiRegionData


DEFAULT_JITTER: float = 1e-10


class VEMARDEngine(InferenceEngine):
    """Time-domain variational EM with ARD emission (mDLAG)."""

    name = "vem_ard"
    required_capabilities = frozenset({"cov_full"})

    def __init__(
        self,
        *,
        learn_gp: bool = True,
        learn_emission: bool = True,
        max_lbfgs_iter: int = 15,
        lbfgs_history: int = 10,
        jitter: float = DEFAULT_JITTER,
        log_every: int = 0,
        alpha_prune_ratio: float = 10.0,
    ) -> None:
        """
        Parameters
        ----------
        learn_gp, learn_emission:
            Toggle the two halves of the M-step. ``learn_gp=False``
            freezes the GP hyperparameters and delay; ``learn_emission=False``
            freezes ``q(C, α, φ, d)``. Both default to ``True``.
        max_lbfgs_iter:
            Inner LBFGS budget per GP M-step (matches DLAG's default).
        lbfgs_history:
            LBFGS history size.
        jitter:
            Diagonal jitter added before each Cholesky.
        log_every:
            Print ELBO every ``log_every`` iterations; ``0`` silences.
        alpha_prune_ratio:
            ARD-aware gating threshold for the GP M-step. A latent
            column ``k`` is considered "pruned" if its max-over-regions
            ``α_mean[r, k]`` exceeds
            ``alpha_prune_ratio · min_k α_mean.max(dim=0).values``.
            Pruned columns' ``(log γ_a, β_{:, k})`` gradients are
            detached during the GP M-step so the data-disconnected δ
            parameter cannot drift on numerical noise. Set to ``inf``
            to disable (legacy behaviour). Matches the spirit of
            fast-mDLAG's ``pruneX`` flag without requiring shape
            mutation.
        """
        if max_lbfgs_iter < 1:
            raise ValueError(f"max_lbfgs_iter must be >= 1; got {max_lbfgs_iter}")
        if lbfgs_history < 1:
            raise ValueError(f"lbfgs_history must be >= 1; got {lbfgs_history}")
        if jitter < 0:
            raise ValueError(f"jitter must be >= 0; got {jitter}")
        if log_every < 0:
            raise ValueError(f"log_every must be >= 0; got {log_every}")
        if alpha_prune_ratio <= 1.0:
            raise ValueError(f"alpha_prune_ratio must be > 1.0 (or inf to disable); got {alpha_prune_ratio}")
        self.learn_gp = learn_gp
        self.learn_emission = learn_emission
        self.max_lbfgs_iter = max_lbfgs_iter
        self.lbfgs_history = lbfgs_history
        self.jitter = jitter
        self.log_every = log_every
        self.alpha_prune_ratio = alpha_prune_ratio

    # ------------------------------------------------------------------
    # Component access (verified at call time so engine stays decoupled
    # from concrete model classes).
    # ------------------------------------------------------------------

    @staticmethod
    def _components(model: BaseModel) -> tuple[object, object]:
        from mbrila.dynamics.exact_gp import ExactGPLatent
        from mbrila.observations.ard import ARDObservation

        dyn = model.dynamics
        obs = model.observation
        if not isinstance(dyn, ExactGPLatent):
            raise TypeError(f"VEMARDEngine requires ExactGPLatent dynamics; got {type(dyn).__name__}")
        if not isinstance(obs, ARDObservation):
            raise TypeError(f"VEMARDEngine requires ARDObservation emission; got {type(obs).__name__}")
        return dyn, obs

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------

    def _e_step(self, model: BaseModel, data: MultiRegionData) -> dict[str, Tensor]:
        """Run the variational E-step at the current ``(q(C, α, φ, d), GP)``.

        Returns a dict of detached tensors:

        - ``x_hat``        : ``(B, T, M)`` posterior means.
        - ``P_full``       : ``(M·T, M·T)`` shared posterior covariance.
        - ``P_per_time``   : ``(T, M, M)`` per-bin covariance blocks.
        - ``S``            : ``(M·T, M·T)`` GP M-step sufficient stat.
        - ``logdet_K``     : ``log|K_big|`` (scalar).
        - ``logdet_Sigma`` : ``log|Σ_X|``  (scalar).
        - ``trace_KinvS``  : ``tr(K_big⁻¹ S)`` (scalar).
        """
        from mbrila.dynamics.exact_gp import ExactGPLatent
        from mbrila.observations.ard import ARDObservation

        dyn, obs = self._components(model)
        assert isinstance(dyn, ExactGPLatent)
        assert isinstance(obs, ARDObservation)

        B, T, _ = data.y.shape
        M = dyn.state_dim_per_time
        MT = M * T
        R = obs.n_regions
        k = obs.n_obs_per_region
        if R * k != M:
            raise ValueError(
                f"ARDObservation has R·k={R * k} but dynamics state_dim={M}; "
                "VEMARDEngine assumes a flat (R, k) emission with no within "
                "latents (set MDLAG's n_within=0)."
            )

        with torch.no_grad():
            K_big = dyn.cov_full(T)  # (MT, MT)
            d_mean = obs.d_mean  # (n_y,)

            dtype = K_big.dtype
            device = K_big.device
            eye_MT = torch.eye(MT, dtype=dtype, device=device)
            eye_T = torch.eye(T, dtype=dtype, device=device)

            # Engine-agnostic variational expected moments.
            CPhi = compute_CPhi(obs)  # (M, n_y)
            CPhiC_block = compute_CPhiC_block(obs)  # (M, M)
            # I_T ⊗ CPhiC_block — dense-engine specific (the Kalman
            # engine consumes CPhiC_block per-time instead of Kron-ing).
            blk_CPhiC = torch.kron(eye_T, CPhiC_block)  # (MT, MT)

            # Cholesky of K_big and Λ.
            L_K = torch.linalg.cholesky(K_big + self.jitter * eye_MT)
            K_inv = torch.cholesky_solve(eye_MT, L_K)
            logdet_K = 2.0 * torch.diagonal(L_K).log().sum()

            Lambda = K_inv + blk_CPhiC
            L_Lambda = torch.linalg.cholesky(Lambda + self.jitter * eye_MT)
            P_full = torch.cholesky_solve(eye_MT, L_Lambda)
            logdet_Lambda = 2.0 * torch.diagonal(L_Lambda).log().sum()
            # log|Σ_X| = -log|Λ|
            logdet_Sigma = -logdet_Lambda

            # term1_b = blkdiag(CPhi) · (y_b - d_mean).vec()
            # Per-time form: CPhi (M, n_y) → einsum gives (B, T, M).
            dif = data.y - d_mean  # (B, T, n_y)
            term1_per_t = torch.einsum("ij,btj->bti", CPhi, dif)
            term1 = term1_per_t.reshape(B, MT)

            # x_hat_b = K_big · (I - blk_CPhiC · P_full) · term1_b
            blkCPhiC_P = blk_CPhiC @ P_full
            K_big_proj = K_big @ (eye_MT - blkCPhiC_P)
            x_hat = term1 @ K_big_proj.transpose(0, 1)  # (B, MT)
            x_hat_per_t = x_hat.reshape(B, T, M)

            # Per-time blocks of P_full.
            P_blocks = P_full.reshape(T, M, T, M).diagonal(dim1=0, dim2=2)  # (M, M, T)
            P_per_time = P_blocks.permute(2, 0, 1).contiguous()

            # GP M-step sufficient stat S = Σ_b x_hat_b x_hat_bᵀ + B · P_full.
            S = x_hat.transpose(0, 1) @ x_hat + B * P_full
            # tr(K_big⁻¹ S) — used both in lb_gp and the GP M-step closure.
            K_inv_S = torch.cholesky_solve(S, L_K)
            trace_KinvS = torch.diagonal(K_inv_S).sum()

        return {
            "x_hat": x_hat_per_t,
            "P_full": P_full,
            "P_per_time": P_per_time,
            "S": S,
            "logdet_K": logdet_K.detach(),
            "logdet_Sigma": logdet_Sigma.detach(),
            "trace_KinvS": trace_KinvS.detach(),
            "B": torch.tensor(B, dtype=dtype, device=device),
        }

    # ------------------------------------------------------------------
    # Sufficient-statistic aggregation for the emission M-step
    # ------------------------------------------------------------------
    # Delegates to :func:`mbrila.inference.ard_helpers.aggregate_emission_stats`
    # so the hybrid VBEM-Kalman engine reuses the same code. Kept here
    # as a thin staticmethod wrapper for direct callers (notebooks, tests).

    @staticmethod
    def _aggregate_emission_stats(
        data: MultiRegionData,
        x_hat: Tensor,
        P_per_time: Tensor,
        y_dims: tuple[int, ...],
        k: int,
    ) -> dict[str, Tensor]:
        return aggregate_emission_stats(data=data, x_hat=x_hat, P_per_time=P_per_time, y_dims=y_dims, k=k)

    # ------------------------------------------------------------------
    # GP M-step (identical to DLAG's _update_gp)
    # ------------------------------------------------------------------

    def _m_step_gp(self, model: BaseModel, data: MultiRegionData, posterior: dict[str, Tensor]) -> None:
        from mbrila.dynamics.exact_gp import ExactGPLatent
        from mbrila.observations.ard import ARDObservation

        dyn = model.dynamics
        obs = model.observation
        assert isinstance(dyn, ExactGPLatent)
        is_mdlag_emission = isinstance(obs, ARDObservation)

        S = posterior["S"]
        B = int(data.y.shape[0])
        T = int(data.y.shape[1])
        M = dyn.state_dim_per_time
        MT = M * T

        gp_params = [p for p in dyn.parameters() if p.requires_grad]
        if not gp_params:
            return

        # ARD-aware pruning gate. Only meaningful for mDLAG (ARDObservation);
        # DLAG paths via ExactEMEngine don't reach this method.
        if is_mdlag_emission:
            assert isinstance(obs, ARDObservation)
            with torch.no_grad():
                max_alpha = obs.alpha_mean.max(dim=0).values  # (K_a,)
                min_alpha = max_alpha.min().clamp(min=1e-12)
                prune_mask_K = max_alpha > (self.alpha_prune_ratio * min_alpha)
        else:
            prune_mask_K = torch.zeros(dyn.n_across, dtype=torch.bool, device=S.device)

        optimiser = torch.optim.LBFGS(
            gp_params,
            max_iter=self.max_lbfgs_iter,
            history_size=self.lbfgs_history,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )
        eye_MT = torch.eye(MT, dtype=S.dtype, device=S.device)

        def closure() -> Tensor:
            optimiser.zero_grad()
            K_big = dyn.cov_full(T)
            L_K = torch.linalg.cholesky(K_big + self.jitter * eye_MT)
            logdet_K = 2.0 * torch.diagonal(L_K).log().sum()
            K_inv_S = torch.cholesky_solve(S, L_K)
            trace = torch.diagonal(K_inv_S).sum()
            neg_Q: Tensor = 0.5 * (B * logdet_K + trace)
            neg_Q.backward()  # type: ignore[no-untyped-call]
            # Zero gradients for pruned latent columns so LBFGS keeps
            # their (kernel params, δ_{:, k}) fixed. Each pruned across
            # column ``k`` owns its own kernel instance (post-Stage-2
            # decoupling) — iterate over its parameters and clear grad.
            if prune_mask_K.any():
                pruned_idx = prune_mask_K.nonzero(as_tuple=False).flatten().tolist()
                for k in pruned_idx:
                    for p in dyn.kernel_across[k].parameters():
                        if p.grad is not None:
                            p.grad.zero_()
                if dyn.delay.beta.grad is not None:
                    # delay.beta shape: (n_regions - 1, n_across)
                    dyn.delay.beta.grad[:, prune_mask_K] = 0.0
            return neg_Q

        optimiser.step(closure)  # type: ignore[no-untyped-call]

    # ------------------------------------------------------------------
    # Emission M-step — dispatches to ARDObservation's four blocks
    # ------------------------------------------------------------------

    def _m_step_emission(
        self,
        model: BaseModel,
        data: MultiRegionData,
        posterior: dict[str, Tensor],
    ) -> None:
        from mbrila.observations.ard import ARDObservation

        _, obs = self._components(model)
        assert isinstance(obs, ARDObservation)

        B, T = int(data.y.shape[0]), int(data.y.shape[1])
        NT = B * T
        stats = aggregate_emission_stats(
            data=data,
            x_hat=posterior["x_hat"],
            P_per_time=posterior["P_per_time"],
            y_dims=obs.y_dims,
            k=obs.n_obs_per_region,
        )
        # Canonical d → C → α → φ order — see ard_helpers.run_emission_m_step.
        run_emission_m_step(obs, stats, NT=NT)

    # ------------------------------------------------------------------
    # ELBO
    # ------------------------------------------------------------------

    def _compute_elbo(
        self,
        model: BaseModel,
        data: MultiRegionData,
        posterior: dict[str, Tensor],
    ) -> Tensor:
        """Full mDLAG ELBO.

        Decomposition: ARDObservation supplies the emission-side terms
        (data likelihood + KLs over C, α, φ, d); this engine adds the
        latent KL.
        """
        from mbrila.dynamics.exact_gp import ExactGPLatent
        from mbrila.observations.ard import ARDObservation

        dyn, obs = self._components(model)
        assert isinstance(dyn, ExactGPLatent)
        assert isinstance(obs, ARDObservation)

        B = int(data.y.shape[0])
        T = int(data.y.shape[1])
        NT = B * T
        M = dyn.state_dim_per_time

        emission = obs.elbo_emission(NT)
        # GP-side terms: ½·B·M·T + lb_gp + ½·B·log|Σ_X|
        #   where lb_gp = -½·B·log|K_big| - ½·tr(K_big⁻¹ S).
        gp_term = (
            0.5 * B * M * T
            - 0.5 * B * posterior["logdet_K"]
            - 0.5 * posterior["trace_KinvS"]
            + 0.5 * B * posterior["logdet_Sigma"]
        )
        return emission + gp_term

    # ------------------------------------------------------------------
    # Fit loop
    # ------------------------------------------------------------------

    def fit(
        self,
        model: BaseModel,
        data: MultiRegionData,
        *,
        max_iter: int,
        tol: float,
        **kwargs: object,
    ) -> FitResult:
        del kwargs
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1; got {max_iter}")

        from mbrila.observations.ard import ARDObservation

        _, obs = self._components(model)
        assert isinstance(obs, ARDObservation)

        # One-time per-fit setup — shared with the hybrid Kalman engine.
        setup_ard_posteriors(obs, data, learn_emission=self.learn_emission)

        score_trace: list[float] = []
        wall_start = time.perf_counter()
        prev_elbo = -math.inf
        converged = False

        # mDLAG ELBO uses the identity ``b_φ - b_φ⁰ = 0.5·⟨||residual||²⟩``
        # which only holds right after the φ-update. Measure ELBO at the
        # END of each iteration (after the M-step) using a fresh E-step
        # so logdet_K / logdet_Σ_X / trace(K⁻¹·S) reflect the new
        # parameters. Matches em_mdlag.m's per-iter LB cadence.

        for iteration in range(max_iter):
            posterior = self._e_step(model, data)
            if self.learn_gp:
                self._m_step_gp(model, data, posterior)
            if self.learn_emission:
                self._m_step_emission(model, data, posterior)
            # Fresh E-step under the post-M-step parameters.
            posterior_end = self._e_step(model, data)
            elbo_value = float(self._compute_elbo(model, data, posterior_end).item())
            score_trace.append(elbo_value)

            if self.log_every > 0 and (iteration + 1) % self.log_every == 0:
                print(f"[vem_ard] iter {iteration + 1}/{max_iter}  ELBO = {elbo_value:.3f}")

            if iteration > 0 and abs(elbo_value - prev_elbo) < tol * max(abs(elbo_value), 1.0):
                converged = True
                break
            prev_elbo = elbo_value

        wall = time.perf_counter() - wall_start
        return FitResult(
            score_trace=score_trace,
            converged=converged,
            n_iter=len(score_trace),
            wall_time_s=wall,
            reason="converged" if converged else "completed max_iter",
        )

    # ------------------------------------------------------------------
    # Inference / scoring
    # ------------------------------------------------------------------

    def infer(self, model: BaseModel, data: MultiRegionData) -> Posterior:
        info = self._e_step(model, data)
        x_hat = info["x_hat"]
        return Posterior(
            mean=x_hat,
            cov=info["P_per_time"].unsqueeze(0).expand(x_hat.shape[0], -1, -1, -1),
            cov_form="per_time_block",
            extras={"P_full": info["P_full"]},
        )

    def score(self, model: BaseModel, data: MultiRegionData) -> float:
        posterior = self._e_step(model, data)
        return float(self._compute_elbo(model, data, posterior).item())
