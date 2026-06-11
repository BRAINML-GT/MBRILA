"""Exact-GP time-domain EM inference engine (DLAG path).

The engine pairs with :class:`mbrila.dynamics.exact_gp.ExactGPLatent` and
:class:`mbrila.observations.multi_region.MultiRegionLinearObservation`.
It expects every trial to share the same length ``T`` so the GP prior
``K_big`` of shape ``(M·T, M·T)`` can be built once per fit iteration and
reused across the batch.

E-step
------
Given the current parameters ``(K_big, C, d, diag(R))``, the joint
posterior over the stacked latent ``x ∈ ℝ^{M·T}`` is Gaussian with

    Λ = K_big⁻¹ + I_T ⊗ (Cᵀ R⁻¹ C)      # precision
    P = Λ⁻¹                              # posterior covariance (M·T, M·T)
    x̂_b = K_big · (I − I_T ⊗ (Cᵀ R⁻¹ C) · Λ⁻¹) · term1_b
    term1_b = I_T ⊗ (Cᵀ R⁻¹) · (y_b − d).vec()

The marginal log-likelihood is

    log p(y_b) = −½ [T log|R| + log|K_big| + log|Λ|
                       + n_y T log 2π
                       + Σ_t (y_t − d)ᵀ R⁻¹ (y_t − d)
                       − term1_bᵀ P term1_b]

summed across the batch.

M-step
------
Three groups of parameters are updated independently:

* ``(C, d, diag(R))``: closed-form Bayesian-regression update via
  :meth:`MultiRegionLinearObservation.update_from_smoothed`. The
  per-region per-time first / second moments are read from ``x̂`` and
  the diagonal blocks of ``P``.
* ``(log γ_a, log γ_w, β)``: a few iterations of LBFGS on the Q-function
  for the GP prior, ``Q(θ_GP) = −½ [B log|K_big| + tr(K_big⁻¹ S)]``
  where ``S = Σ_b (x̂_b x̂_bᵀ + P) = Σ_b x̂_b x̂_bᵀ + B · P``.
  Autograd flows through ``cov_full → Cholesky → logdet/cholesky_solve``;
  no manual analytical-gradient wiring is needed. At ``M·T ≲ 1000`` the
  Cholesky backward is cheap and a manual analytical implementation
  would only complicate testing without a real perf win.

Trial-batched: there are no Python loops over the trial axis anywhere
inside the engine. Loops over ``R`` (regions) and ``K_a`` (latent
counts) are structural and small; CI's static check exempts them by
variable name.

Float64
-------
Per :mod:`mbrila` policy the engine assumes ``float64``. Cholesky on
``(M·T)²`` matrices is sensitive to precision in the float32 regime —
the small jitter that keeps the recursion stable in float64 isn't
enough at float32, and there is no benefit at the typical ``T ≲ 200``
scale where the exact path is competitive with the Markov path.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mbrila.core.inference_engine import FitResult, InferenceEngine, Posterior

if TYPE_CHECKING:
    from mbrila.core.base_model import BaseModel
    from mbrila.core.data import MultiRegionData

# Tiny additive diagonal jitter for Cholesky of K_big and Λ. The GP
# prior already includes ``eps_across`` / ``eps_within`` on the structural
# diagonal so the additive jitter only guards against floating-point
# round-off in the ``M·T``-sized factorisation.
DEFAULT_JITTER: float = 1e-10


class ExactEMEngine(InferenceEngine):
    """Time-domain exact-GP EM engine for DLAG / mDLAG."""

    name = "em_exact"
    required_capabilities = frozenset({"cov_full"})

    def __init__(
        self,
        *,
        learn_obs: bool = True,
        learn_gp: bool = True,
        max_lbfgs_iter: int = 15,
        lbfgs_history: int = 10,
        jitter: float = DEFAULT_JITTER,
        log_every: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        learn_obs:
            Whether to run the closed-form observation M-step on
            ``(C, d, diag(R))`` each iteration.
        learn_gp:
            Whether to optimise the GP hyperparameters (timescales and
            delays) each iteration.
        max_lbfgs_iter:
            Maximum LBFGS iterations inside each GP M-step. The default
            of 15 mirrors DLAG's ``minFunc`` cadence and is plenty when
            warm-started by the previous iteration.
        lbfgs_history:
            LBFGS history size (number of past gradients kept for the
            approximate Hessian).
        jitter:
            Diagonal jitter added to ``K_big`` and ``Λ`` before each
            Cholesky factorisation.
        log_every:
            If positive, print the marginal LL every ``log_every`` EM
            iterations.
        """
        if max_lbfgs_iter < 1:
            raise ValueError(f"max_lbfgs_iter must be >= 1; got {max_lbfgs_iter}")
        if lbfgs_history < 1:
            raise ValueError(f"lbfgs_history must be >= 1; got {lbfgs_history}")
        if jitter < 0:
            raise ValueError(f"jitter must be >= 0; got {jitter}")
        if log_every < 0:
            raise ValueError(f"log_every must be >= 0; got {log_every}")
        self.learn_obs = learn_obs
        self.learn_gp = learn_gp
        self.max_lbfgs_iter = max_lbfgs_iter
        self.lbfgs_history = lbfgs_history
        self.jitter = jitter
        self.log_every = log_every

    # ------------------------------------------------------------------
    # E-step (no autograd; posterior held fixed during M-step)
    # ------------------------------------------------------------------

    def _e_step(self, model: BaseModel, data: MultiRegionData) -> dict[str, Tensor]:
        """Run the E-step at the current parameters.

        Returns a dict with detached tensors:

        - ``x_hat``: posterior mean ``(B, T, M)``.
        - ``P_full``: full posterior covariance ``(M·T, M·T)``.
        - ``P_per_time``: per-time-bin covariance blocks ``(T, M, M)``.
        - ``ll``: scalar marginal log-likelihood (sum over trials).
        - ``S``: GP M-step sufficient statistic ``Σ_b x̂_b x̂_bᵀ + B · P``,
          shape ``(M·T, M·T)``.
        """
        from mbrila.dynamics.exact_gp import ExactGPLatent
        from mbrila.observations.multi_region import MultiRegionLinearObservation

        dyn = model.dynamics
        assert isinstance(dyn, ExactGPLatent)
        obs = model.observation
        assert isinstance(obs, MultiRegionLinearObservation)

        B, T, n_y = data.y.shape
        M = dyn.state_dim_per_time
        MT = M * T

        with torch.no_grad():
            K_big = dyn.cov_full(T)  # (MT, MT)
            C_blk = obs.block_diag_C()  # (n_y, M)
            diag_R = obs.diag_R()  # (n_y,)
            d_off = obs.offset()  # (n_y,)

            dtype = K_big.dtype
            device = K_big.device
            eye_MT = torch.eye(MT, dtype=dtype, device=device)
            eye_T = torch.eye(T, dtype=dtype, device=device)

            inv_R = 1.0 / diag_R  # (n_y,)
            # C^T R^{-1}: shape (M, n_y)
            CRinv = C_blk.transpose(0, 1) * inv_R.unsqueeze(0)
            CRinvC = CRinv @ C_blk  # (M, M)
            blk_CRinvC = torch.kron(eye_T, CRinvC)  # (MT, MT), block-diag of T copies

            # Cholesky of K_big.
            L_K = torch.linalg.cholesky(K_big + self.jitter * eye_MT)
            K_inv = torch.cholesky_solve(eye_MT, L_K)
            logdet_K = 2.0 * torch.diagonal(L_K).log().sum()

            # Precision Λ = K_inv + blkdiag(CRinvC); Cholesky for log|Λ| + Λ⁻¹.
            Lambda = K_inv + blk_CRinvC
            L_Lambda = torch.linalg.cholesky(Lambda + self.jitter * eye_MT)
            P_full = torch.cholesky_solve(eye_MT, L_Lambda)  # full (MT, MT) posterior cov
            logdet_Lambda = 2.0 * torch.diagonal(L_Lambda).log().sum()

            # term1_b = blkdiag(CRinv) · (y_b - d).vec() per trial.
            dif = data.y - d_off  # (B, T, n_y)
            term1_per_t = torch.einsum("ij,btj->bti", CRinv, dif)  # (B, T, M)
            term1 = term1_per_t.reshape(B, MT)  # (B, MT)

            # Posterior means: x̂_b = K_big · (I − blkCRinvC · P_full) · term1_b
            blkCRinvC_P = blk_CRinvC @ P_full  # (MT, MT)
            K_big_proj = K_big @ (eye_MT - blkCRinvC_P)  # (MT, MT)
            x_hat = term1 @ K_big_proj.transpose(0, 1)  # (B, MT)

            x_hat_per_t = x_hat.reshape(B, T, M)

            # Per-time-bin posterior covariance: diagonal blocks of P_full.
            P_blocks = P_full.reshape(T, M, T, M).diagonal(dim1=0, dim2=2)  # (M, M, T)
            P_per_time = P_blocks.permute(2, 0, 1).contiguous()  # (T, M, M)

            # Marginal log-likelihood (sum over trials).
            quad_yRy = (dif.square() * inv_R).sum()
            quad_term1 = (term1 @ P_full * term1).sum()
            log_det_R = torch.log(diag_R).sum()
            ll = -0.5 * (
                B * T * log_det_R
                + B * logdet_K
                + B * logdet_Lambda
                + B * T * n_y * math.log(2.0 * math.pi)
                + quad_yRy
                - quad_term1
            )

            # GP M-step sufficient statistic: Σ_b x̂_b x̂_bᵀ + B · P_full.
            x_outer = x_hat.transpose(0, 1) @ x_hat  # (MT, MT)
            S = x_outer + B * P_full

        return {
            "x_hat": x_hat_per_t,
            "P_full": P_full,
            "P_per_time": P_per_time,
            "ll": ll.detach(),
            "S": S,
        }

    # ------------------------------------------------------------------
    # Closed-form observation M-step
    # ------------------------------------------------------------------

    def _update_observation(
        self,
        model: BaseModel,
        data: MultiRegionData,
        posterior: dict[str, Tensor],
    ) -> None:
        from mbrila.dynamics.exact_gp import ExactGPLatent
        from mbrila.observations.multi_region import MultiRegionLinearObservation

        dyn = model.dynamics
        obs = model.observation
        assert isinstance(dyn, ExactGPLatent)
        assert isinstance(obs, MultiRegionLinearObservation)

        x_hat = posterior["x_hat"]  # (B, T, M)
        P_per_time = posterior["P_per_time"]  # (T, M, M)

        B, T, _ = x_hat.shape
        R = obs.n_regions
        n_obs_per_region = obs.n_obs_per_region
        # DLAG layout requires uniform per-region observable count; the
        # DLAG model class enforces this at construction time.
        if R * n_obs_per_region != dyn.state_dim_per_time:
            raise ValueError(
                f"ExactEMEngine assumes uniform per-region observable count: "
                f"R · n_obs_per_region={R * n_obs_per_region} must equal M="
                f"{dyn.state_dim_per_time}"
            )

        x_per_region = x_hat.view(B, T, R, n_obs_per_region)
        # Per-region posterior cov block from P_per_time.
        #
        # Despite its docstring's "Pass full second moments" wording,
        # :meth:`MultiRegionLinearObservation.update_from_smoothed`
        # internally forms ``ExxT = cov_sum + Σ x x^T`` where the mean
        # outer product is recomputed from the passed-in means. The
        # third argument therefore plays the role of the **covariance**
        # contribution ``Σ Cov_q[x]`` only; double-counting μμᵀ here
        # would break EM monotonicity.
        P_reshaped = P_per_time.view(T, R, n_obs_per_region, R, n_obs_per_region)
        cov_per_region = P_reshaped.diagonal(dim1=1, dim2=3).permute(0, 3, 1, 2).contiguous()
        cov_per_region_b = cov_per_region.unsqueeze(0).expand(B, -1, -1, -1, -1).contiguous()

        obs.update_from_smoothed(data.y, x_per_region, cov_per_region_b)

    # ------------------------------------------------------------------
    # GP-hyperparameter M-step
    # ------------------------------------------------------------------

    def _update_gp(
        self,
        model: BaseModel,
        data: MultiRegionData,
        posterior: dict[str, Tensor],
    ) -> None:
        from mbrila.dynamics.exact_gp import ExactGPLatent

        dyn = model.dynamics
        assert isinstance(dyn, ExactGPLatent)

        S = posterior["S"]  # (MT, MT)  detached
        B = int(data.y.shape[0])
        T = int(data.y.shape[1])
        M = dyn.state_dim_per_time
        MT = M * T

        gp_params = [p for p in dyn.parameters() if p.requires_grad]
        if not gp_params:
            return

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
            # Negative Q (minimise): ½ [B log|K_big| + tr(K_big⁻¹ S)].
            neg_Q: Tensor = 0.5 * (B * logdet_K + trace)
            neg_Q.backward()  # type: ignore[no-untyped-call]
            return neg_Q

        optimiser.step(closure)  # type: ignore[no-untyped-call]

    # ------------------------------------------------------------------
    # InferenceEngine interface
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

        score_trace: list[float] = []
        wall_start = time.perf_counter()
        prev_ll = -math.inf
        converged = False

        for iteration in range(max_iter):
            posterior = self._e_step(model, data)
            ll_value = float(posterior["ll"].item())
            score_trace.append(ll_value)

            if self.learn_obs:
                self._update_observation(model, data, posterior)
            if self.learn_gp:
                self._update_gp(model, data, posterior)

            if self.log_every > 0 and (iteration + 1) % self.log_every == 0:
                print(f"[em_exact] iter {iteration + 1}/{max_iter}  log p(y) = {ll_value:.3f}")

            if iteration > 0 and abs(ll_value - prev_ll) < tol * max(abs(ll_value), 1.0):
                converged = True
                break
            prev_ll = ll_value

        wall = time.perf_counter() - wall_start
        return FitResult(
            score_trace=score_trace,
            converged=converged,
            n_iter=len(score_trace),
            wall_time_s=wall,
            reason="converged" if converged else "completed max_iter",
        )

    def infer(self, model: BaseModel, data: MultiRegionData) -> Posterior:
        info = self._e_step(model, data)
        return Posterior(
            mean=info["x_hat"],
            cov=info["P_per_time"].unsqueeze(0).expand(info["x_hat"].shape[0], -1, -1, -1),
            cov_form="per_time_block",
            extras={"P_full": info["P_full"]},
        )

    def score(self, model: BaseModel, data: MultiRegionData) -> float:
        info = self._e_step(model, data)
        return float(info["ll"].item())
