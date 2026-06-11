"""Hybrid VBEM-Kalman engine for mDLAG-SSM.

The "SSM version" of mDLAG: same model family (R-region shared
across-latents with fixed per-region delays + ARD on the loading
matrix ``C``), but the latent prior is the Markovian-GP lift
(AR(``P``) state space) rather than the dense ``T × T`` GP covariance
the time-domain :class:`VEMARDEngine` uses. Inference is ``O(T)`` per
Kalman pass instead of ``O(T³)``.

Components
----------
- **Latent E-step**: :func:`build_variational_kalman_inputs` converts
  the variational ARD moments to synthetic standard-Kalman inputs
  ``(y_pseudo, H_eff, R_eff=I_M)``. The parallel-scan
  :func:`kalman_filter_parallel` then produces the variational latent
  posterior — no Kalman re-implementation needed.
- **ARD M-step**: :func:`aggregate_emission_stats` +
  :func:`run_emission_m_step` — identical to the dense engine's
  emission update.
- **GP / kernel / delay M-step**: frozen-q variational EM. Run
  filter+smoother once in ``no_grad`` (parallel scan), then compute
  the analytic expected dynamics log-density
  ``E_q[log p(x_t | x_{t-1}; A, Q)]`` and backward through the
  formula. Smoother and filter never enter the autograd graph.
- **Proxy "ELBO" (monitoring)**:
  ``log p(y_pseudo) + obs.elbo_emission(NT)``. ``log p(y_pseudo)``
  comes from the sequential Kalman filter's
  ``return_log_marginal=True`` output — the parallel-scan filter does
  not accumulate log marginals, so this one call stays sequential.

Caveats
-------
- The proxy is **not the true ELBO** and not the data evidence
  ``log p(y)``. ``y_pseudo`` is a Cholesky-trick synthetic
  observation, so ``log p(y_pseudo)`` differs from the true data
  log-likelihood by an iter-dependent Jacobian offset (the offset
  shifts with the ARD posterior, which determines how ``y_pseudo`` is
  built). Cross-iteration trends within a single fit are still a
  valid convergence monitor; cross-model and cross-engine comparisons
  are not supported.
- Per-trial filter covariances are identical (the standard Kalman cov
  recursion is data-independent given the model). We take
  ``filt_covs[0]`` for the ARD ``P_per_time`` aggregation.
- The GP M-step uses one Adam step per outer EM iteration by default
  (``gp_steps_per_em=1``). This is true VBEM: alternate the latent
  E-step with both M-steps cleanly. Increase ``gp_steps_per_em`` for
  faster kernel convergence at the cost of a slight VBEM bias.
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
    build_variational_kalman_inputs,
    run_emission_m_step,
    setup_ard_posteriors,
)
from mbrila.inference.kalman.parallel import (
    kalman_filter_parallel,
    rts_smoother_parallel,
)
from mbrila.inference.kalman.sequential import kalman_filter, rts_smoother
from mbrila.inference.kalman_em import _expected_log_p_dynamics

if TYPE_CHECKING:
    from mbrila.core.base_model import BaseModel
    from mbrila.core.data import MultiRegionData
    from mbrila.dynamics.markov_gp import BlockDiagonalDynamics
    from mbrila.observations.ard import ARDObservation


DEFAULT_JITTER: float = 1e-10


class VEMKalmanARDEngine(InferenceEngine):
    """Hybrid VBEM-Kalman engine for mDLAG-SSM."""

    name = "vem_kalman_ard"
    required_capabilities = frozenset({"to_lds"})

    def __init__(
        self,
        *,
        learn_gp: bool = True,
        learn_emission: bool = True,
        lr: float = 4e-2,
        weight_decay: float = 1e-2,
        gp_steps_per_em: int = 1,
        cosine_anneal: bool = True,
        lr_min: float = 1e-3,
        jitter: float = DEFAULT_JITTER,
        log_every: int = 0,
        alpha_prune_ratio: float = 10.0,
        use_parallel: bool | None = None,
    ) -> None:
        """
        Parameters
        ----------
        learn_gp, learn_emission:
            Toggle the two M-step halves (mirrors :class:`VEMARDEngine`).
        lr, weight_decay:
            Adam hyperparameters for the GP / kernel / delay M-step.
            Defaults match ADM's :class:`KalmanEMEngine`.
        gp_steps_per_em:
            Number of Adam steps the GP M-step takes per outer EM iter.
            ``1`` (default) is the standard VBEM rhythm; raise it to
            give the kernel a faster pre-convergence at the cost of a
            slight VBEM bias.
        cosine_anneal:
            If ``True``, anneal the Adam LR with a cosine schedule over
            the outer EM iteration count. Matches the ADM
            :class:`KalmanEMEngine` pattern for consistency across all
            Adam-based engines in the framework. End-of-horizon LR is
            ``lr_min``.
        lr_min:
            Floor for the cosine annealing schedule. Defaults to ``1e-3``;
            set to ``0`` to recover decaying-to-zero behaviour. A
            non-zero floor avoids the late-iter "infinitesimal step
            / σ drift" failure mode.
        jitter:
            Diagonal jitter added to ``CPhiC_block`` before Cholesky
            (variational-input construction) and to ``Q`` /``P0`` in the
            Kalman recursion's internal Choleskys.
        log_every:
            Print ELBO every ``log_every`` iterations; ``0`` silences.
        alpha_prune_ratio:
            ARD-aware gating threshold for the GP M-step. A latent
            column ``k`` is considered "pruned" if its max-over-regions
            ``α_mean[r, k]`` exceeds
            ``alpha_prune_ratio · min_k α_mean.max(dim=0).values``.
            Pruned columns' kernel ``log_sigma`` and delay ``β``
            gradients are zeroed before each Adam step so the
            data-disconnected parameters don't drift on numerical
            noise. Set to ``inf`` to disable. Default ``10.0`` matches
            :class:`VEMARDEngine`.
        use_parallel:
            Kalman filter+smoother backend. ``None`` (default) uses the
            parallel scan; pass ``True`` / ``False`` to force an
            explicit choice. The proxy-ELBO ``_compute_elbo`` always
            uses the sequential filter because the parallel scan does
            not accumulate the log marginal.
        """
        if lr <= 0:
            raise ValueError(f"lr must be positive; got {lr}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0; got {weight_decay}")
        if gp_steps_per_em < 1:
            raise ValueError(f"gp_steps_per_em must be >= 1; got {gp_steps_per_em}")
        if jitter < 0:
            raise ValueError(f"jitter must be >= 0; got {jitter}")
        if log_every < 0:
            raise ValueError(f"log_every must be >= 0; got {log_every}")
        if lr_min < 0:
            raise ValueError(f"lr_min must be >= 0; got {lr_min}")
        if lr_min > lr:
            raise ValueError(f"lr_min ({lr_min}) must be <= lr ({lr})")
        if alpha_prune_ratio <= 1.0:
            raise ValueError(f"alpha_prune_ratio must be > 1.0 (or inf to disable); got {alpha_prune_ratio}")
        self.learn_gp = learn_gp
        self.learn_emission = learn_emission
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.gp_steps_per_em = int(gp_steps_per_em)
        self.cosine_anneal = cosine_anneal
        self.lr_min = float(lr_min)
        self.jitter = float(jitter)
        self.log_every = log_every
        self.alpha_prune_ratio = float(alpha_prune_ratio)
        self.use_parallel = use_parallel
        self._cpu_warned = False

    # ------------------------------------------------------------------
    # Component access
    # ------------------------------------------------------------------

    @staticmethod
    def _components(
        model: BaseModel,
    ) -> tuple[BlockDiagonalDynamics, ARDObservation]:
        from mbrila.dynamics.markov_gp import BlockDiagonalDynamics
        from mbrila.observations.ard import ARDObservation

        dyn = model.dynamics
        obs = model.observation
        if not isinstance(dyn, BlockDiagonalDynamics):
            raise TypeError(f"VEMKalmanARDEngine requires BlockDiagonalDynamics; got {type(dyn).__name__}")
        if not isinstance(obs, ARDObservation):
            raise TypeError(f"VEMKalmanARDEngine requires ARDObservation; got {type(obs).__name__}")
        return dyn, obs

    def _resolve_use_parallel(self, device: torch.device) -> bool:
        """Decide parallel vs sequential based on ``use_parallel`` + device.

        ``use_parallel=None`` (the default) → always use the parallel
        scan. Pass ``True`` / ``False`` to override.
        """
        del device  # currently unused — kept for API symmetry / future use
        if self.use_parallel is not None:
            return self.use_parallel
        return True

    # ------------------------------------------------------------------
    # E-step: variational Kalman filter + smoother
    # ------------------------------------------------------------------

    def _e_step(self, model: BaseModel, data: MultiRegionData) -> dict[str, Tensor]:
        """Variational latent posterior via synthetic-obs + Kalman.

        Returns dict with:

        - ``x_hat``        : ``(B, T, M)`` posterior means in observable space.
        - ``P_per_time``   : ``(T, M, M)`` per-time covariance in observable space
          (shared across trials — Kalman cov recursion is data-independent).
        - ``smooth_means_state``, ``smooth_covs_state``: full-state versions
          for diagnostics.
        """
        from mbrila.dynamics.markov_gp import BlockDiagonalDynamics
        from mbrila.observations.ard import ARDObservation

        dyn, obs = self._components(model)
        assert isinstance(dyn, BlockDiagonalDynamics)
        assert isinstance(obs, ARDObservation)

        with torch.no_grad():
            A, Q = dyn.forward()
            H_select = dyn.H_select
            var_inputs = build_variational_kalman_inputs(obs, H_select, data.y, jitter=self.jitter)

            D = A.shape[-1]
            m0 = torch.zeros(D, dtype=A.dtype, device=A.device)
            P0 = torch.eye(D, dtype=A.dtype, device=A.device)

            if self._resolve_use_parallel(A.device):
                filt_means, filt_covs = kalman_filter_parallel(
                    y=var_inputs["y_pseudo"],
                    F=A,
                    Q=Q,
                    H=var_inputs["H_eff"],
                    R=var_inputs["R_eff"],
                    m0=m0,
                    P0=P0,
                )
                smooth_means, smooth_covs, _ = rts_smoother_parallel(filt_means, filt_covs, A, Q)
            else:
                filt_means, filt_covs, _ = kalman_filter(
                    y=var_inputs["y_pseudo"],
                    F=A,
                    Q=Q,
                    H=var_inputs["H_eff"],
                    R=var_inputs["R_eff"],
                    m0=m0,
                    P0=P0,
                    return_log_marginal=False,
                )
                smooth_means, smooth_covs, _ = rts_smoother(filt_means, filt_covs, A, Q)

            # Project to observable space (M-dim) — the layout the ARD
            # helpers expect.
            # x_hat_obs:   (B, T, M) — per-trial means.
            # P_per_time_obs: (T, M, M) — shared per-time cov (taken from
            # any trial since the cov recursion is data-independent).
            x_hat_obs = torch.einsum("md,btd->btm", H_select, smooth_means)
            P_state_one = smooth_covs[0]  # (T, D, D)
            P_per_time_obs = torch.einsum("md,tdj,kj->tmk", H_select, P_state_one, H_select)

        return {
            "x_hat": x_hat_obs,
            "P_per_time": P_per_time_obs,
            "smooth_means_state": smooth_means,
            "smooth_covs_state": smooth_covs,
        }

    # ------------------------------------------------------------------
    # ARD M-step — delegates to ard_helpers
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
        run_emission_m_step(obs, stats, NT=NT)

    # ------------------------------------------------------------------
    # GP / kernel / delay M-step — Adam autograd through Kalman filter
    # ------------------------------------------------------------------

    def _m_step_gp(
        self,
        model: BaseModel,
        data: MultiRegionData,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Adam step(s) on kernel / delay via frozen-q joint-LL EM.

        Mirrors :class:`KalmanEMEngine`'s generalized-EM pattern:

        1. Run filter + smoother once in :func:`torch.no_grad` (parallel
           scan, ``O(log T)`` work-depth on GPU) to get the frozen
           variational posterior ``q(x) = N(μ_s, Σ_s, Σ_{t,t+1})``.
        2. With ``q(x)`` frozen, compute the analytic dynamics-side
           expected log-density
           ``E_q[log p(x_t | x_{t-1}; A(θ), Q(θ))]`` summed over
           ``t = 1, …, T-1``. Only ``(A, Q)`` carry grad; the smoother
           output is detached. The initial-state and (synthetic)
           observation terms are constants w.r.t. ``(A, Q)`` under
           variational EM, so they contribute zero gradient and are
           omitted from ``loss`` for efficiency.
        3. A single ``loss.backward()`` populates ``.grad`` on kernel
           ``log_sigma`` and delay ``β`` via autograd through the
           analytic formula — no Kalman filter in the graph.

        The EM theorem guarantees this frozen-posterior gradient is the
        marginal-LL gradient at ``θ_old``, with no implicit-``q``
        chain-rule noise.
        """
        from mbrila.dynamics.markov_gp import BlockDiagonalDynamics

        dyn, obs = self._components(model)
        assert isinstance(dyn, BlockDiagonalDynamics)

        # --- Frozen E-step (parallel by default; sequential on CPU) ------
        with torch.no_grad():
            H_select = dyn.H_select
            var_inputs = build_variational_kalman_inputs(obs, H_select, data.y, jitter=self.jitter)
            A_frozen, Q_frozen = dyn.forward()
            D = A_frozen.shape[-1]
            m0 = torch.zeros(D, dtype=A_frozen.dtype, device=A_frozen.device)
            P0 = torch.eye(D, dtype=A_frozen.dtype, device=A_frozen.device)

            if self._resolve_use_parallel(A_frozen.device):
                filt_m, filt_c = kalman_filter_parallel(
                    var_inputs["y_pseudo"],
                    A_frozen,
                    Q_frozen,
                    var_inputs["H_eff"],
                    var_inputs["R_eff"],
                    m0,
                    P0,
                )
                smooth_m, smooth_c, pair_c = rts_smoother_parallel(filt_m, filt_c, A_frozen, Q_frozen)
            else:
                filt_m, filt_c, _ = kalman_filter(
                    var_inputs["y_pseudo"],
                    A_frozen,
                    Q_frozen,
                    var_inputs["H_eff"],
                    var_inputs["R_eff"],
                    m0,
                    P0,
                    return_log_marginal=False,
                )
                smooth_m, smooth_c, pair_c = rts_smoother(filt_m, filt_c, A_frozen, Q_frozen)

        # --- ARD-aware pruning mask --------------------------------------
        # Columns whose ``max-over-regions α`` is far above the minimum
        # are effectively pruned. Their kernel ``log_sigma`` and delay
        # ``β`` have weak (or no) data signal, so without gating Adam
        # would still step them on numerical noise and they drift away
        # from any meaningful value. Mirrors :class:`VEMARDEngine`'s
        # gate on the dense-GP path.
        with torch.no_grad():
            max_alpha = obs.alpha_mean.max(dim=0).values  # (K,)
            min_alpha = max_alpha.min().clamp(min=1e-12)
            prune_mask_K = max_alpha > (self.alpha_prune_ratio * min_alpha)

        # --- M-step: analytic joint-LL with autograd on (A, Q) -----------
        for _ in range(self.gp_steps_per_em):
            optimizer.zero_grad()
            A, Q = dyn.forward()  # fresh forward with grad through kernel/delay
            ll_dyn = _expected_log_p_dynamics(A, Q, smooth_m, smooth_c, pair_c)
            loss = -ll_dyn
            loss.backward()  # type: ignore[no-untyped-call]
            if prune_mask_K.any():
                for k in range(len(dyn.blocks)):
                    if not bool(prune_mask_K[k]):
                        continue
                    block = dyn.blocks[k]
                    for p in block.parameters():
                        if p.grad is not None:
                            p.grad.zero_()
            optimizer.step()
            # Hard-reset pruned columns' delay back to its init value
            # (``β = 0`` for :class:`FixedDelay`). Zeroing the gradient
            # alone isn't enough on Adam: the optimiser's first-moment
            # buffer (``exp_avg``) carries momentum accumulated from
            # iters before the prune gate engaged, so a zero-grad step
            # still nudges β by ``lr · exp_avg / √(exp_avg_sq + ε)``.
            # Over many iters this drifts pruned δ away from 0 even
            # though no data signal supports it. LBFGS-based
            # :class:`VEMARDEngine` doesn't hit this because it has no
            # momentum; for Adam the gate has to be paired with a hard
            # reset of the parameter value. σ is left alone — it
            # doesn't surface in plots and a pruned column's kernel
            # timescale has no observable consequence.
            if prune_mask_K.any():
                with torch.no_grad():
                    for k in range(len(dyn.blocks)):
                        if not bool(prune_mask_K[k]):
                            continue
                        block_delay = dyn.blocks[k].delay
                        beta = getattr(block_delay, "beta", None)
                        if isinstance(beta, torch.nn.Parameter):
                            beta.data.zero_()

    # ------------------------------------------------------------------
    # ELBO (proxy — see module docstring)
    # ------------------------------------------------------------------

    def _compute_elbo(self, model: BaseModel, data: MultiRegionData) -> Tensor:
        """Proxy ELBO = ``log p(y_pseudo) + obs.elbo_emission(NT)``.

        **Not the true ELBO.** The Cholesky trick rewrites the
        variational ARD observation model as a standard Kalman system
        with synthetic observations ``y_pseudo``; ``log p(y_pseudo)``
        is that synthetic system's marginal log-likelihood, not the
        data evidence ``log p(y)``. The two differ by a Jacobian
        determinant that shifts with the ARD posterior, so this proxy
        is informative within a single fit but **not** comparable
        across models or engines.

        Implementation note: this is the only place in
        :class:`VEMKalmanARDEngine` that calls the sequential Kalman
        filter — the parallel-scan filter does not accumulate log
        marginals, so ``return_log_marginal=True`` is only available
        on the sequential path.
        """
        from mbrila.dynamics.markov_gp import BlockDiagonalDynamics
        from mbrila.observations.ard import ARDObservation

        dyn, obs = self._components(model)
        assert isinstance(dyn, BlockDiagonalDynamics)
        assert isinstance(obs, ARDObservation)

        B, T = int(data.y.shape[0]), int(data.y.shape[1])
        NT = B * T

        with torch.no_grad():
            A, Q = dyn.forward()
            H_select = dyn.H_select
            var_inputs = build_variational_kalman_inputs(obs, H_select, data.y, jitter=self.jitter)
            D = A.shape[-1]
            m0 = torch.zeros(D, dtype=A.dtype, device=A.device)
            P0 = torch.eye(D, dtype=A.dtype, device=A.device)

            # Sequential filter is intentional — parallel-scan filter
            # does not return ``log_marginal``. This is the one
            # remaining sequential Kalman call in the engine.
            _, _, log_ml_per_trial = kalman_filter(
                var_inputs["y_pseudo"],
                A,
                Q,
                var_inputs["H_eff"],
                var_inputs["R_eff"],
                m0,
                P0,
                return_log_marginal=True,
            )
            emission_elbo = obs.elbo_emission(NT)

        return log_ml_per_trial.sum() + emission_elbo

    # ------------------------------------------------------------------
    # Fit / infer / score
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

        dyn, obs = self._components(model)
        assert isinstance(obs, ARDObservation)
        setup_ard_posteriors(obs, data, learn_emission=self.learn_emission)

        # One Adam optimiser per fit() call. Recreating it across fits
        # means each fit gets a fresh optimiser state — matches
        # KalmanEMEngine's pattern.
        gp_params = [p for p in dyn.parameters() if p.requires_grad] if self.learn_gp else []
        gp_optimizer: torch.optim.Optimizer | None = (
            torch.optim.Adam(gp_params, lr=self.lr, weight_decay=self.weight_decay) if gp_params else None
        )
        # Cosine annealing on the outer EM iter count (matches the
        # KalmanEMEngine pattern). T_max counts outer iters, not Adam
        # inner steps — the inner gp_steps_per_em loop runs at the
        # current LR.
        gp_scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
            torch.optim.lr_scheduler.CosineAnnealingLR(gp_optimizer, T_max=max_iter, eta_min=self.lr_min)
            if (gp_optimizer is not None and self.cosine_anneal)
            else None
        )

        score_trace: list[float] = []
        wall_start = time.perf_counter()
        prev_elbo = -math.inf
        converged = False

        for iteration in range(max_iter):
            posterior = self._e_step(model, data)
            # GP M-step first, then emission — matches :class:`VEMARDEngine`'s
            # ordering so the next iter's E-step sees parameters updated in
            # the same sequence on both engines.
            if self.learn_gp and gp_optimizer is not None:
                self._m_step_gp(model, data, gp_optimizer)
            if self.learn_emission:
                self._m_step_emission(model, data, posterior)
            if gp_scheduler is not None:
                gp_scheduler.step()

            elbo_value = float(self._compute_elbo(model, data).item())
            score_trace.append(elbo_value)

            if self.log_every > 0 and (iteration + 1) % self.log_every == 0:
                print(f"[vem_kalman_ard] iter {iteration + 1}/{max_iter}  proxy-ELBO = {elbo_value:.3f}")

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

    def infer(self, model: BaseModel, data: MultiRegionData) -> Posterior:
        info = self._e_step(model, data)
        x_hat = info["x_hat"]
        return Posterior(
            mean=x_hat,
            cov=info["P_per_time"].unsqueeze(0).expand(x_hat.shape[0], -1, -1, -1),
            cov_form="per_time_block",
        )

    def score(self, model: BaseModel, data: MultiRegionData) -> float:
        return float(self._compute_elbo(model, data).item())
