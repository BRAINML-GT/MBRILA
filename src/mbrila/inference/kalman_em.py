"""Kalman-EM inference engine for SDE-LDS models.

Trains by gradient ascent on the **generalized-EM objective**
(Dempster, Laird & Rubin 1977; Ghahramani & Hinton 1996):

- E-step: ``q ← p(x | y; θ_old)`` via parallel Kalman filter + smoother
  (Sarkka & Garcia-Fernandez 2021) inside ``torch.no_grad``.
- M-step: gradient ascent on ``E_{q_old}[log p(x, y; θ)]`` with ``q``
  held constant. The gradient flows only through the explicit
  appearances of ``θ`` in ``log p`` — no implicit ``∂q/∂θ`` chain.
  The EM theorem guarantees the marginal log-likelihood monotonically
  non-decreases under each step.

A closed-form Bayesian-regression M-step on ``(C, d, diag(R))`` is
available via the ``update_obs_every`` parameter. It is disabled by
default — Adam already optimises every parameter through the
joint-LL gradient — but can be enabled for tight-iteration-budget
scenarios where one-step-conjugate emission updates pay for themselves.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from mbrila.core.inference_engine import FitResult, InferenceEngine, Posterior
from mbrila.inference.kalman.parallel import (
    kalman_filter_parallel,
    rts_smoother_parallel,
)
from mbrila.inference.kalman.sequential import kalman_filter, rts_smoother

if TYPE_CHECKING:
    from mbrila.core.base_model import BaseModel
    from mbrila.core.data import MultiRegionData


# ---------------------------------------------------------------------------
# Joint-LL helpers (E_q[log p(x_0)] + E_q[log p(x_t | x_{t-1})] + E_q[log p(y_t | x_t)])
# ---------------------------------------------------------------------------


def _expected_log_p_initial(
    smoothed_means: Tensor,
    smoothed_covs: Tensor,
    mu0: Tensor,
    sigma0: Tensor,
) -> Tensor:
    """``Σ_b E_q[log p(x_0; θ)]`` summed over trials.

    Treats the prior ``(μ_0, Σ_0)`` as fixed (not learned). We add a
    tiny jitter to ``Σ_0`` for the inverse / log-det in case the caller
    passes a rank-deficient prior.
    """
    B = smoothed_means.shape[0]
    D = mu0.shape[-1]
    eye = torch.eye(D, dtype=mu0.dtype, device=mu0.device)
    sigma0_inv = torch.linalg.inv(sigma0 + 1e-12 * eye)
    log_det = torch.linalg.slogdet(sigma0 + 1e-12 * eye).logabsdet

    diff = smoothed_means[:, 0] - mu0  # (B, D)
    Exx0 = smoothed_covs[:, 0] + diff.unsqueeze(-1) * diff.unsqueeze(-2)  # (B, D, D)
    trace_term = torch.einsum("ij,bji->b", sigma0_inv, Exx0).sum()
    out: Tensor = -0.5 * (B * log_det + trace_term + B * D * math.log(2.0 * math.pi))
    return out


def _expected_log_p_dynamics(
    A: Tensor,
    Q: Tensor,
    smoothed_means: Tensor,
    smoothed_covs: Tensor,
    pairwise_covs: Tensor,
) -> Tensor:
    """``Σ_{t=1}^{T-1} Σ_b E_q[log p(x_t | x_{t-1}; θ)]``.

    Parameters
    ----------
    A, Q:
        ``(T, D, D)`` dynamics. ``A[t]`` propagates ``x_{t-1} → x_t``.
        ``A[0]`` and ``Q[0]`` are unused (we only condition for
        ``t = 1, …, T-1``).
    smoothed_means:
        ``(B, T, D)``.
    smoothed_covs:
        ``(B, T, D, D)``.
    pairwise_covs:
        ``(B, T-1, D, D)`` storing ``Cov(x_t, x_{t+1} | y)`` for
        ``t = 0, …, T-2`` (centred cross-time covariance, same
        convention as :func:`rts_smoother`).
    """
    B, T, D = smoothed_means.shape

    A_t = A[1:]  # (T-1, D, D); A_t[k] propagates x_k → x_{k+1}
    Q_t = Q[1:]

    # E[x_t x_t^T] for t = 1..T-1
    Exx_t = smoothed_covs[:, 1:] + smoothed_means[:, 1:].unsqueeze(-1) * smoothed_means[:, 1:].unsqueeze(-2)
    # E[x_{t-1} x_{t-1}^T] for t = 1..T-1
    Exx_prev = smoothed_covs[:, :-1] + smoothed_means[:, :-1].unsqueeze(-1) * smoothed_means[
        :, :-1
    ].unsqueeze(-2)
    # E[x_{t-1} x_t^T] = mean_{t-1} mean_tᵀ + Cov(x_{t-1}, x_t)
    Ex_prev_x_t = smoothed_means[:, :-1].unsqueeze(-1) * smoothed_means[:, 1:].unsqueeze(-2) + pairwise_covs
    Ex_t_x_prev = Ex_prev_x_t.transpose(-1, -2)

    # Σ_b E_q[(x_t - A_t x_{t-1})(x_t - A_t x_{t-1})ᵀ] per (T-1, D, D)
    # = E[x_t x_tᵀ] - A E[x_{t-1} x_tᵀ] - E[x_t x_{t-1}ᵀ] Aᵀ + A E[x_{t-1} x_{t-1}ᵀ] Aᵀ
    residual = (
        Exx_t
        - torch.einsum("tij,btjk->btik", A_t, Ex_prev_x_t)
        - torch.einsum("btij,tkj->btik", Ex_t_x_prev, A_t)
        + torch.einsum("tij,btjk,tlk->btil", A_t, Exx_prev, A_t)
    )
    S_x = residual.sum(dim=0)  # (T-1, D, D), summed over batch

    log_det_Q = torch.linalg.slogdet(Q_t).logabsdet  # (T-1,)
    Q_inv = torch.linalg.inv(Q_t)
    trace_term = torch.einsum("tij,tji->t", Q_inv, S_x)  # (T-1,)

    out: Tensor = -0.5 * (B * log_det_Q.sum() + trace_term.sum() + B * (T - 1) * D * math.log(2.0 * math.pi))
    return out


def _expected_log_p_obs(
    H_eff: Tensor,
    diag_R: Tensor,
    offset: Tensor,
    y: Tensor,
    smoothed_means: Tensor,
    smoothed_covs: Tensor,
) -> Tensor:
    """``Σ_t Σ_b E_q[log p(y_t | x_t; θ)]`` for diagonal ``R``.

    Decomposition::

        E_q[log p(y_t|x_t)] = -½ [‖y_t − d − H E[x_t]‖²_{R⁻¹}
                                  + tr(R⁻¹ H Cov[x_t] Hᵀ)
                                  + log|R|
                                  + n_y log(2π)]
    """
    B, T, _ = smoothed_means.shape
    n_y = y.shape[-1]

    y_centred = y - offset  # (B, T, n_y)
    pred_y = torch.einsum("ij,btj->bti", H_eff, smoothed_means)  # (B, T, n_y)
    diff = y_centred - pred_y

    # H Cov H^T diagonal: Σ_{j, k} H[i, j] Cov[j, k] H[i, k]
    HCov = torch.einsum("ij,btjk->btik", H_eff, smoothed_covs)
    HCovHT_diag = torch.einsum("btij,ij->bti", HCov, H_eff)  # (B, T, n_y)
    obs_residual = diff.square() + HCovHT_diag  # (B, T, n_y)
    inv_R = 1.0 / diag_R
    weighted = obs_residual * inv_R
    sum_obs = weighted.sum()
    log_det_R = torch.log(diag_R).sum()
    return -0.5 * (sum_obs + B * T * (log_det_R + n_y * math.log(2.0 * math.pi)))


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _is_kalman_compatible_dynamics(dyn: object) -> bool:
    """Structural check for a Kalman-compatible dynamics module.

    Two concrete implementations: :class:`BlockDiagonalDynamics`
    (Markovian-GP lifted LDS, used by ADM / DLAG-SSM / GPFA) and
    :class:`FreeLDSLatent` (naive LDS, no kernel). The engine never
    touches anything beyond ``H_select`` and ``forward()``, so any
    future dynamics that produce ``(A, Q)`` with the right shapes plus
    an ``H_select`` selector slots in without touching this file.

    We use a hasattr/callable duck check rather than a
    ``runtime_checkable`` Protocol because Protocol's runtime
    ``isinstance`` does not see :class:`nn.Module` *buffers* (registered
    via ``register_buffer`` and exposed through ``__getattr__``) — they
    pass ``hasattr`` but fail Protocol's class-MRO check. The duck-typing
    check below works in both cases.
    """
    H = getattr(dyn, "H_select", None)
    if not isinstance(H, Tensor):
        return False
    fwd = getattr(dyn, "forward", None)
    return callable(fwd)


class KalmanEMEngine(InferenceEngine):
    """SDE-LDS inference engine: generalized-EM ascent + optional closed-form M-step on emission."""

    name = "kalman_em"
    required_capabilities = frozenset({"to_lds"})

    def __init__(
        self,
        *,
        lr: float = 1e-2,
        weight_decay: float = 1e-2,
        update_obs_every: int = 0,
        cosine_anneal: bool = True,
        lr_min: float = 1e-3,
        log_every: int = 0,
        closed_form_obs_refit: bool = True,
        scale_anchor: bool = True,
        grouped_weight_decay: bool = True,
        use_parallel: bool | None = None,
    ) -> None:
        """
        Parameters
        ----------
        lr, weight_decay:
            Adam hyperparameters.
        update_obs_every:
            How often (in training iterations) to run the closed-form
            Bayesian-regression M-step on ``(C, d, diag(R))``. ``0``
            (default) disables it — Adam optimises every parameter via
            the joint-LL gradient. Set to e.g. ``20`` to run the
            closed-form update every 20 iterations.
        cosine_anneal:
            If ``True``, anneal the Adam LR with a cosine schedule over
            the full training horizon. End-of-horizon LR is ``lr_min``.
        lr_min:
            Floor for the cosine annealing schedule (``eta_min`` of
            :class:`torch.optim.lr_scheduler.CosineAnnealingLR`). Set to
            ``0`` to decay all the way to zero.
        log_every:
            If ``> 0``, print the current loss every ``log_every``
            iterations.
        closed_form_obs_refit:
            If ``True`` (default), run one closed-form ``(C, d, R)``
            LSE refit (via :meth:`_closed_form_obs_step`) once at the
            start of :meth:`fit`, before the main optimiser loop.
            Brings the pCCA emission seed to the joint optimum under
            the current dynamics — Adam starts from a much better basin.
        scale_anchor:
            If ``True`` (default), call
            :func:`mbrila.normalize_latent_scales` once at the start
            (after the closed-form refit if applicable) and once at the
            end of :meth:`fit`. Removes the ``y = C·g`` gauge freedom
            that otherwise makes Adam wander in scale space. Requires
            ``model.observation`` to be a
            :class:`MultiRegionLinearObservation` — silently skipped
            for ARD observations.
        grouped_weight_decay:
            If ``True`` (default), build a grouped AdamW that excludes
            parameters named ``raw_delay`` / ``beta`` / ``d_param``
            (linear shifts: ADM δ(t), FixedDelay δ, emission bias) from
            weight decay. Set to ``False`` to recover plain
            :class:`torch.optim.Adam` (uniform decay) behaviour.
        use_parallel:
            Which Kalman implementation to use for filter + smoother.
            ``None`` (default) uses the parallel-scan implementation
            (Sarkka & Garcia-Fernandez 2021), which gives ``O(log T)``
            work-depth on GPU and remains competitive on CPU thanks to
            batched matrix ops. Pass ``True`` / ``False`` to force an
            explicit choice (useful for parity testing).
        """
        if update_obs_every < 0:
            raise ValueError(f"update_obs_every must be >= 0; got {update_obs_every}")
        if lr_min < 0:
            raise ValueError(f"lr_min must be >= 0; got {lr_min}")
        if lr_min > lr:
            raise ValueError(f"lr_min ({lr_min}) must be <= lr ({lr})")
        self.lr = lr
        self.weight_decay = weight_decay
        self.update_obs_every = update_obs_every
        self.cosine_anneal = cosine_anneal
        self.lr_min = float(lr_min)
        self.log_every = log_every
        self.closed_form_obs_refit = closed_form_obs_refit
        self.scale_anchor = scale_anchor
        self.grouped_weight_decay = grouped_weight_decay
        self.use_parallel = use_parallel
        self._cpu_warned = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _assemble(self, model: BaseModel) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Pull (A, Q, H_eff, diag_R, d) from the model components.

        ``H_eff = block_diag_C @ H_select`` collapses the per-region
        observation matrix and the latent-state selection so that
        ``E[y | s_t] = H_eff s_t + d``.
        """
        if not _is_kalman_compatible_dynamics(model.dynamics):
            raise TypeError(
                "KalmanEMEngine requires a dynamics module exposing ``H_select`` and "
                f"``forward() -> (A, Q)``; got {type(model.dynamics).__name__}"
            )
        A, Q = model.dynamics.forward()
        H_select = model.dynamics.H_select
        # mypy: nn.Module.__getattr__ widens H_select to Tensor | Module; the
        # duck check above guarantees Tensor at runtime.
        assert isinstance(H_select, Tensor)
        C_blk = model.observation.block_diag_C()
        H_eff = C_blk @ H_select
        diag_R = model.observation.diag_R()
        d = model.observation.offset()
        return A, Q, H_eff, diag_R, d

    def _initial_prior(self, A: Tensor) -> tuple[Tensor, Tensor]:
        D = A.shape[-1]
        m0 = torch.zeros(D, dtype=A.dtype, device=A.device)
        P0 = torch.eye(D, dtype=A.dtype, device=A.device)
        return m0, P0

    def _resolve_use_parallel(self, device: torch.device) -> bool:
        """Decide parallel vs sequential based on ``use_parallel`` + device.

        ``use_parallel=None`` (the default) → always use the parallel
        scan: on CUDA this is the standard ``O(log T)`` win; on CPU the
        parallel scan vectorises the batched matrix ops so it remains
        competitive with (and sometimes faster than) the sequential
        Python-loop implementation. Pass ``True`` / ``False`` to
        override.
        """
        del device  # currently unused — kept for API symmetry / future use
        if self.use_parallel is not None:
            return self.use_parallel
        return True

    def _filter_then_smooth(
        self,
        y_centred: Tensor,
        A: Tensor,
        Q: Tensor,
        H_eff: Tensor,
        R: Tensor,
        m0: Tensor,
        P0: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Run filter + smoother, dispatched per ``self.use_parallel``.

        Returns (filtered_means, filtered_covs, smoothed_means,
        smoothed_covs, pairwise_covs).
        """
        if self._resolve_use_parallel(A.device):
            f_means, f_covs = kalman_filter_parallel(y_centred, A, Q, H_eff, R, m0, P0)
            s_means, s_covs, pair = rts_smoother_parallel(f_means, f_covs, A, Q)
        else:
            f_means, f_covs, _ = kalman_filter(y_centred, A, Q, H_eff, R, m0, P0, return_log_marginal=False)
            s_means, s_covs, pair = rts_smoother(f_means, f_covs, A, Q)
        return f_means, f_covs, s_means, s_covs, pair

    def _marginal_ll(self, model: BaseModel, data: MultiRegionData) -> Tensor:
        """Marginal log-likelihood ``log p(y; θ)`` summed over trials.

        Internal helper for held-out validation scoring. Uses the
        sequential Kalman filter (the parallel scan does not accumulate
        the per-step log marginal).
        """
        A, Q, H_eff, diag_R, d = self._assemble(model)
        R = torch.diag(diag_R)
        m0, P0 = self._initial_prior(A)
        y_centred = data.y - d
        _, _, log_ml = kalman_filter(y_centred, A, Q, H_eff, R, m0, P0)
        return log_ml.sum()

    def _joint_ll_em(self, model: BaseModel, data: MultiRegionData) -> Tensor:
        """Generalized EM objective: joint LL with q frozen at the current θ.

        E-step:
            q ← p(x | y; θ)  computed via filter + smoother in
            ``torch.no_grad`` and detached from the autograd graph.
        M-step (this function's return value):
            ``E_{q_old}[log p(x, y; θ)]``  — the gradient w.r.t. ``θ``
            flows ONLY through the explicit ``θ``-dependent terms
            (``A``, ``Q``, ``H_eff``, ``diag(R)``, ``d``); no implicit
            ``∂q/∂θ`` chain.

        By the EM theorem (Dempster, Laird & Rubin 1977), each gradient
        ascent step on this Q-function monotonically improves the
        marginal log-likelihood.
        """
        # E-step: smoother posterior at current θ, no autograd.
        with torch.no_grad():
            A0, Q0, H_eff0, diag_R0, d0 = self._assemble(model)
            R0 = torch.diag(diag_R0)
            m0_, P0_ = self._initial_prior(A0)
            y_centred0 = data.y - d0
            _, _, s_means, s_covs, pair = self._filter_then_smooth(y_centred0, A0, Q0, H_eff0, R0, m0_, P0_)

        # M-step: rebuild the parameter-dependent quantities with autograd.
        A, Q, H_eff, diag_R, d = self._assemble(model)
        m0, P0 = self._initial_prior(A)

        ll_init = _expected_log_p_initial(s_means, s_covs, m0, P0)
        ll_x = _expected_log_p_dynamics(A, Q, s_means, s_covs, pair)
        ll_y = _expected_log_p_obs(H_eff, diag_R, d, data.y, s_means, s_covs)
        return ll_init + ll_x + ll_y

    def _loss_value(self, model: BaseModel, data: MultiRegionData) -> Tensor:
        return self._joint_ll_em(model, data)

    def _smoother_posterior(self, model: BaseModel, data: MultiRegionData) -> dict[str, Tensor]:
        """E-step: smoothed means + covs (no autograd through the smoother).

        Used only by the closed-form emission M-step.
        """
        with torch.no_grad():
            A, Q, H_eff, diag_R, d = self._assemble(model)
            R = torch.diag(diag_R)
            m0, P0 = self._initial_prior(A)
            y_centred = data.y - d
            _, _, s_means, s_covs, pair = self._filter_then_smooth(y_centred, A, Q, H_eff, R, m0, P0)
        return {"means": s_means, "covs": s_covs, "pairwise": pair}

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
        del kwargs  # currently unused
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1; got {max_iter}")

        from mbrila.inference.optim import build_grouped_adamw

        # --- Closed-form (C, d, R) LSE refit (one-shot, pre-loop) --------
        if self.closed_form_obs_refit:
            self._closed_form_obs_step(model, data)

        # --- Initial scale anchor (silently skipped if not applicable) ----
        if self.scale_anchor:
            self._maybe_normalize_latent_scales(model, data)

        if self.grouped_weight_decay:
            optimizer = build_grouped_adamw(model, lr=self.lr, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=self.lr_min)
            if self.cosine_anneal
            else None
        )

        score_trace: list[float] = []
        wall_start = time.perf_counter()
        prev_ll = -math.inf
        converged = False
        for iteration in range(max_iter):
            optimizer.zero_grad()
            ll = self._loss_value(model, data)
            loss = -ll
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            ll_value = float(ll.detach().item())
            score_trace.append(ll_value)

            if self.update_obs_every > 0 and (iteration + 1) % self.update_obs_every == 0:
                self._closed_form_obs_step(model, data)

            if self.log_every > 0 and (iteration + 1) % self.log_every == 0:
                print(f"[kalman_em] iter {iteration + 1}/{max_iter}  E_q[log p(x,y)] = {ll_value:.3f}")

            # Convergence: relative change in LL below tol.
            if iteration > 0 and abs(ll_value - prev_ll) < tol * max(abs(ll_value), 1.0):
                converged = True
                break
            prev_ll = ll_value

        # --- Final scale anchor ------------------------------------------
        if self.scale_anchor:
            self._maybe_normalize_latent_scales(model, data)

        wall = time.perf_counter() - wall_start
        return FitResult(
            score_trace=score_trace,
            converged=converged,
            n_iter=len(score_trace),
            wall_time_s=wall,
            reason="converged" if converged else "completed max_iter",
        )

    def _maybe_normalize_latent_scales(self, model: BaseModel, data: MultiRegionData) -> None:
        """Call :func:`normalize_latent_scales` when applicable, else skip.

        Requires both a :class:`MultiRegionLinearObservation` (so the
        ``y = C·g + d`` factoring is well-defined) AND a
        :class:`BlockDiagonalDynamics` with per-block kernel scales
        (so each block's gauge can be normalised independently).
        Skipped silently for ARD emissions (mDLAG, where ARD's ``α``
        priors already regularise the latent scale) and for flat-state
        dynamics like :class:`FreeLDSLatent` that have no per-block
        ``K(0)`` to anchor to.
        """
        from mbrila.dynamics.markov_gp import BlockDiagonalDynamics
        from mbrila.init.scale_anchor import normalize_latent_scales
        from mbrila.observations.multi_region import MultiRegionLinearObservation

        if not isinstance(model.observation, MultiRegionLinearObservation):
            return
        if not isinstance(model.dynamics, BlockDiagonalDynamics):
            return
        normalize_latent_scales(model, data)

    def infer(self, model: BaseModel, data: MultiRegionData) -> Posterior:
        info = self._smoother_posterior(model, data)
        return Posterior(
            mean=info["means"],
            cov=info["covs"],
            cov_form="per_time_block",
            extras={"pairwise_covs": info["pairwise"]},
        )

    def score(self, model: BaseModel, data: MultiRegionData) -> float:
        with torch.no_grad():
            ll = self._loss_value(model, data)
        return float(ll.item())

    # ------------------------------------------------------------------
    # Closed-form emission M-step
    # ------------------------------------------------------------------

    def _closed_form_obs_step(self, model: BaseModel, data: MultiRegionData) -> None:
        from mbrila.observations.multi_region import MultiRegionLinearObservation

        observation = model.observation
        if not isinstance(observation, MultiRegionLinearObservation):
            raise TypeError(
                f"closed-form emission update requires MultiRegionLinearObservation; "
                f"got {type(observation).__name__}"
            )

        info = self._smoother_posterior(model, data)
        s_means = info["means"]  # (B, T, D)
        s_covs = info["covs"]  # (B, T, D, D)

        assert _is_kalman_compatible_dynamics(model.dynamics)
        H_select = model.dynamics.H_select
        assert isinstance(H_select, Tensor)
        g_means = torch.einsum("ij,btj->bti", H_select, s_means)
        g_covs = torch.einsum("ij,btjk,lk->btil", H_select, s_covs, H_select)

        n_regions = observation.n_regions
        n_obs_per_region = observation.n_obs_per_region

        g_means_pr = g_means.view(*g_means.shape[:-1], n_regions, n_obs_per_region)
        g_second_pr_list: list[Tensor] = []
        for r in range(n_regions):
            start = r * n_obs_per_region
            end = start + n_obs_per_region
            block_cov = g_covs[..., start:end, start:end]
            block_mean = g_means_pr[..., r, :]
            block_second = block_cov + block_mean.unsqueeze(-1) * block_mean.unsqueeze(-2)
            g_second_pr_list.append(block_second.unsqueeze(-3))
        g_second_pr = torch.cat(g_second_pr_list, dim=-3)

        observation.update_from_smoothed(data.y, g_means_pr, g_second_pr)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @classmethod
    def from_kwargs(cls, kwargs: Mapping[str, Any]) -> KalmanEMEngine:
        return cls(**dict(kwargs))
