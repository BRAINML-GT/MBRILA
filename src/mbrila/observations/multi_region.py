"""Multi-region linear-Gaussian observation model.

Each region ``r`` has a private ``(y_dim_r × n_obs_per_region)`` loading
matrix ``C_r``. The full emission is

::

    y = block_diag(C_0, C_1, …, C_{R-1}) · g + d + ε,    ε ~ N(0, diag(R))

where ``g`` is the per-region observable latent vector (length
``R · n_obs_per_region``), ``d`` the per-neuron offset, and ``R`` the
diagonal observation noise. The mapping from the dynamics' lifted
state ``s_t`` to ``g`` is handled by
:class:`mbrila.dynamics.markov_gp.BlockDiagonalDynamics`'s ``H_select``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from mbrila.core.observation_spec import Observation


class MultiRegionLinearObservation(Observation):
    """Per-region block-diagonal linear-Gaussian emission.

    Parameters
    ----------
    y_dims:
        Per-region neuron counts.
    n_obs_per_region:
        Number of observable latents per region (``xdima + xdimw`` in
        ADM-speak).
    init_R:
        Initial diagonal observation noise variance for every neuron.
    init_C_scale:
        Scale of the random Xavier-style initialisation of ``C_r``.
        ADM uses ``xavier_uniform``, which is roughly ``√(6 / (fan_in +
        fan_out))``; we keep the same heuristic here.
    """

    # Class-level annotations for the registered parameters; see the
    # corresponding note on BlockDiagonalDynamics for the rationale.
    # The capital R in `diag_R_param` mirrors the math notation
    # (observation noise matrix R) and is used throughout the codebase.
    diag_R_param: nn.Parameter  # noqa: N815
    d_param: nn.Parameter
    n_obs_per_region: int

    def __init__(
        self,
        y_dims: tuple[int, ...],
        n_obs_per_region: int,
        *,
        init_R: float = 0.1,
        init_C_scale: float | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if n_obs_per_region < 1:
            raise ValueError(f"n_obs_per_region must be >= 1; got {n_obs_per_region}")
        n_total_obs = len(y_dims) * n_obs_per_region
        super().__init__(y_dims=y_dims, n_latent_total=n_total_obs)
        self.n_obs_per_region = n_obs_per_region

        # Per-region C matrices, registered as Parameters via setattr loop so
        # state_dict picks them up automatically. nn.ParameterList would also
        # work but accessing ``self.Cs[r]`` is awkward in mypy.
        Cs: list[nn.Parameter] = []
        for r, y_r in enumerate(y_dims):
            scale = init_C_scale if init_C_scale is not None else (6.0 / (y_r + n_obs_per_region)) ** 0.5
            C_r = (2 * torch.rand(y_r, n_obs_per_region, dtype=dtype) - 1) * scale
            param = nn.Parameter(C_r)
            Cs.append(param)
            self.register_parameter(f"C_{r}", param)
        self._Cs = Cs

        self.diag_R_param = nn.Parameter(torch.full((self.n_neurons,), init_R, dtype=dtype))
        self.d_param = nn.Parameter(torch.zeros(self.n_neurons, dtype=dtype))

    @property
    def Cs(self) -> list[nn.Parameter]:
        return list(self._Cs)

    def block_diag_C(self) -> Tensor:
        """Assemble the block-diagonal emission matrix.

        Shape ``(sum(y_dims), n_regions * n_obs_per_region)``.
        """
        out: Tensor = torch.block_diag(*self._Cs)  # type: ignore[no-untyped-call]
        return out

    def diag_R(self) -> Tensor:
        # Clamp to keep R strictly positive — gradient descent occasionally
        # drives a noise variance below zero numerically; the closed-form
        # M-step clamps to ``jitter`` already.
        return torch.clamp(self.diag_R_param, min=1e-8)

    def offset(self) -> Tensor:
        return self.d_param

    def forward(self, x: Tensor) -> Tensor:
        """Predict noiseless ``E[y | g]`` from observable latents ``g``.

        ``x`` here is the *observable* latent vector of shape
        ``(n_trials, T, n_obs_per_region * n_regions)``, **not** the
        lifted dynamics state. Callers (the engine) project the lifted
        state through ``H_select`` before invoking this.
        """
        C = self.block_diag_C()
        return torch.einsum("ij,btj->bti", C, x) + self.d_param

    # Closed-form M-step ---------------------------------------------------

    def update_from_smoothed(
        self,
        ys: Tensor,
        x_means: Tensor,
        x_second_moments: Tensor,
        *,
        prior_strength: float = 1e-8,
    ) -> None:
        """Closed-form Bayesian regression M-step.

        Parameters
        ----------
        ys:
            Observed neural data of shape ``(n_trials, T, sum(y_dims))``.
        x_means:
            Smoothed posterior means of *observable* latents, shape
            ``(n_trials, T, n_regions, n_obs_per_region)`` — i.e. the
            engine should already have projected the lifted state via
            ``H_select`` and reshaped per region before calling.
        x_second_moments:
            Per-region per-time second moment matrices
            ``E[x_r,t x_r,tᵀ]`` of shape
            ``(n_trials, T, n_regions, n_obs_per_region, n_obs_per_region)``.
            Pass full second moments (mean·meanᵀ + Cov), not just means.
        prior_strength:
            Weight of a tiny ridge prior on ``C_r`` to keep updates
            well-conditioned for short trials.
        """
        from mbrila.observations.linear_regression import bayesian_linear_regression

        n_trials, T, sum_y = ys.shape
        if sum_y != self.n_neurons:
            raise ValueError(f"ys last dim ({sum_y}) must equal sum(y_dims)={self.n_neurons}")
        if x_means.shape != (n_trials, T, self.n_regions, self.n_obs_per_region):
            raise ValueError(f"x_means must have shape (B, T, R, k); got {tuple(x_means.shape)}")
        if x_second_moments.shape != (
            n_trials,
            T,
            self.n_regions,
            self.n_obs_per_region,
            self.n_obs_per_region,
        ):
            raise ValueError(
                f"x_second_moments must have shape (B, T, R, k, k); got {tuple(x_second_moments.shape)}"
            )

        device = ys.device
        dtype = ys.dtype
        new_diag_Rs: list[Tensor] = []
        new_ds: list[Tensor] = []
        cum = 0
        for r, y_r_count in enumerate(self.y_dims):
            x_r = x_means[:, :, r, :].reshape(-1, self.n_obs_per_region)  # (B*T, k)
            y_r = ys[:, :, cum : cum + y_r_count].reshape(-1, y_r_count)  # (B*T, y_r)

            # Build sufficient stats with second-moment correction. The naive
            # bayesian_linear_regression would treat x_r as observed; here x_r
            # is only the posterior mean and we must add per-trial covariance
            # contribution to ExxT.
            ones = torch.ones(x_r.shape[0], 1, dtype=dtype, device=device)
            x_aug = torch.cat([x_r, ones], dim=1)  # (B*T, k+1)
            ExxT_mean = x_aug.transpose(0, 1) @ x_aug  # (k+1, k+1)
            ExyT = x_aug.transpose(0, 1) @ y_r  # (k+1, y_r)
            EyyT = y_r.transpose(0, 1) @ y_r  # (y_r, y_r)
            # Add Σ_{b,t} Cov_{b,t}[x] to the upper-left block of ExxT.
            cov_sum = x_second_moments[:, :, r, :, :].sum(dim=(0, 1))  # (k, k)
            mean_outer = (x_r.unsqueeze(-1) * x_r.unsqueeze(-2)).sum(dim=0)  # (k, k)
            # ExxT_mean[:k, :k] currently equals Σ x_r x_rᵀ; replace with second-moment.
            ExxT = ExxT_mean.clone()
            ExxT[:-1, :-1] = cov_sum + (ExxT[:-1, :-1] - mean_outer) + mean_outer  # i.e. cov_sum + Σ x xᵀ
            # Simplify: ExxT[:-1, :-1] = cov_sum + Σ x_r x_rᵀ_total. The
            # subtraction-and-re-addition is a no-op but keeps the intent
            # explicit; we just want cov_sum added to the existing total.
            ExxT[:-1, :-1] = cov_sum + ExxT_mean[:-1, :-1]

            weight_sum = torch.tensor(float(x_r.shape[0]), dtype=dtype, device=device)
            prior_ExxT = prior_strength * torch.eye(self.n_obs_per_region + 1, dtype=dtype, device=device)
            res = bayesian_linear_regression(
                expectations=(ExxT + prior_ExxT, ExyT, EyyT, weight_sum),
                fit_intercept=True,
            )

            with torch.no_grad():
                self._Cs[r].copy_(res.W)
            new_ds.append(res.b.detach())
            new_diag_Rs.append(res.sigma_diag.detach())
            cum += y_r_count

        with torch.no_grad():
            self.diag_R_param.copy_(torch.cat(new_diag_Rs))
            self.d_param.copy_(torch.cat(new_ds))
