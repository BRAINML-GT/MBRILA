"""Delayed Latent Aligned GP model (DLAG).

DLAG models multi-region recordings as observations of a shared set of
across-region latent factors with per-region **fixed scalar time delays**,
plus optional per-region within-region private factors.

Two inference paths:

- ``engine="exact"`` (default):
  Non-Markovian Gaussian-process prior over the full ``(M·T, M·T)``
  joint covariance. ``ExactEMEngine`` does Cholesky-based EM. Cost is
  ``O(T³)`` per step but no AR(``P``) approximation.
- ``engine="kalman"`` (DLAG-SSM):
  Same model structure — across latents with fixed per-region delays
  + within latents — but each latent is lifted into a lag-``P`` LDS
  via the AR(``P``) bridge (``lagged_cov_grid`` +
  ``kernel_to_lds``). ``KalmanEMEngine`` runs filter + smoother in
  ``O(T)``. With non-Markovian kernels (e.g. MOSE) the lift is an
  AR(``P``) approximation — the user picks ``P`` via ``lag_across`` /
  ``lag_within``.

Same emission ``MultiRegionLinearObservation`` and the same pCCA
initialisation work for both paths.

State layout per time bin
-------------------------
::

    s_t = [reg0_across (K_a), reg0_within (K_w),
           reg1_across (K_a), reg1_within (K_w),
           …,
           reg_{R-1}_across (K_a), reg_{R-1}_within (K_w)]

DLAG (this implementation, v1) requires a uniform within-latent count
``K_w = n_within[0] = … = n_within[R-1]``. This mirrors the DLAG MATLAB
reference's most common deployment and keeps the emission layer's
``MultiRegionLinearObservation`` (which expects uniform
``n_obs_per_region``) directly usable.

Initialisation
--------------
Like every model in this family, DLAG is sensitive to the initial value
of the per-region loading matrices ``C_r``. The recommended workflow is

::

    model = DLAG(latent_spec=..., y_dims=..., T=...)
    model.initialize_from_data(train_data)         # seeds (C, d, R)
    result = model.fit(train_data, max_iter=200)

Two initialisation modes are supported:

* ``"pcca"`` (default, matches DLAG MATLAB's ``init_pCCA_dlag.m``):
  multi-view probabilistic CCA — a shared-latent factor analysis over
  the stacked regional data — produces emission loadings ``C_r`` with
  the correct data scale (``Var[y_r] ≈ W_r W_rᵀ + diag(ψ_r)``).
  Recommended for DLAG since :class:`ExactEMEngine` has no in-loop
  scale-anchor mechanism.
* ``"fa"``: per-region (independent) factor analysis. Used only when
  ``n_across = 0``; matches the ``n_across = 0`` branch of DLAG's
  ``initialize_dlag.m``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

import torch

from mbrila.core.base_model import BaseModel
from mbrila.core.data import MultiRegionData
from mbrila.core.latent_spec import LatentSpec
from mbrila.delays.fixed import FixedDelay
from mbrila.dynamics.exact_gp import ExactGPLatent
from mbrila.dynamics.markov_gp import BlockDiagonalDynamics, MarkovianGPLatent
from mbrila.inference.em_exact import ExactEMEngine
from mbrila.inference.kalman_em import KalmanEMEngine
from mbrila.kernels.base import BaseKernel
from mbrila.models.gpfa import build_observable_to_state
from mbrila.observations.multi_region import MultiRegionLinearObservation

InitMode = Literal["pcca", "fa"]
EngineKind = Literal["exact", "kalman"]


def _uniform_within(spec: LatentSpec) -> int:
    """DLAG requires every region to share the same within-region latent count."""
    if not spec.n_within:
        raise ValueError("LatentSpec must declare at least one region in n_within")
    first = spec.n_within[0]
    if any(w != first for w in spec.n_within):
        raise ValueError(f"DLAG requires uniform n_within across regions; got {spec.n_within}")
    return first


class DLAG(BaseModel):
    """Delayed Latent Aligned GP model.

    Parameters
    ----------
    latent_spec:
        Latent geometry. ``n_across`` is the number of cross-region
        latent factors; ``n_within`` is the per-region within-region
        latent counts (must be uniform — see module docstring).
    y_dims:
        Per-region neuron counts.
    T:
        Trial length in bins. All trials are assumed to share this length.
    kernel_factory_across, kernel_factory_within:
        **Required.** Zero-arg callables returning fresh
        :class:`~mbrila.kernels.base.BaseKernel` instances. Each across
        / within block owns an independent kernel (called once per
        block). Both engines (``"exact"`` and ``"kalman"``) consume the
        kernel through :meth:`BaseKernel.cov` (time domain). MOSE is the
        common default — pass ``lambda: MOSEKernel(num_regions=R, init_sigma=σ)``.
    eps_across, eps_within:
        White-noise floors on the GP kernels. Fixed (not optimised) per
        DLAG's MATLAB convention.
    max_delay:
        Bound on the effective delay magnitude in bins. Delays are
        parameterised as ``D_max · tanh(β/2)`` with a learnable ``β``.
        ``None`` defaults to ``floor(T/2)`` so the bound is data-driven.
    engine:
        Optional pre-configured :class:`ExactEMEngine`. ``None`` builds
        the default engine.
    """

    def __init__(
        self,
        latent_spec: LatentSpec,
        y_dims: tuple[int, ...],
        T: int,
        *,
        kernel_factory_across: Callable[[], BaseKernel],
        kernel_factory_within: Callable[[], BaseKernel],
        eps_across: float = 1e-3,
        eps_within: float = 1e-3,
        max_delay: float | None = None,
        engine: EngineKind = "exact",
        engine_override: ExactEMEngine | KalmanEMEngine | None = None,
        # SSM-only kwargs (``engine="kalman"``).
        lag_across: int = 5,
        lag_within: int = 2,
        cov_jitter: float = 1e-4,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if len(y_dims) != latent_spec.n_regions:
            raise ValueError(f"y_dims has {len(y_dims)} regions but latent_spec has {latent_spec.n_regions}")
        if T < 1:
            raise ValueError(f"T must be >= 1; got {T}")
        if engine not in ("exact", "kalman"):
            raise ValueError(f"engine must be 'exact' or 'kalman'; got {engine!r}")
        if engine == "kalman" and (lag_across < 1 or lag_within < 1):
            raise ValueError(
                f"lag_across and lag_within must be >= 1 when engine='kalman'; got {lag_across}, {lag_within}"
            )
        self._y_dims = tuple(int(d) for d in y_dims)
        self._T = int(T)
        self._eps_across = float(eps_across)
        self._eps_within = float(eps_within)
        self._max_delay = float(max_delay) if max_delay is not None else max(1.0, math.floor(self._T / 2))
        self._engine_kind: EngineKind = engine
        self._engine_override = engine_override
        self._lag_across = int(lag_across)
        self._lag_within = int(lag_within)
        self._cov_jitter = float(cov_jitter)
        self._kernel_factory_across = kernel_factory_across
        self._kernel_factory_within = kernel_factory_within
        self._xdimw = _uniform_within(latent_spec)
        super().__init__(latent_spec=latent_spec, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Component wiring
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        if self._engine_kind == "exact":
            self._init_components_exact()
        else:
            self._init_components_kalman()

    def _init_components_exact(self) -> None:
        """Historical DLAG path: dense GP prior + ExactEMEngine."""
        spec = self.latent_spec
        n_regions = spec.n_regions
        n_across = spec.n_across
        n_within = self._xdimw

        delay: FixedDelay | None = (
            FixedDelay(
                n_regions=n_regions,
                n_latent=n_across,
                max_delay=self._max_delay,
                dtype=self._dtype,
            )
            if n_across > 0
            else None
        )

        # n_within tuple matches the LatentSpec (uniform), so we pass it through.
        within_tuple = tuple(n_within for _ in range(n_regions))

        self.dynamics = ExactGPLatent(
            n_regions=n_regions,
            n_across=n_across,
            n_within=within_tuple,
            delay=delay,
            kernel_factory_across=self._kernel_factory_across,
            kernel_factory_within=self._kernel_factory_within,
            eps_across=self._eps_across,
            eps_within=self._eps_within,
            dtype=self._dtype,
        )

        # BaseModel's introspection requires ``kernel`` / ``delay`` attrs.
        # DLAG's GP hyperparameters live inside the dynamics module; we
        # expose the dynamics' delay (which is always populated — a
        # degenerate placeholder is constructed internally when
        # ``n_across == 0``).
        assert isinstance(self.dynamics, ExactGPLatent)
        self.delay = self.dynamics.delay

        # DLAG's kernel form is implicit (RBF inside ExactGPLatent); there's
        # no separate Kernel submodule. We still need to populate
        # BaseModel's slot for save/load to register a sane Module.
        self.kernel = torch.nn.Identity()

        n_obs_per_region = n_across + n_within
        self.observation = MultiRegionLinearObservation(
            y_dims=self._y_dims,
            n_obs_per_region=n_obs_per_region,
            init_R=0.1,
            dtype=self._dtype,
        )

        if self._engine_override is not None:
            if not isinstance(self._engine_override, ExactEMEngine):
                raise TypeError(
                    "engine_override must be ExactEMEngine when engine='exact'; "
                    f"got {type(self._engine_override).__name__}"
                )
            self.inference = self._engine_override
        else:
            self.inference = ExactEMEngine()

    def _init_components_kalman(self) -> None:
        """DLAG-SSM path: AR(P) lifted LDS prior + KalmanEMEngine.

        Structurally identical to :class:`~mbrila.models.gpfa.GPFA` but with
        :class:`~mbrila.delays.fixed.FixedDelay` on the across blocks. Reuses
        the shared :func:`~mbrila.models.gpfa.build_observable_to_state`
        H_select layout.
        """
        spec = self.latent_spec
        n_regions = spec.n_regions
        n_across = spec.n_across
        n_within = self._xdimw
        T = self._T
        lag_a = self._lag_across
        lag_w = self._lag_within

        # Build across blocks: one FixedDelay (shared across all latents inside
        # the block — the standard layout convention) wrapped in n_across
        # MarkovianGPLatent blocks, each with its own kernel (from the
        # caller-supplied factory).
        across_blocks: list[MarkovianGPLatent] = []
        for _ in range(n_across):
            kernel_a = self._kernel_factory_across()
            # FixedDelay with n_latent=1 — one delay-trajectory per across block,
            # matching MarkovianGPLatent's "single latent factor" contract.
            delay = FixedDelay(
                n_regions=n_regions,
                n_latent=1,
                max_delay=self._max_delay,
                dtype=self._dtype,
            )
            across_blocks.append(
                MarkovianGPLatent(
                    kernel=kernel_a,
                    lag=lag_a,
                    T=T,
                    delay=delay,
                    num_dim=n_regions,
                    cov_jitter=self._cov_jitter,
                )
            )

        # Within blocks: one MarkovianGPLatent per (region, within-latent), no delay.
        within_blocks: list[MarkovianGPLatent] = []
        for _ in range(n_regions * n_within):
            kernel_w = self._kernel_factory_within()
            within_blocks.append(
                MarkovianGPLatent(
                    kernel=kernel_w,
                    lag=lag_w,
                    T=T,
                    delay=None,
                    num_dim=1,
                    cov_jitter=self._cov_jitter,
                )
            )

        all_blocks = across_blocks + within_blocks
        n_obs_per_region = n_across + n_within
        n_observable = n_regions * n_obs_per_region

        observable_to_state = build_observable_to_state(
            xdima=n_across,
            xdimw=n_within,
            num_regions=n_regions,
            lag_a=lag_a,
            lag_w=lag_w,
        )

        self.dynamics = BlockDiagonalDynamics(
            all_blocks,
            n_observable=n_observable,
            observable_to_state_indices=observable_to_state,
        )

        self.observation = MultiRegionLinearObservation(
            y_dims=self._y_dims,
            n_obs_per_region=n_obs_per_region,
            init_R=0.1,
            dtype=self._dtype,
        )

        # Representative kernel/delay attributes for BaseModel introspection.
        # When n_across=0 we fall back to a placeholder; this is consistent with
        # the exact path's degenerate-delay treatment.
        if across_blocks:
            rep_kernel = across_blocks[0].kernel
            assert isinstance(rep_kernel, BaseKernel)
            self.kernel = rep_kernel
            rep_delay = across_blocks[0].delay
            assert isinstance(rep_delay, FixedDelay)
            self.delay = rep_delay
        else:
            within_kernel = within_blocks[0].kernel
            assert isinstance(within_kernel, BaseKernel)
            self.kernel = within_kernel
            self.delay = FixedDelay(
                n_regions=n_regions,
                n_latent=max(1, n_across),
                max_delay=self._max_delay,
                dtype=self._dtype,
            )

        if self._engine_override is not None:
            if not isinstance(self._engine_override, KalmanEMEngine):
                raise TypeError(
                    "engine_override must be KalmanEMEngine when engine='kalman'; "
                    f"got {type(self._engine_override).__name__}"
                )
            self.inference = self._engine_override
        else:
            self.inference = KalmanEMEngine()

    # ------------------------------------------------------------------
    # Initialisation from data
    # ------------------------------------------------------------------

    def initialize_from_data(
        self,
        data: MultiRegionData,
        *,
        mode: InitMode = "pcca",
        estimate_R: bool = True,
        zero_offset: bool = True,
        fa_max_iter: int = 50,
    ) -> None:
        """Seed ``(C, d, diag(R))`` from observed data.

        Parameters
        ----------
        data:
            Training observations.
        mode:
            ``"pcca"`` (default, matches DLAG MATLAB's
            ``init_pCCA_dlag.m``): multi-view probabilistic CCA — fits
            a shared latent across regions and returns loadings,
            offset, and noise variance in their natural data scale.
            The :class:`ExactEMEngine` has no in-loop scale anchor, so
            the init must already produce correctly-scaled ``C``.
            ``"fa"``: per-region (independent) factor analysis. Useful
            only when ``n_across = 0``.
        estimate_R:
            If ``True``, reset ``diag(R)`` to pCCA / FA's noise
            estimate.
        zero_offset:
            If ``True``, reset the per-neuron offset ``d`` to zero.
            With ``mode="pcca"`` setting this to ``False`` keeps
            pCCA's mean estimate as the initial offset.
        fa_max_iter:
            FA-EM iteration cap.
        """
        n_across = self.latent_spec.n_across
        n_within = self._xdimw
        observation = self.observation
        assert isinstance(observation, MultiRegionLinearObservation)
        y = data.y.to(device=self._device, dtype=self._dtype)

        if mode == "pcca":
            # When n_across == 0 there is no shared latent → fall back to
            # per-region FA. This matches DLAG MATLAB's behaviour
            # (initialize_dlag.m skips pCCA when xDim_across == 0).
            if n_across == 0:
                self.initialize_from_data(
                    data,
                    mode="fa",
                    estimate_R=estimate_R,
                    zero_offset=zero_offset,
                    fa_max_iter=fa_max_iter,
                )
                return

            from mbrila.init.pcca import pcca_init_C

            Cs, diag_R, mu = pcca_init_C(
                y,
                y_dims=self._y_dims,
                n_across=n_across,
                n_within=n_within,
                max_iter=fa_max_iter,
            )
            with torch.no_grad():
                for r, C_r in enumerate(Cs):
                    observation.Cs[r].data.copy_(
                        C_r.to(dtype=observation.Cs[r].dtype, device=observation.Cs[r].device)
                    )
                if estimate_R:
                    observation.diag_R_param.data.copy_(
                        diag_R.to(
                            dtype=observation.diag_R_param.dtype,
                            device=observation.diag_R_param.device,
                        )
                    )
                if zero_offset:
                    observation.d_param.data.zero_()
                else:
                    observation.d_param.data.copy_(
                        mu.to(dtype=observation.d_param.dtype, device=observation.d_param.device)
                    )
            return

        if mode == "fa":
            from mbrila.init.factor_analysis import fa_init_per_region

            n_per_region = n_across + n_within
            Cs, diag_R = fa_init_per_region(
                y,
                y_dims=self._y_dims,
                n_per_region=n_per_region,
                max_iter=fa_max_iter,
            )
            with torch.no_grad():
                for r, C_r in enumerate(Cs):
                    observation.Cs[r].data.copy_(
                        C_r.to(dtype=observation.Cs[r].dtype, device=observation.Cs[r].device)
                    )
                if estimate_R:
                    observation.diag_R_param.data.copy_(
                        diag_R.to(
                            dtype=observation.diag_R_param.dtype,
                            device=observation.diag_R_param.device,
                        )
                    )
                if zero_offset:
                    observation.d_param.data.zero_()
            return

        raise ValueError(f"unknown init mode: {mode!r}; expected 'pcca' or 'fa'")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        n_trials: int,
        T: int,
        *,
        seed: int | None = None,
    ) -> MultiRegionData:
        """Draw a synthetic dataset from the model's prior.

        ``T`` must equal the model's configured trial length (the GP
        kernel / lifted LDS is built at ``T``). The exact and Kalman
        paths sample from different parameterisations of the same model
        family; they will produce statistically similar but not identical
        draws (the AR(``P``) lift introduces a small approximation error).
        """
        if T != self._T:
            raise ValueError(f"sample T must match model T={self._T}; got {T}")
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1; got {n_trials}")

        if self._engine_kind == "exact":
            return self._sample_exact(n_trials, T, seed=seed)
        return self._sample_kalman(n_trials, T, seed=seed)

    def _sample_exact(
        self,
        n_trials: int,
        T: int,
        *,
        seed: int | None = None,
    ) -> MultiRegionData:
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)

        assert isinstance(self.dynamics, ExactGPLatent)
        assert isinstance(self.observation, MultiRegionLinearObservation)

        with torch.no_grad():
            K_big = self.dynamics.cov_full(T)  # (MT, MT)
            MT = K_big.shape[0]
            M = self.dynamics.state_dim_per_time
            eye = torch.eye(MT, dtype=self._dtype, device=self._device)
            L = torch.linalg.cholesky(K_big + 1e-10 * eye)

            z = torch.randn(n_trials, MT, generator=gen, dtype=self._dtype).to(self._device)
            x_flat = z @ L.transpose(0, 1)  # (n_trials, MT)
            x_per_t = x_flat.reshape(n_trials, T, M)

            C_blk = self.observation.block_diag_C()
            d_off = self.observation.offset()
            diag_R = self.observation.diag_R()

            y_clean = torch.einsum("ij,btj->bti", C_blk, x_per_t) + d_off
            noise_unit = torch.randn(n_trials, T, y_clean.shape[-1], generator=gen, dtype=self._dtype).to(
                self._device
            )
            noise = noise_unit * diag_R.sqrt()
            y = y_clean + noise

        return MultiRegionData(y=y, y_dims=self._y_dims, bin_width=1.0)

    def _sample_kalman(
        self,
        n_trials: int,
        T: int,
        *,
        seed: int | None = None,
    ) -> MultiRegionData:
        """Forward-simulate the lifted LDS (same shape as ADM/GPFA sampling)."""
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)

        assert isinstance(self.dynamics, BlockDiagonalDynamics)
        assert isinstance(self.observation, MultiRegionLinearObservation)

        with torch.no_grad():
            A, Q = self.dynamics.forward()
            H_select = self.dynamics.H_select
            C = self.observation.block_diag_C()
            d = self.observation.offset()
            diag_R = self.observation.diag_R()

            D = A.shape[-1]
            x_prev = torch.randn(n_trials, D, generator=gen, dtype=self._dtype).to(self._device)
            xs: list[torch.Tensor] = []
            for t in range(T):  # trial-loop: ok  (time loop in sampling code)
                noise = torch.randn(n_trials, D, generator=gen, dtype=self._dtype).to(self._device)
                L_Q = torch.linalg.cholesky(Q[t])
                x_t = torch.einsum("ij,bj->bi", A[t], x_prev) + noise @ L_Q.T
                xs.append(x_t)
                x_prev = x_t
            x_full = torch.stack(xs, dim=1)  # (B, T, D)

            g = torch.einsum("ij,btj->bti", H_select, x_full)
            y_clean = torch.einsum("ij,btj->bti", C, g) + d
            obs_noise = (
                torch.randn(n_trials, T, y_clean.shape[-1], generator=gen, dtype=self._dtype).to(self._device)
                * diag_R.sqrt()
            )
            y = y_clean + obs_noise

        return MultiRegionData(y=y, y_dims=self._y_dims, bin_width=1.0)

    # ------------------------------------------------------------------
    # Config (save / load)
    # ------------------------------------------------------------------

    def to_config(self) -> dict[str, Any]:
        return {
            "n_across": self.latent_spec.n_across,
            "n_within": list(self.latent_spec.n_within),
            "y_dims": list(self._y_dims),
            "T": self._T,
            "eps_across": self._eps_across,
            "eps_within": self._eps_within,
            "max_delay": self._max_delay,
            "engine": self._engine_kind,
            "lag_across": self._lag_across,
            "lag_within": self._lag_within,
            "cov_jitter": self._cov_jitter,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> DLAG:
        """Restore DLAG from config + caller-supplied kernel factories.

        ``kernel_factory_across`` / ``kernel_factory_within`` are not
        serialisable — both engines require them and the caller must
        supply them via ``**kwargs``.
        """
        spec = LatentSpec(
            n_across=int(config["n_across"]),
            n_within=tuple(int(x) for x in config["n_within"]),
        )
        return cls(
            latent_spec=spec,
            y_dims=tuple(int(x) for x in config["y_dims"]),
            T=int(config["T"]),
            eps_across=float(config["eps_across"]),
            eps_within=float(config["eps_within"]),
            max_delay=float(config["max_delay"]),
            engine=config.get("engine", "exact"),
            lag_across=int(config.get("lag_across", 5)),
            lag_within=int(config.get("lag_within", 2)),
            cov_jitter=float(config.get("cov_jitter", 1e-4)),
            **kwargs,
        )
