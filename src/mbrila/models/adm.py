"""Adaptive Delay Model (ADM) — multi-region Markovian-GP with time-varying delays.

This is the Phase-1 reference implementation: it composes a list of
:class:`~mbrila.dynamics.markov_gp.MarkovianGPLatent` blocks (``xdima``
across-region with time-varying delays + ``num_regions × xdimw``
within-region without delays) and trains them jointly with the
:class:`~mbrila.inference.kalman_em.KalmanEMEngine`.

State layout
------------
The lifted state ``s_t`` concatenates per-block states::

    [ across_block_0 | across_block_1 | … | across_block_{xdima-1}
      | within_block_(r=0, k=0) | … | within_block_(r=0, k=xdimw-1)
      | within_block_(r=1, k=0) | …
      | within_block_(r=R-1, k=xdimw-1) ]

Each across block has dimension ``lag_a × num_regions``; each within
block has dimension ``lag_w × 1``. Inside each across block the
per-time slot lays out region 0 first, then region 1, …, then time
``t - 1``'s regions, etc.

The observation selector ``H_select`` extracts the per-region
observable latent vector ``g_t`` (length ``num_regions × (xdima +
xdimw)``) from the lifted state so the emission collapses to
``y_t = block_diag(C_0, …, C_{R-1}) g_t + d + ε``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import torch

from mbrila.core.base_model import BaseModel
from mbrila.core.data import MultiRegionData
from mbrila.core.latent_spec import LatentSpec
from mbrila.delays.time_varying import TimeVaryingDelay, smoothing_params_for_kernel_sigma
from mbrila.dynamics.markov_gp import BlockDiagonalDynamics, MarkovianGPLatent
from mbrila.inference.kalman_em import KalmanEMEngine
from mbrila.kernels.base import BaseKernel
from mbrila.observations.multi_region import MultiRegionLinearObservation

InitMode = Literal["pcca"]


def _validate_within(spec: LatentSpec) -> int:
    """ADM requires every region to share the same within-region latent count."""
    if not spec.n_within:
        raise ValueError("LatentSpec must declare at least one region in n_within")
    first = spec.n_within[0]
    if any(w != first for w in spec.n_within):
        raise ValueError(f"ADM requires uniform n_within across regions; got {spec.n_within}")
    return first


class ADM(BaseModel):
    """Adaptive Delay Model.

    Parameters
    ----------
    latent_spec:
        Latent geometry. ``n_across`` is the number of cross-region
        latent factors (each with its own MOSE block + time-varying
        delay); every region carries ``n_within[r]`` within-region
        latent factors (uniform across regions, ADM-style).
    y_dims:
        Per-region neuron counts. Must agree with ``len(latent_spec.n_within)``.
    T:
        Trial length in bins.
    lag_across, lag_within:
        Markov order of the across-region and within-region blocks
        respectively.
    init_sigma_across, init_sigma_within:
        Initial timescale parameter ``σ`` for the MOSE kernel of the
        respective block family (inverse squared timescale).
    eps:
        Diagonal jitter added before the Cholesky in the kernel→LDS
        construction (matches ADM's ``eps``).
    init_R:
        Initial diagonal observation noise variance.
    """

    DEFAULT_LR: float = 4e-2
    DEFAULT_WD: float = 1e-2

    def __init__(
        self,
        latent_spec: LatentSpec,
        y_dims: tuple[int, ...],
        T: int,
        *,
        kernel_factory_across: Callable[[], BaseKernel],
        kernel_factory_within: Callable[[], BaseKernel],
        lag_across: int = 5,
        lag_within: int = 2,
        delay_smoothing_sigma_across: float = 0.05,
        eps: float = 1e-4,
        init_R: float = 0.1,
        engine: KalmanEMEngine | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """
        Parameters
        ----------
        kernel_factory_across, kernel_factory_within:
            **Required** zero-arg callables that produce fresh
            :class:`BaseKernel` instances for the across / within blocks.
            One kernel is constructed per block, so a typical factory
            for ``K`` across blocks calls e.g.
            ``lambda: MOSEKernel(num_regions=R, init_sigma=0.05)``. The
            library does not pick a default kernel — kernel choice is
            the 4-axis architecture's "kernel" axis and must be
            specified explicitly. For the canonical MOSE setup pass:
            ``kernel_factory_across=lambda: MOSEKernel(num_regions=R, init_sigma=0.05)``.
        delay_smoothing_sigma_across:
            Kernel timescale used **only** to size the Gaussian
            smoothing window applied to ``δ(t)``. Default 0.05 mirrors
            the historical ADM default. This is independent of the
            ``kernel_factory_*`` above — δ(t)'s smoothing kernel is an
            ADM design choice (TimeVaryingDelay smoothing) and is not
            tied to the latent's GP kernel.
        """
        if len(y_dims) != latent_spec.n_regions:
            raise ValueError(f"y_dims has {len(y_dims)} regions but latent_spec has {latent_spec.n_regions}")
        if T < 1:
            raise ValueError(f"T must be >= 1; got {T}")
        if lag_across < 1 or lag_within < 1:
            raise ValueError(f"lag_across and lag_within must be >= 1; got {lag_across}, {lag_within}")
        if delay_smoothing_sigma_across <= 0:
            raise ValueError(
                f"delay_smoothing_sigma_across must be positive; got {delay_smoothing_sigma_across}"
            )
        self._y_dims = tuple(y_dims)
        self._T = T
        self._lag_across = lag_across
        self._lag_within = lag_within
        self._kernel_factory_across = kernel_factory_across
        self._kernel_factory_within = kernel_factory_within
        self._delay_smoothing_sigma_across = float(delay_smoothing_sigma_across)
        self._eps = eps
        self._init_R = init_R
        self._engine_override = engine
        self._xdimw = _validate_within(latent_spec)
        super().__init__(latent_spec=latent_spec, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Component construction
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        spec = self.latent_spec
        num_regions = spec.n_regions
        T = self._T
        lag_a = self._lag_across
        lag_w = self._lag_within
        xdima = spec.n_across
        xdimw = self._xdimw

        # δ(t) smoothing window is sized from ``delay_smoothing_sigma_across`` —
        # an ADM-specific knob independent of the latent's GP kernel.
        # The window is capped at T: the smoother's reflect padding
        # requires kernel width <= T, and a window wider than the trial
        # would smooth over everything.
        smooth_size, smooth_sigma = smoothing_params_for_kernel_sigma(self._delay_smoothing_sigma_across)
        smooth_size = min(smooth_size, T)

        # Build across-region blocks: each owns its own kernel (from
        # the user-supplied factory) plus a per-block TimeVaryingDelay.
        across_blocks: list[MarkovianGPLatent] = []
        for _ in range(xdima):
            kernel_a = self._kernel_factory_across()
            delay = TimeVaryingDelay(
                n_regions=num_regions,
                n_latent=1,
                T=T,
                smoothing_size=smooth_size,
                smoothing_sigma=smooth_sigma,
                dtype=self._dtype,
            )
            across_blocks.append(
                MarkovianGPLatent(
                    kernel=kernel_a,
                    lag=lag_a,
                    T=T,
                    delay=delay,
                    num_dim=num_regions,
                    cov_jitter=self._eps,
                )
            )

        # Build within-region blocks: per (region, within-latent), no delay.
        within_blocks: list[MarkovianGPLatent] = []
        for _ in range(num_regions * xdimw):
            kernel_w = self._kernel_factory_within()
            within_blocks.append(
                MarkovianGPLatent(
                    kernel=kernel_w,
                    lag=lag_w,
                    T=T,
                    delay=None,
                    num_dim=1,
                    cov_jitter=self._eps,
                )
            )

        all_blocks = across_blocks + within_blocks
        n_obs_per_region = xdima + xdimw
        n_observable = num_regions * n_obs_per_region

        observable_to_state = self._build_selection(
            xdima=xdima,
            xdimw=xdimw,
            num_regions=num_regions,
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
            init_R=self._init_R,
            dtype=self._dtype,
        )

        # The kernel & delay belong to the dynamics; expose stand-in attributes
        # so BaseModel's invariants (kernel/delay/inference set) are satisfied.
        # We use the first across block's kernel/delay as the "representative"
        # for the BaseModel attributes (capabilities are aggregated from
        # dynamics, so this is just for introspection).
        if across_blocks:
            first_block = across_blocks[0]
            kernel = first_block.kernel
            delay_obj = first_block.delay
            assert isinstance(kernel, BaseKernel)
            assert isinstance(delay_obj, TimeVaryingDelay)
            self.kernel = kernel
            self.delay = delay_obj
        else:
            kernel = within_blocks[0].kernel
            assert isinstance(kernel, BaseKernel)
            self.kernel = kernel
            self.delay = TimeVaryingDelay(
                n_regions=num_regions,
                n_latent=1,
                T=T,
                smoothing_size=smooth_size,
                smoothing_sigma=smooth_sigma,
                dtype=self._dtype,
            )

        self.inference = self._engine_override or KalmanEMEngine(
            lr=self.DEFAULT_LR, weight_decay=self.DEFAULT_WD
        )

    @staticmethod
    def _build_selection(
        *,
        xdima: int,
        xdimw: int,
        num_regions: int,
        lag_a: int,
        lag_w: int,
    ) -> list[tuple[int, int]]:
        """Compute (observable_idx, state_idx) pairs for ``H_select``.

        Observable indexing: ``g[r, k]`` is at flat index
        ``r * (xdima + xdimw) + k`` where ``k ∈ [0, xdima)`` are the
        cross-region latents and ``k ∈ [xdima, xdima + xdimw)`` are
        the within-region latents.

        State indexing follows the layout described at the top of the
        module.
        """
        n_obs_per_region = xdima + xdimw
        across_total = xdima * lag_a * num_regions
        pairs: list[tuple[int, int]] = []
        # Across observables: for each (r, i), state index = i * lag_a * num_regions + r.
        for r in range(num_regions):
            for i in range(xdima):
                obs_idx = r * n_obs_per_region + i
                state_idx = i * lag_a * num_regions + r
                pairs.append((obs_idx, state_idx))
        # Within observables: for each (r, k), state index = across_total
        # + (r * xdimw + k) * lag_w (the "current time" entry of that block).
        for r in range(num_regions):
            for k in range(xdimw):
                obs_idx = r * n_obs_per_region + xdima + k
                state_idx = across_total + (r * xdimw + k) * lag_w
                pairs.append((obs_idx, state_idx))
        return pairs

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_from_data(
        self,
        data: MultiRegionData,
        *,
        mode: InitMode = "pcca",
        estimate_R: bool = True,
        zero_offset: bool = True,
        pcca_max_iter: int = 50,
    ) -> None:
        """Seed emission parameters from observed data using probabilistic CCA.

        DLAG-family models are sensitive to the initial value of the
        per-region loading matrices. Probabilistic CCA — a shared-z
        multi-view factor analysis on the stacked
        ``(B·T, sum y_dims)`` data (MATLAB ``init_pCCA_dlag.m``) — is
        the canonical init: it produces loadings on the data scale
        (``Var[y_r] ≈ W_r W_rᵀ + diag(ψ_r)``) and generalises to any
        number of regions.

        Call this before :meth:`fit` for best results::

            model.initialize_from_data(train_data)
            result = model.fit(train_data, max_iter=200)

        Parameters
        ----------
        data:
            Training observations.
        mode:
            Init recipe. Currently only ``"pcca"`` is supported.
        estimate_R:
            If ``True``, reset ``diag(R)`` to pCCA's ``ψ`` estimate.
        zero_offset:
            If ``True``, reset ``d`` to zero. Setting ``False``
            preserves pCCA's mean estimate as the initial offset.
        pcca_max_iter:
            Maximum FA-EM iterations inside the pCCA solver.
        """
        n_across = self.latent_spec.n_across
        n_within = self._xdimw
        y = data.y.to(device=self._device, dtype=self._dtype)
        observation = self.observation
        if not isinstance(observation, MultiRegionLinearObservation):
            raise TypeError(
                f"initialize_from_data requires MultiRegionLinearObservation; "
                f"got {type(observation).__name__}"
            )

        if mode != "pcca":
            raise ValueError(f"unknown init mode: {mode!r}; expected 'pcca'")
        self._initialize_from_data_pcca(
            y,
            observation=observation,
            n_across=n_across,
            n_within=n_within,
            estimate_R=estimate_R,
            zero_offset=zero_offset,
            pcca_max_iter=pcca_max_iter,
        )

    def _initialize_from_data_pcca(
        self,
        y: torch.Tensor,
        *,
        observation: MultiRegionLinearObservation,
        n_across: int,
        n_within: int,
        estimate_R: bool,
        zero_offset: bool,
        pcca_max_iter: int,
    ) -> None:
        """Multi-view pCCA emission init (shared-z FA across R regions).

        Falls back to per-region FA when ``n_across == 0`` — matches
        DLAG's ``initialize_dlag.m`` branch where the n_across == 0
        leg skips the pCCA call entirely and runs per-region FA on
        each region in isolation.
        """
        if n_across == 0:
            from mbrila.init.factor_analysis import fa_init_per_region

            n_per_region = n_across + n_within
            Cs, diag_R = fa_init_per_region(
                y,
                y_dims=self._y_dims,
                n_per_region=n_per_region,
                max_iter=pcca_max_iter,
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

        from mbrila.init.pcca import pcca_init_C

        Cs, diag_R, mu = pcca_init_C(
            y,
            y_dims=self._y_dims,
            n_across=n_across,
            n_within=n_within,
            max_iter=pcca_max_iter,
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

    # ------------------------------------------------------------------
    # Sampling and config
    # ------------------------------------------------------------------

    def sample(self, n_trials: int, T: int, *, seed: int | None = None) -> MultiRegionData:
        """Draw synthetic data from the model's prior.

        Forward-simulates the lifted LDS using the current parameters,
        then projects through ``H_select`` and the per-region emission.
        ``T`` must equal the model's configured trial length (the
        time-varying delay is anchored to it).
        """
        if T != self._T:
            raise ValueError(f"sample T must match model T={self._T}; got {T}")
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)

        assert isinstance(self.dynamics, BlockDiagonalDynamics)
        with torch.no_grad():
            A, Q = self.dynamics.forward()
            H_select = self.dynamics.H_select
            C = self.observation.block_diag_C()
            d = self.observation.offset()
            diag_R = self.observation.diag_R()

            # Initial state ~ N(0, I). Sample on CPU to keep the generator
            # local; move to the model's device after.
            D = A.shape[-1]
            x_prev = torch.randn(n_trials, D, generator=gen, dtype=self._dtype).to(self._device)
            xs: list[torch.Tensor] = []
            for t in range(T):  # trial-loop: ok  (time loop in sampling code)
                noise = torch.randn(n_trials, D, generator=gen, dtype=self._dtype).to(self._device)
                # Cholesky of Q[t] for noise injection.
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

    def to_config(self) -> dict[str, Any]:
        return {
            "n_across": self.latent_spec.n_across,
            "n_within": list(self.latent_spec.n_within),
            "y_dims": list(self._y_dims),
            "T": self._T,
            "lag_across": self._lag_across,
            "lag_within": self._lag_within,
            "delay_smoothing_sigma_across": self._delay_smoothing_sigma_across,
            "eps": self._eps,
            "init_R": self._init_R,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> ADM:
        """Restore ADM from config + caller-supplied kernel factories.

        Callable kernel factories cannot be serialised into a JSON
        config; ``from_config`` therefore requires the caller to pass
        ``kernel_factory_across`` and ``kernel_factory_within`` via
        ``**kwargs`` (matching the now-required constructor params).
        """
        spec = LatentSpec(
            n_across=int(config["n_across"]),
            n_within=tuple(int(x) for x in config["n_within"]),
        )
        return cls(
            latent_spec=spec,
            y_dims=tuple(int(x) for x in config["y_dims"]),
            T=int(config["T"]),
            lag_across=int(config["lag_across"]),
            lag_within=int(config["lag_within"]),
            delay_smoothing_sigma_across=float(config.get("delay_smoothing_sigma_across", 0.05)),
            eps=float(config["eps"]),
            init_R=float(config["init_R"]),
            **kwargs,
        )
