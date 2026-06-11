"""Multi-group DLAG with ARD column prior (mDLAG).

Port of fast-mDLAG's variational mDLAG to mbrila. The latent layout is

::

    s_t = [reg0_across (K_a), reg1_across (K_a), …, reg_{R-1}_across (K_a)]

so the per-time state has ``M = R · K_a`` slots, and each region's
emission ``C_r`` is a dense ``(y_dim_r, K_a)`` matrix. There are **no
within-region latents** in mDLAG MATLAB; ARD on ``C_r``'s columns
naturally discovers within/across structure (a latent that ends up
high-``α`` for all but one region behaves as a "within" latent for
that one region).

Components
----------
- :class:`mbrila.dynamics.exact_gp.ExactGPLatent` with ``n_within = (0, …, 0)``
- :class:`mbrila.delays.fixed.FixedDelay` (per-region tanh-bounded delay)
- :class:`mbrila.observations.ard.ARDObservation` (mDLAG emission with
  ARD on C)
- :class:`mbrila.inference.vem_ard.VEMARDEngine` (time-domain VEM)

Initialisation
--------------
mDLAG inherits DLAG's sensitivity to the initial emission scale:
:class:`VEMARDEngine` has no in-loop scale anchor, so the first fit
iteration's ``q(C)`` must have the right magnitude. Use pCCA — call
``model.initialize_from_data(data)`` once before ``fit``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

import torch

from mbrila.core.base_model import BaseModel
from mbrila.core.data import MultiRegionData
from mbrila.core.inference_engine import InferenceEngine
from mbrila.core.latent_spec import ARDPriorConfig, LatentSpec
from mbrila.delays.fixed import FixedDelay
from mbrila.dynamics.exact_gp import ExactGPLatent
from mbrila.dynamics.markov_gp import BlockDiagonalDynamics, MarkovianGPLatent
from mbrila.inference.vem_ard import VEMARDEngine
from mbrila.inference.vem_kalman_ard import VEMKalmanARDEngine
from mbrila.kernels.base import BaseKernel
from mbrila.models.gpfa import build_observable_to_state
from mbrila.observations.ard import ARDObservation

EngineKind = Literal["time", "freq", "kalman"]


def _check_within_zero(spec: LatentSpec) -> None:
    """mDLAG MATLAB has no within latents — enforce the same here."""
    if any(w != 0 for w in spec.n_within):
        raise ValueError(
            f"MDLAG requires n_within = (0, …, 0); got {spec.n_within}. "
            "ARD on C discovers within/across structure — explicit within "
            "latents are redundant and not supported in v1."
        )


def _check_ard_selection(spec: LatentSpec) -> None:
    if spec.selection != "ard":
        raise ValueError(f"MDLAG requires LatentSpec(selection='ard'); got selection={spec.selection!r}.")
    if spec.n_across < 1:
        raise ValueError(f"MDLAG requires n_across >= 1 (no within latents); got n_across={spec.n_across}.")


class MDLAG(BaseModel):
    """Multi-group DLAG with ARD prior on the emission matrix.

    Parameters
    ----------
    latent_spec:
        Must have ``selection="ard"`` and ``n_within = (0, …, 0)``. The
        ``n_across`` field is treated as an **upper bound**: ARD may
        collapse some columns of ``C_r`` to zero, effectively pruning
        the corresponding latent for region ``r``.
    y_dims:
        Per-region neuron counts.
    T:
        Trial length in bins.
    kernel_factory_across:
        **Required.** Zero-arg callable returning a fresh
        :class:`~mbrila.kernels.base.BaseKernel`. Called once per across
        latent so each owns an independent parameter set. All three
        engines (``"time"`` / ``"freq"`` / ``"kalman"``) consume the
        kernel — ``"freq"`` additionally requires
        :meth:`BaseKernel.spectral_density`. For MOSE pass
        ``lambda: MOSEKernel(num_regions=R, init_sigma=σ)``.
    eps_across:
        White-noise floor on the GP kernel (fixed, matching DLAG MATLAB
        convention).
    max_delay:
        Bound on per-region delay magnitude in bins. Defaults to
        ``floor(T/2)``.
    min_var_frac:
        Per-neuron variance floor for ``1/φ_mean`` (matches fast-mDLAG's
        ``minVarFrac``).
    engine:
        Optional pre-configured :class:`VEMARDEngine`.
    """

    def __init__(
        self,
        latent_spec: LatentSpec,
        y_dims: tuple[int, ...],
        T: int,
        *,
        kernel_factory_across: Callable[[], BaseKernel],
        eps_across: float = 1e-3,
        max_delay: float | None = None,
        min_var_frac: float = 1e-3,
        engine: EngineKind = "time",
        engine_override: InferenceEngine | None = None,
        # SSM-only kwargs (``engine="kalman"``).
        lag_across: int = 5,
        cov_jitter: float = 1e-4,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        _check_ard_selection(latent_spec)
        _check_within_zero(latent_spec)
        if len(y_dims) != latent_spec.n_regions:
            raise ValueError(f"y_dims has {len(y_dims)} regions but latent_spec has {latent_spec.n_regions}")
        if T < 1:
            raise ValueError(f"T must be >= 1; got {T}")
        if engine not in ("time", "freq", "kalman"):
            raise ValueError(f"engine must be 'time', 'freq', or 'kalman'; got {engine!r}")
        if engine == "kalman" and lag_across < 1:
            raise ValueError(f"lag_across must be >= 1 when engine='kalman'; got {lag_across}")
        self._y_dims = tuple(int(d) for d in y_dims)
        self._T = int(T)
        self._eps_across = float(eps_across)
        self._max_delay = float(max_delay) if max_delay is not None else max(1.0, math.floor(self._T / 2))
        self._min_var_frac = float(min_var_frac)
        self._engine_kind: EngineKind = engine
        self._engine_override = engine_override
        self._lag_across = int(lag_across)
        self._cov_jitter = float(cov_jitter)
        self._kernel_factory_across = kernel_factory_across
        # ARD hyperprior parameters live on the spec; carry them into the
        # observation constructor below.
        ard_prior = latent_spec.ard_prior or ARDPriorConfig()
        self._ard_prior = ard_prior
        super().__init__(latent_spec=latent_spec, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Component wiring
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        # The observation is shared across all engines — mDLAG always uses
        # ARDObservation regardless of how the latent prior is inferred.
        self.observation = ARDObservation(
            y_dims=self._y_dims,
            n_obs_per_region=self.latent_spec.n_across,
            prior_alpha_a=self._ard_prior.shape,
            prior_alpha_b=self._ard_prior.rate,
            min_var_frac=self._min_var_frac,
            dtype=self._dtype,
        )
        # Dispatch on engine kind for the latent dynamics + inference engine.
        if self._engine_kind == "kalman":
            self._init_components_kalman()
        else:
            # "time" and "freq" share the dense-GP dynamics; only the engine
            # instance differs (user passes VEMARDFreqEngine via engine_override).
            self._init_components_dense()

    def _init_components_dense(self) -> None:
        """Historical mDLAG path: dense GP prior + VEMARDEngine (or freq)."""
        spec = self.latent_spec
        n_regions = spec.n_regions
        n_across = spec.n_across

        delay = FixedDelay(
            n_regions=n_regions,
            n_latent=n_across,
            max_delay=self._max_delay,
            dtype=self._dtype,
        )

        self.dynamics = ExactGPLatent(
            n_regions=n_regions,
            n_across=n_across,
            n_within=tuple(0 for _ in range(n_regions)),
            delay=delay,
            kernel_factory_across=self._kernel_factory_across,
            kernel_factory_within=self._kernel_factory_across,  # placeholder; no within latents
            eps_across=self._eps_across,
            eps_within=self._eps_across,
            dtype=self._dtype,
        )
        assert isinstance(self.dynamics, ExactGPLatent)
        self.delay = self.dynamics.delay
        # mDLAG has no separate Kernel submodule (RBF is hard-coded inside
        # ExactGPLatent.cov_full); populate a placeholder for BaseModel's slot.
        self.kernel = torch.nn.Identity()

        if self._engine_override is not None:
            # User-supplied engine instance (e.g. VEMARDFreqEngine for the
            # freq path). No type check — VEMARDEngine and VEMARDFreqEngine
            # share enough surface that the engine itself validates at fit time.
            self.inference = self._engine_override
        else:
            self.inference = VEMARDEngine()

    def _init_components_kalman(self) -> None:
        """mDLAG-SSM path: Markovian-GP lifted LDS + VEMKalmanARDEngine.

        Structurally identical to :class:`~mbrila.models.dlag.DLAG`'s
        ``engine="kalman"`` branch but with ``n_within=0`` everywhere
        (mDLAG has no within latents — ARD discovers within/across
        structure on ``C``). Shares the H_select layout via the
        :func:`build_observable_to_state` helper from GPFA.
        """
        spec = self.latent_spec
        n_regions = spec.n_regions
        n_across = spec.n_across
        T = self._T
        lag_a = self._lag_across

        across_blocks: list[MarkovianGPLatent] = []
        for _ in range(n_across):
            kernel_a = self._kernel_factory_across()
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

        # No within blocks for mDLAG-SSM (matches dense mDLAG and ARD's
        # role in discovering within/across structure).
        n_obs_per_region = n_across
        n_observable = n_regions * n_obs_per_region
        observable_to_state = build_observable_to_state(
            xdima=n_across,
            xdimw=0,
            num_regions=n_regions,
            lag_a=lag_a,
            lag_w=1,  # ignored when xdimw=0 (no within observables)
        )
        self.dynamics = BlockDiagonalDynamics(
            across_blocks,
            n_observable=n_observable,
            observable_to_state_indices=observable_to_state,
        )

        # Representative kernel/delay for BaseModel introspection — the
        # real ones live inside the dynamics blocks.
        rep_kernel = across_blocks[0].kernel
        assert isinstance(rep_kernel, BaseKernel)
        self.kernel = rep_kernel
        rep_delay = across_blocks[0].delay
        assert isinstance(rep_delay, FixedDelay)
        self.delay = rep_delay

        if self._engine_override is not None:
            if not isinstance(self._engine_override, VEMKalmanARDEngine):
                raise TypeError(
                    "engine_override must be VEMKalmanARDEngine when engine='kalman'; "
                    f"got {type(self._engine_override).__name__}"
                )
            self.inference = self._engine_override
        else:
            self.inference = VEMKalmanARDEngine()

    # ------------------------------------------------------------------
    # Initialisation from data
    # ------------------------------------------------------------------

    def initialize_from_data(self, data: MultiRegionData, *, zero_offset: bool = False) -> None:
        """Seed ``q(C, α, φ, d)`` via multi-view pCCA on ``data``.

        Equivalent to DLAG's ``initialize_from_data(mode='pcca')`` but
        with the ARD posterior layout. Sets ``var_floor`` to
        ``min_var_frac · Var[y_i]`` so the φ-update can never push the
        private variance ``1/φ_mean`` below the floor.
        """
        obs = self.observation
        assert isinstance(obs, ARDObservation)
        y = data.y.to(device=self._device, dtype=self._dtype)
        obs.initialize_from_pcca(y, zero_offset=zero_offset)

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
        """Sample from the *posterior-mean* parameters.

        Treats ``C_mean``, ``d_mean`` and ``1/φ_mean`` as point
        estimates of the loadings, offset, and diagonal noise. Latent
        draws come from the prior (dense GP Cholesky for ``"time"``/
        ``"freq"`` engines, lifted LDS forward simulation for
        ``"kalman"``).
        """
        if T != self._T:
            raise ValueError(f"sample T must match model T={self._T}; got {T}")
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1; got {n_trials}")

        if self._engine_kind == "kalman":
            return self._sample_kalman(n_trials, T, seed=seed)
        return self._sample_dense(n_trials, T, seed=seed)

    def _sample_dense(
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
        assert isinstance(self.observation, ARDObservation)

        with torch.no_grad():
            K_big = self.dynamics.cov_full(T)  # (MT, MT)
            MT = K_big.shape[0]
            M = self.dynamics.state_dim_per_time
            eye = torch.eye(MT, dtype=self._dtype, device=self._device)
            L = torch.linalg.cholesky(K_big + 1e-10 * eye)

            z = torch.randn(n_trials, MT, generator=gen, dtype=self._dtype).to(self._device)
            x_flat = z @ L.transpose(0, 1)
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
        """Forward-simulate the lifted LDS (same pattern as ADM/GPFA/DLAG-SSM)."""
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)

        assert isinstance(self.dynamics, BlockDiagonalDynamics)
        assert isinstance(self.observation, ARDObservation)

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
            x_full = torch.stack(xs, dim=1)

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
            "max_delay": self._max_delay,
            "min_var_frac": self._min_var_frac,
            "ard_prior_shape": self._ard_prior.shape,
            "ard_prior_rate": self._ard_prior.rate,
            "engine": self._engine_kind,
            "lag_across": self._lag_across,
            "cov_jitter": self._cov_jitter,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> MDLAG:
        """Restore MDLAG from config + caller-supplied kernel factory.

        ``kernel_factory_across`` is not serialisable — all engines
        require it and the caller must supply it via ``**kwargs``.
        """
        spec = LatentSpec(
            n_across=int(config["n_across"]),
            n_within=tuple(int(x) for x in config["n_within"]),
            selection="ard",
            ard_prior=ARDPriorConfig(
                shape=float(config["ard_prior_shape"]),
                rate=float(config["ard_prior_rate"]),
            ),
        )
        return cls(
            latent_spec=spec,
            y_dims=tuple(int(x) for x in config["y_dims"]),
            T=int(config["T"]),
            eps_across=float(config["eps_across"]),
            max_delay=float(config["max_delay"]),
            min_var_frac=float(config["min_var_frac"]),
            engine=config.get("engine", "time"),
            lag_across=int(config.get("lag_across", 5)),
            cov_jitter=float(config.get("cov_jitter", 1e-4)),
            **kwargs,
        )
