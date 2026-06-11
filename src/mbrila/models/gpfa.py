"""Multi-region GPFA — Gaussian-Process Factor Analysis via the SSM path.

GPFA is what you get from the "compose 4 axes" framework with:

- delay: :class:`~mbrila.delays.none.NoDelay` (no per-region time shifts)
- prior: any GP kernel exposing ``cov`` — supplied as a required
  ``kernel_factory_across`` callable
- observation: standard linear-Gaussian
- engine: :class:`KalmanEMEngine` (the SSM path — O(T) per Kalman step)

**Shared-only by definition** (Yu et al. 2009): GPFA has K_a shared latents
across all regions and **no per-region (within) latents**. ``n_within`` on
the :class:`LatentSpec` must be all-zero; non-zero within counts are
rejected. Users who want "shared + within without delay" should compose
:class:`~mbrila.models.dlag.DLAG` directly (or wait for the Stage 3
``DLAG(delay=NoDelay)`` preset).

In the ``n_regions = 1`` degenerate case this is classic single-region
GPFA; for ``n_regions > 1`` it is multi-region GPFA — similar in structure
to mDLAG but without per-region delays and without ARD on ``C``.

State layout & H_select
-----------------------
Mirrors ADM's layout for the across blocks. With no within blocks, the
state is just ``[reg0_lat0_t, …, reg0_lat0_{t-P+1}, reg1_lat0_t, …]``
laid out region-first within each across latent.

Kernel pluggability
-------------------
``kernel_factory_across`` is a required zero-arg callable returning a fresh
:class:`BaseKernel`. The library does not pick a default — for MOSE pass
``lambda: MOSEKernel(num_regions=R, init_sigma=0.05)``, for Matérn-3/2 pass
``lambda: Matern32Kernel(lengthscale=2.0)``, etc.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from mbrila.core.base_model import BaseModel
from mbrila.core.data import MultiRegionData
from mbrila.core.latent_spec import LatentSpec
from mbrila.delays.none import NoDelay
from mbrila.dynamics.markov_gp import BlockDiagonalDynamics, MarkovianGPLatent
from mbrila.inference.kalman_em import KalmanEMEngine
from mbrila.kernels.base import BaseKernel
from mbrila.observations.multi_region import MultiRegionLinearObservation


def _validate_shared_only(spec: LatentSpec) -> None:
    """GPFA is shared-only (Yu et al. 2009): every region's within count must be 0."""
    if not spec.n_within:
        raise ValueError("LatentSpec must declare at least one region in n_within")
    if any(w != 0 for w in spec.n_within):
        raise ValueError(
            "GPFA requires n_within = 0 for every region (shared-only). "
            f"Got n_within={spec.n_within}. For shared + within-without-delay use DLAG."
        )


def build_observable_to_state(
    *,
    xdima: int,
    xdimw: int,
    num_regions: int,
    lag_a: int,
    lag_w: int,
) -> list[tuple[int, int]]:
    """``(observable, state)`` pairs for ``H_select`` in the SSM layout.

    Layout convention shared with :class:`~mbrila.models.adm.ADM` — see its
    module docstring for the precise state ordering. This helper exists at
    module level so multiple presets (ADM, GPFA, future DLAG-SSM) can build
    the same selector without duplicating the indexing arithmetic.
    """
    n_obs_per_region = xdima + xdimw
    across_total = xdima * lag_a * num_regions
    pairs: list[tuple[int, int]] = []
    # Across observables: g[r, i] picks slot ``i * lag_a * R + r`` from the
    # current-time block of across-latent i.
    for r in range(num_regions):
        for i in range(xdima):
            obs_idx = r * n_obs_per_region + i
            state_idx = i * lag_a * num_regions + r
            pairs.append((obs_idx, state_idx))
    # Within observables: g[r, k] picks the "current time" entry of within
    # block ``(r, k)`` from the within-block stack.
    for r in range(num_regions):
        for k in range(xdimw):
            obs_idx = r * n_obs_per_region + xdima + k
            state_idx = across_total + (r * xdimw + k) * lag_w
            pairs.append((obs_idx, state_idx))
    return pairs


class GPFA(BaseModel):
    """Multi-region GPFA via the SSM (Kalman) path. Shared latents only.

    Structurally identical to :class:`~mbrila.models.adm.ADM`'s across
    blocks with :class:`NoDelay`, no within blocks, and a time-invariant
    lifted ``(A, Q)``.

    Parameters
    ----------
    latent_spec:
        Latent geometry. ``n_across`` is the number of shared latents.
        ``n_within`` **must be all-zero** (GPFA is shared-only); a non-zero
        value raises :class:`ValueError`.
    y_dims:
        Per-region neuron counts.
    T:
        Trial length in bins.
    lag_across:
        Markov order ``P`` for the AR(P) lifting of the across kernel.
        For non-Markovian kernels (RBF / MOSE) larger ``P`` gives a tighter
        approximation; for exact-SDE kernels (Matérn) it sets the redundancy
        of the lifted state but does not affect inference quality.
    kernel_factory_across:
        **Required** zero-arg callable returning a fresh :class:`BaseKernel`.
        The library does not pick a default kernel — pass e.g.
        ``lambda: MOSEKernel(num_regions=R, init_sigma=0.05)``.
    eps:
        Diagonal jitter added before the Cholesky in the kernel→LDS bridge.
    init_R:
        Initial diagonal observation noise.
    engine:
        Optional pre-configured :class:`KalmanEMEngine`. Defaults to one
        constructed with this class's ``DEFAULT_LR`` / ``DEFAULT_WD``.
    """

    DEFAULT_LR: float = 4e-2
    DEFAULT_WD: float = 1e-2

    # nn.Module __getattr__ widens attribute access; declare types so mypy
    # can see them on instances.
    delay: NoDelay

    def __init__(
        self,
        latent_spec: LatentSpec,
        y_dims: tuple[int, ...],
        T: int,
        *,
        kernel_factory_across: Callable[[], BaseKernel],
        lag_across: int = 5,
        eps: float = 1e-4,
        init_R: float = 0.1,
        engine: KalmanEMEngine | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if len(y_dims) != latent_spec.n_regions:
            raise ValueError(f"y_dims has {len(y_dims)} regions but latent_spec has {latent_spec.n_regions}")
        if T < 1:
            raise ValueError(f"T must be >= 1; got {T}")
        if lag_across < 1:
            raise ValueError(f"lag_across must be >= 1; got {lag_across}")
        _validate_shared_only(latent_spec)
        self._y_dims = tuple(y_dims)
        self._T = T
        self._lag_across = lag_across
        self._kernel_factory_across = kernel_factory_across
        self._eps = eps
        self._init_R = init_R
        self._engine_override = engine
        super().__init__(latent_spec=latent_spec, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Component construction
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        spec = self.latent_spec
        num_regions = spec.n_regions
        T = self._T
        lag_a = self._lag_across
        xdima = spec.n_across

        across_blocks: list[MarkovianGPLatent] = []
        for _ in range(xdima):
            kernel_a = self._kernel_factory_across()
            across_blocks.append(
                MarkovianGPLatent(
                    kernel=kernel_a,
                    lag=lag_a,
                    T=T,
                    delay=NoDelay(n_regions=num_regions, n_latent=1, dtype=self._dtype),
                    num_dim=num_regions,
                    cov_jitter=self._eps,
                )
            )

        n_obs_per_region = xdima
        n_observable = num_regions * n_obs_per_region

        # ``xdimw=0`` and ``lag_w=1`` are passed as placeholders; the helper
        # short-circuits the within section when xdimw is zero.
        observable_to_state = build_observable_to_state(
            xdima=xdima,
            xdimw=0,
            num_regions=num_regions,
            lag_a=lag_a,
            lag_w=1,
        )

        self.dynamics = BlockDiagonalDynamics(
            across_blocks,
            n_observable=n_observable,
            observable_to_state_indices=observable_to_state,
        )

        self.observation = MultiRegionLinearObservation(
            y_dims=self._y_dims,
            n_obs_per_region=n_obs_per_region,
            init_R=self._init_R,
            dtype=self._dtype,
        )

        # Representative kernel/delay attributes for BaseModel introspection.
        rep_block = across_blocks[0]
        rep_kernel = rep_block.kernel
        assert isinstance(rep_kernel, nn.Module)
        self.kernel = rep_kernel
        rep_delay = rep_block.delay
        assert isinstance(rep_delay, NoDelay)
        self.delay = rep_delay

        self.inference = self._engine_override or KalmanEMEngine(
            lr=self.DEFAULT_LR, weight_decay=self.DEFAULT_WD
        )

    # ------------------------------------------------------------------
    # Sampling and config
    # ------------------------------------------------------------------

    def sample(self, n_trials: int, T: int, *, seed: int | None = None) -> MultiRegionData:
        """Draw synthetic data from the model's prior.

        Forward-simulates the lifted LDS and projects through the
        observation. ``T`` must equal the configured trial length (the
        dynamics blocks are anchored to it).
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

    def to_config(self) -> dict[str, Any]:
        # NB: kernel factories are not JSON-serialisable. ``to_config``
        # records only the kernel-agnostic structural fields; callers
        # restoring via ``from_config`` must pass ``kernel_factory_across``
        # via ``**kwargs``.
        return {
            "n_across": self.latent_spec.n_across,
            "n_within": list(self.latent_spec.n_within),
            "y_dims": list(self._y_dims),
            "T": self._T,
            "lag_across": self._lag_across,
            "eps": self._eps,
            "init_R": self._init_R,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> GPFA:
        """Restore GPFA from config + caller-supplied kernel factory.

        ``kernel_factory_across`` is a required constructor param and
        cannot be serialised — caller must supply it via ``**kwargs``.
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
            eps=float(config["eps"]),
            init_R=float(config["init_R"]),
            **kwargs,
        )
