"""Naive linear dynamical system preset.

A learnable linear-Gaussian SSM with no GP / Markov-kernel structure.
The prior on ``x_t`` is

    x_{t+1} = A · x_t + ε,   ε ~ N(0, Q),
    x_0 ~ N(0, I).

``A`` and ``Q`` are free :class:`nn.Parameter`s in
:class:`~mbrila.dynamics.free_lds.FreeLDSLatent` (no temporal smoothness
prior beyond what ``A`` itself encodes). The observation is a standard
multi-region linear-Gaussian emission with one ``C_r`` per region, all
acting on the same shared ``x_t``. Inference is the same
:class:`KalmanEMEngine` ADM/GPFA use — naive SSM differs only in the
*dynamics* axis.

Use cases:

- Baseline for "GP prior actually helps" claims (paired with ADM/DLAG/GPFA
  on the same dataset and the same emission spec).
- Lightweight multi-region LDS when you don't have a reason to commit to
  a particular GP timescale or kernel family.
- Pure single-region LDS (``n_regions = 1``) for classical Kalman-EM
  workflows.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mbrila.core.base_model import BaseModel
from mbrila.core.data import MultiRegionData
from mbrila.core.latent_spec import LatentSpec
from mbrila.delays.none import NoDelay
from mbrila.dynamics.free_lds import FreeLDSLatent
from mbrila.inference.kalman_em import KalmanEMEngine
from mbrila.observations.multi_region import MultiRegionLinearObservation


class LDS(BaseModel):
    """Naive multi-region linear dynamical system.

    Parameters
    ----------
    n_latent:
        Dimensionality of the shared latent state ``x_t``.
    y_dims:
        Per-region neuron counts. ``len(y_dims)`` is the number of
        regions; use ``y_dims=(N,)`` for the classical single-region LDS.
    T:
        Trial length in bins.
    init_R:
        Initial diagonal observation noise variance.
    init_Q_diag:
        Initial scalar value on the diagonal of ``Q`` (``Q = q · I``
        initially).
    init_A:
        Optional ``(n_latent, n_latent)`` tensor for the initial transition
        matrix. Defaults to ``0.95 · I`` (contractive — keeps the Kalman
        filter Cholesky well-conditioned on a freshly-built model).
    engine:
        Optional pre-configured :class:`KalmanEMEngine`.
    """

    DEFAULT_LR: float = 4e-2
    DEFAULT_WD: float = 1e-2

    # mypy: nn.Module __getattr__ widens these; declare so the class is typed.
    delay: NoDelay

    def __init__(
        self,
        n_latent: int,
        y_dims: tuple[int, ...],
        T: int,
        *,
        init_R: float = 0.1,
        init_Q_diag: float = 0.1,
        init_A: torch.Tensor | None = None,
        engine: KalmanEMEngine | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if n_latent < 1:
            raise ValueError(f"n_latent must be >= 1; got {n_latent}")
        if not y_dims or any(d <= 0 for d in y_dims):
            raise ValueError(f"y_dims must be a non-empty tuple of positive ints; got {y_dims}")
        if T < 1:
            raise ValueError(f"T must be >= 1; got {T}")
        self._n_latent = int(n_latent)
        self._y_dims = tuple(int(d) for d in y_dims)
        self._T = int(T)
        self._init_R = float(init_R)
        self._init_Q_diag = float(init_Q_diag)
        self._init_A = init_A
        self._engine_override = engine

        # Reuse LatentSpec to carry the structural info: n_across encodes the
        # shared flat latent count; n_within is identically 0 across regions
        # (LDS has no per-region "private" latents in this formulation).
        n_regions = len(self._y_dims)
        spec = LatentSpec(
            n_across=self._n_latent,
            n_within=tuple(0 for _ in range(n_regions)),
        )
        super().__init__(latent_spec=spec, device=device, dtype=dtype)

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------

    def _init_components(self) -> None:
        n_regions = self.latent_spec.n_regions

        self.dynamics = FreeLDSLatent(
            n_latent=self._n_latent,
            n_regions=n_regions,
            T=self._T,
            init_A=self._init_A,
            init_Q_diag=self._init_Q_diag,
            dtype=self._dtype,
        )

        # Each region's emission sees the full shared latent — n_obs_per_region
        # equals the latent dim (the per-region observable is just x_t,
        # replicated by the H_select inside FreeLDSLatent).
        self.observation = MultiRegionLinearObservation(
            y_dims=self._y_dims,
            n_obs_per_region=self._n_latent,
            init_R=self._init_R,
            dtype=self._dtype,
        )

        # LDS has no kernel and no delay; populate placeholders so
        # BaseModel's introspection (capability aggregation, save/load
        # state_dict scoping) sees fully-typed Modules.
        self.kernel = nn.Identity()
        self.delay = NoDelay(n_regions=n_regions, n_latent=self._n_latent, dtype=self._dtype)

        self.inference = self._engine_override or KalmanEMEngine(
            lr=self.DEFAULT_LR, weight_decay=self.DEFAULT_WD
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n_trials: int, T: int, *, seed: int | None = None) -> MultiRegionData:
        """Forward-simulate the LDS and emit through the per-region ``C_r``.

        ``T`` must equal the model's configured trial length.
        """
        if T != self._T:
            raise ValueError(f"sample T must match model T={self._T}; got {T}")
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1; got {n_trials}")

        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)

        assert isinstance(self.dynamics, FreeLDSLatent)
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
            "n_latent": self._n_latent,
            "y_dims": list(self._y_dims),
            "T": self._T,
            "init_R": self._init_R,
            "init_Q_diag": self._init_Q_diag,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> LDS:
        return cls(
            n_latent=int(config["n_latent"]),
            y_dims=tuple(int(x) for x in config["y_dims"]),
            T=int(config["T"]),
            init_R=float(config["init_R"]),
            init_Q_diag=float(config.get("init_Q_diag", 0.1)),
            **kwargs,
        )
