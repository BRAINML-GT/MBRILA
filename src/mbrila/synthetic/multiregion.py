"""Synthetic multi-region SSM data generator with configurable delay
shapes, SNR, trial count, and within-region perturbations.

Used for benchmarking ADM (and the upcoming DLAG / mDLAG / MRM-GP
ports). The generator builds the ground truth on top of mbrila's own
ADM components — ``MOSEKernel``, ``TimeVaryingDelay``,
``BlockDiagonalDynamics``, ``MultiRegionLinearObservation`` — so the
recovery target is well-defined within the model class itself.

Public API
----------
- :class:`MultiRegionScenario`: dataclass describing the data-generating
  configuration. Supports 5 delay shapes (sin / box / gaussian / ramp /
  constant), per-region within-latent oscillation injection, and
  arbitrary (T, n_trials, n_across, n_within, SNR) settings.
- :func:`generate_multiregion_synthetic`: returns a
  :class:`SyntheticDataset` with the data + all latent / parameter
  ground truth needed for evaluation.

The legacy ``generate_adm_synthetic`` (used by the recovery test) is a
thin compatibility shim that wraps a default sin-shaped scenario.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor

from mbrila.core.data import MultiRegionData

DelayShape = Literal["sin", "box", "gaussian", "ramp", "constant"]


@dataclass(slots=True, frozen=True)
class MultiRegionScenario:
    """Configuration for a synthetic multi-region SSM dataset."""

    n_trials: int = 100
    T: int = 200
    y_dims: tuple[int, ...] = (30, 30, 30, 30, 30)

    # Latent structure
    n_across: int = 1
    n_within: int = 1
    lag_across: int = 5
    lag_within: int = 2
    sigma_across: float = 0.05
    sigma_within: float = 0.05

    # Delay configuration
    delay_shape: DelayShape = "sin"
    delay_amplitude: float = 1.0
    delay_box_start: float = 0.125  # fraction of T at which the box plateau begins
    delay_box_end: float = 0.5  # fraction of T at which the box plateau ends
    delay_gaussian_centre: float = 0.5  # fraction of T for gaussian envelope centre
    delay_gaussian_width: float = 0.125  # fraction of T for gaussian envelope std

    # Observation noise
    snr: float = 5.0

    # Optional within-region oscillation injection (additive deterministic
    # component on each within latent's observable g_r,k(t)). Length must
    # equal n_within or be empty (no oscillation). Each entry is a
    # frequency in cycles/trial.
    within_oscillation_freqs: tuple[float, ...] = field(default_factory=tuple)
    within_oscillation_amplitude: float = 0.5

    # Per-latent overrides: when provided, length must equal n_across.
    # Each across latent gets its own shape / amplitude / kernel
    # timescale so the multi-latent regime can carry genuinely
    # heterogeneous communication channels (the case where xcorr-init
    # confounds latent identity).
    per_latent_shapes: tuple[DelayShape, ...] | None = None
    per_latent_amplitudes: tuple[float, ...] | None = None
    per_latent_sigma_across: tuple[float, ...] | None = None

    # Region-wise delay heterogeneity in [0, 1].
    # 0 = all regions share same time-varying shape (just magnitude scaling)
    # 1 = each region gets distinct freq/phase/shape-param offset
    # Higher values = pairwise Δ(t) is genuinely different per pair →
    # time-varying δ becomes more identifiable.
    region_heterogeneity: float = 0.0

    # Reproducibility
    seed: int = 0
    dtype: torch.dtype = torch.float64
    device: str | torch.device = "cpu"


@dataclass(slots=True, frozen=True)
class SyntheticDataset:
    """Ground-truth payload produced by the generator."""

    data: MultiRegionData
    true_latents: Tensor  # (n_trials, T, n_obs_total) — observable latent g_t
    true_delay: Tensor  # (T, n_regions, n_across) — incl. zero ref column
    true_sigma_across: float
    true_sigma_within: float
    Cs: tuple[Tensor, ...]  # per-region emission, each (y_r, n_obs_per_region)
    diag_R: Tensor  # noqa: N815  (sum y_dims,) — capital R kept for math convention
    scenario: MultiRegionScenario | None = None


# ---------------------------------------------------------------------
# Delay-trajectory builders
# ---------------------------------------------------------------------
def _build_delay_trajectory(
    *,
    shape: DelayShape | tuple[DelayShape, ...],
    T: int,
    n_regions: int,
    n_across: int,
    amplitude: float | tuple[float, ...],
    box_start: float,
    box_end: float,
    gaussian_centre: float,
    gaussian_width: float,
    region_heterogeneity: float = 0.0,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> Tensor:
    """Return the raw (un-smoothed) delay tensor of shape (T, R-1, n_across).

    Each (region r > 0, across-latent k) pair gets a distinct phase / sign.
    Region 0 is the fixed reference (no entry in returned tensor).

    ``region_heterogeneity`` (0..1) controls how distinct each region's
    time-varying delay pattern is from the others:

    - **0.0** (synchronous): all regions share the same time-varying
      shape, only magnitude differs by ~15% across regions. Pairwise
      Δ(t) is the same shape for all pairs (just scaled). This is the
      partially pathological regime — time-varying δ is barely
      identifiable beyond magnitude-on-shared-shape.
    - **1.0** (fully independent): each region gets a distinct
      frequency / phase / shape-parameter offset, large enough that
      pairwise Δ(t) is genuinely different for each pair.

    Effect by shape:
      sin: per-region freq multiplier in [1, 1+0.4·het] and phase in [0, 1.5·het]
      box: per-region plateau shift up to ±0.15·het in T fraction
      gaussian: per-region centre shift up to ±0.15·het in T fraction
      ramp: per-region peak time shift up to ±0.15·het in T fraction
      constant: only magnitude differs (constant has no time variation)
    """
    if n_regions < 2:
        return torch.zeros(T, 0, n_across, dtype=dtype, device=device)
    if not 0.0 <= region_heterogeneity <= 1.0:
        raise ValueError(f"region_heterogeneity must be in [0, 1]; got {region_heterogeneity}")

    ts = torch.arange(T, dtype=dtype, device=device)
    out = torch.zeros(T, n_regions - 1, n_across, dtype=dtype, device=device)
    het = region_heterogeneity

    # Per-latent fractional position in [0, 1]: spreads diversity over K.
    def _k_frac(k_: int) -> float:
        return k_ / max(1, n_across - 1) if n_across > 1 else 0.5

    for r_idx in range(n_regions - 1):
        # Region-relative position in [0, 1] for scaling per-region perturbations.
        r_frac = r_idx / max(1, n_regions - 2) if n_regions > 2 else 0.5

        for k in range(n_across):
            k_frac = _k_frac(k)
            magnitude = 1.0 - (0.15 + 0.35 * het) * r_frac
            base_phase = k * 1.7  # latent-index phase
            phase_per_region = het * 1.5 * r_frac  # 0..1.5 rad per region
            shape_k_local = shape[k] if isinstance(shape, tuple) else shape
            amplitude_k = amplitude[k] if isinstance(amplitude, tuple) else amplitude

            if shape_k_local == "sin":
                # Per-latent freq spread 1 → 2; per-region het multiplier on top.
                freq_lat = 1.0 + 1.0 * k_frac
                freq_mult = freq_lat * (1.0 + 0.4 * het * r_frac)
                out[:, r_idx, k] = (
                    magnitude
                    * amplitude_k
                    * torch.sin(2 * math.pi * freq_mult * ts / T + base_phase + phase_per_region)
                )
            elif shape_k_local == "constant":
                out[:, r_idx, k] = magnitude * amplitude_k
            elif shape_k_local == "box":
                # Per-latent: spread plateau centre across [0.2, 0.8] of T.
                # Per-region: het-scaled shift on top.
                lat_shift = (k_frac - 0.5) * 0.6  # ±0.3
                reg_shift = het * 0.15 * (r_frac - 0.5)  # ±0.075
                width_default = box_end - box_start  # default plateau width
                centre_default = (box_start + box_end) / 2
                centre = centre_default + lat_shift + reg_shift
                start = int(max(0.0, centre - width_default / 2) * T)
                end = int(min(1.0, centre + width_default / 2) * T)
                if end > start:
                    out[start:end, r_idx, k] = magnitude * amplitude_k
            elif shape_k_local == "ramp":
                # Per-latent peak position spread across [0.25, 0.75] of T.
                lat_peak_frac = 0.25 + 0.5 * k_frac
                reg_shift = het * 0.15 * (r_frac - 0.5)
                mid = int((lat_peak_frac + reg_shift) * T)
                mid = max(1, min(T - 1, mid))
                tri = torch.zeros(T, dtype=dtype, device=device)
                tri[:mid] = torch.linspace(0, 1, mid, dtype=dtype, device=device)
                tri[mid:] = torch.linspace(1, 0, T - mid, dtype=dtype, device=device)
                out[:, r_idx, k] = magnitude * amplitude_k * tri
            elif shape_k_local == "gaussian":
                # Per-latent: centre spread across [0.2, 0.8] of T;
                #             width spread across [0.04, 0.12] of T.
                # Per-region: het-scaled centre shift.
                centre_lat = 0.2 + 0.6 * k_frac
                width_lat = 0.04 + 0.08 * k_frac
                centre = (centre_lat + het * 0.15 * (r_frac - 0.5)) * T
                std = width_lat * T
                bell = torch.exp(-0.5 * ((ts - centre) / std).pow(2))
                out[:, r_idx, k] = magnitude * amplitude_k * bell
            else:
                raise ValueError(f"unknown delay shape {shape_k_local!r}")
    return out


# ---------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------
def generate_multiregion_synthetic(scenario: MultiRegionScenario) -> SyntheticDataset:  # noqa: PLR0912
    """Generate a synthetic dataset under ``scenario``.

    The pipeline:
    1. Build an ADM model with the requested structure.
    2. Override its delay parameter with the chosen ground-truth shape.
    3. Override its emission ``C`` with random orthonormal columns and
       set ``diag(R)`` so per-neuron SNR ≈ ``scenario.snr``.
    4. Forward-simulate the lifted LDS once to produce latent trajectories.
    5. (Optionally) add a deterministic per-trial within-region
       oscillation directly on the observable latent ``g_t`` so the
       within channels carry non-Gaussian structure.
    6. Project through ``C`` and add Gaussian observation noise.
    """
    from mbrila.core.latent_spec import LatentSpec
    from mbrila.delays.time_varying import TimeVaryingDelay
    from mbrila.dynamics.markov_gp import BlockDiagonalDynamics, MarkovianGPLatent
    from mbrila.kernels.mose import MOSEKernel
    from mbrila.models.adm import ADM
    from mbrila.observations.multi_region import MultiRegionLinearObservation

    s = scenario
    if s.n_across < 1:
        raise ValueError(f"n_across must be >= 1; got {s.n_across}")
    if s.n_within < 0:
        raise ValueError(f"n_within must be >= 0; got {s.n_within}")
    if s.within_oscillation_freqs and len(s.within_oscillation_freqs) != s.n_within:
        raise ValueError(
            "within_oscillation_freqs must have length n_within "
            f"({s.n_within}); got {len(s.within_oscillation_freqs)}"
        )
    for fld_name, fld in (
        ("per_latent_shapes", s.per_latent_shapes),
        ("per_latent_amplitudes", s.per_latent_amplitudes),
        ("per_latent_sigma_across", s.per_latent_sigma_across),
    ):
        if fld is not None and len(fld) != s.n_across:
            raise ValueError(f"{fld_name} must have length n_across ({s.n_across}); got {len(fld)}")

    num_regions = len(s.y_dims)
    g_cpu = torch.Generator(device="cpu").manual_seed(s.seed)

    spec = LatentSpec(n_across=s.n_across, n_within=tuple([s.n_within] * num_regions))
    # Synthetic data is generated using ADM's MOSE-kernel structure
    # (across + within blocks), even though the data generator no longer
    # uses ADM's AR(P) forward simulation (see the GP-sampling note in
    # ``_sample_observable_via_gp``). The ADM here is just scaffolding —
    # it stores the kernel σ, time-varying delay, and emission C/d/R that
    # the sampler reads back.
    _sigma_across = float(s.sigma_across)
    _sigma_within = float(s.sigma_within)
    model = ADM(
        latent_spec=spec,
        y_dims=s.y_dims,
        T=s.T,
        kernel_factory_across=lambda: MOSEKernel(num_regions=num_regions, init_sigma=_sigma_across),
        kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=_sigma_within),
        lag_across=s.lag_across,
        lag_within=s.lag_within,
        delay_smoothing_sigma_across=_sigma_across,
        eps=1e-4,
        init_R=1e-6,  # placeholder
        device=s.device,
        dtype=s.dtype,
    )
    assert isinstance(model.dynamics, BlockDiagonalDynamics)
    assert isinstance(model.observation, MultiRegionLinearObservation)
    dyn = model.dynamics
    obs = model.observation

    # 0. (Optional) per-latent kernel timescale override on across blocks.
    if s.per_latent_sigma_across is not None:
        for k_block, block in enumerate(list(dyn.blocks)[: s.n_across]):
            assert isinstance(block, MarkovianGPLatent)
            kernel = block.kernel
            assert isinstance(kernel, MOSEKernel)
            with torch.no_grad():
                kernel.log_sigma.data.fill_(math.log(s.per_latent_sigma_across[k_block]))

    # 1. Override delay trajectory (supports per-latent shape / amplitude).
    shape_arg: DelayShape | tuple[DelayShape, ...] = (
        s.per_latent_shapes if s.per_latent_shapes is not None else s.delay_shape
    )
    amp_arg: float | tuple[float, ...] = (
        s.per_latent_amplitudes if s.per_latent_amplitudes is not None else s.delay_amplitude
    )
    raw_delay = _build_delay_trajectory(
        shape=shape_arg,
        T=s.T,
        n_regions=num_regions,
        n_across=s.n_across,
        amplitude=amp_arg,
        box_start=s.delay_box_start,
        box_end=s.delay_box_end,
        gaussian_centre=s.delay_gaussian_centre,
        gaussian_width=s.delay_gaussian_width,
        region_heterogeneity=s.region_heterogeneity,
        dtype=s.dtype,
        device=s.device,
    )
    if num_regions > 1:
        for k_block, block in enumerate(list(dyn.blocks)[: s.n_across]):
            assert isinstance(block, MarkovianGPLatent)
            assert isinstance(block.delay, TimeVaryingDelay)
            with torch.no_grad():
                block.delay.raw_delay.copy_(raw_delay[:, :, k_block : k_block + 1])

    # 2. Override emission matrices.
    Cs: list[Tensor] = []
    n_obs_per_region = s.n_across + s.n_within
    for r, y_r in enumerate(s.y_dims):
        raw = torch.randn(y_r, n_obs_per_region, generator=g_cpu, dtype=s.dtype).to(s.device)
        if y_r >= n_obs_per_region:
            q, _ = torch.linalg.qr(raw)
            C_r = q[:, :n_obs_per_region]
        else:
            C_r = raw / raw.norm(dim=1, keepdim=True).clamp(min=1e-6)
        with torch.no_grad():
            obs.Cs[r].copy_(C_r)
        Cs.append(C_r.detach().clone())

    # 3. Set R for the requested SNR.
    noise_var = 1.0 / max(s.snr, 1e-6)
    with torch.no_grad():
        obs.diag_R_param.data.fill_(noise_var)
        obs.d_param.data.zero_()

    # 4. Sample observable latents directly from the GP prior.
    # We DON'T use AR(P) forward simulation here — that path is numerically
    # unstable for fast-decaying kernels (large σ + small lag → AR
    # eigenvalues > 1 on some delay configurations → divergence to 1e30+).
    # Direct GP sampling is mathematically exact and stable for any σ.
    with torch.no_grad():
        # Read per-latent kernel σ from the model (was set in step 0 if
        # per_latent_sigma_across was given; else uniformly init_sigma_across).
        sigmas_across_k: list[float] = []
        for k in range(s.n_across):
            blk_k = dyn.blocks[k]
            assert isinstance(blk_k, MarkovianGPLatent)
            blk_kernel = blk_k.kernel
            assert isinstance(blk_kernel, MOSEKernel)
            sigmas_across_k.append(float(blk_kernel.sigma.item()))
        # Get smoothed delays (matches what model.dynamics would use internally).
        delays_full = torch.zeros(s.T, num_regions, s.n_across, dtype=s.dtype, device=s.device)
        for k in range(s.n_across):
            block = dyn.blocks[k]
            assert isinstance(block, MarkovianGPLatent) and block.delay is not None
            smoothed = block.delay.as_tensor(s.T)  # (T, R, 1)
            delays_full[:, :, k] = smoothed[:, :, 0]

        g_true = _sample_observable_via_gp(
            sigmas_across=sigmas_across_k,
            sigma_within=s.sigma_within,
            delays_across=delays_full,
            n_trials=s.n_trials,
            T=s.T,
            n_regions=num_regions,
            n_across=s.n_across,
            n_within=s.n_within,
            seed=s.seed + 1,
            dtype=s.dtype,
            device=s.device,
        )

        # 5. Inject within-region oscillation if requested. The within
        #    observables live at offsets `r * n_obs_per_region + s.n_across + k`.
        if s.within_oscillation_freqs:
            ts = torch.arange(s.T, dtype=s.dtype, device=s.device)
            for k, freq in enumerate(s.within_oscillation_freqs):
                for r in range(num_regions):
                    obs_idx = r * n_obs_per_region + s.n_across + k
                    phase = (r * 0.7 + k * 1.3) % (2 * math.pi)
                    osc = s.within_oscillation_amplitude * torch.sin(2 * math.pi * freq * ts / s.T + phase)
                    g_true[:, :, obs_idx] = g_true[:, :, obs_idx] + osc

        # 6. Project through C and add Gaussian noise.
        C_blk = obs.block_diag_C()
        y_clean = torch.einsum("ij,btj->bti", C_blk, g_true) + obs.offset()
        noise = (
            torch.randn(s.n_trials, s.T, y_clean.shape[-1], generator=g_cpu, dtype=s.dtype).to(s.device)
            * obs.diag_R().sqrt()
        )
        y = y_clean + noise

    sample_data = MultiRegionData(y=y, y_dims=s.y_dims, bin_width=1.0)

    # Recover the smoothed delay trajectory for evaluation.
    with torch.no_grad():
        delay_blocks: list[Tensor] = []
        blocks_list = list(dyn.blocks)
        for k in range(s.n_across):
            block = blocks_list[k]
            assert isinstance(block, MarkovianGPLatent)
            assert isinstance(block.delay, TimeVaryingDelay)
            delay_blocks.append(block.delay.as_tensor(s.T))  # (T, R, 1)
        true_delay = torch.cat(delay_blocks, dim=-1)  # (T, R, n_across)

    return SyntheticDataset(
        data=sample_data,
        true_latents=g_true.detach(),
        true_delay=true_delay.detach(),
        true_sigma_across=s.sigma_across,
        true_sigma_within=s.sigma_within,
        Cs=tuple(Cs),
        diag_R=obs.diag_R().detach().clone(),
        scenario=s,
    )


def _sample_observable_via_gp(
    *,
    sigmas_across: list[float],
    sigma_within: float,
    delays_across: Tensor,
    n_trials: int,
    T: int,
    n_regions: int,
    n_across: int,
    n_within: int,
    seed: int,
    dtype: torch.dtype,
    device: str | torch.device,
) -> Tensor:
    """Sample observable latents directly from the GP prior.

    Replaces AR(P) forward simulation, which is numerically unstable for
    fast-decaying kernels (large σ) — observed |g| → 1e30+ at σ=0.5,
    lag=2 because the AR(P) approximation has eigenvalues > 1 on some
    delay configurations.

    For each across latent k:
      g_k_underlying(t) ~ MVN(0, K_full(σ_k))  with K_full[i,j] = exp(-σ_k/2 (i-j)²)
      g_k_observable[r, t] = g_k_underlying(t − δ_{r,k}(t))   (linear interp)

    For each (region, within latent):
      g_within_{r,w}(t) ~ MVN(0, K_full(σ_within))  independent per (r, w)

    Returns ``(n_trials, T, n_regions × (n_across + n_within))`` with
    layout per region ``[across_0..K-1, within_0..n_within-1]`` (matches
    ``MultiRegionLinearObservation.n_obs_per_region`` convention).
    """
    g_cpu = torch.Generator(device="cpu").manual_seed(seed)
    n_obs_per_region = n_across + n_within
    out = torch.zeros(n_trials, T, n_regions * n_obs_per_region, dtype=dtype, device=device)

    ts = torch.arange(T, dtype=dtype, device=device)
    diff = ts.unsqueeze(0) - ts.unsqueeze(1)  # (T, T)
    eye_T = torch.eye(T, dtype=dtype, device=device)

    # Across latents: shared g per latent, per-region delayed view.
    for k in range(n_across):
        sigma_k = sigmas_across[k]
        K_full = torch.exp(-0.5 * sigma_k * diff.square()) + 1e-6 * eye_T
        L = torch.linalg.cholesky(K_full)  # (T, T)
        eps = torch.randn(n_trials, T, generator=g_cpu, dtype=dtype).to(device)
        g_underlying = eps @ L.T  # (n_trials, T)

        for r in range(n_regions):
            d_rk = delays_across[:, r, k]  # (T,)
            t_shifted = (ts - d_rk).clamp(0, T - 1)  # (T,)
            t_floor = t_shifted.floor().long()
            t_ceil = (t_floor + 1).clamp(max=T - 1)
            frac = t_shifted - t_floor.to(dtype)
            g_lo = g_underlying[:, t_floor]
            g_hi = g_underlying[:, t_ceil]
            g_obs_rk = (1 - frac) * g_lo + frac * g_hi  # (n_trials, T)
            out[:, :, r * n_obs_per_region + k] = g_obs_rk

    # Within latents: independent per (region, w_idx).
    if n_within > 0:
        K_within = torch.exp(-0.5 * sigma_within * diff.square()) + 1e-6 * eye_T
        L_within = torch.linalg.cholesky(K_within)  # (T, T)
        for r in range(n_regions):
            for w in range(n_within):
                eps = torch.randn(n_trials, T, generator=g_cpu, dtype=dtype).to(device)
                g_within_rw = eps @ L_within.T  # (n_trials, T)
                out[:, :, r * n_obs_per_region + n_across + w] = g_within_rw

    return out
