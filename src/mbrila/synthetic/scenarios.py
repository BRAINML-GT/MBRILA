"""Pre-defined scenarios for benchmarking multi-region SSM methods.

Each preset is a :class:`MultiRegionScenario` covering one of the
challenge axes that distinguishes a generative model (ADM / DLAG / etc.)
from a non-parametric estimator (cross-correlation, simple FA).

The 5 presets exposed here mirror the ablation axes worth showing in a
paper:

1. :data:`EASY` — high SNR, many trials, sharp delay structure.
   Cross-correlation already does well; ADM should match.
2. :data:`LOW_SNR` — low SNR. Cross-correlation peak is buried in
   noise; ADM's joint smoother + multi-trial pooling should still
   recover delay.
3. :data:`FEW_TRIALS` — few trials. Per-window xcorr estimates have
   large variance; ADM benefits from explicit prior + joint fit.
4. :data:`SMOOTH_DELAY` — Gaussian-bell delay envelope.
   Cross-correlation has no sharp peak; ADM's GP-smoothed parameter
   recovers a continuous trajectory.
5. :data:`MULTI_LATENT` — two across-latent factors.
   Cross-correlation cannot disentangle which lag belongs to which
   latent; ADM separates them via the joint observation model.
6. :data:`COMPLEX_WITHIN` — within-region oscillations.
   Cross-correlation conflates within and across signals; ADM's
   structured latent decomposition disambiguates.

Each scenario is also constructible via simple keyword overrides on
:class:`MultiRegionScenario`, so the presets are starting points rather
than a fixed catalog.
"""

from __future__ import annotations

from mbrila.synthetic.multiregion import MultiRegionScenario

# -----------------------------------------------------------------
# 1. EASY  — matches the ADM-paper sim_data.mat regime
# -----------------------------------------------------------------
EASY = MultiRegionScenario(
    n_trials=100,
    T=200,
    y_dims=(30, 30, 30, 30, 30),
    n_across=1,
    n_within=1,
    delay_shape="box",
    delay_amplitude=3.0,
    snr=5.0,
    seed=0,
)

# -----------------------------------------------------------------
# 2. LOW_SNR — observation noise dominates the signal
# -----------------------------------------------------------------
LOW_SNR = MultiRegionScenario(
    n_trials=100,
    T=200,
    y_dims=(30, 30, 30, 30, 30),
    n_across=1,
    n_within=1,
    delay_shape="box",
    delay_amplitude=3.0,
    snr=0.3,
    seed=0,
)

# -----------------------------------------------------------------
# 3. FEW_TRIALS — only a handful of trials available
# -----------------------------------------------------------------
FEW_TRIALS = MultiRegionScenario(
    n_trials=10,
    T=200,
    y_dims=(30, 30, 30, 30, 30),
    n_across=1,
    n_within=1,
    delay_shape="box",
    delay_amplitude=3.0,
    snr=5.0,
    seed=0,
)

# -----------------------------------------------------------------
# 4. SMOOTH_DELAY — continuous gaussian envelope
# -----------------------------------------------------------------
SMOOTH_DELAY = MultiRegionScenario(
    n_trials=100,
    T=200,
    y_dims=(30, 30, 30, 30, 30),
    n_across=1,
    n_within=1,
    delay_shape="gaussian",
    delay_amplitude=3.0,
    delay_gaussian_centre=0.5,
    delay_gaussian_width=0.15,
    snr=5.0,
    seed=0,
)

# -----------------------------------------------------------------
# 5. MULTI_LATENT — multiple across-region factors with distinct delays
# -----------------------------------------------------------------
MULTI_LATENT = MultiRegionScenario(
    n_trials=100,
    T=200,
    y_dims=(30, 30, 30, 30, 30),
    n_across=2,
    n_within=1,
    delay_shape="sin",
    delay_amplitude=3.0,
    snr=5.0,
    seed=0,
)

# -----------------------------------------------------------------
# 6. COMPLEX_WITHIN — within-region oscillations confound xcorr
# -----------------------------------------------------------------
COMPLEX_WITHIN = MultiRegionScenario(
    n_trials=100,
    T=200,
    y_dims=(30, 30, 30, 30, 30),
    n_across=1,
    n_within=2,
    delay_shape="box",
    delay_amplitude=3.0,
    within_oscillation_freqs=(2.0, 5.0),  # 2 cycles/trial, 5 cycles/trial
    within_oscillation_amplitude=0.7,
    snr=5.0,
    seed=0,
)


# -----------------------------------------------------------------
# 7. HARD_MULTI_LATENT — K=5 across latents over R=8 regions, each with
#    a distinct timescale, delay shape, and amplitude. xcorr-init
#    confounds latent identity (CCA latent ordering ≠ true ordering),
#    so the per-latent windowed xcorr places each row's lag on the
#    wrong latent. ADM's joint emission + smoother is the only path
#    that can disentangle.
# -----------------------------------------------------------------
HARD_MULTI_LATENT = MultiRegionScenario(
    n_trials=50,
    T=200,
    y_dims=(30, 30, 30, 30, 30, 30),  # R=6
    n_across=4,
    n_within=1,
    lag_across=3,  # smaller lifted-state lag → tractable D
    lag_within=2,
    delay_shape="sin",  # fallback; per_latent_shapes overrides
    delay_amplitude=2.0,
    snr=3.0,
    # Each across latent: different shape, amplitude, timescale.
    per_latent_shapes=("sin", "box", "gaussian", "ramp"),
    per_latent_amplitudes=(2.0, 2.5, 1.5, 2.0),
    per_latent_sigma_across=(0.03, 0.06, 0.1, 0.04),
    seed=0,
)


PRESETS: dict[str, MultiRegionScenario] = {
    "easy": EASY,
    "low_snr": LOW_SNR,
    "few_trials": FEW_TRIALS,
    "smooth_delay": SMOOTH_DELAY,
    "multi_latent": MULTI_LATENT,
    "complex_within": COMPLEX_WITHIN,
    "hard_multi_latent": HARD_MULTI_LATENT,
}
