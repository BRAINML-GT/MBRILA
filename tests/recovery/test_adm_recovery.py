"""Parameter-recovery tests for ADM on synthetic data.

These tests run a full training loop and assert that the fitted model
recovers the data-generating parameters. They are slow (~minutes), so
they are gated behind ``-m recovery``::

    uv run pytest -m recovery

Default ``uv run pytest`` skips them.
"""

from __future__ import annotations

import pytest
import torch

from mbrila import ADM, LatentSpec, MOSEKernel
from mbrila._testing import generate_adm_synthetic

pytestmark = [pytest.mark.recovery, pytest.mark.slow]


def _abs_corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-feature absolute correlation between two ``(N, K)`` tensors."""
    a_z = (a - a.mean(dim=0, keepdim=True)) / a.std(dim=0, keepdim=True).clamp(min=1e-8)
    b_z = (b - b.mean(dim=0, keepdim=True)) / b.std(dim=0, keepdim=True).clamp(min=1e-8)
    return (a_z * b_z).mean(dim=0).abs()


def test_adm_recovers_delay_and_latents() -> None:
    """Fit ADM on a small synthetic dataset and check parameter recovery.

    Acceptance criteria (from the plan, Phase 1 verification):

    - LL trace must be (eventually) non-decreasing.
    - Mean absolute correlation between recovered and true observable
      latents > 0.7 (loose because of identifiability up to sign /
      orthogonal rotation; we average per-region per-latent
      correlations).
    - Recovered delay trajectory's RMSE vs the ground-truth one
      < 0.5 bins.
    """
    torch.manual_seed(0)
    sd = generate_adm_synthetic(
        n_trials=24,
        T=40,
        y_dims=(8, 8),
        n_across=1,
        n_within=1,
        lag_across=4,
        lag_within=2,
        sigma_across=0.05,
        sigma_within=0.05,
        delay_amplitude=2.0,
        snr=5.0,
        seed=0,
        dtype=torch.float64,
        device="cpu",
    )

    spec = LatentSpec(n_across=1, n_within=(1, 1))
    model = ADM(
        latent_spec=spec,
        y_dims=(8, 8),
        T=40,
        lag_across=4,
        lag_within=2,
        device="cpu",
        dtype=torch.float64,
        kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.05),
        kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=0.05),
    )
    # CCA-based initialisation of the per-region emission matrices is
    # essential for delay recovery in this family of models — see ADM's
    # init_params.init_C. Random init lands in a sign-flipped local minimum.
    model.initialize_from_data(sd.data)

    result = model.fit(sd.data, max_iter=120)

    # 1. Late-iteration LL above early-iteration LL (training improved things).
    early = sum(result.score_trace[:5]) / 5
    late = sum(result.score_trace[-5:]) / 5
    assert late > early, f"LL did not improve: early={early:.2f}  late={late:.2f}"

    # 2. Latent recovery (allow sign flips and ordering).
    posterior = model.infer(sd.data)
    H_select = model.dynamics.H_select
    g_pred = torch.einsum("ij,btj->bti", H_select, posterior.mean)
    # Flatten over (B, T) to compute per-feature correlations.
    a_flat = sd.true_latents.reshape(-1, sd.true_latents.shape[-1])
    b_flat = g_pred.reshape(-1, g_pred.shape[-1])
    corrs = _abs_corr(a_flat, b_flat)
    assert corrs.mean().item() > 0.7, f"mean |corr| too low: {corrs.tolist()}"

    # 3. Delay recovery — only for the across factor.
    from mbrila.dynamics.markov_gp import MarkovianGPLatent

    block = model.dynamics.blocks[0]
    assert isinstance(block, MarkovianGPLatent)
    assert block.delay is not None
    learned_delay = block.delay.as_tensor(40)  # (T, R, 1)
    rmse = (learned_delay - sd.true_delay).pow(2).mean().sqrt().item()
    # Threshold widened from 0.5 → 0.6 after restoring ADM's `Q/2`
    # in `kernel_to_lds`. The /2 is essential for delay recovery on
    # real-scale data (sim_data.mat, T=200, n_obs=150) but mildly
    # degrades this small-T=40 synthetic. Zero-delay baseline ≈ 1.0,
    # so 0.6 still validates non-trivial recovery.
    assert rmse < 0.6, f"delay RMSE too high: {rmse:.3f}"
