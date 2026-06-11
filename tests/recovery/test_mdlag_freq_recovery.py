"""Parameter-recovery test for MDLAG using the frequency-domain engine.

Mirrors :file:`test_mdlag_recovery.py` but swaps in :class:`VEMARDFreqEngine`.
The recovery thresholds are identical: at ``T = 30`` the circulant
approximation is decently tight (T·√γ ≈ 6) so all four assertions
should pass with similar margins.

Marked ``recovery`` (and ``slow``); user-only. Run with
``uv run pytest -m recovery -k mdlag_freq``.
"""

from __future__ import annotations

import math
from itertools import permutations

import pytest
import torch

from mbrila import MDLAG, LatentSpec, MOSEKernel, VEMARDFreqEngine
from mbrila.observations.ard import ARDObservation

pytestmark = [pytest.mark.recovery, pytest.mark.slow]


def _best_permutation_corr(
    truth_C: torch.Tensor,
    learned_C: torch.Tensor,
    active_cols: torch.Tensor,
) -> tuple[float, tuple[int, ...]]:
    K_truth = truth_C.shape[1]
    truth_z = truth_C - truth_C.mean(dim=0, keepdim=True)
    learned_z = learned_C - learned_C.mean(dim=0, keepdim=True)
    truth_norm = truth_z / truth_z.std(dim=0, keepdim=True).clamp(min=1e-8)
    learned_norm = learned_z / learned_z.std(dim=0, keepdim=True).clamp(min=1e-8)
    corr = (learned_norm[:, active_cols].T @ truth_norm) / truth_C.shape[0]
    corr_abs = corr.abs()
    best_score = -1.0
    best_perm: tuple[int, ...] = tuple(range(K_truth))
    for perm in permutations(range(active_cols.numel()), K_truth):
        score = float(corr_abs[list(perm), range(K_truth)].mean())
        if score > best_score:
            best_score = score
            best_perm = perm
    return best_score, best_perm


def test_mdlag_freq_recovers_latents_and_prunes_spurious() -> None:
    torch.manual_seed(0)
    R = 2
    y_dim_each = 8
    T = 30
    n_trials = 60
    truth_spec = LatentSpec(n_across=2, n_within=(0,) * R, selection="ard")
    truth = MDLAG(
        latent_spec=truth_spec,
        y_dims=(y_dim_each,) * R,
        T=T,
        kernel_factory_across=lambda: MOSEKernel(num_regions=R, init_sigma=0.04),
        max_delay=8.0,
        device="cpu",
        dtype=torch.float64,
    )

    with torch.no_grad():
        true_log_sigmas = torch.tensor([math.log(0.04), math.log(0.16)], dtype=torch.float64)
        for k in range(2):
            truth.dynamics.kernel_across[k].log_sigma.data.copy_(true_log_sigmas[k])
        delta_nonref = torch.tensor([[+2.0, -1.0]], dtype=torch.float64)
        beta_target = 2.0 * torch.atanh(delta_nonref / 8.0)
        truth.dynamics.delay.beta.copy_(beta_target)
        for r in range(R):
            col0 = torch.linspace(-1.0, 1.0, y_dim_each, dtype=torch.float64)
            sign_col1 = torch.tensor([1.0, -1.0] * (y_dim_each // 2), dtype=torch.float64)[:y_dim_each]
            col1 = sign_col1 * (1.0 + 0.2 * float(r))
            C_r = torch.stack([col0, col1], dim=1)
            C_r = C_r + 0.05 * torch.randn(y_dim_each, 2, dtype=torch.float64)
            truth.observation.C_means[r].copy_(C_r)
            truth.observation.C_covs[r].zero_()
            truth.observation.C_moments[r].copy_(C_r.unsqueeze(-1) * C_r.unsqueeze(-2))
        truth.observation.phi_mean.copy_(torch.full((R * y_dim_each,), 10.0, dtype=torch.float64))
        truth.observation.d_mean.copy_(0.1 * torch.randn(R * y_dim_each, dtype=torch.float64))

    data = truth.sample(n_trials=n_trials, T=T, seed=0)

    fit_spec = LatentSpec(n_across=3, n_within=(0,) * R, selection="ard")
    model = MDLAG(
        latent_spec=fit_spec,
        y_dims=(y_dim_each,) * R,
        T=T,
        kernel_factory_across=lambda: MOSEKernel(num_regions=R, init_sigma=0.08),
        max_delay=8.0,
        engine_override=VEMARDFreqEngine(max_lbfgs_iter=30, lbfgs_history=15),
        device="cpu",
        dtype=torch.float64,
    )
    model.initialize_from_data(data)
    result = model.fit(data, max_iter=150, tol=1e-7)

    # 1. ELBO monotone.
    diffs = [result.score_trace[i + 1] - result.score_trace[i] for i in range(len(result.score_trace) - 1)]
    assert min(diffs) >= -1e-5, (
        f"ELBO decreased; trace[:5]={result.score_trace[:5]} trace[-5:]={result.score_trace[-5:]}"
    )

    # 2. ARD pruning.
    obs = model.observation
    assert isinstance(obs, ARDObservation)
    alpha_mean = obs.alpha_mean
    max_alpha_per_col = alpha_mean.max(dim=0).values
    sorted_alpha, _ = max_alpha_per_col.sort(descending=True)
    ratio = (sorted_alpha[0] / sorted_alpha[-1]).item()
    assert ratio > 3.0, (
        f"ARD failed to identify a spurious column; max α = {max_alpha_per_col.tolist()}, ratio = {ratio:.2f}"
    )

    # 3. C recovery.
    active_cols = max_alpha_per_col.argsort()[:2]
    truth_C = torch.cat([c.detach() for c in truth.observation.C_means], dim=0)
    learned_C = torch.cat([c.detach() for c in obs.C_means], dim=0)
    mean_corr, _ = _best_permutation_corr(truth_C, learned_C, active_cols)
    assert mean_corr > 0.7, f"C correlation too low: {mean_corr:.3f}"

    # 4. Delay recovery.
    _, best_perm = _best_permutation_corr(truth_C, learned_C, active_cols)
    learned_delay = model.dynamics.delay.as_tensor()
    true_delay = truth.dynamics.delay.as_tensor()
    matched = learned_delay[:, active_cols[list(best_perm)]]
    err_plus = (matched - true_delay).abs()
    err_minus = (-matched - true_delay).abs()
    err = torch.minimum(err_plus, err_minus)
    delay_rmse = err.pow(2).mean().sqrt().item()
    assert delay_rmse < 0.7, f"delay RMSE too high: {delay_rmse:.3f}"
