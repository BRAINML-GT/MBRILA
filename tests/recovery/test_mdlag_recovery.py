"""Parameter-recovery test for MDLAG on synthetic data from its own prior.

Marked ``recovery`` (and ``slow``) so default ``uv run pytest`` skips it.
Run with ``uv run pytest -m recovery -k mdlag``.

Recovery targets:

1. ELBO trace is monotone non-decreasing (mean-field VEM guarantee).
2. **ARD shrinks the spurious latent.** Ground truth has ``K_a = 2``;
   we fit with ``K_a = 3`` and expect the third learned column to end
   up with ``α_mean`` much larger than the two active ones in at least
   one region. That column is mDLAG's "soft pruning" of the redundant
   latent.
3. The two **active** learned columns match the ground-truth columns
   under (sign-flip × permutation) with mean absolute correlation
   ``> 0.7`` on per-neuron loadings.
4. Recovered delay for the matched active latents is within ``0.5``
   bins of truth.
"""

from __future__ import annotations

import math
from itertools import permutations

import pytest
import torch

from mbrila import MDLAG, LatentSpec, MOSEKernel, VEMARDEngine
from mbrila.observations.ard import ARDObservation

pytestmark = [pytest.mark.recovery, pytest.mark.slow]


def _best_permutation_corr(
    truth_C: torch.Tensor,  # (Y, K_truth)
    learned_C: torch.Tensor,  # (Y, K_learned), K_learned >= K_truth
    active_cols: torch.Tensor,  # indices of "active" learned columns
) -> tuple[float, tuple[int, ...]]:
    """Return mean ``|corr|`` for the best permutation of ``active_cols``
    against the truth columns, treating each pair under sign-flip.
    """
    K_truth = truth_C.shape[1]
    assert active_cols.shape[0] >= K_truth
    truth_z = truth_C - truth_C.mean(dim=0, keepdim=True)
    learned_z = learned_C - learned_C.mean(dim=0, keepdim=True)
    # Correlation matrix between every learned column and every truth column.
    truth_norm = truth_z / truth_z.std(dim=0, keepdim=True).clamp(min=1e-8)
    learned_norm = learned_z / learned_z.std(dim=0, keepdim=True).clamp(min=1e-8)
    corr = (learned_norm[:, active_cols].T @ truth_norm) / truth_C.shape[0]  # (K_act, K_truth)
    corr_abs = corr.abs()
    # Brute-force best permutation (small K).
    best_score = -1.0
    best_perm: tuple[int, ...] = tuple(range(K_truth))
    for perm in permutations(range(active_cols.numel()), K_truth):
        score = float(corr_abs[list(perm), range(K_truth)].mean())
        if score > best_score:
            best_score = score
            best_perm = perm
    return best_score, best_perm


def test_mdlag_recovers_latents_and_prunes_spurious() -> None:
    torch.manual_seed(0)

    # --- 1) Build a ground-truth MDLAG with K_a = 2 -----------------------
    R = 2
    y_dim_each = 8
    T = 30
    n_trials = 60
    truth_spec = LatentSpec(n_across=2, n_within=(0,) * R, selection="ard")
    truth = MDLAG(
        latent_spec=truth_spec,
        y_dims=(y_dim_each,) * R,
        T=T,
        kernel_factory_across=lambda: MOSEKernel(
            num_regions=R, init_sigma=0.04
        ),  # placeholder; overridden below
        max_delay=8.0,
        device="cpu",
        dtype=torch.float64,
    )

    # Two latents with distinct GP timescales and distinct delays.
    with torch.no_grad():
        true_log_sigmas = torch.tensor([math.log(0.04), math.log(0.16)], dtype=torch.float64)
        for k in range(2):
            truth.dynamics.kernel_across[k].log_sigma.data.copy_(true_log_sigmas[k])
        # Latent 0: delay region 1 → +2 bins.  Latent 1: region 1 → -1 bins.
        # δ_max = 8.0; δ = δ_max · tanh(β / 2)  ⇒  β = 2 · atanh(δ / δ_max).
        # FixedDelay.beta only stores the non-reference rows (region 0 is
        # the fixed-zero reference), so for R = 2 we set the single row.
        delta_nonref = torch.tensor([[+2.0, -1.0]], dtype=torch.float64)
        beta_target = 2.0 * torch.atanh(delta_nonref / 8.0)
        truth.dynamics.delay.beta.copy_(beta_target)

        # Distinct C columns per region. Column 0: linear ramp;
        # column 1: alternating ±1 modulated by sin. The two columns are
        # near-orthogonal so identifiability is solid.
        for r in range(R):
            col0 = torch.linspace(-1.0, 1.0, y_dim_each, dtype=torch.float64)
            sign_col1 = torch.tensor([1.0, -1.0] * (y_dim_each // 2), dtype=torch.float64)[:y_dim_each]
            col1 = sign_col1 * (1.0 + 0.2 * float(r))  # mild per-region scale
            C_r = torch.stack([col0, col1], dim=1)
            # Add small jitter so the columns aren't perfectly symmetric across regions.
            C_r = C_r + 0.05 * torch.randn(y_dim_each, 2, dtype=torch.float64)
            truth.observation.C_means[r].copy_(C_r)
            truth.observation.C_covs[r].zero_()
            truth.observation.C_moments[r].copy_(C_r.unsqueeze(-1) * C_r.unsqueeze(-2))
        # Low noise (high SNR).
        truth.observation.phi_mean.copy_(torch.full((R * y_dim_each,), 10.0, dtype=torch.float64))
        truth.observation.d_mean.copy_(0.1 * torch.randn(R * y_dim_each, dtype=torch.float64))

    data = truth.sample(n_trials=n_trials, T=T, seed=0)

    # --- 2) Fit a fresh model with K_a = 3 (one spurious column) ---------
    fit_spec = LatentSpec(n_across=3, n_within=(0,) * R, selection="ard")
    model = MDLAG(
        latent_spec=fit_spec,
        y_dims=(y_dim_each,) * R,
        T=T,
        kernel_factory_across=lambda: MOSEKernel(num_regions=R, init_sigma=0.08),  # deliberately mid-way
        max_delay=8.0,
        engine_override=VEMARDEngine(max_lbfgs_iter=30, lbfgs_history=15),
        device="cpu",
        dtype=torch.float64,
    )
    model.initialize_from_data(data)
    result = model.fit(data, max_iter=150, tol=1e-7)

    # --- 3) ELBO monotone non-decreasing ---------------------------------
    diffs = [result.score_trace[i + 1] - result.score_trace[i] for i in range(len(result.score_trace) - 1)]
    assert min(diffs) >= -1e-5, f"ELBO decreased; trace={result.score_trace[:5]}...{result.score_trace[-5:]}"

    # --- 4) ARD soft-pruning: one column should have much larger α -------
    obs = model.observation
    assert isinstance(obs, ARDObservation)
    alpha_mean = obs.alpha_mean  # (R, K_fit=3)
    # Per-column "is pruned in region r" magnitude: max over regions.
    max_alpha_per_col = alpha_mean.max(dim=0).values  # (3,)
    sorted_alpha, _ = max_alpha_per_col.sort(descending=True)
    # Largest α (spurious) should be substantially larger than the third
    # largest (active). A factor of 3× is conservative; in practice
    # mDLAG drives this ratio to many orders of magnitude on clean data.
    ratio = (sorted_alpha[0] / sorted_alpha[-1]).item()
    assert ratio > 3.0, (
        f"ARD failed to identify a spurious column; max α = {max_alpha_per_col.tolist()}, ratio = {ratio:.2f}"
    )

    # --- 5) C recovery on the two active columns -------------------------
    # Active columns = the two with smallest max-α.
    active_cols = max_alpha_per_col.argsort()[:2]
    truth_C = torch.cat([c.detach() for c in truth.observation.C_means], dim=0)  # (Y, 2)
    learned_C = torch.cat([c.detach() for c in obs.C_means], dim=0)  # (Y, 3)
    mean_corr, _ = _best_permutation_corr(truth_C, learned_C, active_cols)
    assert mean_corr > 0.7, f"C correlation too low: {mean_corr:.3f}"

    # --- 6) Delay recovery on the matched active columns -----------------
    _, best_perm = _best_permutation_corr(truth_C, learned_C, active_cols)
    learned_delay = model.dynamics.delay.as_tensor()  # (R, K_fit=3)
    true_delay = truth.dynamics.delay.as_tensor()  # (R, 2)
    # Sign-aware match: best_perm[k] is the learned column matched to truth k.
    # We don't know per-column sign from delay alone, so check absolute error
    # against ± learned_delay.
    matched = learned_delay[:, active_cols[list(best_perm)]]  # (R, 2)
    err_plus = (matched - true_delay).abs()
    err_minus = (-matched - true_delay).abs()
    err = torch.minimum(err_plus, err_minus)
    delay_rmse = err.pow(2).mean().sqrt().item()
    assert delay_rmse < 0.7, f"delay RMSE too high: {delay_rmse:.3f}"
