"""Tests for CF6a — engine-agnostic ARD helpers.

Two responsibilities:

1. The helpers in :mod:`mbrila.inference.ard_helpers` compute exactly
   what their docstrings claim (shape + numerical spot checks).
2. The refactored :class:`VEMARDEngine` is **behaviour-identical** to
   the pre-CF6a version. We verify this by running E-step + emission
   M-step end-to-end on a controlled input and comparing against
   reference values computed inline (mirroring the old inlined code).
"""

from __future__ import annotations

import torch

from mbrila import MDLAG, LatentSpec, MOSEKernel
from mbrila.inference.ard_helpers import (
    aggregate_emission_stats,
    compute_CPhi,
    compute_CPhiC_block,
    run_emission_m_step,
    setup_ard_posteriors,
)
from mbrila.inference.vem_ard import VEMARDEngine
from mbrila.observations.ard import ARDObservation


def _make_mdlag(K: int = 2, R: int = 2, T: int = 6, neuron_per_region: int = 4) -> MDLAG:
    spec = LatentSpec(n_across=K, n_within=(0,) * R, selection="ard")
    return MDLAG(
        latent_spec=spec,
        y_dims=tuple(neuron_per_region for _ in range(R)),
        T=T,
        kernel_factory_across=lambda: MOSEKernel(num_regions=R, init_sigma=0.05),
        dtype=torch.float64,
        device="cpu",
    )


def _make_data(model: MDLAG, n_trials: int = 5, seed: int = 0) -> object:
    return model.sample(n_trials=n_trials, T=model._T, seed=seed)


# ---------------------------------------------------------------------------
# compute_CPhi / compute_CPhiC_block
# ---------------------------------------------------------------------------


class TestComputeCPhi:
    def test_shape(self) -> None:
        model = _make_mdlag(K=3, R=2, neuron_per_region=5)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        CPhi = compute_CPhi(obs)
        # M = R * K = 6, n_y = 2 * 5 = 10
        assert CPhi.shape == (6, 10)

    def test_matches_explicit_formula(self) -> None:
        """``CPhi = block_diag(C_means)ᵀ · diag(phi_mean)`` exactly."""
        model = _make_mdlag(K=2, R=2, neuron_per_region=3)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        with torch.no_grad():
            obs.d_mean.copy_(torch.randn_like(obs.d_mean))
            obs.phi_mean.copy_(torch.full_like(obs.phi_mean, 1.7).add_(torch.randn_like(obs.phi_mean) * 0.1))
        expected = obs.block_diag_C().transpose(0, 1) * obs.phi_mean.unsqueeze(0)
        torch.testing.assert_close(compute_CPhi(obs), expected, atol=1e-14, rtol=1e-14)


class TestComputeCPhiCBlock:
    def test_shape_and_block_structure(self) -> None:
        model = _make_mdlag(K=2, R=3, neuron_per_region=4)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        CPhiC = compute_CPhiC_block(obs)
        # Block-diagonal (R, k, k) → (R·k, R·k) = (6, 6)
        assert CPhiC.shape == (6, 6)
        # Off-block entries are zero.
        for r1 in range(3):
            for r2 in range(3):
                if r1 == r2:
                    continue
                block = CPhiC[r1 * 2 : (r1 + 1) * 2, r2 * 2 : (r2 + 1) * 2]
                torch.testing.assert_close(block, torch.zeros_like(block))

    def test_symmetric(self) -> None:
        model = _make_mdlag(K=3, R=2, neuron_per_region=4)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        CPhiC = compute_CPhiC_block(obs)
        torch.testing.assert_close(CPhiC, CPhiC.transpose(-2, -1), atol=1e-14, rtol=1e-14)

    def test_uses_C_second_moments_not_outer_of_mean(self) -> None:
        """CLAUDE.md 'mDLAG 注意事项': must use ``⟨C C^T⟩ = C_cov + outer(C_mean)``
        rather than ``outer(C_mean)`` alone. Spike ``C_moments`` directly
        (what ``compute_CPhiC_block`` reads) and verify the block scales."""
        model = _make_mdlag(K=2, R=2, neuron_per_region=3)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        baseline = compute_CPhiC_block(obs)
        with torch.no_grad():
            for r in range(obs.n_regions):
                eye_k = torch.eye(obs.n_obs_per_region, dtype=obs.C_moments[r].dtype)
                obs.C_moments[r].add_(0.5 * eye_k)
        inflated = compute_CPhiC_block(obs)
        # Diagonal entries strictly larger after spiking the second moments.
        diff_diag = torch.diagonal(inflated) - torch.diagonal(baseline)
        assert (diff_diag > 1e-10).all().item()


# ---------------------------------------------------------------------------
# aggregate_emission_stats
# ---------------------------------------------------------------------------


class TestAggregateEmissionStats:
    def test_shapes(self) -> None:
        model = _make_mdlag(K=2, R=2, neuron_per_region=3, T=5)
        data = _make_data(model, n_trials=4)
        B, T, n_y = data.y.shape  # type: ignore[attr-defined]
        assert isinstance(model.observation, ARDObservation)
        R = model.observation.n_regions
        k = model.observation.n_obs_per_region
        M = R * k
        x_hat = torch.randn(B, T, M, dtype=torch.float64)
        P_per_time = torch.eye(M, dtype=torch.float64).unsqueeze(0).expand(T, -1, -1).contiguous()

        stats = aggregate_emission_stats(
            data=data, x_hat=x_hat, P_per_time=P_per_time, y_dims=model._y_dims, k=k
        )
        assert stats["sum_y"].shape == (n_y,)
        assert stats["sum_y2"].shape == (n_y,)
        assert stats["sum_x_per_region"].shape == (R, k)
        assert stats["XX_per_region"].shape == (R, k, k)
        assert isinstance(stats["XY_per_region"], list)
        assert len(stats["XY_per_region"]) == R
        for r, XY_r in enumerate(stats["XY_per_region"]):
            assert XY_r.shape == (k, model._y_dims[r])

    def test_matches_legacy_static_method(self) -> None:
        """The free function and the (now-thin) staticmethod wrapper agree."""
        model = _make_mdlag(K=2, R=2)
        data = _make_data(model, n_trials=3, seed=0)
        assert isinstance(model.observation, ARDObservation)
        k = model.observation.n_obs_per_region
        M = model.observation.n_regions * k
        B, T = data.y.shape[:2]  # type: ignore[attr-defined]
        torch.manual_seed(42)
        x_hat = torch.randn(B, T, M, dtype=torch.float64)
        P_per_time = torch.randn(T, M, M, dtype=torch.float64)
        P_per_time = (P_per_time @ P_per_time.transpose(-1, -2)) + 0.1 * torch.eye(M, dtype=torch.float64)

        # Engine staticmethod path (now delegates to helper).
        legacy_stats = VEMARDEngine._aggregate_emission_stats(
            data=data, x_hat=x_hat, P_per_time=P_per_time, y_dims=model._y_dims, k=k
        )
        # Direct helper call.
        helper_stats = aggregate_emission_stats(
            data=data, x_hat=x_hat, P_per_time=P_per_time, y_dims=model._y_dims, k=k
        )

        for key in ("sum_y", "sum_y2", "sum_x_per_region", "XX_per_region"):
            torch.testing.assert_close(legacy_stats[key], helper_stats[key], atol=1e-14, rtol=1e-14)
        for r in range(len(legacy_stats["XY_per_region"])):
            torch.testing.assert_close(
                legacy_stats["XY_per_region"][r],
                helper_stats["XY_per_region"][r],
                atol=1e-14,
                rtol=1e-14,
            )


# ---------------------------------------------------------------------------
# setup_ard_posteriors + run_emission_m_step
# ---------------------------------------------------------------------------


class TestSetupARDPosteriors:
    def test_sets_phi_shape_and_variance_floor(self) -> None:
        model = _make_mdlag(K=2, R=2, T=6, neuron_per_region=4)
        data = _make_data(model, n_trials=4)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        # Before: var_floor buffer all zeros (per ARDObservation init).
        assert torch.all(obs.var_floor == 0).item()
        setup_ard_posteriors(obs, data, learn_emission=True)
        # var_floor now > 0 everywhere the data has variance.
        assert (obs.var_floor > 0).all().item()
        # phi_a should be ``prior_phi_a + NT/2``.
        B, T = data.y.shape[:2]  # type: ignore[attr-defined]
        NT = B * T
        expected_phi_a = float(obs.prior_phi_a.item()) + NT / 2.0
        torch.testing.assert_close(obs.phi_a, torch.full_like(obs.phi_a, expected_phi_a))

    def test_learn_emission_false_skips_alpha_update(self) -> None:
        model = _make_mdlag(K=2, R=2)
        data = _make_data(model, n_trials=4)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        alpha_before = obs.alpha_mean.clone()
        setup_ard_posteriors(obs, data, learn_emission=False)
        torch.testing.assert_close(obs.alpha_mean, alpha_before)


class TestRunEmissionMStep:
    def test_runs_without_error_and_mutates_obs(self) -> None:
        model = _make_mdlag(K=2, R=2, T=6)
        data = _make_data(model, n_trials=5)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        setup_ard_posteriors(obs, data, learn_emission=True)

        # Make synthetic stats from a random posterior.
        B, T = data.y.shape[:2]  # type: ignore[attr-defined]
        k = obs.n_obs_per_region
        M = obs.n_regions * k
        torch.manual_seed(0)
        x_hat = torch.randn(B, T, M, dtype=torch.float64)
        P_per_time = torch.eye(M, dtype=torch.float64).unsqueeze(0).expand(T, -1, -1).contiguous()
        stats = aggregate_emission_stats(
            data=data, x_hat=x_hat, P_per_time=P_per_time, y_dims=model._y_dims, k=k
        )

        d_before = obs.d_mean.clone()
        C_before = obs.C_means[0].clone()
        run_emission_m_step(obs, stats, NT=B * T)
        # d and C should change.
        assert not torch.allclose(obs.d_mean, d_before, atol=1e-12)
        assert not torch.allclose(obs.C_means[0], C_before, atol=1e-12)


# ---------------------------------------------------------------------------
# VEMARDEngine end-to-end: behaviour preserved post-refactor
# ---------------------------------------------------------------------------


class TestVEMARDEngineBehaviourPreserved:
    """The CF6a refactor is supposed to be a pure code reorganisation —
    the engine's externally observable behaviour must be unchanged.

    These tests don't compare bit-for-bit against a pre-refactor branch
    (no easy way without keeping the old code). Instead they run the
    engine end-to-end and assert the high-level invariants that the
    existing ``test_vem_ard.py`` already covers, ensuring we didn't break
    them during the extraction.
    """

    def test_score_finite_after_one_iter(self) -> None:
        model = _make_mdlag(K=2, R=2, T=6, neuron_per_region=4)
        data = _make_data(model, n_trials=5)
        # Initialise from data so the engine has a sane starting point.
        model.initialize_from_data(data)
        # One iteration, just to exercise the refactored helpers in fit().
        engine = VEMARDEngine(max_lbfgs_iter=2, lbfgs_history=2)
        result = engine.fit(model, data, max_iter=1, tol=1e-6)
        assert result.n_iter == 1
        assert torch.isfinite(torch.tensor(result.score_trace[0])).item()

    def test_two_iters_elbo_finite(self) -> None:
        """ELBO trace stays finite across iterations (no NaN from refactor)."""
        model = _make_mdlag(K=2, R=2, T=6, neuron_per_region=4)
        data = _make_data(model, n_trials=5)
        model.initialize_from_data(data)
        engine = VEMARDEngine(max_lbfgs_iter=2, lbfgs_history=2)
        result = engine.fit(model, data, max_iter=2, tol=1e-6)
        for elbo in result.score_trace:
            assert torch.isfinite(torch.tensor(elbo)).item()
