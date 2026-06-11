"""Tests for CF6c — mDLAG-SSM (``MDLAG(engine="kalman")``).

Three layers:

1. ``VEMKalmanARDEngine`` standalone — constructs, runs E-step / ARD
   M-step / GP M-step / ELBO. Smoke tests only; recovery is CF6d.
2. ``MDLAG(engine="kalman")`` preset wiring — dispatches to the
   SSM stack (MarkovianGPLatent + BlockDiagonalDynamics + ARD +
   VEMKalmanARDEngine), and sample/score/save-load round-trip work.
3. ``build_model("mdlag", engine="kalman", ...)`` dispatch + legacy
   default (``engine="time"``) unchanged.

No recovery tests — those are CF6d / user-run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mbrila import (
    MDLAG,
    LatentSpec,
    VEMARDEngine,
    VEMKalmanARDEngine,
    build_model,
)
from mbrila.delays.fixed import FixedDelay
from mbrila.dynamics.exact_gp import ExactGPLatent
from mbrila.dynamics.markov_gp import BlockDiagonalDynamics, MarkovianGPLatent
from mbrila.kernels.mose import MOSEKernel
from mbrila.observations.ard import ARDObservation


def _make_mdlag_kalman(K: int = 2, R: int = 2, T: int = 8, neuron_per_region: int = 4) -> MDLAG:
    spec = LatentSpec(n_across=K, n_within=(0,) * R, selection="ard")
    return MDLAG(
        latent_spec=spec,
        y_dims=tuple(neuron_per_region for _ in range(R)),
        T=T,
        engine="kalman",
        lag_across=3,
        kernel_factory_across=lambda: MOSEKernel(num_regions=R, init_sigma=0.1),
        dtype=torch.float64,
        device="cpu",
    )


def _make_mdlag_time(K: int = 2, R: int = 2, T: int = 8, neuron_per_region: int = 4) -> MDLAG:
    spec = LatentSpec(n_across=K, n_within=(0,) * R, selection="ard")
    return MDLAG(
        latent_spec=spec,
        y_dims=tuple(neuron_per_region for _ in range(R)),
        T=T,
        kernel_factory_across=lambda: MOSEKernel(num_regions=R, init_sigma=0.1),
        dtype=torch.float64,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# MDLAG preset dispatch on engine kind
# ---------------------------------------------------------------------------


class TestEngineDispatch:
    def test_default_engine_is_time(self) -> None:
        model = _make_mdlag_time()
        assert isinstance(model.dynamics, ExactGPLatent)
        assert isinstance(model.inference, VEMARDEngine)

    def test_engine_kalman_builds_ssm_stack(self) -> None:
        model = _make_mdlag_kalman(K=2, R=2)
        assert isinstance(model.dynamics, BlockDiagonalDynamics)
        assert isinstance(model.inference, VEMKalmanARDEngine)
        # Each across block carries a FixedDelay and a MOSE kernel.
        for block in model.dynamics.blocks:
            assert isinstance(block, MarkovianGPLatent)
            assert isinstance(block.delay, FixedDelay)
            assert isinstance(block.kernel, MOSEKernel)
        # ARD observation regardless of engine kind.
        assert isinstance(model.observation, ARDObservation)

    def test_state_dim_layout(self) -> None:
        spec = LatentSpec(n_across=3, n_within=(0,) * 4, selection="ard")
        model = MDLAG(
            latent_spec=spec,
            y_dims=(4, 4, 4, 4),
            T=10,
            engine="kalman",
            lag_across=4,
            kernel_factory_across=lambda: MOSEKernel(num_regions=4, init_sigma=0.1),
        )
        assert isinstance(model.dynamics, BlockDiagonalDynamics)
        # n_across * lag_a * R = 3 * 4 * 4 = 48
        assert model.dynamics.total_state_dim == 48
        # n_observable = R * n_across = 4 * 3 = 12
        assert model.dynamics.H_select.shape == (12, 48)

    def test_unknown_engine_rejected(self) -> None:
        spec = LatentSpec(n_across=2, n_within=(0, 0), selection="ard")
        with pytest.raises(ValueError, match="engine must be"):
            MDLAG(
                latent_spec=spec,
                y_dims=(3, 4),
                T=5,
                engine="bogus",  # type: ignore[arg-type]
                kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.1),
            )

    def test_kalman_requires_positive_lag(self) -> None:
        spec = LatentSpec(n_across=2, n_within=(0, 0), selection="ard")
        with pytest.raises(ValueError, match="lag_across must be >= 1"):
            MDLAG(
                latent_spec=spec,
                y_dims=(3, 4),
                T=5,
                engine="kalman",
                lag_across=0,
                kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.1),
            )


# ---------------------------------------------------------------------------
# engine_override type safety
# ---------------------------------------------------------------------------


class TestEngineOverride:
    def test_kalman_override_with_kalman_engine(self) -> None:
        spec = LatentSpec(n_across=2, n_within=(0, 0), selection="ard")
        engine = VEMKalmanARDEngine()
        model = MDLAG(
            latent_spec=spec,
            y_dims=(3, 4),
            T=5,
            engine="kalman",
            lag_across=3,
            engine_override=engine,
            kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.1),
        )
        assert model.inference is engine

    def test_mismatched_kalman_with_time_engine(self) -> None:
        spec = LatentSpec(n_across=2, n_within=(0, 0), selection="ard")
        with pytest.raises(TypeError, match="must be VEMKalmanARDEngine"):
            MDLAG(
                latent_spec=spec,
                y_dims=(3, 4),
                T=5,
                engine="kalman",
                lag_across=3,
                engine_override=VEMARDEngine(),
                kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.1),
            )


# ---------------------------------------------------------------------------
# Sample + ARD initialise_from_data + score smoke
# ---------------------------------------------------------------------------


class TestSampleScoreInit:
    def test_sample_kalman(self) -> None:
        model = _make_mdlag_kalman(K=2, R=2, T=6)
        data = model.sample(n_trials=3, T=6, seed=0)
        assert data.y.shape == (3, 6, 4 + 4)
        assert torch.isfinite(data.y).all().item()

    def test_initialize_from_pcca_works(self) -> None:
        """Same pCCA init code path works on the SSM model — ARDObservation
        is identical regardless of engine."""
        model = _make_mdlag_kalman()
        sampler = _make_mdlag_time()  # use dense-GP path to sample seed data
        data = sampler.sample(n_trials=8, T=8, seed=0)
        model.initialize_from_data(data)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        # C should be non-degenerate after pCCA seeding.
        assert obs.C_means[0].abs().max().item() > 0.0

    def test_score_finite(self) -> None:
        """End-to-end E-step + ELBO via the new VEMKalmanARDEngine."""
        model = _make_mdlag_kalman()
        sampler = _make_mdlag_time()
        data = sampler.sample(n_trials=4, T=8, seed=0)
        model.initialize_from_data(data)
        proxy_elbo = model.score(data)
        assert torch.isfinite(torch.tensor(proxy_elbo)).item()


# ---------------------------------------------------------------------------
# Fit smoke — one iteration of full VBEM cycle runs end-to-end
# ---------------------------------------------------------------------------


class TestFitSmoke:
    def test_one_iter_runs(self) -> None:
        """A single outer iteration exercises every code path:
        E-step (Kalman) → ARD M-step → GP M-step (Adam) → ELBO.
        """
        model = _make_mdlag_kalman(K=2, R=2, T=6)
        sampler = _make_mdlag_time(K=2, R=2, T=6)
        data = sampler.sample(n_trials=4, T=6, seed=0)
        model.initialize_from_data(data)
        result = model.fit(data, max_iter=1, tol=1e-6)
        assert result.n_iter == 1
        assert torch.isfinite(torch.tensor(result.score_trace[0])).item()

    def test_two_iters_elbo_finite(self) -> None:
        model = _make_mdlag_kalman(K=2, R=2, T=6)
        sampler = _make_mdlag_time(K=2, R=2, T=6)
        data = sampler.sample(n_trials=4, T=6, seed=0)
        model.initialize_from_data(data)
        result = model.fit(data, max_iter=2, tol=1e-9)
        for elbo in result.score_trace:
            assert torch.isfinite(torch.tensor(elbo)).item()


# ---------------------------------------------------------------------------
# Learn-flag freezing
# ---------------------------------------------------------------------------


class TestLearnFlagsKalman:
    def test_learn_gp_false_freezes_kernel(self) -> None:
        model = _make_mdlag_kalman(K=2, R=2, T=5)
        sampler = _make_mdlag_time(K=2, R=2, T=5)
        data = sampler.sample(n_trials=3, T=5, seed=0)
        model.initialize_from_data(data)
        assert isinstance(model.dynamics, BlockDiagonalDynamics)
        kernel_param_before = next(model.dynamics.parameters()).detach().clone()
        engine = VEMKalmanARDEngine(learn_gp=False)
        model._engine_override = engine
        model.inference = engine
        model.fit(data, max_iter=2, tol=1e-9)
        kernel_param_after = next(model.dynamics.parameters()).detach()
        torch.testing.assert_close(kernel_param_after, kernel_param_before)

    def test_learn_emission_false_freezes_C(self) -> None:
        model = _make_mdlag_kalman(K=2, R=2, T=5)
        sampler = _make_mdlag_time(K=2, R=2, T=5)
        data = sampler.sample(n_trials=3, T=5, seed=0)
        model.initialize_from_data(data)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        C_before = obs.C_means[0].clone()
        engine = VEMKalmanARDEngine(learn_emission=False)
        model._engine_override = engine
        model.inference = engine
        model.fit(data, max_iter=2, tol=1e-9)
        torch.testing.assert_close(obs.C_means[0], C_before)


# ---------------------------------------------------------------------------
# Save / load preserves engine choice
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_kalman_round_trip(self, tmp_path: Path) -> None:
        model = _make_mdlag_kalman(K=2, R=2, T=5)
        path = tmp_path / "mdlag_kalman.pt"
        model.save(path)
        loaded = MDLAG.load(path, kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.1))
        assert isinstance(loaded.dynamics, BlockDiagonalDynamics)
        assert isinstance(loaded.inference, VEMKalmanARDEngine)
        assert loaded._lag_across == 3

    def test_legacy_config_defaults_to_time(self) -> None:
        """Old mDLAG checkpoints (pre-CF6c) don't have ``engine`` field —
        ``from_config`` must default to ``"time"`` for backwards compat."""
        legacy_config = {
            "n_across": 2,
            "n_within": [0, 0],
            "y_dims": [3, 4],
            "T": 5,
            "init_gamma_across": 0.01,
            "eps_across": 1e-3,
            "max_delay": 2.0,
            "min_var_frac": 1e-3,
            "ard_prior_shape": 1e-3,
            "ard_prior_rate": 1e-3,
        }
        model = MDLAG.from_config(
            legacy_config,
            kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.1),
        )
        assert isinstance(model.inference, VEMARDEngine)


# ---------------------------------------------------------------------------
# build_model dispatch
# ---------------------------------------------------------------------------


class TestBuildModelMDLAGSSM:
    def test_build_model_mdlag_kalman(self) -> None:
        spec = LatentSpec(n_across=2, n_within=(0, 0), selection="ard")
        model = build_model(
            "mdlag",
            latent_spec=spec,
            y_dims=(3, 4),
            T=5,
            engine="kalman",
            lag_across=3,
            kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.1),
        )
        assert isinstance(model, MDLAG)
        assert isinstance(model.dynamics, BlockDiagonalDynamics)
        assert isinstance(model.inference, VEMKalmanARDEngine)

    def test_build_model_default_engine_unchanged(self) -> None:
        """Backwards compat: build_model('mdlag', ...) without engine kwarg
        must still produce the historical dense-GP mDLAG."""
        spec = LatentSpec(n_across=2, n_within=(0, 0), selection="ard")
        model = build_model(
            "mdlag",
            latent_spec=spec,
            y_dims=(3, 4),
            T=5,
            kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.1),
        )
        assert isinstance(model, MDLAG)
        assert isinstance(model.dynamics, ExactGPLatent)
        assert isinstance(model.inference, VEMARDEngine)
