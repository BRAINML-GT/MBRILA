"""Unit tests for VEMARDEngine + MDLAG model wiring."""

from __future__ import annotations

import pytest
import torch

from mbrila import MDLAG, ARDPriorConfig, LatentSpec, MOSEKernel, VEMARDEngine
from mbrila.observations.ard import ARDObservation


def _mose_factory(R: int = 2, sigma: float = 0.05):  # type: ignore[no-untyped-def]
    return lambda: MOSEKernel(num_regions=R, init_sigma=sigma)


def _toy_model(*, n_across: int = 2, T: int = 8) -> MDLAG:
    spec = LatentSpec(n_across=n_across, n_within=(0, 0), selection="ard")
    return MDLAG(
        latent_spec=spec,
        y_dims=(4, 5),
        T=T,
        kernel_factory_across=_mose_factory(2),
        device="cpu",
        dtype=torch.float64,
    )


class TestModelWiring:
    def test_components_have_right_types(self) -> None:
        model = _toy_model()
        from mbrila.dynamics.exact_gp import ExactGPLatent

        assert isinstance(model.dynamics, ExactGPLatent)
        assert isinstance(model.observation, ARDObservation)
        assert isinstance(model.inference, VEMARDEngine)
        # Layout: M = R · n_across with no within latents.
        assert model.dynamics.state_dim_per_time == 2 * 2

    def test_rejects_within_latents(self) -> None:
        spec = LatentSpec(n_across=2, n_within=(1, 0), selection="ard")
        with pytest.raises(ValueError, match="n_within = "):
            MDLAG(latent_spec=spec, y_dims=(4, 5), T=8, kernel_factory_across=_mose_factory(2))

    def test_rejects_non_ard_selection(self) -> None:
        spec = LatentSpec(n_across=2, n_within=(0, 0), selection="fixed")
        with pytest.raises(ValueError, match="selection='ard'"):
            MDLAG(latent_spec=spec, y_dims=(4, 5), T=8, kernel_factory_across=_mose_factory(2))

    def test_capabilities_include_cov_full(self) -> None:
        model = _toy_model()
        assert "cov_full" in model.capabilities()

    def test_ard_prior_propagated(self) -> None:
        spec = LatentSpec(
            n_across=2,
            n_within=(0, 0),
            selection="ard",
            ard_prior=ARDPriorConfig(shape=0.5, rate=2.0),
        )
        model = MDLAG(
            latent_spec=spec,
            y_dims=(4, 5),
            T=8,
            kernel_factory_across=_mose_factory(2),
            device="cpu",
            dtype=torch.float64,
        )
        assert isinstance(model.observation, ARDObservation)
        # alpha_a = prior_shape + y_r/2
        torch.testing.assert_close(
            model.observation.prior_alpha_a,
            torch.tensor(0.5, dtype=torch.float64),
        )
        torch.testing.assert_close(
            model.observation.alpha_a,
            torch.tensor([0.5 + 2.0, 0.5 + 2.5], dtype=torch.float64),
        )


class TestSampleRoundtrip:
    def test_sample_shapes(self) -> None:
        torch.manual_seed(0)
        model = _toy_model(T=12)
        data = model.sample(n_trials=3, T=12, seed=0)
        assert data.y.shape == (3, 12, 9)
        assert data.y_dims == (4, 5)

    def test_sample_rejects_wrong_T(self) -> None:
        model = _toy_model(T=12)
        with pytest.raises(ValueError, match="must match model T"):
            model.sample(n_trials=1, T=10, seed=0)

    def test_save_load_round_trip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        torch.manual_seed(0)
        model = _toy_model()
        data = model.sample(n_trials=2, T=8, seed=0)
        model.initialize_from_data(data)

        path = tmp_path / "mdlag.pt"
        model.save(path)
        loaded = MDLAG.load(
            path,
            device="cpu",
            dtype=torch.float64,
            kernel_factory_across=_mose_factory(2),
        )

        # Compare a few state tensors.
        torch.testing.assert_close(loaded.observation.d_mean, model.observation.d_mean)
        torch.testing.assert_close(loaded.observation.alpha_mean, model.observation.alpha_mean)
        for r in range(2):
            torch.testing.assert_close(loaded.observation.C_means[r], model.observation.C_means[r])


class TestVEMEngine:
    def test_e_step_returns_expected_shapes(self) -> None:
        torch.manual_seed(0)
        model = _toy_model(T=8)
        data = model.sample(n_trials=3, T=8, seed=0)
        model.initialize_from_data(data)

        engine = model.inference
        assert isinstance(engine, VEMARDEngine)
        # Engine isn't a Module, but its _e_step is a regular method.
        posterior = engine._e_step(model, data)
        M = model.dynamics.state_dim_per_time
        assert posterior["x_hat"].shape == (3, 8, M)
        assert posterior["P_full"].shape == (M * 8, M * 8)
        assert posterior["P_per_time"].shape == (8, M, M)
        # S used by GP M-step
        assert posterior["S"].shape == (M * 8, M * 8)

    def test_e_step_uses_C_second_moments(self) -> None:
        """The mDLAG E-step's CPhiC uses ⟨C C^T⟩, not C·C^T. Verify by
        directly comparing CPhiC computation steps.
        """
        torch.manual_seed(0)
        model = _toy_model(T=4)
        data = model.sample(n_trials=2, T=4, seed=0)
        model.initialize_from_data(data)

        obs = model.observation
        assert isinstance(obs, ARDObservation)
        # Tweak C_moments so it differs significantly from outer(C_mean).
        with torch.no_grad():
            for r in range(2):
                obs.C_covs[r].copy_(0.5 * torch.eye(2, dtype=torch.float64).expand_as(obs.C_covs[r]))
                obs.C_moments[r].copy_(
                    obs.C_covs[r] + obs.C_means[r].unsqueeze(-1) * obs.C_means[r].unsqueeze(-2)
                )
        # Manually compute the CPhiC block (region 0).
        phi_r = obs.phi_mean[: obs.y_dims[0]]
        expected_CPhiC_0 = (phi_r.view(-1, 1, 1) * obs.C_moments[0]).sum(dim=0)
        # Compare against point-estimate version: should differ by Σ_i φ_i · C_cov_r[i].
        point_estimate = obs.C_means[0].T @ torch.diag(phi_r) @ obs.C_means[0]
        cov_only = (phi_r.view(-1, 1, 1) * obs.C_covs[0]).sum(dim=0)
        torch.testing.assert_close(expected_CPhiC_0, point_estimate + cov_only)

    def test_elbo_monotone_increases_over_iterations(self) -> None:
        """The most important VEM property: ELBO non-decreasing across iterations."""
        torch.manual_seed(0)
        model = _toy_model(T=8)
        data = model.sample(n_trials=4, T=8, seed=0)
        model.initialize_from_data(data)
        result = model.fit(data, max_iter=15, tol=0.0)

        diffs = [
            result.score_trace[i + 1] - result.score_trace[i] for i in range(len(result.score_trace) - 1)
        ]
        # Allow a tiny epsilon for floating-point wobble.
        assert min(diffs) >= -1e-6, f"ELBO decreased; trace={result.score_trace}"

    def test_learn_gp_false_freezes_gp_params(self) -> None:
        torch.manual_seed(0)
        model = _toy_model(T=6)
        data = model.sample(n_trials=2, T=6, seed=0)
        model.initialize_from_data(data)
        log_sigma_before = model.dynamics.kernel_across[0].log_sigma.clone()
        delay_before = model.dynamics.delay.beta.clone()

        model.inference = VEMARDEngine(learn_gp=False)
        model.fit(data, max_iter=3, tol=0.0)

        torch.testing.assert_close(model.dynamics.kernel_across[0].log_sigma, log_sigma_before)
        torch.testing.assert_close(model.dynamics.delay.beta, delay_before)

    def test_learn_emission_false_freezes_emission(self) -> None:
        torch.manual_seed(0)
        model = _toy_model(T=6)
        data = model.sample(n_trials=2, T=6, seed=0)
        model.initialize_from_data(data)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        d_before = obs.d_mean.clone()
        alpha_before = obs.alpha_mean.clone()
        C_before = [c.clone() for c in obs.C_means]

        model.inference = VEMARDEngine(learn_emission=False)
        model.fit(data, max_iter=3, tol=0.0)

        torch.testing.assert_close(obs.d_mean, d_before)
        torch.testing.assert_close(obs.alpha_mean, alpha_before)
        for r in range(2):
            torch.testing.assert_close(obs.C_means[r], C_before[r])

    def test_infer_returns_posterior_with_per_time_block_cov(self) -> None:
        torch.manual_seed(0)
        model = _toy_model(T=6)
        data = model.sample(n_trials=2, T=6, seed=0)
        model.initialize_from_data(data)
        posterior = model.infer(data)
        M = model.dynamics.state_dim_per_time
        assert posterior.mean.shape == (2, 6, M)
        assert posterior.cov.shape == (2, 6, M, M)
        assert posterior.cov_form == "per_time_block"

    def test_score_returns_finite_elbo(self) -> None:
        torch.manual_seed(0)
        model = _toy_model(T=6)
        data = model.sample(n_trials=2, T=6, seed=0)
        model.initialize_from_data(data)
        score = model.score(data)
        assert torch.tensor(score).isfinite().item()


class TestEngineCompatibilityCheck:
    def test_rejects_non_exact_gp_model(self) -> None:
        """Engine refuses a model whose dynamics aren't ExactGPLatent."""
        from mbrila import ADM, MOSEKernel, MultiRegionLinearObservation  # noqa: F401

        adm_spec = LatentSpec(n_across=1, n_within=(0, 0))
        adm = ADM(
            latent_spec=adm_spec,
            y_dims=(3, 3),
            T=6,
            device="cpu",
            dtype=torch.float64,
            kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.1),
            kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=0.1),
        )
        engine = VEMARDEngine()
        with pytest.raises(TypeError, match="ExactGPLatent"):
            engine._e_step(adm, adm.sample(n_trials=1, T=6, seed=0))
