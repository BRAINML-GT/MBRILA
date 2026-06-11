"""Unit tests for the frequency-domain mDLAG VEM engine (PR 5d)."""

from __future__ import annotations

import pytest
import torch

from mbrila import MDLAG, LatentSpec, MOSEKernel, VEMARDEngine, VEMARDFreqEngine
from mbrila.observations.ard import ARDObservation


def _build_model(*, n_across: int = 2, T: int = 30, engine: object | None = None) -> MDLAG:
    """CF6c-era helper: the legacy ``engine=<instance>`` API was replaced by
    ``engine=str`` + ``engine_override=<instance>``. We accept an engine
    instance here for test ergonomics and route it through ``engine_override``."""
    spec = LatentSpec(n_across=n_across, n_within=(0, 0), selection="ard")
    kwargs: dict[str, object] = {
        "latent_spec": spec,
        "y_dims": (4, 4),
        "T": T,
        "kernel_factory_across": lambda: MOSEKernel(num_regions=2, init_sigma=0.05),
        "device": "cpu",
        "dtype": torch.float64,
    }
    if engine is not None:
        kwargs["engine_override"] = engine
    return MDLAG(**kwargs)  # type: ignore[arg-type]


class TestEngineCompatibility:
    def test_engine_requires_cov_freq_capability(self) -> None:
        engine = VEMARDFreqEngine()
        assert "cov_freq" in engine.required_capabilities

    def test_advertised_by_exact_gp(self) -> None:
        from mbrila.dynamics.exact_gp import ExactGPLatent

        assert "cov_freq" in ExactGPLatent.CAPABILITIES


class TestESTepShapes:
    def test_posterior_dict_has_expected_shapes(self) -> None:
        torch.manual_seed(0)
        engine = VEMARDFreqEngine()
        model = _build_model(T=16, engine=engine)
        data = model.sample(n_trials=5, T=16, seed=0)
        model.initialize_from_data(data)

        posterior = engine._e_step(model, data)
        T = 16
        K_a = 2
        R = 2
        B = 5
        assert posterior["mu_X"].shape == (B, T, K_a)
        assert posterior["Sigma_X"].shape == (T, K_a, K_a)
        assert posterior["A_f"].shape == (T, K_a, K_a)
        assert posterior["yfft"].shape == (B, T, sum(model._y_dims))
        assert posterior["y0fft"].shape == (B, T, sum(model._y_dims))
        assert posterior["Sx"].shape == (T, K_a)
        assert posterior["Q"].shape == (T, R, K_a)
        assert posterior["CPhiC"].shape == (R, K_a, K_a)
        assert posterior["yX"].shape == (T, R, K_a)
        # Posterior cov is Hermitian.
        Sigma = posterior["Sigma_X"]
        herm_err = (Sigma - Sigma.conj().transpose(-2, -1)).abs().max().item()
        assert herm_err < 1e-10


class TestELBOMonotone:
    def test_elbo_non_decreasing_over_iterations(self) -> None:
        """The freq-domain ELBO is monotone non-decreasing up to its own
        measurement noise.

        Unlike the time engine's exact ELBO, the freq ELBO is an
        *approximation*: the circulant diagonalisation, plus the
        ``b_φ - b_φ⁰ = ½⟨residual²⟩`` identity (which uses the M-step's
        E-step statistics) combined with a ``logdet_Σ_X`` taken from the
        post-M-step fresh E-step. These are mutually consistent only at
        an exact fixed point, so near convergence the *measured* ELBO
        wobbles by O(1e-5 · |ELBO|). The underlying variational updates
        are still each non-decreasing — we therefore allow a small
        scale-relative tolerance rather than demanding bitwise monotonicity.
        """
        torch.manual_seed(0)
        engine = VEMARDFreqEngine()
        model = _build_model(T=24, engine=engine)
        data = model.sample(n_trials=8, T=24, seed=0)
        model.initialize_from_data(data)
        result = model.fit(data, max_iter=15, tol=0.0)

        diffs = [
            result.score_trace[i + 1] - result.score_trace[i] for i in range(len(result.score_trace) - 1)
        ]
        # Scale-relative tolerance for converged-regime measurement noise.
        tol = 1e-4 * abs(result.score_trace[-1])
        assert min(diffs) >= -tol, f"ELBO decreased beyond noise; trace={result.score_trace}"
        # The fit must still make real overall progress.
        assert result.score_trace[-1] > result.score_trace[0]


class TestTimeVsFreqAgreement:
    def test_elbo_matches_time_domain_at_moderate_T(self) -> None:
        """At T ≳ a few times 1/√γ, the circulant approximation is tight
        and freq ELBO should match the time ELBO to ≲ 1% relative error
        after several EM iterations.
        """
        torch.manual_seed(0)
        truth = _build_model(T=60)
        data = truth.sample(n_trials=15, T=60, seed=0)

        torch.manual_seed(1)
        model_t = _build_model(T=60, engine=VEMARDEngine())
        model_t.initialize_from_data(data)
        result_t = model_t.fit(data, max_iter=10, tol=0.0)

        torch.manual_seed(1)
        model_f = _build_model(T=60, engine=VEMARDFreqEngine())
        model_f.initialize_from_data(data)
        result_f = model_f.fit(data, max_iter=10, tol=0.0)

        elbo_t = result_t.score_trace[-1]
        elbo_f = result_f.score_trace[-1]
        rel_err = abs(elbo_t - elbo_f) / abs(elbo_t)
        assert rel_err < 0.01, (
            f"time vs freq ELBO disagree: t={elbo_t:.2f} f={elbo_f:.2f} rel_err={rel_err:.4f}"
        )

    def test_freq_gp_mstep_recovers_delay_with_nonzero_truth(self) -> None:
        """Regression test for the freq GP M-step δ objective.

        The earlier ``test_elbo_matches_time_domain`` used a truth model
        with δ = 0 (default ``FixedDelay`` init), under which every phase
        ``Q_m(f) ≡ 1`` and the quad term's conjugation index is
        irrelevant — so it could not catch a swapped ``A_f`` index in
        ``tr(diag(Q^H)·CPhiC·diag(Q)·A_f)``. Here the truth carries a
        **non-zero** delay; the freq engine must drive δ toward it
        rather than to a runaway value.
        """
        torch.manual_seed(0)
        truth = _build_model(T=48, n_across=2)
        # Seed a clear non-zero per-region delay on the truth model.
        with torch.no_grad():
            # FixedDelay.beta is (n_regions-1, n_across); δ = max_delay·tanh(β/2).
            beta = truth.dynamics.delay.beta
            target_delta = torch.tensor([[2.0, -1.5]], dtype=torch.float64)
            max_delay = truth.dynamics.delay.max_delay
            beta.copy_(2.0 * torch.atanh(target_delta / max_delay))
        data = truth.sample(n_trials=20, T=48, seed=1)
        true_delay = truth.dynamics.delay.as_tensor().detach()  # (R, K_a)

        torch.manual_seed(1)
        model_f = _build_model(T=48, n_across=2, engine=VEMARDFreqEngine())
        model_f.initialize_from_data(data)
        model_f.fit(data, max_iter=40, tol=0.0)
        learned_delay = model_f.dynamics.delay.as_tensor().detach()  # (R, K_a)

        # The freq engine must not let δ run away. Pre-fix, the quad-term
        # index bug drove δ into the tens; the true |δ| here is ≤ 2.
        assert learned_delay.abs().max().item() < 8.0, (
            f"freq GP M-step delay ran away: learned={learned_delay.tolist()}"
        )
        # And it should land within a few bins of truth (loose — only 40
        # iters, identifiability is up to the usual symmetries).
        delay_rmse = (learned_delay - true_delay).pow(2).mean().sqrt().item()
        assert delay_rmse < 2.5, f"delay RMSE too high: {delay_rmse:.3f}"


class TestEStepCorrectness:
    def test_zero_data_gives_zero_mean_posterior(self) -> None:
        torch.manual_seed(0)
        engine = VEMARDFreqEngine()
        model = _build_model(T=16, engine=engine)
        data = model.sample(n_trials=3, T=16, seed=0)
        model.initialize_from_data(data)
        # Zero out the data.
        from mbrila.core.data import MultiRegionData

        zero_data = MultiRegionData(
            y=torch.zeros_like(data.y),
            y_dims=data.y_dims,
            bin_width=data.bin_width,
        )
        # Set d_mean to zero so y0fft = yfft = 0.
        with torch.no_grad():
            model.observation.d_mean.zero_()
        posterior = engine._e_step(model, zero_data)
        # μ_X must be zero.
        assert posterior["mu_X"].abs().max().item() < 1e-12


class TestFreezing:
    def test_learn_gp_false_freezes_gp_params(self) -> None:
        torch.manual_seed(0)
        model = _build_model(T=20, engine=VEMARDFreqEngine(learn_gp=False))
        data = model.sample(n_trials=4, T=20, seed=0)
        model.initialize_from_data(data)
        log_gamma_before = torch.stack([k.log_sigma.detach().clone() for k in model.dynamics.kernel_across])
        beta_before = model.dynamics.delay.beta.clone()
        model.fit(data, max_iter=3, tol=0.0)
        log_gamma_after = torch.stack([k.log_sigma.detach().clone() for k in model.dynamics.kernel_across])
        torch.testing.assert_close(log_gamma_after, log_gamma_before)
        torch.testing.assert_close(model.dynamics.delay.beta, beta_before)

    def test_learn_emission_false_freezes_emission(self) -> None:
        torch.manual_seed(0)
        model = _build_model(T=20, engine=VEMARDFreqEngine(learn_emission=False))
        data = model.sample(n_trials=4, T=20, seed=0)
        model.initialize_from_data(data)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        d_before = obs.d_mean.clone()
        C0_before = obs.C_means[0].clone()
        model.fit(data, max_iter=3, tol=0.0)
        torch.testing.assert_close(obs.d_mean, d_before)
        torch.testing.assert_close(obs.C_means[0], C0_before)


class TestInferAndScore:
    def test_infer_returns_per_region_time_domain_mean(self) -> None:
        """``infer`` returns the per-region (R·K_a) layout matching
        :class:`VEMARDEngine.infer` so :meth:`ARDObservation.forward`
        accepts it directly. Each region's view is the delayed shared
        signal, computed by applying ``Q_m(f)`` in frequency before IFFT.
        """
        torch.manual_seed(0)
        model = _build_model(T=16, engine=VEMARDFreqEngine())
        data = model.sample(n_trials=3, T=16, seed=0)
        model.initialize_from_data(data)
        post = model.infer(data)
        R = 2
        K_a = 2
        assert post.mean.shape == (3, 16, R * K_a)
        assert post.mean.dtype == torch.float64
        # Forward must accept the inferred mean shape.
        y_recon = model.observation.forward(post.mean)
        assert y_recon.shape == data.y.shape
        # Extras include the freq-domain pieces and the shared (un-delayed) signal.
        assert "Sigma_X_freq" in post.extras
        assert "mu_X_freq" in post.extras
        assert "x_shared" in post.extras
        assert post.extras["x_shared"].shape == (3, 16, K_a)

    def test_score_returns_finite_value(self) -> None:
        torch.manual_seed(0)
        model = _build_model(T=16, engine=VEMARDFreqEngine())
        data = model.sample(n_trials=3, T=16, seed=0)
        model.initialize_from_data(data)
        s = model.score(data)
        assert torch.tensor(s).isfinite().item()


class TestARDPruningGate:
    def test_pruned_column_delay_is_frozen_under_gp_mstep(self) -> None:
        """When ARD soft-prunes a latent column (huge ``α``), the engine
        must hold that column's ``(log γ_k, β_{:, k})`` constant in the
        GP M-step so LBFGS does not drift the data-disconnected δ on
        numerical noise (mDLAG MATLAB achieves this via hard ``pruneX``).

        We force a heavy α-imbalance directly on the ARD posterior
        (``α[:, 1]`` 100× larger than ``α[:, 0]``), run one GP M-step,
        and assert that ``β[:, 1]`` and ``log_γ[1]`` did not move.
        """
        torch.manual_seed(0)
        engine = VEMARDFreqEngine(alpha_prune_ratio=10.0)
        model = _build_model(T=20, n_across=2, engine=engine)
        data = model.sample(n_trials=4, T=20, seed=0)
        model.initialize_from_data(data)

        # Force the imbalance.
        with torch.no_grad():
            obs = model.observation
            assert isinstance(obs, ARDObservation)
            obs.alpha_mean[:, 0] = 0.1
            obs.alpha_mean[:, 1] = 50.0  # pruned
            obs.alpha_b[:, 0] = obs.alpha_a / obs.alpha_mean[:, 0]
            obs.alpha_b[:, 1] = obs.alpha_a / obs.alpha_mean[:, 1]
        # Snapshot pre-M-step params.
        log_gamma_before = [k.log_sigma.detach().clone() for k in model.dynamics.kernel_across]
        beta_before = model.dynamics.delay.beta.clone()

        posterior = engine._e_step(model, data)
        engine._m_step_gp(model, data, posterior)

        # Pruned (col 1) — should be unchanged.
        torch.testing.assert_close(
            model.dynamics.kernel_across[1].log_sigma, log_gamma_before[1], rtol=0.0, atol=0.0
        )
        torch.testing.assert_close(model.dynamics.delay.beta[:, 1], beta_before[:, 1], rtol=0.0, atol=0.0)
        # Active (col 0) — should be free to update; allow movement (we
        # just verify the gate is selective, not that LBFGS converged).

    def test_alpha_prune_ratio_inf_disables_gate(self) -> None:
        """``alpha_prune_ratio=inf`` reverts to legacy behaviour where
        every column's (γ, δ) is updated regardless of α magnitude.
        """
        torch.manual_seed(0)
        engine = VEMARDFreqEngine(alpha_prune_ratio=float("inf"))
        model = _build_model(T=20, n_across=2, engine=engine)
        data = model.sample(n_trials=4, T=20, seed=0)
        model.initialize_from_data(data)
        with torch.no_grad():
            obs = model.observation
            assert isinstance(obs, ARDObservation)
            obs.alpha_mean[:, 0] = 0.1
            obs.alpha_mean[:, 1] = 50.0
            obs.alpha_b[:, 0] = obs.alpha_a / obs.alpha_mean[:, 0]
            obs.alpha_b[:, 1] = obs.alpha_a / obs.alpha_mean[:, 1]
        log_gamma_before = [k.log_sigma.detach().clone() for k in model.dynamics.kernel_across]
        beta_before = model.dynamics.delay.beta.clone()
        posterior = engine._e_step(model, data)
        engine._m_step_gp(model, data, posterior)
        # No assertion of equality — disabled gate means pruned col CAN move.
        # We only assert that no error was raised. (A behavioural sanity
        # check on movement would be brittle since LBFGS may legitimately
        # find a near-fixed-point for tiny gradients.)
        for k in model.dynamics.kernel_across:
            assert torch.isfinite(k.log_sigma).all()
        assert torch.isfinite(model.dynamics.delay.beta).all()
        del log_gamma_before, beta_before


class TestEngineRejectsBadModels:
    def test_rejects_within_latents_in_dynamics(self) -> None:
        """Freq engine assumes K_a only; if a model somehow had within latents
        it would fail (defensive check).
        """
        from mbrila.delays.fixed import FixedDelay
        from mbrila.dynamics.exact_gp import ExactGPLatent
        from mbrila.observations.ard import ARDObservation

        # Construct a fake module with ExactGPLatent that has within latents.
        # We exercise the engine's _components validator directly.
        dyn = ExactGPLatent(
            n_regions=2,
            n_across=1,
            n_within=(1, 0),
            delay=FixedDelay(n_regions=2, n_latent=1, max_delay=4.0),
            kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.05),
            kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=0.05),
        )
        obs = ARDObservation(y_dims=(3, 3), n_obs_per_region=1)

        class _DummyModel:
            pass

        dummy = _DummyModel()
        dummy.dynamics = dyn  # type: ignore[attr-defined]
        dummy.observation = obs  # type: ignore[attr-defined]

        engine = VEMARDFreqEngine()
        with pytest.raises(ValueError, match=r"n_within = \(0"):
            engine._components(dummy)  # type: ignore[arg-type]
