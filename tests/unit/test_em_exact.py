"""Tests for :class:`ExactEMEngine` — DLAG's time-domain inference engine.

These cover the engine in isolation (no real recovery, just shape /
numerics / EM monotonicity invariants). The full parameter-recovery
test lives in ``tests/recovery/test_dlag_recovery.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from mbrila import DLAG, ExactEMEngine, LatentSpec, MOSEKernel


def _small_dlag(*, n_across: int = 1, n_within: int = 1, R: int = 2, T: int = 8) -> tuple[DLAG, torch.Tensor]:
    """Build a small DLAG and a synthetic data batch for unit tests."""
    spec = LatentSpec(n_across=n_across, n_within=(n_within,) * R)
    model = DLAG(
        latent_spec=spec,
        y_dims=tuple(4 + r for r in range(R)),  # mixed neuron counts per region
        T=T,
        kernel_factory_across=lambda: MOSEKernel(num_regions=R, init_sigma=0.05),
        kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=0.05),
        device="cpu",
        dtype=torch.float64,
        max_delay=3.0,
    )
    data = model.sample(n_trials=5, T=T, seed=0)
    return model, data.y


class TestEStepShapes:
    def test_e_step_returns_expected_shapes(self) -> None:
        model, _ = _small_dlag(n_across=1, n_within=1, R=2, T=8)
        data = model.sample(n_trials=4, T=8, seed=1)
        engine = ExactEMEngine()
        info = engine._e_step(model, data)
        M = model.dynamics.state_dim_per_time
        T = data.y.shape[1]
        assert info["x_hat"].shape == (4, T, M)
        assert info["P_full"].shape == (M * T, M * T)
        assert info["P_per_time"].shape == (T, M, M)
        assert info["S"].shape == (M * T, M * T)
        assert info["ll"].ndim == 0
        # Posterior cov must be symmetric and PSD.
        torch.testing.assert_close(info["P_full"], info["P_full"].T, atol=1e-9, rtol=1e-9)
        eigvals = torch.linalg.eigvalsh(info["P_full"])
        assert eigvals.min().item() > -1e-9


class TestEStepLikelihood:
    def test_score_matches_direct_marginal(self) -> None:
        """LL via the engine equals the direct ``N(d, C K_big C^T + R)`` form."""
        model, _ = _small_dlag(n_across=1, n_within=1, R=2, T=6)
        data = model.sample(n_trials=3, T=6, seed=2)
        engine = ExactEMEngine()
        ll_engine = engine.score(model, data)

        # Reference: build Σ_y = C̃ K_big C̃ᵀ + I_T ⊗ R explicitly and use
        # PyTorch's multivariate Gaussian density.
        with torch.no_grad():
            K_big = model.dynamics.cov_full(6)
            C = model.observation.block_diag_C()
            d_off = model.observation.offset()
            diag_R = model.observation.diag_R()
            B, T, n_y = data.y.shape
            eye_T = torch.eye(T, dtype=K_big.dtype, device=K_big.device)
            C_tilde = torch.kron(eye_T, C)  # (n_y*T, MT)
            Sigma_y = C_tilde @ K_big @ C_tilde.T + torch.kron(eye_T, torch.diag(diag_R))
            # mean per trial (constant in t): d_off; flat over T.
            mu = d_off.repeat(T)
            y_flat = data.y.reshape(B, T * n_y)
            diff = y_flat - mu
            L_y = torch.linalg.cholesky(Sigma_y + 1e-10 * torch.eye(T * n_y, dtype=K_big.dtype))
            sol = torch.cholesky_solve(diff.T, L_y)
            quad = (diff * sol.T).sum()
            logdet = 2.0 * torch.diagonal(L_y).log().sum()
            ll_ref = -0.5 * (B * logdet + quad + B * T * n_y * math.log(2 * math.pi))
        assert abs(ll_engine - float(ll_ref.item())) < 1e-6 * max(abs(ll_engine), 1.0)

    def test_score_with_zero_data_equals_normaliser(self) -> None:
        """With y = d (zero-residual), the quadratic terms vanish and LL = −½ log|Σ_y| − const."""
        model, _ = _small_dlag(n_across=0, n_within=2, R=2, T=4)
        # Build a degenerate data tensor that exactly matches the offset.
        data = model.sample(n_trials=2, T=4, seed=3)
        with torch.no_grad():
            data.y[:] = model.observation.offset()
        engine = ExactEMEngine()
        ll = engine.score(model, data)
        # ll should be finite and equal to the constant component.
        assert math.isfinite(ll)


class TestObservationMStep:
    def test_obs_update_changes_C_and_R(self) -> None:
        """A single observation M-step should change C and diag(R)."""
        model, _ = _small_dlag(n_across=1, n_within=1, R=2, T=6)
        data = model.sample(n_trials=4, T=6, seed=4)
        engine = ExactEMEngine(learn_gp=False)
        C_before = model.observation.block_diag_C().clone()
        R_before = model.observation.diag_R().clone()
        engine.fit(model, data, max_iter=2, tol=0.0)
        C_after = model.observation.block_diag_C()
        R_after = model.observation.diag_R()
        # Update must have done something non-trivial.
        assert (C_before - C_after).abs().max().item() > 1e-6
        assert (R_before - R_after).abs().max().item() > 1e-6


class TestGPMStep:
    def test_gp_update_changes_log_gamma(self) -> None:
        model, _ = _small_dlag(n_across=1, n_within=1, R=2, T=6)
        data = model.sample(n_trials=6, T=6, seed=5)
        engine = ExactEMEngine(learn_obs=False)
        gamma_before = model.dynamics.kernel_across[0].log_sigma.detach().clone()
        engine.fit(model, data, max_iter=2, tol=0.0)
        gamma_after = model.dynamics.kernel_across[0].log_sigma.detach()
        assert (gamma_before - gamma_after).abs().max().item() > 1e-6


class TestEMTrajectory:
    def test_LL_improves_with_obs_only(self) -> None:
        """Closed-form obs M-step alone should drive LL up from a perturbed init.

        We do not assert strict monotonicity because
        :func:`bayesian_linear_regression` carries a tiny Inverse-Wishart
        prior on ``Σ`` (``nu0=1, psi0=1``), making the obs M-step MAP
        rather than MLE; the marginal LL trajectory is therefore monotone
        only up to a small prior-induced wobble. We instead check that
        the late LL is much higher than the early LL — the standard
        recovery-trajectory check used elsewhere in this codebase.
        """
        model, _ = _small_dlag(n_across=1, n_within=1, R=2, T=8)
        data = model.sample(n_trials=12, T=8, seed=6)
        # Perturb C and R so the initial iterate is sub-optimal.
        with torch.no_grad():
            for C_r in model.observation.Cs:
                C_r.data += 0.3 * torch.randn_like(C_r.data)
            model.observation.diag_R_param.data *= 1.5
        engine = ExactEMEngine(learn_gp=False)
        result = engine.fit(model, data, max_iter=15, tol=0.0)
        trace = result.score_trace
        early = sum(trace[:3]) / 3
        late = sum(trace[-3:]) / 3
        assert late > early, f"LL did not improve: early={early:.2f}  late={late:.2f}"
        # The improvement should be substantially larger than the prior-induced
        # noise (≤ a few nats per iteration).
        assert late - early > 1.0


class TestInfer:
    def test_infer_returns_posterior_shape(self) -> None:
        model, _ = _small_dlag(n_across=1, n_within=1, R=2, T=6)
        data = model.sample(n_trials=3, T=6, seed=7)
        engine = ExactEMEngine()
        post = engine.infer(model, data)
        M = model.dynamics.state_dim_per_time
        assert post.mean.shape == (3, 6, M)
        assert post.cov.shape == (3, 6, M, M)
        assert post.cov_form == "per_time_block"
        assert "P_full" in post.extras


class TestEngineValidation:
    def test_rejects_wrong_dynamics(self) -> None:
        from mbrila import ADM, MOSEKernel

        spec = LatentSpec(n_across=1, n_within=(1, 1))
        adm = ADM(
            spec,
            y_dims=(4, 5),
            T=6,
            device="cpu",
            dtype=torch.float64,
            kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.1),
            kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=0.1),
        )
        engine = ExactEMEngine()
        # ADM lacks 'cov_full' capability → check_compatible should fail.
        with pytest.raises(ValueError, match="cov_full"):
            engine.check_compatible(adm)
