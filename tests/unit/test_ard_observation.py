"""Unit tests for the ARDObservation emission model (mDLAG PR 5a)."""

from __future__ import annotations

import pytest
import torch

from mbrila.observations.ard import ARDObservation


def _ones_priors(
    y_dims: tuple[int, ...],
    k: int,
    *,
    prior_d_beta: float = 1e-3,
    prior_phi_a: float = 1e-3,
    prior_phi_b: float = 1e-3,
    prior_alpha_a: float = 1e-3,
    prior_alpha_b: float = 1e-3,
    init_alpha_mean: float = 1.0,
    init_phi_mean: float = 1.0,
) -> ARDObservation:
    return ARDObservation(
        y_dims=y_dims,
        n_obs_per_region=k,
        prior_d_beta=prior_d_beta,
        prior_phi_a=prior_phi_a,
        prior_phi_b=prior_phi_b,
        prior_alpha_a=prior_alpha_a,
        prior_alpha_b=prior_alpha_b,
        init_alpha_mean=init_alpha_mean,
        init_phi_mean=init_phi_mean,
    )


class TestConstruction:
    def test_state_shapes(self) -> None:
        y_dims = (3, 5, 2)
        k = 4
        obs = _ones_priors(y_dims, k)
        Y = sum(y_dims)
        R = len(y_dims)

        assert obs.d_mean.shape == (Y,)
        assert obs.d_cov.shape == (Y,)
        assert obs.phi_a.shape == ()
        assert obs.phi_b.shape == (Y,)
        assert obs.phi_mean.shape == (Y,)
        assert obs.alpha_a.shape == (R,)
        assert obs.alpha_b.shape == (R, k)
        assert obs.alpha_mean.shape == (R, k)
        for r, y_r in enumerate(y_dims):
            assert obs.C_means[r].shape == (y_r, k)
            assert obs.C_covs[r].shape == (y_r, k, k)
            assert obs.C_moments[r].shape == (y_r, k, k)

    def test_alpha_a_set_from_prior_plus_half_yr(self) -> None:
        obs = _ones_priors((3, 5), k=2, prior_alpha_a=0.5)
        torch.testing.assert_close(obs.alpha_a, torch.tensor([0.5 + 1.5, 0.5 + 2.5], dtype=torch.float64))

    def test_block_diag_C_layout(self) -> None:
        obs = _ones_priors((2, 3), k=2)
        # Seed deterministic, distinguishable per-region means.
        with torch.no_grad():
            obs.C_means[0].copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64))
            obs.C_means[1].copy_(torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=torch.float64))
        C = obs.block_diag_C()
        assert C.shape == (5, 4)
        torch.testing.assert_close(C[:2, :2], obs.C_means[0])
        torch.testing.assert_close(C[2:, 2:], obs.C_means[1])
        assert torch.all(C[:2, 2:] == 0)
        assert torch.all(C[2:, :2] == 0)

    def test_phi_shape_setter(self) -> None:
        obs = _ones_priors((2,), k=2, prior_phi_a=0.1, init_phi_mean=4.0)
        obs.set_phi_shape_from_NT(20)
        assert obs.phi_a.item() == pytest.approx(0.1 + 10.0)
        # phi_mean preserved across the rescale.
        torch.testing.assert_close(obs.phi_mean, torch.tensor([4.0, 4.0], dtype=torch.float64))
        # phi_b consistent.
        torch.testing.assert_close(obs.phi_b, obs.phi_a / obs.phi_mean)


class TestUpdateD:
    def test_d_mean_recovers_empirical_offset_with_strong_evidence(self) -> None:
        # Single region, k=1, no offset between y and C·x → d should fit residual mean.
        obs = _ones_priors((2,), k=1, prior_d_beta=1e-3)
        # Set C_mean = 0 so blkdiag(C)·x = 0 → d_mean = sum_y / (1 + NT·phi)·phi
        with torch.no_grad():
            obs.C_means[0].zero_()
        sum_y = torch.tensor([100.0, -50.0], dtype=torch.float64)
        sum_x = torch.zeros(1, 1, dtype=torch.float64)
        NT = 1000

        obs.update_d(sum_y, sum_x, NT)
        expected_cov = 1.0 / (1e-3 + NT * 1.0)
        torch.testing.assert_close(obs.d_cov, torch.full((2,), expected_cov, dtype=torch.float64))
        torch.testing.assert_close(obs.d_mean, expected_cov * 1.0 * sum_y)

    def test_d_subtracts_C_dot_sum_x(self) -> None:
        obs = _ones_priors((1, 1), k=2, prior_d_beta=1e-3)
        with torch.no_grad():
            obs.C_means[0].copy_(torch.tensor([[1.0, 2.0]], dtype=torch.float64))
            obs.C_means[1].copy_(torch.tensor([[-1.0, 0.5]], dtype=torch.float64))
        sum_y = torch.tensor([10.0, 20.0], dtype=torch.float64)
        sum_x = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float64)  # (R=2, k=2)
        NT = 50

        obs.update_d(sum_y, sum_x, NT)
        # sum_Cx[0] = 1·1 + 2·1 = 3 ; sum_Cx[1] = -1·2 + 0.5·2 = -1
        expected_cov = 1.0 / (1e-3 + NT * 1.0)
        expected_mean = expected_cov * 1.0 * torch.tensor([10.0 - 3.0, 20.0 - (-1.0)], dtype=torch.float64)
        torch.testing.assert_close(obs.d_mean, expected_mean)


class TestUpdateC:
    def test_scalar_case_matches_hand_computation(self) -> None:
        # Single region, single neuron, k=1, scalar everything.
        obs = _ones_priors((1,), k=1, init_alpha_mean=1.0, init_phi_mean=1.0)
        with torch.no_grad():
            obs.d_mean.copy_(torch.tensor([0.5], dtype=torch.float64))
        XX = torch.tensor([[[10.0]]], dtype=torch.float64)  # (R=1, k=1, k=1)
        XY = [torch.tensor([[15.0]], dtype=torch.float64)]  # one (k=1, y_r=1) tensor
        sum_x = torch.tensor([[5.0]], dtype=torch.float64)  # (R=1, k=1)

        obs.update_C(XX, XY, sum_x)

        # XY0 = 15 - 5·0.5 = 12.5
        # Σ_inv = α_mean + φ·XX = 1 + 1·10 = 11
        # Σ = 1/11
        # μ_C = Σ·φ·XY0 = (1/11)·1·12.5
        expected_mean = torch.tensor([[12.5 / 11.0]], dtype=torch.float64)
        expected_cov = torch.tensor([[[1.0 / 11.0]]], dtype=torch.float64)
        torch.testing.assert_close(obs.C_means[0], expected_mean)
        torch.testing.assert_close(obs.C_covs[0], expected_cov)
        # Second moment invariant.
        torch.testing.assert_close(
            obs.C_moments[0],
            expected_cov + expected_mean.unsqueeze(-1) * expected_mean.unsqueeze(-2),
        )

    def test_cov_symmetric_and_psd(self) -> None:
        obs = _ones_priors((4, 3), k=3)
        with torch.no_grad():
            obs.d_mean.copy_(torch.linspace(-1.0, 1.0, 7, dtype=torch.float64))
        gen = torch.Generator().manual_seed(0)
        # Build XX as A·A^T + I to ensure PSD and non-trivial.
        A = torch.randn(2, 3, 5, generator=gen, dtype=torch.float64)
        XX = A @ A.transpose(-2, -1) + torch.eye(3, dtype=torch.float64)  # (R, k, k)
        XY = [
            torch.randn(3, 4, generator=gen, dtype=torch.float64),
            torch.randn(3, 3, generator=gen, dtype=torch.float64),
        ]
        sum_x = torch.randn(2, 3, generator=gen, dtype=torch.float64)

        obs.update_C(XX, XY, sum_x)

        for r in range(2):
            cov_r = obs.C_covs[r]
            asym = (cov_r - cov_r.transpose(-2, -1)).abs().max()
            assert asym.item() < 1e-12
            # Cholesky should succeed → cov is PD.
            torch.linalg.cholesky(cov_r)
            # Second moment invariant.
            outer = obs.C_means[r].unsqueeze(-1) * obs.C_means[r].unsqueeze(-2)
            torch.testing.assert_close(obs.C_moments[r], cov_r + outer)

    def test_shape_validation(self) -> None:
        obs = _ones_priors((2, 3), k=2)
        bad_XX = torch.zeros(3, 2, 2)  # wrong R
        with pytest.raises(ValueError, match="XX must have shape"):
            obs.update_C(bad_XX, [torch.zeros(2, 2), torch.zeros(2, 3)], torch.zeros(2, 2))

        XX_ok = torch.zeros(2, 2, 2)
        with pytest.raises(ValueError, match="XY must have 2 entries"):
            obs.update_C(XX_ok, [torch.zeros(2, 2)], torch.zeros(2, 2))

        with pytest.raises(ValueError, match=r"XY\[1\] must have shape"):
            obs.update_C(XX_ok, [torch.zeros(2, 2), torch.zeros(2, 4)], torch.zeros(2, 2))


class TestUpdateAlpha:
    def test_b_equals_b0_plus_half_diagonal_sum(self) -> None:
        obs = _ones_priors((3,), k=2, prior_alpha_b=0.1)
        # Set a specific moment tensor.
        with torch.no_grad():
            mom = torch.tensor(
                [
                    [[2.0, 0.0], [0.0, 4.0]],
                    [[3.0, 0.0], [0.0, 5.0]],
                    [[1.0, 0.0], [0.0, 2.0]],
                ],
                dtype=torch.float64,
            )
            obs.C_moments[0].copy_(mom)
        obs.update_alpha()
        # diag sum = (2+3+1, 4+5+2) = (6, 11)
        expected_b = torch.tensor([0.1 + 3.0, 0.1 + 5.5], dtype=torch.float64)
        torch.testing.assert_close(obs.alpha_b[0], expected_b)
        torch.testing.assert_close(obs.alpha_mean[0], obs.alpha_a[0] / expected_b)


class TestUpdatePhi:
    def test_scalar_case_matches_hand_computation(self) -> None:
        obs = _ones_priors((1,), k=1, prior_phi_b=0.2, init_phi_mean=1.0)
        obs.set_phi_shape_from_NT(10)  # phi_a = 1e-3 + 5 = 5.001
        with torch.no_grad():
            obs.d_mean.copy_(torch.tensor([0.4], dtype=torch.float64))
            obs.d_cov.copy_(torch.tensor([0.05], dtype=torch.float64))
            obs.C_means[0].copy_(torch.tensor([[0.9]], dtype=torch.float64))
            obs.C_moments[0].copy_(torch.tensor([[[0.85]]], dtype=torch.float64))

        sum_y = torch.tensor([10.0], dtype=torch.float64)
        sum_y2 = torch.tensor([18.0], dtype=torch.float64)
        XX = torch.tensor([[[6.0]]], dtype=torch.float64)
        XY = [torch.tensor([[7.0]], dtype=torch.float64)]
        sum_x = torch.tensor([[3.0]], dtype=torch.float64)

        obs.update_phi(NT=10, sum_y=sum_y, sum_y2=sum_y2, XX=XX, XY=XY, sum_x_per_region=sum_x)

        # XY0 = 7 - 3·0.4 = 5.8
        # term1 = 10·(0.05+0.16) + 18 - 2·0.4·10 = 2.1 + 18 - 8 = 12.1
        # term2 = -2·0.9·5.8 = -10.44
        # term3 = 0.85·6 = 5.1
        # residual = 12.1 - 10.44 + 5.1 = 6.76
        # b = 0.2 + 0.5·6.76 = 3.58
        expected_b = torch.tensor([0.2 + 0.5 * 6.76], dtype=torch.float64)
        torch.testing.assert_close(obs.phi_b, expected_b)
        torch.testing.assert_close(obs.phi_mean, obs.phi_a / expected_b)

    def test_variance_floor_caps_phi_mean(self) -> None:
        obs = ARDObservation(
            y_dims=(1,),
            n_obs_per_region=1,
            prior_phi_b=1e-6,
            init_phi_mean=1.0,
            min_var_frac=0.1,
        )
        obs.set_phi_shape_from_NT(100)
        # Set var_floor → 1/φ_mean must stay above 0.1·1.0 = 0.1, so φ_mean ≤ 10.
        obs.set_variance_floor(torch.tensor([1.0], dtype=torch.float64))
        with torch.no_grad():
            obs.d_mean.zero_()
            obs.d_cov.copy_(torch.tensor([1e-6], dtype=torch.float64))
            obs.C_means[0].zero_()
            obs.C_moments[0].zero_()
        # Residual = NT·0 + sum_y2 - 0 - 0 + 0 = 0 → b would go near 0 → φ_mean huge.
        sum_y = torch.zeros(1, dtype=torch.float64)
        sum_y2 = torch.zeros(1, dtype=torch.float64)
        XX = torch.zeros(1, 1, 1, dtype=torch.float64)
        XY = [torch.zeros(1, 1, dtype=torch.float64)]
        sum_x = torch.zeros(1, 1, dtype=torch.float64)
        obs.update_phi(NT=100, sum_y=sum_y, sum_y2=sum_y2, XX=XX, XY=XY, sum_x_per_region=sum_x)
        # φ_mean clipped at 1/var_floor = 10.0
        assert obs.phi_mean.item() <= 10.0 + 1e-9


class TestELBO:
    @staticmethod
    def _build_random_state(seed: int) -> ARDObservation:
        """Build an ARDObservation in a generic non-degenerate state."""
        torch.manual_seed(seed)
        obs = _ones_priors((3, 4), k=2)
        obs.set_phi_shape_from_NT(20)
        obs.set_variance_floor(torch.full((7,), 1e-6, dtype=torch.float64))
        with torch.no_grad():
            obs.d_mean.copy_(torch.randn(7, dtype=torch.float64) * 0.2)
            obs.d_cov.copy_(torch.rand(7, dtype=torch.float64) * 0.1 + 0.01)
            obs.phi_mean.copy_(torch.rand(7, dtype=torch.float64) * 2.0 + 0.1)
            obs.phi_b.copy_(obs.phi_a / obs.phi_mean)
            obs.alpha_mean.copy_(torch.rand(2, 2, dtype=torch.float64) * 5.0 + 0.1)
            obs.alpha_b.copy_(obs.alpha_a.unsqueeze(-1) / obs.alpha_mean)
            for r, y_r in enumerate(obs.y_dims):
                cov_seed = torch.randn(y_r, 2, 2, dtype=torch.float64)
                cov_r = cov_seed @ cov_seed.transpose(-2, -1) + 0.1 * torch.eye(
                    2, dtype=torch.float64
                ).expand(y_r, 2, 2)
                cov_r = 0.5 * (cov_r + cov_r.transpose(-2, -1))
                mean_r = torch.randn(y_r, 2, dtype=torch.float64)
                obs.C_covs[r].copy_(cov_r)
                obs.C_means[r].copy_(mean_r)
                obs.C_moments[r].copy_(cov_r + mean_r.unsqueeze(-1) * mean_r.unsqueeze(-2))
        return obs

    def test_pieces_are_finite(self) -> None:
        obs = self._build_random_state(seed=42)
        for name, val in [
            ("data_lik", obs.elbo_data_likelihood(20)),
            ("C", obs.elbo_C()),
            ("alpha", obs.elbo_alpha()),
            ("phi", obs.elbo_phi()),
            ("d", obs.elbo_d()),
        ]:
            assert torch.isfinite(val), f"{name} ELBO term is not finite: {val}"

    def test_emission_total_matches_sum_of_pieces(self) -> None:
        obs = self._build_random_state(seed=7)
        NT = 20
        total = obs.elbo_emission(NT)
        pieces = (
            obs.elbo_data_likelihood(NT) + obs.elbo_C() + obs.elbo_alpha() + obs.elbo_phi() + obs.elbo_d()
        )
        torch.testing.assert_close(total, pieces)

    def test_elbo_monotone_increase_across_full_emission_round(self) -> None:
        """ELBO measured at the end of each full ``(d, C, α, φ)`` round
        is monotone non-decreasing when X moments are held fixed.

        We don't check intermediate substep ELBOs because the data-
        likelihood term uses the identity ``b_φ - b_φ⁰ = 0.5·⟨residual²⟩``
        which only holds after the φ update; reading the ELBO between
        substeps gives a stale value that may dip transiently. This is
        exactly why em_mdlag.m computes the lower bound at the end of
        each iteration only.
        """
        torch.manual_seed(123)
        y_dims = (3, 4)
        k = 2
        B, T = 8, 5
        NT = B * T
        Y = sum(y_dims)
        R = len(y_dims)

        # Synthesise a small dataset and fix sufficient statistics.
        y = torch.randn(B, T, Y, dtype=torch.float64) * 0.7 + 0.3
        sum_y = y.sum(dim=(0, 1))
        sum_y2 = (y * y).sum(dim=(0, 1))

        # Fake an X posterior. Per-region means + per-region second moments,
        # both of shape that ARDObservation requires.
        x_means_per_region = torch.randn(B, T, R, k, dtype=torch.float64)
        sum_x_per_region = x_means_per_region.sum(dim=(0, 1))  # (R, k)
        # Per-time covariance shared across (b, t) for simplicity → second
        # moment = NT · cov + Σ_{b,t} mean·meanᵀ.
        cov_t = torch.tensor([[1.0, 0.3], [0.3, 1.2]], dtype=torch.float64)
        outer = (x_means_per_region.unsqueeze(-1) * x_means_per_region.unsqueeze(-2)).sum(
            dim=(0, 1)
        )  # (R, k, k)
        XX = NT * cov_t.unsqueeze(0).expand(R, k, k) + outer

        XY: list[torch.Tensor] = []
        cum = 0
        for r, y_r in enumerate(y_dims):
            y_r_data = y[:, :, cum : cum + y_r]
            XY.append(torch.einsum("btk,bti->ki", x_means_per_region[:, :, r, :], y_r_data))
            cum += y_r

        obs = _ones_priors(y_dims, k=k)
        obs.set_phi_shape_from_NT(NT)
        obs.set_variance_floor(y.reshape(-1, Y).var(dim=0, unbiased=False))

        # Run one full warm-up round so the ELBO formula's b_φ identity
        # holds, then measure across subsequent full rounds.
        obs.update_d(sum_y, sum_x_per_region, NT)
        obs.update_C(XX, XY, sum_x_per_region)
        obs.update_alpha()
        obs.update_phi(NT, sum_y, sum_y2, XX, XY, sum_x_per_region)

        history: list[float] = [obs.elbo_emission(NT).item()]
        for _ in range(8):
            obs.update_d(sum_y, sum_x_per_region, NT)
            obs.update_C(XX, XY, sum_x_per_region)
            obs.update_alpha()
            obs.update_phi(NT, sum_y, sum_y2, XX, XY, sum_x_per_region)
            history.append(obs.elbo_emission(NT).item())

        diffs = [history[i + 1] - history[i] for i in range(len(history) - 1)]
        min_diff = min(diffs)
        assert min_diff >= -1e-6, f"ELBO decreased by {-min_diff}; trace={history}"


class TestInitializeFromPCCA:
    def test_pcca_init_shapes_and_invariants(self) -> None:
        torch.manual_seed(0)
        y_dims = (4, 5)
        k = 2
        obs = _ones_priors(y_dims, k=k)
        y = torch.randn(6, 10, sum(y_dims), dtype=torch.float64)

        obs.initialize_from_pcca(y, zero_offset=False)

        for r, y_r in enumerate(y_dims):
            assert obs.C_means[r].shape == (y_r, k)
            assert obs.C_covs[r].shape == (y_r, k, k)
            assert torch.isfinite(obs.C_means[r]).all()
            # Second moment = cov + outer(mean).
            outer = obs.C_means[r].unsqueeze(-1) * obs.C_means[r].unsqueeze(-2)
            torch.testing.assert_close(obs.C_moments[r], obs.C_covs[r] + outer)
        # phi_mean came from 1/psi.
        assert (obs.phi_mean > 0).all()
        # var_floor set to min_var_frac · sample var.
        sample_var = y.reshape(-1, sum(y_dims)).var(dim=0, unbiased=False)
        torch.testing.assert_close(obs.var_floor, 1e-3 * sample_var)
        # d_mean = sample mean.
        torch.testing.assert_close(obs.d_mean, y.reshape(-1, sum(y_dims)).mean(dim=0), atol=1e-6, rtol=1e-6)

    def test_pcca_init_zero_offset(self) -> None:
        torch.manual_seed(0)
        obs = _ones_priors((3, 4), k=2)
        y = torch.randn(4, 8, 7, dtype=torch.float64)
        obs.initialize_from_pcca(y, zero_offset=True)
        torch.testing.assert_close(obs.d_mean, torch.zeros(7, dtype=torch.float64))


class TestObservationContract:
    def test_forward_uses_block_diag(self) -> None:
        obs = _ones_priors((2, 3), k=2)
        with torch.no_grad():
            obs.C_means[0].copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64))
            obs.C_means[1].copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float64))
            obs.d_mean.copy_(torch.arange(5, dtype=torch.float64))
        x = torch.randn(2, 4, 4, dtype=torch.float64)
        y_hat = obs(x)
        assert y_hat.shape == (2, 4, 5)
        # Manual: C ∈ (5, 4) → y_hat = C·x + d.
        expected = torch.einsum("ij,btj->bti", obs.block_diag_C(), x) + obs.d_mean
        torch.testing.assert_close(y_hat, expected)

    def test_diag_R_reciprocal_of_phi_mean(self) -> None:
        obs = _ones_priors((3,), k=2)
        with torch.no_grad():
            obs.phi_mean.copy_(torch.tensor([2.0, 4.0, 8.0], dtype=torch.float64))
        torch.testing.assert_close(obs.diag_R(), torch.tensor([0.5, 0.25, 0.125], dtype=torch.float64))
