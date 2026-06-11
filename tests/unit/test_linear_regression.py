"""Tests for the conjugate Bayesian linear-Gaussian regression M-step."""

from __future__ import annotations

import pytest
import torch

from mbrila.observations import bayesian_linear_regression


def _make_data(
    n: int, p: int, d: int, *, noise: float = 0.5, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    W_true = torch.randn(d, p, generator=g, dtype=torch.float64)
    b_true = torch.randn(d, generator=g, dtype=torch.float64)
    X = torch.randn(n, p, generator=g, dtype=torch.float64)
    Y = X @ W_true.T + b_true + noise * torch.randn(n, d, generator=g, dtype=torch.float64)
    return X, Y, W_true, b_true


class TestSampleStats:
    def test_recovers_true_weights_in_low_noise(self) -> None:
        X, Y, W_true, b_true = _make_data(n=2000, p=3, d=2, noise=0.05, seed=0)
        result = bayesian_linear_regression(X, Y, fit_intercept=True, nu0=0.0, psi0=0.0)
        # Tight tolerance: large n, small noise.
        torch.testing.assert_close(result.W, W_true, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(result.b, b_true, atol=5e-3, rtol=5e-3)
        # Sigma_diag should hover around noise^2 = 2.5e-3.
        assert (result.sigma_diag - 0.05**2).abs().max().item() < 1e-3

    def test_no_intercept(self) -> None:
        X, Y, W_true, _ = _make_data(n=2000, p=4, d=3, noise=0.05, seed=1)
        # Force intercept to zero by centring.
        Y = Y - Y.mean(dim=0, keepdim=True)
        X = X - X.mean(dim=0, keepdim=True)
        result = bayesian_linear_regression(X, Y, fit_intercept=False, nu0=0.0, psi0=0.0)
        assert torch.equal(result.b, torch.zeros(3, dtype=torch.float64))
        torch.testing.assert_close(result.W, W_true, atol=2e-2, rtol=2e-2)

    def test_weights(self) -> None:
        # Weighting one half of the samples by 0 should equal fitting on the other half.
        X = torch.randn(40, 2, dtype=torch.float64)
        Y = torch.randn(40, 1, dtype=torch.float64)
        w = torch.cat([torch.ones(20, dtype=torch.float64), torch.zeros(20, dtype=torch.float64)])
        weighted = bayesian_linear_regression(X, Y, weights=w, fit_intercept=True, nu0=0.0, psi0=0.0)
        unweighted = bayesian_linear_regression(X[:20], Y[:20], fit_intercept=True, nu0=0.0, psi0=0.0)
        torch.testing.assert_close(weighted.W, unweighted.W, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(weighted.b, unweighted.b, atol=1e-9, rtol=1e-9)


class TestPriors:
    def test_strong_prior_shrinks_weights(self) -> None:
        X, Y, _, _ = _make_data(n=5, p=3, d=2, noise=0.1, seed=2)
        x_dim = 3 + 1  # intercept column
        # Strong prior pulling weights toward zero: V_0 = lambda I, prior_ExyT = 0.
        prior_ExxT = 1e6 * torch.eye(x_dim, dtype=torch.float64)
        prior_ExyT = torch.zeros(x_dim, 2, dtype=torch.float64)
        result = bayesian_linear_regression(
            X,
            Y,
            fit_intercept=True,
            prior_ExxT=prior_ExxT,
            prior_ExyT=prior_ExyT,
        )
        assert result.W.abs().max().item() < 1e-3
        assert result.b.abs().max().item() < 1e-3

    def test_zero_prior_recovers_no_prior(self) -> None:
        X, Y, _, _ = _make_data(n=80, p=2, d=2, seed=3)
        with_prior = bayesian_linear_regression(
            X,
            Y,
            fit_intercept=True,
            prior_ExxT=torch.zeros(3, 3, dtype=torch.float64),
            prior_ExyT=torch.zeros(3, 2, dtype=torch.float64),
        )
        without_prior = bayesian_linear_regression(X, Y, fit_intercept=True)
        torch.testing.assert_close(with_prior.W, without_prior.W, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(with_prior.b, without_prior.b, atol=1e-9, rtol=1e-9)


class TestExpectationsAPI:
    def test_expectations_match_raw_path(self) -> None:
        X, Y, _, _ = _make_data(n=100, p=3, d=2, seed=4)
        # Build the sufficient stats by hand and feed them as `expectations`.
        x_dim = 4
        X_aug = torch.cat([X, torch.ones(100, 1, dtype=torch.float64)], dim=1)
        ExxT = X_aug.T @ X_aug
        ExyT = X_aug.T @ Y
        EyyT = Y.T @ Y
        weight_sum = torch.tensor(100.0, dtype=torch.float64)

        from_raw = bayesian_linear_regression(X, Y, fit_intercept=True)
        from_stats = bayesian_linear_regression(
            expectations=(ExxT, ExyT, EyyT, weight_sum), fit_intercept=True
        )
        torch.testing.assert_close(from_raw.W, from_stats.W, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(from_raw.b, from_stats.b, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(from_raw.sigma_diag, from_stats.sigma_diag, atol=1e-9, rtol=1e-9)
        # Sanity: x_dim must agree with the bookkeeping of fit_intercept.
        assert from_stats.W.shape == (2, 3)
        assert ExxT.shape == (x_dim, x_dim)


class TestValidation:
    def test_missing_inputs(self) -> None:
        with pytest.raises(ValueError, match="either expectations or both X and Y"):
            bayesian_linear_regression()

    def test_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="leading shape"):
            bayesian_linear_regression(torch.zeros(10, 3), torch.zeros(11, 2))

    def test_bad_prior_shape(self) -> None:
        X, Y, _, _ = _make_data(n=20, p=2, d=2, seed=5)
        with pytest.raises(ValueError, match="prior_ExxT"):
            bayesian_linear_regression(
                X, Y, fit_intercept=True, prior_ExxT=torch.zeros(2, 2, dtype=torch.float64)
            )

    def test_bad_expectations_shape(self) -> None:
        with pytest.raises(ValueError, match="ExxT must be square"):
            bayesian_linear_regression(
                expectations=(
                    torch.zeros(3, 4, dtype=torch.float64),
                    torch.zeros(3, 1, dtype=torch.float64),
                    torch.zeros(1, 1, dtype=torch.float64),
                    torch.tensor(1.0),
                )
            )
