"""Tests for the FA-based emission init helpers (DLAG warm-start)."""

from __future__ import annotations

import pytest
import torch

from mbrila.init.factor_analysis import fa_em, fa_init_per_region


def _sample_fa(W_true: torch.Tensor, psi_true: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    """Sample ``n`` rows from ``y = W z + ε`` with ``z ~ N(0, I)``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    k = W_true.shape[1]
    d = W_true.shape[0]
    z = torch.randn(n, k, generator=gen, dtype=W_true.dtype)
    eps = torch.randn(n, d, generator=gen, dtype=W_true.dtype) * psi_true.sqrt().unsqueeze(0)
    return z @ W_true.T + eps


class TestFAEM:
    def test_recovers_loadings_up_to_rotation(self) -> None:
        torch.manual_seed(0)
        d, k = 10, 2
        # Construct a well-conditioned ground-truth loading matrix.
        W_true = torch.randn(d, k, dtype=torch.float64)
        psi_true = torch.full((d,), 0.1, dtype=torch.float64)
        Y = _sample_fa(W_true, psi_true, n=4000, seed=0)
        W_hat, psi_hat, mu_hat = fa_em(Y, k=k, max_iter=100, tol=1e-6)

        # WWᵀ is rotation-invariant; compare those.
        WWt_true = W_true @ W_true.T
        WWt_hat = W_hat @ W_hat.T
        rel = (WWt_hat - WWt_true).norm() / WWt_true.norm()
        assert rel.item() < 0.1

        # ψ should be close to its true value (per-channel).
        rel_psi = (psi_hat - psi_true).abs().mean() / psi_true.mean()
        assert rel_psi.item() < 0.2

        # μ̂ should be close to the empirical mean of Y.
        torch.testing.assert_close(mu_hat, Y.mean(dim=0), atol=1e-9, rtol=0)

    def test_shape_invariants(self) -> None:
        Y = torch.randn(200, 5, dtype=torch.float64)
        W, psi, mu = fa_em(Y, k=2, max_iter=5)
        assert W.shape == (5, 2)
        assert psi.shape == (5,)
        assert mu.shape == (5,)
        # ψ floor.
        assert (psi >= 0).all().item()

    def test_rejects_invalid_args(self) -> None:
        Y = torch.randn(100, 4, dtype=torch.float64)
        with pytest.raises(ValueError, match="k must satisfy"):
            fa_em(Y, k=0)
        with pytest.raises(ValueError, match="k must satisfy"):
            fa_em(Y, k=5)
        with pytest.raises(ValueError, match="N=1"):
            fa_em(torch.randn(1, 4, dtype=torch.float64), k=2)


class TestFAInitPerRegion:
    def test_shapes(self) -> None:
        y = torch.randn(20, 8, 6 + 4, dtype=torch.float64)  # (B, T, sum y_dims)
        Cs, diag_R = fa_init_per_region(
            y,
            y_dims=(6, 4),
            n_per_region=2,
            max_iter=10,
        )
        assert len(Cs) == 2
        assert Cs[0].shape == (6, 2)
        assert Cs[1].shape == (4, 2)
        assert diag_R.shape == (10,)
        assert (diag_R > 0).all().item()

    def test_rejects_too_large_k(self) -> None:
        y = torch.randn(10, 5, 8, dtype=torch.float64)
        with pytest.raises(ValueError, match="exceeds the smallest y_dim"):
            fa_init_per_region(y, y_dims=(3, 5), n_per_region=4)
