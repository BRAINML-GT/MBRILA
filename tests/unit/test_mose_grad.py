"""Analytical-vs-autograd tests for the RBF / MOSE kernel gradient helpers.

These verify the chain-rule building blocks DLAG's M-step relies on so we
never have to autograd through a ``(M·T, M·T)`` Cholesky.
"""

from __future__ import annotations

import torch

from mbrila.kernels.mose import (
    rbf_grad_delta_t,
    rbf_grad_log_gamma,
    rbf_kernel_with_eps,
)


class TestRBFKernelValue:
    def test_zero_delta_t_returns_one_on_diag(self) -> None:
        # When δ_mask = 1 on the diagonal entry and Δt = 0, kernel value
        # should be (1 - ε)·1 + ε·1 = 1.
        delta_t = torch.zeros(3, 3, dtype=torch.float64)
        log_gamma = torch.tensor(0.0, dtype=torch.float64)
        eps = torch.tensor(0.1, dtype=torch.float64)
        diag_mask = torch.eye(3, dtype=torch.float64)
        K = rbf_kernel_with_eps(delta_t, log_gamma, eps, diag_mask=diag_mask)
        torch.testing.assert_close(torch.diagonal(K), torch.ones(3, dtype=torch.float64))

    def test_far_apart_values_decay(self) -> None:
        delta_t = torch.tensor([0.0, 5.0], dtype=torch.float64)
        log_gamma = torch.tensor(0.0, dtype=torch.float64)
        eps = torch.tensor(0.0, dtype=torch.float64)
        K = rbf_kernel_with_eps(delta_t, log_gamma, eps)
        assert K[0].item() > K[1].item()


class TestRBFAnalyticalGrads:
    def test_grad_log_gamma_matches_autograd(self) -> None:
        torch.manual_seed(42)
        delta_t = torch.randn(6, 6, dtype=torch.float64)
        # Symmetrise just to mirror real usage; the helper itself does not
        # require symmetry.
        delta_t = 0.5 * (delta_t + delta_t.T)
        log_gamma = torch.tensor(-2.0, dtype=torch.float64, requires_grad=True)
        eps = torch.tensor(0.05, dtype=torch.float64)

        K = rbf_kernel_with_eps(delta_t, log_gamma, eps)
        # Scalarise via sum so .backward populates log_gamma.grad with
        # Σ_{i,j} ∂K[i,j] / ∂(log γ), which equals the sum of our analytical
        # entrywise gradient.
        K.sum().backward()
        autograd_grad = log_gamma.grad
        analytical = rbf_grad_log_gamma(delta_t, log_gamma.detach(), eps).sum()
        assert autograd_grad is not None
        torch.testing.assert_close(autograd_grad, analytical, atol=1e-10, rtol=1e-10)

    def test_grad_delta_t_matches_autograd(self) -> None:
        torch.manual_seed(7)
        delta_t = torch.randn(5, 5, dtype=torch.float64, requires_grad=True)
        log_gamma = torch.tensor(-1.0, dtype=torch.float64)
        eps = torch.tensor(0.02, dtype=torch.float64)

        K = rbf_kernel_with_eps(delta_t, log_gamma, eps)
        # Use a random covector so autograd populates the full
        # entrywise gradient (Σ_{i,j} c[i,j]·∂K[i,j]/∂(Δt[k,l]) =
        # c[k,l]·∂K[k,l]/∂(Δt[k,l]) since the kernel acts pointwise).
        c = torch.randn(5, 5, dtype=torch.float64)
        (c * K).sum().backward()
        autograd_grad = delta_t.grad
        analytical = c * rbf_grad_delta_t(delta_t.detach(), log_gamma, eps)
        assert autograd_grad is not None
        torch.testing.assert_close(autograd_grad, analytical, atol=1e-10, rtol=1e-10)

    def test_grad_at_zero_delta_t_is_zero(self) -> None:
        # ∂K / ∂(Δt) = -γ · Δt · ... vanishes at Δt = 0.
        delta_t = torch.zeros(4, dtype=torch.float64)
        log_gamma = torch.tensor(0.5, dtype=torch.float64)
        eps = torch.tensor(0.01, dtype=torch.float64)
        g = rbf_grad_delta_t(delta_t, log_gamma, eps)
        torch.testing.assert_close(g, torch.zeros(4, dtype=torch.float64))
