"""Tests for the parallel-scan Kalman filter.

The decisive contract is that the parallel filter must reproduce the
sequential filter to within floating-point tolerance for every problem
size — that's both a correctness check on the scan and an implicit
verification of the Sarkka–Garcia-Fernandez formulas.
"""

from __future__ import annotations

import pytest
import torch

from mbrila.inference.kalman import (
    kalman_filter,
    kalman_filter_parallel,
    rts_smoother,
    rts_smoother_parallel,
)
from mbrila.inference.kalman.parallel import (
    _kalman_combine,
    _smoother_combine,
    associative_scan,
)


def _random_problem(
    *, B: int, T: int, D: int, N: int, seed: int, dtype: torch.dtype = torch.float64
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    F = torch.eye(D, dtype=dtype) + 0.05 * torch.randn(D, D, generator=g, dtype=dtype)
    F = F.unsqueeze(0).expand(T, D, D).contiguous()
    # Q PSD via L L^T
    Lq = 0.3 * torch.tril(torch.randn(D, D, generator=g, dtype=dtype))
    Q = (Lq @ Lq.T + 0.05 * torch.eye(D, dtype=dtype)).unsqueeze(0).expand(T, D, D).contiguous()
    H = torch.randn(N, D, generator=g, dtype=dtype)
    Lr = 0.2 * torch.tril(torch.randn(N, N, generator=g, dtype=dtype))
    R = Lr @ Lr.T + 0.05 * torch.eye(N, dtype=dtype)
    m0 = 0.1 * torch.randn(D, generator=g, dtype=dtype)
    P0 = torch.eye(D, dtype=dtype) * 1.0
    y = 0.5 * torch.randn(B, T, N, generator=g, dtype=dtype)
    return {"F": F, "Q": Q, "H": H, "R": R, "m0": m0, "P0": P0, "y": y}


# ---------------------------------------------------------------------------
# associative_scan — generic correctness on a known operator
# ---------------------------------------------------------------------------


class TestAssociativeScan:
    def _add_op(self, a: tuple[torch.Tensor, ...], b: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        return (a[0] + b[0],)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33])
    def test_inclusive_prefix_sum_matches_cumsum(self, n: int) -> None:
        x = torch.arange(n, dtype=torch.float64)
        (out,) = associative_scan(self._add_op, (x,), dim=0)
        torch.testing.assert_close(out, torch.cumsum(x, dim=0))

    @pytest.mark.parametrize("n", [3, 8, 11])
    def test_reverse_scan_matches_reverse_cumsum(self, n: int) -> None:
        x = torch.arange(n, dtype=torch.float64) + 1.0
        (out,) = associative_scan(self._add_op, (x,), dim=0, reverse=True)
        ref = torch.flip(torch.cumsum(torch.flip(x, [0]), dim=0), [0])
        torch.testing.assert_close(out, ref)

    def test_scan_along_inner_axis(self) -> None:
        x = torch.arange(2 * 5, dtype=torch.float64).reshape(2, 5)
        (out,) = associative_scan(self._add_op, (x,), dim=1)
        torch.testing.assert_close(out, torch.cumsum(x, dim=1))

    def test_combine_operator_is_associative(self) -> None:
        # Sanity: the published Kalman combine operator is associative — we
        # check this empirically on small random elements (rather than
        # symbolically) so any formula bug would surface here.
        torch.manual_seed(0)
        D, B = 3, 2

        def make_elem() -> tuple[torch.Tensor, ...]:
            A = 0.5 * torch.randn(B, D, D, dtype=torch.float64)
            b = torch.randn(B, D, dtype=torch.float64)
            L = 0.3 * torch.tril(torch.randn(B, D, D, dtype=torch.float64))
            C = L @ L.transpose(-1, -2) + 0.1 * torch.eye(D, dtype=torch.float64)
            Lj = 0.3 * torch.tril(torch.randn(B, D, D, dtype=torch.float64))
            J = Lj @ Lj.transpose(-1, -2) + 0.1 * torch.eye(D, dtype=torch.float64)
            eta = torch.randn(B, D, dtype=torch.float64)
            return (A, b, C, J, eta)

        e1, e2, e3 = make_elem(), make_elem(), make_elem()
        left = _kalman_combine(_kalman_combine(e1, e2), e3)
        right = _kalman_combine(e1, _kalman_combine(e2, e3))
        for li, ri in zip(left, right, strict=True):
            torch.testing.assert_close(li, ri, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# Parallel == Sequential
# ---------------------------------------------------------------------------


class TestParallelMatchesSequential:
    @pytest.mark.parametrize("T", [2, 3, 5, 8, 16, 17, 32])
    @pytest.mark.parametrize("B", [1, 4])
    def test_means_and_covs_match(self, T: int, B: int) -> None:
        D, N = 3, 4
        sys = _random_problem(B=B, T=T, D=D, N=N, seed=T * 100 + B)
        seq_means, seq_covs, _ = kalman_filter(
            sys["y"], sys["F"], sys["Q"], sys["H"], sys["R"], sys["m0"], sys["P0"]
        )
        par_means, par_covs = kalman_filter_parallel(
            sys["y"], sys["F"], sys["Q"], sys["H"], sys["R"], sys["m0"], sys["P0"]
        )
        torch.testing.assert_close(par_means, seq_means, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(par_covs, seq_covs, atol=1e-8, rtol=1e-8)

    def test_constant_dynamics_broadcast(self) -> None:
        # Pass F, Q as (D, D) (constant in time) — both filters must handle it.
        D, N, T, B = 2, 2, 6, 2
        torch.manual_seed(0)
        F = torch.eye(D, dtype=torch.float64) + 0.05 * torch.randn(D, D, dtype=torch.float64)
        Q = 0.1 * torch.eye(D, dtype=torch.float64)
        H = torch.randn(N, D, dtype=torch.float64)
        R = 0.2 * torch.eye(N, dtype=torch.float64)
        m0 = torch.zeros(D, dtype=torch.float64)
        P0 = torch.eye(D, dtype=torch.float64)
        y = torch.randn(B, T, N, dtype=torch.float64)

        seq_means, seq_covs, _ = kalman_filter(y, F, Q, H, R, m0, P0)
        par_means, par_covs = kalman_filter_parallel(y, F, Q, H, R, m0, P0)
        torch.testing.assert_close(par_means, seq_means, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(par_covs, seq_covs, atol=1e-9, rtol=1e-9)

    def test_per_trial_priors_match(self) -> None:
        D, N, T, B = 2, 2, 5, 3
        sys = _random_problem(B=B, T=T, D=D, N=N, seed=999)
        m0_per_trial = torch.randn(B, D, dtype=torch.float64)
        P0_per_trial = torch.eye(D, dtype=torch.float64).expand(B, D, D).contiguous() * 0.5

        seq_means, seq_covs, _ = kalman_filter(
            sys["y"], sys["F"], sys["Q"], sys["H"], sys["R"], m0_per_trial, P0_per_trial
        )
        par_means, par_covs = kalman_filter_parallel(
            sys["y"], sys["F"], sys["Q"], sys["H"], sys["R"], m0_per_trial, P0_per_trial
        )
        torch.testing.assert_close(par_means, seq_means, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(par_covs, seq_covs, atol=1e-9, rtol=1e-9)


class TestSmootherCombineAssociativity:
    def test_combine_is_associative(self) -> None:
        torch.manual_seed(7)
        D, B = 3, 2

        def make_elem() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            E = 0.4 * torch.randn(B, D, D, dtype=torch.float64)
            g = torch.randn(B, D, dtype=torch.float64)
            Lc = 0.3 * torch.tril(torch.randn(B, D, D, dtype=torch.float64))
            L = Lc @ Lc.transpose(-1, -2) + 0.1 * torch.eye(D, dtype=torch.float64)
            return (E, g, L)

        e1, e2, e3 = make_elem(), make_elem(), make_elem()
        left = _smoother_combine(_smoother_combine(e1, e2), e3)
        right = _smoother_combine(e1, _smoother_combine(e2, e3))
        for li, ri in zip(left, right, strict=True):
            torch.testing.assert_close(li, ri, atol=1e-9, rtol=1e-9)


class TestParallelSmoother:
    @pytest.mark.parametrize("T", [2, 3, 5, 8, 16, 17, 32])
    @pytest.mark.parametrize("B", [1, 4])
    def test_smoother_matches_sequential(self, T: int, B: int) -> None:
        D, N = 3, 4
        sys = _random_problem(B=B, T=T, D=D, N=N, seed=T * 7 + B)
        # Get filtered states (parallel filter is fine; we already verified it
        # matches sequential).
        f_means, f_covs = kalman_filter_parallel(
            sys["y"], sys["F"], sys["Q"], sys["H"], sys["R"], sys["m0"], sys["P0"]
        )
        seq_sm_means, seq_sm_covs, seq_pair = rts_smoother(f_means, f_covs, sys["F"], sys["Q"])
        par_sm_means, par_sm_covs, par_pair = rts_smoother_parallel(f_means, f_covs, sys["F"], sys["Q"])
        torch.testing.assert_close(par_sm_means, seq_sm_means, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(par_sm_covs, seq_sm_covs, atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(par_pair, seq_pair, atol=1e-8, rtol=1e-8)

    def test_constant_dynamics_broadcast(self) -> None:
        D, N, T, B = 2, 2, 6, 2
        torch.manual_seed(1)
        F = torch.eye(D, dtype=torch.float64) + 0.05 * torch.randn(D, D, dtype=torch.float64)
        Q = 0.1 * torch.eye(D, dtype=torch.float64)
        H = torch.randn(N, D, dtype=torch.float64)
        R = 0.2 * torch.eye(N, dtype=torch.float64)
        m0 = torch.zeros(D, dtype=torch.float64)
        P0 = torch.eye(D, dtype=torch.float64)
        y = torch.randn(B, T, N, dtype=torch.float64)

        f_means, f_covs = kalman_filter_parallel(y, F, Q, H, R, m0, P0)
        seq_sm_means, seq_sm_covs, _ = rts_smoother(f_means, f_covs, F, Q)
        par_sm_means, par_sm_covs, _ = rts_smoother_parallel(f_means, f_covs, F, Q)
        torch.testing.assert_close(par_sm_means, seq_sm_means, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(par_sm_covs, seq_sm_covs, atol=1e-9, rtol=1e-9)

    def test_last_step_unchanged(self) -> None:
        sys = _random_problem(B=2, T=10, D=3, N=2, seed=42)
        f_means, f_covs = kalman_filter_parallel(
            sys["y"], sys["F"], sys["Q"], sys["H"], sys["R"], sys["m0"], sys["P0"]
        )
        sm_means, sm_covs, _ = rts_smoother_parallel(f_means, f_covs, sys["F"], sys["Q"])
        torch.testing.assert_close(sm_means[:, -1], f_means[:, -1])
        torch.testing.assert_close(sm_covs[:, -1], f_covs[:, -1])


class TestAutograd:
    def test_gradients_flow_through_parallel_filter(self) -> None:
        D, N, T, B = 2, 2, 6, 2
        torch.manual_seed(0)
        F = (torch.eye(D) + 0.05 * torch.randn(D, D)).requires_grad_(True)
        Q = (0.1 * torch.eye(D)).clone().requires_grad_(True)
        H = torch.randn(N, D)
        R = 0.2 * torch.eye(N)
        m0 = torch.zeros(D)
        P0 = torch.eye(D)
        y = torch.randn(B, T, N)

        means, _ = kalman_filter_parallel(y, F, Q, H, R, m0, P0)
        loss = means.pow(2).sum()
        loss.backward()
        assert F.grad is not None and torch.isfinite(F.grad).all()
        assert Q.grad is not None and torch.isfinite(Q.grad).all()
