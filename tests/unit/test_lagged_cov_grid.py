"""Tests for the generic kernel→lagged-cov bridge (CF2).

Two responsibilities:

1. Parity — :func:`lagged_cov_grid` must produce numerically identical
   output to the legacy :meth:`MOSEKernel.lagged_cov` on all three delay
   shapes (``None`` / ``(R,)`` / ``(T, R)``). This is what guarantees the
   ADM / DLAG-SSM / mDLAG-SSM paths are bit-identical after the refactor.
2. Genericity — a synthetic non-MOSE kernel exposing only ``cov(tau)``
   plumbs through cleanly. This is the load-bearing test for "custom
   kernel author writes only ``cov``" in CF3.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor, nn

from mbrila import MOSEKernel
from mbrila.dynamics.kernel_to_sde import (
    LaggedCovKernel,
    kernel_to_lds,
    lagged_cov_grid,
)

# ---------------------------------------------------------------------------
# Parity with the legacy MOSE-specific path
# ---------------------------------------------------------------------------


class TestParityWithMOSELaggedCov:
    """The generic bridge must agree bit-identically with MOSE's own method."""

    def test_no_delay(self) -> None:
        torch.manual_seed(0)
        lag, R = 4, 3
        k = MOSEKernel(num_regions=R, init_sigma=0.07)
        tau = torch.stack(
            [torch.arange(lag + 1, dtype=torch.float64) - i for i in range(lag + 1)],
            dim=0,
        )
        legacy = k.lagged_cov(tau)  # (L, L, R, R)
        generic = lagged_cov_grid(
            k,
            lag=lag,
            num_dim=R,
            delays=None,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(generic, legacy, atol=1e-14, rtol=1e-14)

    def test_static_delay(self) -> None:
        lag, R = 3, 4
        k = MOSEKernel(num_regions=R, init_sigma=0.1)
        delays = torch.tensor([0.0, 1.5, -2.0, 0.7], dtype=torch.float64)
        legacy = k.lagged_cov(_tau_grid(lag), delays)  # (L, L, R, R)
        generic = lagged_cov_grid(
            k,
            lag=lag,
            num_dim=R,
            delays=delays,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(generic, legacy, atol=1e-14, rtol=1e-14)

    def test_time_varying_delay(self) -> None:
        torch.manual_seed(0)
        T, lag, R = 7, 3, 3
        k = MOSEKernel(num_regions=R, init_sigma=0.03)
        delays = torch.randn(T, R, dtype=torch.float64)
        delays[:, 0] = 0.0  # reference region pinned per library convention
        legacy = k.lagged_cov(_tau_grid(lag), delays)  # (T, L, L, R, R)
        generic = lagged_cov_grid(
            k,
            lag=lag,
            num_dim=R,
            delays=delays,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        torch.testing.assert_close(generic, legacy, atol=1e-14, rtol=1e-14)

    def test_kernel_to_lds_via_generic_path(self) -> None:
        """End-to-end: legacy lagged_cov → kernel_to_lds vs generic → kernel_to_lds."""
        T, lag, R = 5, 2, 2
        k = MOSEKernel(num_regions=R, init_sigma=0.05)
        delays = torch.tensor([0.0, 1.2], dtype=torch.float64)
        K_legacy = k.lagged_cov(_tau_grid(lag), delays)
        K_generic = lagged_cov_grid(
            k,
            lag=lag,
            num_dim=R,
            delays=delays,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        A_legacy, Q_legacy = kernel_to_lds(K_legacy, lag=lag, num_dim=R)
        A_generic, Q_generic = kernel_to_lds(K_generic, lag=lag, num_dim=R)
        torch.testing.assert_close(A_legacy, A_generic, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(Q_legacy, Q_generic, atol=1e-14, rtol=1e-14)
        del T  # quiet linter; we just wanted a non-trivial T context


# ---------------------------------------------------------------------------
# Genericity: a custom kernel only exposing ``cov`` plumbs through
# ---------------------------------------------------------------------------


class _MaternHalfKernel(nn.Module):
    """Synthetic non-MOSE kernel: ``k(τ) = exp(-|τ| / ℓ)`` (Matérn-½).

    Defined here purely to exercise the generic bridge with a kernel that
    is not MOSE and is not an instance of any mbrila kernel class. It is a
    real-valued scalar stationary kernel with one positive hyperparameter
    ``ℓ`` (lengthscale).
    """

    def __init__(self, lengthscale: float = 1.0) -> None:
        super().__init__()
        self.log_ell = nn.Parameter(torch.log(torch.tensor(lengthscale, dtype=torch.float64)))

    def cov(self, tau: Tensor) -> Tensor:
        ell = torch.exp(self.log_ell)
        return torch.exp(-tau.to(ell.dtype).abs() / ell)


class TestGenericKernelSatisfiesProtocol:
    def test_matern_half_runs_through_bridge(self) -> None:
        T, lag, R = 4, 3, 2
        k = _MaternHalfKernel(lengthscale=2.5)
        # Structural typing: a plain nn.Module exposing ``cov(tau)`` satisfies
        # the runtime-checkable protocol consumed by ``lagged_cov_grid``.
        assert isinstance(k, LaggedCovKernel)

        # No-delay path.
        K_no = lagged_cov_grid(
            k,
            lag=lag,
            num_dim=R,
            delays=None,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        assert K_no.shape == (lag + 1, lag + 1, R, R)
        # Every (r1, r2) slot is identical when there's no delay.
        for r1 in range(R):
            for r2 in range(R):
                torch.testing.assert_close(K_no[..., r1, r2], K_no[..., 0, 0])

        # Static-delay path: shifts cross-region entries deterministically.
        delays = torch.tensor([0.0, 1.0], dtype=torch.float64)
        K_st = lagged_cov_grid(
            k,
            lag=lag,
            num_dim=R,
            delays=delays,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        ell = math.exp(k.log_ell.item())
        # K[0, 0, 0, 1] = exp(-|0 + δ_1 - δ_0| / ℓ) = exp(-1.0 / ℓ)
        expected = math.exp(-1.0 / ell)
        assert abs(K_st[0, 0, 0, 1].item() - expected) < 1e-12

        # Time-varying path.
        delays_tv = torch.zeros(T, R, dtype=torch.float64)
        delays_tv[:, 1] = torch.linspace(-1.0, 1.0, T, dtype=torch.float64)
        K_tv = lagged_cov_grid(
            k,
            lag=lag,
            num_dim=R,
            delays=delays_tv,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        assert K_tv.shape == (T, lag + 1, lag + 1, R, R)

        # End-to-end: bridges into kernel_to_lds successfully.
        A, Q = kernel_to_lds(K_tv, lag=lag, num_dim=R)
        assert A.shape == (T, lag * R, lag * R)
        assert Q.shape == (T, lag * R, lag * R)


# ---------------------------------------------------------------------------
# Shape errors
# ---------------------------------------------------------------------------


class TestLaggedCovGridErrors:
    def test_bad_lag(self) -> None:
        k = MOSEKernel(num_regions=1)
        with pytest.raises(ValueError, match="lag must be >= 1"):
            lagged_cov_grid(k, lag=0, num_dim=1, dtype=torch.float64, device=torch.device("cpu"))

    def test_bad_num_dim(self) -> None:
        k = MOSEKernel(num_regions=1)
        with pytest.raises(ValueError, match="num_dim must be >= 1"):
            lagged_cov_grid(k, lag=2, num_dim=0, dtype=torch.float64, device=torch.device("cpu"))

    def test_bad_1d_delays(self) -> None:
        k = MOSEKernel(num_regions=2)
        with pytest.raises(ValueError, match=r"1-D delays must have shape"):
            lagged_cov_grid(
                k,
                lag=2,
                num_dim=2,
                delays=torch.zeros(3, dtype=torch.float64),
                dtype=torch.float64,
                device=torch.device("cpu"),
            )

    def test_bad_2d_delays(self) -> None:
        k = MOSEKernel(num_regions=2)
        with pytest.raises(ValueError, match=r"2-D delays must have shape"):
            lagged_cov_grid(
                k,
                lag=2,
                num_dim=2,
                delays=torch.zeros(4, 3, dtype=torch.float64),
                dtype=torch.float64,
                device=torch.device("cpu"),
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tau_grid(lag: int) -> Tensor:
    t = torch.arange(lag + 1, dtype=torch.float64)
    return t.unsqueeze(1) - t.unsqueeze(0)
