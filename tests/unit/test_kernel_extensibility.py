"""Tests for the CF3 kernel extensibility surface.

Three responsibilities:

1. :class:`BaseKernel` ABC contract (subclass must implement ``cov``;
   defaults for the optional methods; capability auto-detection).
2. Matérn-½/3/2/5/2 implementations are *exact* (closed-form ``cov``
   matches reference values; SDE form satisfies Lyapunov; SDE ↔ ``cov``
   consistency via ``H · expm(F·τ) · P∞ · Hᵀ``).
3. :func:`check_kernel` accepts the built-in kernels and rejects a
   deliberately broken kernel.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from mbrila import (
    BaseKernel,
    Matern12Kernel,
    Matern32Kernel,
    Matern52Kernel,
    MOSEKernel,
    check_kernel,
    kernel_registry,
)
from mbrila.core.kernel_spec import SDECoefficients
from mbrila.dynamics.kernel_to_sde import kernel_to_lds, lagged_cov_grid

# ---------------------------------------------------------------------------
# BaseKernel contract
# ---------------------------------------------------------------------------


class TestBaseKernelContract:
    def test_cannot_instantiate_without_cov(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            BaseKernel()  # type: ignore[abstract]

    def test_defaults_for_optional_methods(self) -> None:
        class _MinimalKernel(BaseKernel):
            def cov(self, tau: Tensor) -> Tensor:
                return torch.exp(-tau.abs())

        k = _MinimalKernel()
        assert k.sde_form() is None
        assert k.spectral_density(torch.tensor([0.0, 1.0])) is None
        assert k.is_markovian is False
        assert k.is_complex is False

    def test_n_params_auto(self) -> None:
        # MOSE has one parameter (log_sigma); Matern12 has one (log_lengthscale).
        assert MOSEKernel().n_params == 1
        assert Matern12Kernel().n_params == 1

    def test_capabilities_advertise_cov(self) -> None:
        # MOSE: cov only.
        caps = MOSEKernel().capabilities()
        assert "cov" in caps
        assert "sde_form" not in caps
        # Matern: cov + sde_form (no spectral density implemented).
        caps_matern = Matern32Kernel().capabilities()
        assert {"cov", "sde_form"} <= caps_matern


# ---------------------------------------------------------------------------
# Matérn analytical correctness — spot-check cov(τ) values
# ---------------------------------------------------------------------------


class TestMaternCovValues:
    def test_matern12_known_values(self) -> None:
        k = Matern12Kernel(lengthscale=1.0)
        # k(τ) = exp(-|τ|) with ℓ=1.
        tau = torch.tensor([0.0, 0.5, 1.0, 3.0], dtype=torch.float64)
        expected = torch.tensor(
            [1.0, math.exp(-0.5), math.exp(-1.0), math.exp(-3.0)],
            dtype=torch.float64,
        )
        torch.testing.assert_close(k.cov(tau), expected, atol=1e-12, rtol=1e-12)

    def test_matern32_known_values(self) -> None:
        k = Matern32Kernel(lengthscale=2.0)
        # k(τ) = (1 + √3|τ|/2) exp(-√3|τ|/2)
        lam = math.sqrt(3.0) / 2.0
        tau = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
        expected = torch.tensor(
            [(1.0 + lam * t) * math.exp(-lam * t) for t in [0.0, 1.0, 2.0]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(k.cov(tau), expected, atol=1e-12, rtol=1e-12)

    def test_matern52_known_values(self) -> None:
        k = Matern52Kernel(lengthscale=3.0)
        lam = math.sqrt(5.0) / 3.0
        tau = torch.tensor([0.0, 1.0, 4.0], dtype=torch.float64)
        expected = torch.tensor(
            [(1.0 + lam * t + (lam * t) ** 2 / 3.0) * math.exp(-lam * t) for t in [0.0, 1.0, 4.0]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(k.cov(tau), expected, atol=1e-12, rtol=1e-12)

    def test_cov_evenness(self) -> None:
        for cls in (Matern12Kernel, Matern32Kernel, Matern52Kernel):
            k = cls(lengthscale=1.5)
            tau = torch.tensor([0.3, 1.7, 4.2], dtype=torch.float64)
            torch.testing.assert_close(k.cov(tau), k.cov(-tau), atol=1e-14, rtol=1e-14)


# ---------------------------------------------------------------------------
# Matérn SDE consistency — Lyapunov + cov ↔ expm(F·τ)·P∞ identity
# ---------------------------------------------------------------------------


class TestMaternSDEConsistency:
    """The strict test for the Matérn implementations.

    For each Matérn variant, the exact SDE form ``(F, L, Qc, H, P∞)``
    must satisfy:

    - ``F·P∞ + P∞·Fᵀ + L·Qc·Lᵀ = 0`` (Lyapunov).
    - ``H · expm(F·τ) · P∞ · Hᵀ = cov(τ)`` for τ ≥ 0.

    These are what licence calling ``sde_form`` exact rather than
    approximate.
    """

    @pytest.mark.parametrize(
        "cls,state_dim",
        [
            (Matern12Kernel, 1),
            (Matern32Kernel, 2),
            (Matern52Kernel, 3),
        ],
    )
    def test_shapes_and_is_markovian(self, cls: type[BaseKernel], state_dim: int) -> None:
        k = cls(lengthscale=2.0)
        assert k.is_markovian is True
        sde = k.sde_form()
        assert isinstance(sde, SDECoefficients)
        assert sde.state_dim() == state_dim
        assert sde.F.shape == (state_dim, state_dim)
        assert sde.L.shape == (state_dim, 1)
        assert sde.Qc.shape == (1, 1)
        assert sde.H.shape == (1, state_dim)
        assert sde.stationary_cov.shape == (state_dim, state_dim)

    @pytest.mark.parametrize("cls", [Matern12Kernel, Matern32Kernel, Matern52Kernel])
    def test_lyapunov(self, cls: type[BaseKernel]) -> None:
        k = cls(lengthscale=1.7)
        sde = k.sde_form()
        assert sde is not None
        residual = (
            sde.F @ sde.stationary_cov
            + sde.stationary_cov @ sde.F.transpose(-1, -2)
            + sde.L @ sde.Qc @ sde.L.transpose(-1, -2)
        )
        torch.testing.assert_close(
            residual,
            torch.zeros_like(residual),
            atol=1e-12,
            rtol=1e-12,
        )

    @pytest.mark.parametrize("cls", [Matern12Kernel, Matern32Kernel, Matern52Kernel])
    def test_cov_matches_expm_F_tau(self, cls: type[BaseKernel]) -> None:
        """``H · expm(F·τ) · P∞ · Hᵀ`` must equal ``cov(τ)`` exactly."""
        k = cls(lengthscale=2.3)
        sde = k.sde_form()
        assert sde is not None
        for tau_val in [0.0, 0.4, 1.1, 3.0, 7.0]:
            tau_t = torch.tensor(tau_val, dtype=torch.float64)
            transition = torch.linalg.matrix_exp(sde.F * tau_t)
            sde_pred = (sde.H @ transition @ sde.stationary_cov @ sde.H.transpose(-1, -2)).reshape(())
            cov_actual = k.cov(tau_t).reshape(())
            torch.testing.assert_close(sde_pred, cov_actual, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# check_kernel
# ---------------------------------------------------------------------------


class TestCheckKernel:
    def test_accepts_MOSE(self) -> None:
        check_kernel(MOSEKernel(init_sigma=0.05))

    @pytest.mark.parametrize("cls", [Matern12Kernel, Matern32Kernel, Matern52Kernel])
    def test_accepts_matern(self, cls: type[BaseKernel]) -> None:
        check_kernel(cls(lengthscale=1.4))

    def test_rejects_non_PSD_kernel(self) -> None:
        class _BrokenKernel(BaseKernel):
            """Returns a non-PSD covariance (negative on small τ)."""

            def cov(self, tau: Tensor) -> Tensor:
                t = tau.to(torch.float64)
                return torch.exp(-t.abs()) - 1.5

        with pytest.raises(AssertionError, match=r"cov\(0\) must be positive"):
            check_kernel(_BrokenKernel())

    def test_rejects_non_even_kernel(self) -> None:
        class _OddKernel(BaseKernel):
            def cov(self, tau: Tensor) -> Tensor:
                t = tau.to(torch.float64)
                # exp(-|τ|) * (1 + 0.3·τ) — perturbs evenness.
                return torch.exp(-t.abs()) * (1.0 + 0.3 * t)

        with pytest.raises(AssertionError, match="cov evenness"):
            check_kernel(_OddKernel())

    def test_rejects_wrong_sde_form(self) -> None:
        """An SDE form that doesn't satisfy Lyapunov must be rejected."""

        class _MaternWithBrokenSDE(Matern32Kernel):
            def sde_form(self) -> SDECoefficients:
                good = super().sde_form()
                # Perturb Qc to break Lyapunov.
                bad_Qc = good.Qc * 2.0
                return SDECoefficients(
                    F=good.F,
                    L=good.L,
                    Qc=bad_Qc,
                    H=good.H,
                    stationary_cov=good.stationary_cov,
                )

        with pytest.raises(AssertionError, match="Lyapunov"):
            check_kernel(_MaternWithBrokenSDE(lengthscale=1.0))


# ---------------------------------------------------------------------------
# Integration: Matérn plumbs through CF2's lagged_cov_grid + kernel_to_lds
# ---------------------------------------------------------------------------


class TestMaternThroughSSMBridge:
    """Matérn must drop into the CF1+CF2 Kalman path with no special wiring.

    This is the load-bearing test that "custom kernel + Kalman engine"
    is a working combination after CF1–CF3. Matérn ``cov`` plumbs
    through ``lagged_cov_grid`` and ``kernel_to_lds`` like MOSE would,
    producing real-valued ``(A, Q)`` of the expected shape.
    """

    @pytest.mark.parametrize("cls", [Matern12Kernel, Matern32Kernel, Matern52Kernel])
    def test_no_delay(self, cls: type[BaseKernel]) -> None:
        lag, R = 4, 2
        k = cls(lengthscale=2.0)
        K_lag = lagged_cov_grid(
            k,
            lag=lag,
            num_dim=R,
            delays=None,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        assert K_lag.shape == (lag + 1, lag + 1, R, R)
        A, Q = kernel_to_lds(K_lag, lag=lag, num_dim=R)
        assert A.shape == (lag * R, lag * R)
        assert Q.shape == (lag * R, lag * R)
        # Q diagonal must be non-negative (real innovation covariance).
        assert (torch.diagonal(Q) > -1e-9).all().item()

    def test_static_delay(self) -> None:
        T, lag, R = 5, 3, 3
        k = Matern52Kernel(lengthscale=1.5)
        delays = torch.tensor([0.0, 1.0, -0.7], dtype=torch.float64)
        K_lag = lagged_cov_grid(
            k,
            lag=lag,
            num_dim=R,
            delays=delays,
            dtype=torch.float64,
            device=torch.device("cpu"),
        )
        assert K_lag.shape == (lag + 1, lag + 1, R, R)
        # Smoke: bridge runs to completion without numerical failure.
        kernel_to_lds(K_lag, lag=lag, num_dim=R)
        del T


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestKernelRegistry:
    def test_builtin_kernels_registered(self) -> None:
        names = set(kernel_registry.names())
        assert {"mose", "rbf", "matern_12", "matern_32", "matern_52"} <= names

    def test_lookup_returns_classes(self) -> None:
        assert kernel_registry.get("mose") is MOSEKernel
        assert kernel_registry.get("matern_32") is Matern32Kernel
        # Case-insensitive per Registry.
        assert kernel_registry.get("MATERN_52") is Matern52Kernel

    def test_can_register_custom_kernel(self) -> None:
        class _MyKernel(BaseKernel):
            def cov(self, tau: Tensor) -> Tensor:
                return torch.exp(-tau.abs())

        try:
            kernel_registry.register("_test_custom_kernel", _MyKernel)
            assert kernel_registry.get("_test_custom_kernel") is _MyKernel
        finally:
            kernel_registry.unregister("_test_custom_kernel")
