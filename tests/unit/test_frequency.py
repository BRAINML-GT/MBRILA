"""Unit tests for the frequency-domain primitives (PR 5c)."""

from __future__ import annotations

import math

import pytest
import torch

from mbrila.delays.fixed import FixedDelay
from mbrila.dynamics.exact_gp import ExactGPLatent
from mbrila.frequency import centered_freqs, unitary_fft, unitary_ifft, zero_freq_index
from mbrila.kernels.mose import MOSEKernel, rbf_psd, rbf_psd_grad_log_gamma

# ----------------------------------------------------------------------
# Frequency vector convention
# ----------------------------------------------------------------------


class TestCenteredFreqs:
    @pytest.mark.parametrize("T", [4, 5, 8, 9, 16, 17])
    def test_matches_matlab_formula(self, T: int) -> None:
        freqs = centered_freqs(T)
        # MATLAB: (-floor(T/2):floor((T-1)/2))/T
        expected = torch.arange(-(T // 2), (T - 1) // 2 + 1, dtype=torch.float64) / float(T)
        torch.testing.assert_close(freqs, expected)
        # Zero-frequency lives at index floor(T/2).
        assert freqs[zero_freq_index(T)].item() == 0.0

    def test_rejects_invalid_T(self) -> None:
        with pytest.raises(ValueError, match="T must be"):
            centered_freqs(0)


# ----------------------------------------------------------------------
# Unitary FFT / IFFT
# ----------------------------------------------------------------------


class TestUnitaryFFT:
    @pytest.mark.parametrize("T", [4, 5, 7, 8, 16, 33])
    def test_round_trip(self, T: int) -> None:
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(3, T, 5, generator=gen, dtype=torch.float64)
        X = unitary_fft(x, dim=1)
        x_back = unitary_ifft(X, dim=1)
        torch.testing.assert_close(x_back.real, x, rtol=1e-12, atol=1e-12)
        assert x_back.imag.abs().max().item() < 1e-12

    @pytest.mark.parametrize("T", [4, 5, 7, 16])
    def test_parseval(self, T: int) -> None:
        """Unitary transform preserves L2 norm (Parseval's theorem)."""
        gen = torch.Generator().manual_seed(0)
        x = torch.randn(T, dtype=torch.float64, generator=gen)
        X = unitary_fft(x)
        torch.testing.assert_close(x.pow(2).sum(), X.abs().pow(2).sum(), rtol=1e-12, atol=1e-12)

    def test_dc_component_matches_sum_over_sqrtT(self) -> None:
        """At zero frequency, X[zero_idx] = (Σ x) / √T."""
        T = 16
        x = torch.arange(T, dtype=torch.float64) - 7.0  # arbitrary signal
        X = unitary_fft(x)
        expected_dc = x.sum() / math.sqrt(T)
        torch.testing.assert_close(X[zero_freq_index(T)].real, expected_dc, atol=1e-12, rtol=1e-12)
        # DC bin is real (imag = 0) for a real input.
        assert X[zero_freq_index(T)].imag.abs().item() < 1e-12


# ----------------------------------------------------------------------
# RBF spectral density
# ----------------------------------------------------------------------


class TestRBFSpectralDensity:
    def test_value_at_zero_omega(self) -> None:
        """At ω = 0, S(0) = (1-ε)·√(2π/γ) + ε."""
        log_gamma = torch.tensor(math.log(0.5), dtype=torch.float64)
        eps = torch.tensor(0.05, dtype=torch.float64)
        omega = torch.tensor(0.0, dtype=torch.float64)
        expected = (1.0 - 0.05) * math.sqrt(2.0 * math.pi / 0.5) + 0.05
        torch.testing.assert_close(
            rbf_psd(omega, log_gamma, eps), torch.tensor(expected, dtype=torch.float64)
        )

    def test_value_at_nonzero_omega(self) -> None:
        log_gamma = torch.tensor(math.log(2.0), dtype=torch.float64)
        eps = torch.tensor(0.0, dtype=torch.float64)
        omega = torch.tensor(1.5, dtype=torch.float64)
        expected = math.sqrt(2.0 * math.pi / 2.0) * math.exp(-0.5 * 1.5**2 / 2.0)
        torch.testing.assert_close(
            rbf_psd(omega, log_gamma, eps), torch.tensor(expected, dtype=torch.float64)
        )

    def test_psd_is_positive(self) -> None:
        gen = torch.Generator().manual_seed(0)
        omega = torch.randn(20, generator=gen, dtype=torch.float64) * 5.0
        log_gamma = torch.tensor(math.log(0.3), dtype=torch.float64)
        eps = torch.tensor(0.01, dtype=torch.float64)
        S = rbf_psd(omega, log_gamma, eps)
        assert (S > 0).all()

    def test_psd_matches_discrete_dtft_sum(self) -> None:
        """The continuous-form analytical PSD matches the DTFT sum
        ``Σ_τ K(τ) e^{-i 2π f τ}`` to high precision when ``γ`` is
        small enough that the discrete Riemann approximation of
        ``∫ K(t) e^{-iωt} dt`` is tight and aliasing is negligible.

        This is the cleanest test of the closed-form PSD: it does not
        depend on the unitary FFT or the Toeplitz↔circulant boundary
        subtleties — just verifies the analytical formula against the
        direct definition.
        """
        T = 64
        log_gamma = torch.tensor(math.log(0.2), dtype=torch.float64)
        eps = torch.tensor(0.0, dtype=torch.float64)
        # Wide range of τ so K(τ) decays to ~0 at the boundary.
        tau = torch.arange(-2 * T, 2 * T + 1, dtype=torch.float64)
        gamma = torch.exp(log_gamma)
        K_tau = (1.0 - eps) * torch.exp(-0.5 * gamma * tau * tau)
        freqs = centered_freqs(T)
        # DTFT: S(f) = Σ_τ K(τ) e^{-i 2π f τ}, evaluated on the centered grid.
        phase = -1j * 2.0 * torch.pi * freqs.unsqueeze(-1) * tau.unsqueeze(0)  # (T, len(tau))
        psd_dtft = (K_tau.unsqueeze(0) * torch.exp(phase)).sum(dim=-1).real

        psd_analytic = rbf_psd(2.0 * torch.pi * freqs, log_gamma, eps)
        # The PSD drops to ~1e-10 at the spectrum edges, where relative
        # error becomes meaningless (both numerically near zero); compare
        # only where the analytical PSD is at least 1e-6 of its peak.
        mask = psd_analytic > 1e-6 * psd_analytic.max()
        rel_err = ((psd_dtft[mask] - psd_analytic[mask]) / psd_analytic[mask]).abs().max().item()
        assert rel_err < 1e-6, f"PSD vs DTFT relative error too high: {rel_err}"


class TestRBFSpectralGradient:
    def test_grad_log_gamma_matches_autograd(self) -> None:
        """Analytical ∂S/∂(log γ) matches autograd to float64 precision."""
        gen = torch.Generator().manual_seed(0)
        omega = torch.randn(32, generator=gen, dtype=torch.float64) * 3.0
        log_gamma = torch.tensor(math.log(0.4), dtype=torch.float64, requires_grad=True)
        eps = torch.tensor(0.02, dtype=torch.float64)

        S = rbf_psd(omega, log_gamma, eps)
        # Sum so we can scalarise the grad; ∂(Σ S)/∂(log γ) = Σ ∂S/∂(log γ).
        S.sum().backward()
        autograd_grad = log_gamma.grad
        assert autograd_grad is not None

        analytical_grad = rbf_psd_grad_log_gamma(omega, log_gamma.detach(), eps).sum()
        torch.testing.assert_close(autograd_grad, analytical_grad, rtol=1e-12, atol=1e-12)

    def test_grad_at_omega_zero_is_negative_for_increasing_gamma(self) -> None:
        """At ω = 0, ∂S/∂(log γ) = (1-ε)·sqrt(π/2)·γ^{-3/2}·(0 - γ)
        = -(1-ε)·sqrt(π/2)·γ^{-1/2} < 0 (larger γ ⇒ smaller variance).
        """
        log_gamma = torch.tensor(math.log(0.5), dtype=torch.float64)
        eps = torch.tensor(0.05, dtype=torch.float64)
        omega = torch.tensor(0.0, dtype=torch.float64)
        grad = rbf_psd_grad_log_gamma(omega, log_gamma, eps)
        expected = -(1.0 - 0.05) * math.sqrt(math.pi / 2.0) * 0.5 ** (-0.5)
        torch.testing.assert_close(grad, torch.tensor(expected, dtype=torch.float64))


# ----------------------------------------------------------------------
# FixedDelay.phase_at_freq
# ----------------------------------------------------------------------


class TestFixedDelayPhase:
    def test_reference_region_is_unity(self) -> None:
        delay = FixedDelay(n_regions=3, n_latent=2, max_delay=4.0, init_scale=0.5)
        T = 8
        Q = delay.phase_at_freq(centered_freqs(T))
        # Region 0 = reference → Q_0(f) = 1 for every f and every latent.
        torch.testing.assert_close(Q[:, 0, :], torch.ones(T, 2, dtype=torch.complex128))

    def test_value_matches_analytic_formula(self) -> None:
        delay = FixedDelay(n_regions=2, n_latent=1, max_delay=4.0)
        # Set β such that δ_{1,0} = +1.5 bins.
        target_delta = 1.5
        beta_val = 2.0 * math.atanh(target_delta / 4.0)
        with torch.no_grad():
            delay.beta.copy_(torch.tensor([[beta_val]], dtype=torch.float64))
        freqs = centered_freqs(6)
        Q = delay.phase_at_freq(freqs)
        # Q_0(f) = 1; Q_1(f) = exp(-i · 2π · f · 1.5)
        expected = torch.exp(-1j * 2.0 * torch.pi * freqs * target_delta).to(torch.complex128)
        torch.testing.assert_close(Q[:, 1, 0], expected, rtol=1e-12, atol=1e-12)

    def test_magnitude_is_one(self) -> None:
        delay = FixedDelay(n_regions=4, n_latent=3, max_delay=5.0, init_scale=1.0)
        Q = delay.phase_at_freq(centered_freqs(10))
        torch.testing.assert_close(Q.abs(), torch.ones_like(Q.abs()), rtol=1e-12, atol=1e-12)

    def test_shape_validation(self) -> None:
        delay = FixedDelay(n_regions=2, n_latent=1, max_delay=4.0)
        with pytest.raises(ValueError, match="freqs must be 1-D"):
            delay.phase_at_freq(torch.zeros(3, 4))
        with pytest.raises(TypeError, match="real floating tensor"):
            delay.phase_at_freq(torch.zeros(4, dtype=torch.complex128))


# ----------------------------------------------------------------------
# ExactGPLatent.cov_freq
# ----------------------------------------------------------------------


class TestExactGPLatentCovFreq:
    def test_shape_matches_K_a_plus_within(self) -> None:
        delay = FixedDelay(n_regions=2, n_latent=2, max_delay=4.0)
        dyn = ExactGPLatent(
            n_regions=2,
            n_across=2,
            n_within=(1, 0),
            delay=delay,
            kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.05),
            kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=0.05),
        )
        psd = dyn.cov_freq(T=10)
        assert psd.shape == (10, 2 + 1 + 0)
        assert psd.dtype == torch.float64

    def test_values_match_rbf_psd_directly(self) -> None:
        delay = FixedDelay(n_regions=2, n_latent=1, max_delay=4.0)
        dyn = ExactGPLatent(
            n_regions=2,
            n_across=1,
            n_within=(0, 0),
            delay=delay,
            kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.05),
            kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=0.05),
            eps_across=0.01,
        )
        T = 16
        psd = dyn.cov_freq(T=T)  # (T, 1)
        freqs = centered_freqs(T)
        log_sigma_a0 = dyn.kernel_across[0].log_sigma.detach().reshape(1)
        expected = rbf_psd(
            (2.0 * torch.pi * freqs),
            log_sigma_a0,
            dyn.eps_across[0:1],
        ).unsqueeze(-1)  # (T, 1)
        # Wait: rbf_psd returns shape that broadcasts; reshape to match psd.
        torch.testing.assert_close(psd, expected.expand_as(psd))

    def test_within_latents_use_their_own_gamma(self) -> None:
        # ExactGPLatent constructs a placeholder FixedDelay internally
        # when n_across == 0, so we pass delay=None.
        dyn = ExactGPLatent(
            n_regions=2,
            n_across=0,
            n_within=(2, 1),
            delay=None,
            kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.05),
            kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=0.1),
            eps_within=0.0,
        )
        # Manually overwrite a within-latent's kernel σ to a distinct value.
        with torch.no_grad():
            dyn.kernel_within_flat[1].log_sigma.copy_(torch.tensor(math.log(0.4), dtype=torch.float64))
        T = 12
        psd = dyn.cov_freq(T=T)  # (T, 3)  — 0 across + 3 within (2+1 per region)
        assert psd.shape == (T, 3)
        # Column 1 corresponds to the modified γ = 0.4.
        omega = 2.0 * torch.pi * centered_freqs(T)
        expected_col1 = rbf_psd(
            omega,
            torch.tensor(math.log(0.4), dtype=torch.float64),
            torch.tensor(0.0, dtype=torch.float64),
        )
        torch.testing.assert_close(psd[:, 1], expected_col1)

    def test_differentiable_through_log_gamma(self) -> None:
        delay = FixedDelay(n_regions=2, n_latent=1, max_delay=4.0)
        dyn = ExactGPLatent(
            n_regions=2,
            n_across=1,
            n_within=(0, 0),
            delay=delay,
            kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.1),
            kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=0.05),
        )
        psd = dyn.cov_freq(T=8)
        loss = psd.sum()
        loss.backward()
        log_sigma = dyn.kernel_across[0].log_sigma
        assert log_sigma.grad is not None
        assert torch.isfinite(log_sigma.grad).all()

    def test_rejects_T_lt_one(self) -> None:
        delay = FixedDelay(n_regions=2, n_latent=1, max_delay=4.0)
        dyn = ExactGPLatent(
            n_regions=2,
            n_across=1,
            n_within=(0, 0),
            delay=delay,
            kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.05),
            kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=0.05),
        )
        with pytest.raises(ValueError, match="T must be"):
            dyn.cov_freq(0)
