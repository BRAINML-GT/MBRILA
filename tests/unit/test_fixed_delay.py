"""Smoke / shape / gradient tests for :class:`FixedDelay`.

These cover the static contract used by DLAG's exact-GP path:

* shape and reference-region zero invariant;
* tanh bounds for extreme parameter values;
* the autograd gradient ``∂δ/∂β`` matches the analytical
  :meth:`FixedDelay.grad_delta_wrt_beta`;
* the inherited :meth:`phase_shift` returns the expected ``(F, R, L)``
  complex shape.
"""

from __future__ import annotations

import pytest
import torch

from mbrila import FixedDelay


class TestFixedDelayShape:
    def test_shape_and_reference_zero(self) -> None:
        delay = FixedDelay(n_regions=4, n_latent=3, max_delay=10.0)
        d = delay.as_tensor()
        assert d.shape == (4, 3)
        torch.testing.assert_close(d[0], torch.zeros(3, dtype=d.dtype))

    def test_T_argument_is_ignored(self) -> None:
        delay = FixedDelay(n_regions=2, n_latent=2, max_delay=5.0)
        torch.testing.assert_close(delay.as_tensor(), delay.as_tensor(123))

    def test_n_regions_one_returns_zero(self) -> None:
        delay = FixedDelay(n_regions=1, n_latent=4, max_delay=8.0)
        d = delay.as_tensor()
        assert d.shape == (1, 4)
        torch.testing.assert_close(d, torch.zeros(1, 4, dtype=d.dtype))
        # Empty beta has zero rows.
        assert delay.beta.shape == (0, 4)

    def test_zero_init(self) -> None:
        delay = FixedDelay(n_regions=3, n_latent=2, max_delay=4.0, init_scale=0.0)
        d = delay.as_tensor()
        torch.testing.assert_close(d, torch.zeros(3, 2, dtype=d.dtype))


class TestFixedDelayBounds:
    def test_large_positive_beta_saturates_at_max_delay(self) -> None:
        delay = FixedDelay(n_regions=2, n_latent=1, max_delay=7.0)
        with torch.no_grad():
            delay.beta.fill_(50.0)  # very large β → δ ≈ +D_max
        d = delay.as_tensor()
        torch.testing.assert_close(d[1, 0].item(), 7.0, atol=1e-5, rtol=0)

    def test_large_negative_beta_saturates_at_minus_max_delay(self) -> None:
        delay = FixedDelay(n_regions=2, n_latent=1, max_delay=3.5)
        with torch.no_grad():
            delay.beta.fill_(-50.0)
        d = delay.as_tensor()
        torch.testing.assert_close(d[1, 0].item(), -3.5, atol=1e-5, rtol=0)

    def test_zero_beta_is_zero_delay(self) -> None:
        delay = FixedDelay(n_regions=3, n_latent=2, max_delay=5.0)
        # β starts at 0; δ should be exactly 0 (including the reference row).
        d = delay.as_tensor()
        torch.testing.assert_close(d, torch.zeros(3, 2, dtype=d.dtype))


class TestFixedDelayGradient:
    def test_analytical_matches_autograd(self) -> None:
        torch.manual_seed(0)
        delay = FixedDelay(n_regions=4, n_latent=3, max_delay=6.0, init_scale=1.0, dtype=torch.float64)
        # Sum over latents and non-reference rows so the autograd-derived
        # ∂(Σδ)/∂β equals the analytical entrywise ∂δ/∂β on each (r, k).
        d = delay.as_tensor()
        loss = d[1:].sum()
        loss.backward()
        assert delay.beta.grad is not None
        analytical = delay.grad_delta_wrt_beta()
        torch.testing.assert_close(delay.beta.grad, analytical, atol=1e-12, rtol=1e-12)


class TestFixedDelayPhaseShift:
    def test_phase_shift_shape_and_zero_at_zero_freq(self) -> None:
        delay = FixedDelay(n_regions=3, n_latent=2, max_delay=4.0)
        # Include f = 0 explicitly so we can check exp(0) = 1.
        freqs = torch.tensor([-0.4, -0.2, 0.0, 0.2, 0.4])
        phase = delay.phase_shift(freqs)
        assert phase.shape == (5, 3, 2)
        torch.testing.assert_close(phase[2].real, torch.ones(3, 2, dtype=phase.real.dtype), atol=1e-7, rtol=0)
        torch.testing.assert_close(
            phase[2].imag, torch.zeros(3, 2, dtype=phase.real.dtype), atol=1e-7, rtol=0
        )

    def test_phase_shift_reference_row_is_one(self) -> None:
        delay = FixedDelay(n_regions=2, n_latent=1, max_delay=5.0)
        with torch.no_grad():
            delay.beta.fill_(1.0)  # non-zero β so the non-reference row is non-trivial
        freqs = torch.tensor([0.1, 0.2, 0.3])
        phase = delay.phase_shift(freqs)
        # Region 0 has zero delay, so its phase is exactly 1 for every frequency.
        torch.testing.assert_close(phase[:, 0, 0], torch.ones(3, dtype=phase.dtype), atol=1e-7, rtol=0)


class TestFixedDelayValidation:
    def test_rejects_non_positive_max_delay(self) -> None:
        with pytest.raises(ValueError, match="max_delay"):
            FixedDelay(n_regions=2, n_latent=1, max_delay=0.0)

    def test_rejects_negative_init_scale(self) -> None:
        with pytest.raises(ValueError, match="init_scale"):
            FixedDelay(n_regions=2, n_latent=1, max_delay=1.0, init_scale=-0.1)
