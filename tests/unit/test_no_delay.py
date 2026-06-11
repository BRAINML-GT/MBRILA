"""Tests for :class:`NoDelay` and the :class:`MarkovianGPLatent` dispatch
on the three :class:`Delay` flavours (``None`` / ``NoDelay`` / ``FixedDelay``
/ ``TimeVaryingDelay``).

The dispatch tests are the load-bearing part of CF1: they assert that the
no-delay path, the static-delay path, and the time-varying-delay path agree
where they should (zero delay → identical output) and that a static delay
matches a time-varying delay tiled to the same constant.
"""

from __future__ import annotations

import pytest
import torch

from mbrila import FixedDelay, MOSEKernel, NoDelay, TimeVaryingDelay
from mbrila.dynamics.markov_gp import MarkovianGPLatent

# ---------------------------------------------------------------------------
# NoDelay
# ---------------------------------------------------------------------------


class TestNoDelay:
    def test_shape_and_zero(self) -> None:
        delay = NoDelay(n_regions=4, n_latent=3)
        d = delay.as_tensor()
        assert d.shape == (4, 3)
        torch.testing.assert_close(d, torch.zeros(4, 3, dtype=torch.float64))

    def test_T_argument_is_ignored(self) -> None:
        delay = NoDelay(n_regions=2, n_latent=2)
        torch.testing.assert_close(delay.as_tensor(), delay.as_tensor(99))

    def test_is_time_varying_false(self) -> None:
        assert NoDelay(n_regions=3, n_latent=1).is_time_varying is False

    def test_dtype_propagates(self) -> None:
        delay = NoDelay(n_regions=2, n_latent=1, dtype=torch.float32)
        assert delay.as_tensor().dtype == torch.float32

    def test_phase_shift_all_ones(self) -> None:
        delay = NoDelay(n_regions=3, n_latent=2)
        freqs = torch.linspace(-0.5, 0.5, 7, dtype=torch.float64)
        phase = delay.phase_shift(freqs)
        assert phase.shape == (7, 3, 2)
        torch.testing.assert_close(phase, torch.ones_like(phase), atol=1e-12, rtol=1e-12)

    def test_has_no_parameters(self) -> None:
        delay = NoDelay(n_regions=3, n_latent=2)
        assert list(delay.parameters()) == []

    def test_rejects_bad_dims(self) -> None:
        with pytest.raises(ValueError, match="n_regions"):
            NoDelay(n_regions=0, n_latent=1)
        with pytest.raises(ValueError, match="n_latent"):
            NoDelay(n_regions=2, n_latent=0)


# ---------------------------------------------------------------------------
# MarkovianGPLatent dispatch
# ---------------------------------------------------------------------------


class TestMarkovianGPLatentDispatch:
    """All three delay paths should agree where they conceptually should."""

    def test_none_and_NoDelay_match(self) -> None:
        torch.manual_seed(0)
        T, lag, R = 7, 3, 2
        k_none = MOSEKernel(num_regions=R)
        k_nodelay = MOSEKernel(num_regions=R)
        # Share kernel weights so the comparison is meaningful.
        with torch.no_grad():
            k_nodelay.log_sigma.copy_(k_none.log_sigma)

        block_none = MarkovianGPLatent(k_none, lag=lag, T=T, delay=None)
        block_nodelay = MarkovianGPLatent(k_nodelay, lag=lag, T=T, delay=NoDelay(n_regions=R, n_latent=1))
        A1, Q1 = block_none.forward()
        A2, Q2 = block_nodelay.forward()
        assert A1.shape == (T, lag * R, lag * R)
        torch.testing.assert_close(A1, A2, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(Q1, Q2, atol=1e-12, rtol=1e-12)

    def test_zero_FixedDelay_matches_no_delay(self) -> None:
        """A FixedDelay initialised at 0 must reproduce the no-delay (A, Q)."""
        torch.manual_seed(0)
        T, lag, R = 5, 2, 3
        k_a = MOSEKernel(num_regions=R)
        k_b = MOSEKernel(num_regions=R)
        with torch.no_grad():
            k_b.log_sigma.copy_(k_a.log_sigma)

        block_no = MarkovianGPLatent(k_a, lag=lag, T=T, delay=None)
        fixed = FixedDelay(n_regions=R, n_latent=1, max_delay=5.0, init_scale=0.0)
        block_fx = MarkovianGPLatent(k_b, lag=lag, T=T, delay=fixed)

        A_no, Q_no = block_no.forward()
        A_fx, Q_fx = block_fx.forward()
        torch.testing.assert_close(A_no, A_fx, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(Q_no, Q_fx, atol=1e-12, rtol=1e-12)

    def test_FixedDelay_matches_constant_TimeVaryingDelay(self) -> None:
        """Static delay path must match a TVD frozen to the same constant.

        Sets a non-zero FixedDelay and a TimeVaryingDelay whose smoothed
        trajectory is the same constant. Both should produce identical
        ``(A, Q)`` (modulo the broadcast over T)."""
        torch.manual_seed(0)
        T, lag, R = 6, 3, 3
        sigma_seed = 0.1
        k_fx = MOSEKernel(num_regions=R, init_sigma=sigma_seed)
        k_tv = MOSEKernel(num_regions=R, init_sigma=sigma_seed)

        fixed = FixedDelay(n_regions=R, n_latent=1, max_delay=10.0, init_scale=0.0)
        with torch.no_grad():
            # δ_r = max_delay * tanh(β/2). Set β so δ = (0, 1.5, -2.0).
            target_delta = torch.tensor([1.5, -2.0], dtype=fixed.beta.dtype).unsqueeze(-1)
            fixed.beta.copy_(2.0 * torch.atanh(target_delta / fixed.max_delay))
        delta_static = fixed.as_tensor()  # (R, 1)

        tvd = TimeVaryingDelay(n_regions=R, n_latent=1, T=T, init_scale=0.0)
        with torch.no_grad():
            # raw_delay has shape (T, R-1, 1); fill with the same constants
            # as the non-reference rows of `delta_static`. Smoothing of a
            # constant signal with reflect-padded conv is the same constant.
            const = delta_static[1:].to(tvd.raw_delay.dtype)  # (R-1, 1)
            tvd.raw_delay.copy_(const.unsqueeze(0).expand(T, -1, -1))

        block_fx = MarkovianGPLatent(k_fx, lag=lag, T=T, delay=fixed)
        block_tv = MarkovianGPLatent(k_tv, lag=lag, T=T, delay=tvd)
        A_fx, Q_fx = block_fx.forward()
        A_tv, Q_tv = block_tv.forward()
        # Both have shape (T, state_dim, state_dim).
        assert A_fx.shape == A_tv.shape == (T, lag * R, lag * R)
        torch.testing.assert_close(A_fx, A_tv, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(Q_fx, Q_tv, atol=1e-10, rtol=1e-10)

    def test_FixedDelay_shape_and_broadcast(self) -> None:
        """Static-delay path output must still have a T axis."""
        T, lag, R = 4, 2, 2
        kernel = MOSEKernel(num_regions=R)
        fixed = FixedDelay(n_regions=R, n_latent=1, max_delay=3.0)
        block = MarkovianGPLatent(kernel, lag=lag, T=T, delay=fixed)
        A, Q = block.forward()
        assert A.shape == (T, lag * R, lag * R)
        assert Q.shape == (T, lag * R, lag * R)
        # Broadcast: every time slice identical.
        for t in range(1, T):
            torch.testing.assert_close(A[t], A[0], atol=1e-12, rtol=1e-12)
            torch.testing.assert_close(Q[t], Q[0], atol=1e-12, rtol=1e-12)

    def test_rejects_wrong_n_regions(self) -> None:
        kernel = MOSEKernel(num_regions=2)
        with pytest.raises(ValueError, match="n_regions"):
            MarkovianGPLatent(kernel, lag=2, T=4, delay=NoDelay(n_regions=3, n_latent=1))

    def test_rejects_wrong_n_latent(self) -> None:
        kernel = MOSEKernel(num_regions=2)
        with pytest.raises(ValueError, match="n_latent"):
            MarkovianGPLatent(
                kernel,
                lag=2,
                T=4,
                delay=FixedDelay(n_regions=2, n_latent=2, max_delay=1.0),
            )
