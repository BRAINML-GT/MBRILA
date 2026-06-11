"""Tests for :class:`ExactGPLatent` — the DLAG / mDLAG dynamics primitive.

These verify the K_big construction in isolation (no inference engine):

* shape and slot-index layout;
* symmetry and positive-definiteness;
* across-latent block-diagonality (different latents → zero cross-cov);
* the across-block reduces to a known analytical form when delays are zero;
* the across-block delay convention matches the DLAG MATLAB sign
  (``Δt = (t1 - t2) - (δ_{r1} - δ_{r2})``);
* within-latent isolation (within latents are uncorrelated across regions
  and across latent indices);
* :meth:`to_lds` and :meth:`cov_freq` raise :class:`NotImplementedError`
  so engines that need them fail loudly at compatibility-check time.
"""

from __future__ import annotations

import math

import pytest
import torch

from mbrila import ExactGPLatent, FixedDelay, MOSEKernel


def _build(
    R: int,
    K_a: int,
    n_within: tuple[int, ...],
    *,
    init_gamma_across: float = 0.05,
    init_gamma_within: float = 0.05,
    eps: float = 0.0,
    max_delay: float = 5.0,
    dtype: torch.dtype = torch.float64,
) -> ExactGPLatent:
    delay = FixedDelay(n_regions=R, n_latent=K_a, max_delay=max_delay, dtype=dtype) if K_a > 0 else None
    return ExactGPLatent(
        n_regions=R,
        n_across=K_a,
        n_within=n_within,
        delay=delay,
        kernel_factory_across=lambda: MOSEKernel(num_regions=R, init_sigma=init_gamma_across),
        kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=init_gamma_within),
        eps_across=eps,
        eps_within=eps,
        dtype=dtype,
    )


class TestExactGPLatentShape:
    def test_M_layout(self) -> None:
        m = _build(R=3, K_a=2, n_within=(1, 0, 2))
        # Region 0: 2 across + 1 within = 3
        # Region 1: 2 across + 0 within = 2
        # Region 2: 2 across + 2 within = 4
        # M = 9
        assert m.state_dim_per_time == 9
        assert m.region_base == (0, 3, 5)
        assert m.slot_across(0, 0) == 0
        assert m.slot_across(2, 1) == 6
        assert m.slot_within(0, 0) == 2
        assert m.slot_within(2, 1) == 8

    def test_cov_full_shape(self) -> None:
        T = 6
        m = _build(R=2, K_a=1, n_within=(1, 1))
        K = m.cov_full(T)
        # M = (1+1) + (1+1) = 4 → MT = 24
        assert K.shape == (m.state_dim_per_time * T, m.state_dim_per_time * T)


class TestExactGPLatentSymmetryPSD:
    def test_cov_full_symmetric(self) -> None:
        m = _build(R=2, K_a=1, n_within=(1, 1))
        K = m.cov_full(8)
        torch.testing.assert_close(K, K.T, atol=1e-10, rtol=1e-10)

    def test_cov_full_psd(self) -> None:
        # ε > 0 guarantees strict PD; without ε, K can be merely PSD with
        # eigenvalues touching 0. We test the realistic case with a small ε.
        m = _build(R=2, K_a=2, n_within=(1, 1), eps=1e-3)
        K = m.cov_full(10)
        eigvals = torch.linalg.eigvalsh(K)
        assert eigvals.min().item() > -1e-9

    def test_cov_full_psd_no_delay(self) -> None:
        m = _build(R=3, K_a=1, n_within=(0, 0, 0), eps=1e-4)
        K = m.cov_full(12)
        eigvals = torch.linalg.eigvalsh(K)
        assert eigvals.min().item() > -1e-9


class TestExactGPLatentAcrossBlock:
    def test_across_block_diagonal_in_latent(self) -> None:
        """Different across latents should have zero cross-covariance."""
        m = _build(R=2, K_a=2, n_within=(0, 0))
        T = 4
        K = m.cov_full(T)
        M = m.state_dim_per_time
        # Pick (region=0, across=0, time=0) vs (region=0, across=1, time=0)
        i = 0 * M + m.slot_across(0, 0)
        j = 0 * M + m.slot_across(0, 1)
        assert abs(K[i, j].item()) < 1e-12

    def test_zero_delay_reduces_to_rbf(self) -> None:
        """With δ ≡ 0 the across kernel is just the plain RBF in (t1 - t2)."""
        gamma = 0.05
        m = _build(R=2, K_a=1, n_within=(0, 0), init_gamma_across=gamma)
        T = 6
        K = m.cov_full(T)
        M = m.state_dim_per_time  # 2
        # For across latent 0 the entry at (r1=0, t1, r2=1, t2) should be
        # (1 - ε)·exp(-γ/2 · (t1 - t2)²); ε=0 by default in the fixture.
        for t1 in range(T):
            for t2 in range(T):
                i = t1 * M + m.slot_across(0, 0)
                j = t2 * M + m.slot_across(1, 0)
                expected = math.exp(-0.5 * gamma * (t1 - t2) ** 2)
                assert abs(K[i, j].item() - expected) < 1e-10

    def test_delay_sign_convention(self) -> None:
        """Δt = (t1 - t2) - (δ_{r1} - δ_{r2}).

        With δ_{0} = 0 and δ_{1} = +2 the kernel peak between reg0 at t1
        and reg1 at t2 occurs when Δt = 0, i.e. t2 = t1 + δ — reg1 is a
        delayed copy of reg0 by ``+δ`` bins, matching the DLAG MATLAB
        sign in ``make_K_big_plusDelays.m``.
        """
        gamma = 0.1
        max_delay = 6.0
        m = _build(
            R=2,
            K_a=1,
            n_within=(0, 0),
            init_gamma_across=gamma,
            max_delay=max_delay,
        )
        # Set δ_{1,0} = 2.0 directly via β. δ = D_max · tanh(β/2) → β = 2·atanh(δ/D_max).
        target_delay = 2.0
        beta_val = 2.0 * math.atanh(target_delay / max_delay)
        with torch.no_grad():
            m.delay.beta.fill_(beta_val)
        T = 10
        K = m.cov_full(T)
        M = m.state_dim_per_time
        t1 = 5
        best_t2 = -1
        best_val = -1.0
        for t2 in range(T):
            i = t1 * M + m.slot_across(0, 0)
            j = t2 * M + m.slot_across(1, 0)
            v = K[i, j].item()
            if v > best_val:
                best_val = v
                best_t2 = t2
        assert best_t2 == t1 + int(target_delay)
        # And the value at the peak should be (1 - ε)·1 = 1 (no eps in this fixture).
        assert abs(best_val - 1.0) < 1e-8


class TestExactGPLatentWithinBlock:
    def test_within_block_no_cross_region(self) -> None:
        m = _build(R=3, K_a=0, n_within=(1, 1, 1))
        T = 5
        K = m.cov_full(T)
        M = m.state_dim_per_time  # 3
        # Within latent of region 0 vs within latent of region 1: should be 0
        i = 0 * M + m.slot_within(0, 0)
        j = 2 * M + m.slot_within(1, 0)
        assert abs(K[i, j].item()) < 1e-12

    def test_within_block_no_cross_latent(self) -> None:
        m = _build(R=1, K_a=0, n_within=(3,))
        T = 4
        K = m.cov_full(T)
        M = m.state_dim_per_time  # 3
        # Different within latents inside the same region are independent.
        i = 0 * M + m.slot_within(0, 0)
        j = 0 * M + m.slot_within(0, 1)
        assert abs(K[i, j].item()) < 1e-12

    def test_within_block_matches_rbf(self) -> None:
        gamma_w = 0.04
        m = _build(R=1, K_a=0, n_within=(1,), init_gamma_within=gamma_w)
        T = 6
        K = m.cov_full(T)
        M = m.state_dim_per_time  # 1
        for t1 in range(T):
            for t2 in range(T):
                i = t1 * M
                j = t2 * M
                expected = math.exp(-0.5 * gamma_w * (t1 - t2) ** 2)
                assert abs(K[i, j].item() - expected) < 1e-10


class TestExactGPLatentDifferentiable:
    def test_cov_full_is_autograd_friendly(self) -> None:
        """Gradients should flow back into each per-block kernel and the delay."""
        m = _build(R=2, K_a=1, n_within=(1, 0), max_delay=4.0)
        with torch.no_grad():
            m.delay.beta.fill_(0.5)
        K = m.cov_full(5)
        K.sum().backward()
        # Every kernel parameter (across + within) should receive a finite grad.
        for kernel in list(m.kernel_across) + list(m.kernel_within_flat):
            for p in kernel.parameters():
                assert p.grad is not None
                assert torch.isfinite(p.grad).all().item()
        assert m.delay.beta.grad is not None
        assert torch.isfinite(m.delay.beta.grad).all().item()


class TestExactGPLatentCapabilities:
    def test_advertises_cov_full(self) -> None:
        assert "cov_full" in ExactGPLatent.CAPABILITIES

    def test_to_lds_raises(self) -> None:
        m = _build(R=2, K_a=1, n_within=(0, 0))
        with pytest.raises(NotImplementedError, match="to_lds"):
            m.to_lds(8)

    def test_cov_freq_returns_per_latent_psd_shape(self) -> None:
        # PR 5c implemented cov_freq; returns (T, K_a + Σ K_w[r]) PSD values.
        m = _build(R=2, K_a=1, n_within=(0, 0))
        psd = m.cov_freq(8)
        assert psd.shape == (8, 1)
        assert torch.isfinite(psd).all()
        assert (psd > 0).all()


class TestExactGPLatentValidation:
    def test_rejects_n_within_length_mismatch(self) -> None:
        kf = lambda: MOSEKernel(num_regions=1, init_sigma=0.05)  # noqa: E731
        with pytest.raises(ValueError, match="n_within has length"):
            ExactGPLatent(
                n_regions=2,
                n_across=1,
                n_within=(1, 1, 1),  # length 3 != n_regions=2
                delay=FixedDelay(n_regions=2, n_latent=1, max_delay=1.0),
                kernel_factory_across=kf,
                kernel_factory_within=kf,
            )

    def test_requires_delay_when_across_present(self) -> None:
        kf = lambda: MOSEKernel(num_regions=1, init_sigma=0.05)  # noqa: E731
        with pytest.raises(ValueError, match="delay must be provided"):
            ExactGPLatent(
                n_regions=2,
                n_across=2,
                n_within=(0, 0),
                delay=None,
                kernel_factory_across=kf,
                kernel_factory_within=kf,
            )

    def test_rejects_delay_shape_mismatch(self) -> None:
        kf = lambda: MOSEKernel(num_regions=1, init_sigma=0.05)  # noqa: E731
        with pytest.raises(ValueError, match=r"delay\.n_latent"):
            ExactGPLatent(
                n_regions=2,
                n_across=2,
                n_within=(0, 0),
                delay=FixedDelay(n_regions=2, n_latent=1, max_delay=1.0),
                kernel_factory_across=kf,
                kernel_factory_within=kf,
            )

    def test_rejects_no_latents(self) -> None:
        kf = lambda: MOSEKernel(num_regions=1, init_sigma=0.05)  # noqa: E731
        with pytest.raises(ValueError, match="at least one latent"):
            ExactGPLatent(
                n_regions=2,
                n_across=0,
                n_within=(0, 0),
                delay=None,
                kernel_factory_across=kf,
                kernel_factory_within=kf,
            )

    def test_rejects_bad_eps(self) -> None:
        with pytest.raises(ValueError, match="eps_"):
            _build(R=2, K_a=1, n_within=(0, 0), eps=1.0)
