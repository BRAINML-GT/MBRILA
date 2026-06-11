"""Smoke / shape tests for the ADM model components.

These do *not* run a full training loop — see ``tests/recovery/`` for
parameter-recovery tests. The cases here are fast and verify the static
plumbing: kernel & delay shapes, kernel→LDS conversion structure,
block-diagonal assembly, observation block layout, model forward pass.
"""

from __future__ import annotations

import pytest
import torch

from mbrila import ADM, LatentSpec, MOSEKernel, MultiRegionLinearObservation, TimeVaryingDelay
from mbrila.dynamics.kernel_to_sde import kernel_to_lds, lag_pair_grid
from mbrila.dynamics.markov_gp import BlockDiagonalDynamics, MarkovianGPLatent
from mbrila.dynamics.ssm_base import block_diag_time, identity_shift_block

# ---------------------------------------------------------------------------
# MOSE kernel
# ---------------------------------------------------------------------------


class TestMOSEKernel:
    def test_time_invariant_shape(self) -> None:
        k = MOSEKernel(num_regions=3)
        tau = lag_pair_grid(lag=4, dtype=torch.float64, device=torch.device("cpu"))
        K = k.lagged_cov(tau.to(torch.float32))
        assert K.shape == (5, 5, 3, 3)

    def test_time_varying_shape(self) -> None:
        k = MOSEKernel(num_regions=2)
        tau = lag_pair_grid(lag=3, dtype=torch.float32, device=torch.device("cpu"))
        delays = torch.zeros(7, 2)
        K = k.lagged_cov(tau, delays)
        assert K.shape == (7, 4, 4, 2, 2)

    def test_time_invariant_equals_zero_delay(self) -> None:
        k = MOSEKernel(num_regions=2)
        tau = lag_pair_grid(lag=2, dtype=torch.float32, device=torch.device("cpu"))
        K_no = k.lagged_cov(tau)  # (lag+1, lag+1, R, R)
        delays = torch.zeros(4, 2)
        K_yes = k.lagged_cov(tau, delays)  # (4, lag+1, lag+1, R, R)
        for t in range(4):
            torch.testing.assert_close(K_yes[t], K_no, atol=1e-7, rtol=1e-7)

    def test_kernel_decreases_with_tau(self) -> None:
        k = MOSEKernel(num_regions=1, init_sigma=1.0)
        tau = lag_pair_grid(lag=5, dtype=torch.float64, device=torch.device("cpu")).to(torch.float32)
        K = k.lagged_cov(tau)
        # Diagonal entries (τ = 0) should be 1; off-diagonal should be smaller.
        diag = torch.diagonal(K[..., 0, 0])  # (lag+1,)
        torch.testing.assert_close(diag, torch.ones_like(diag), atol=1e-6, rtol=1e-6)
        # Cov falls monotonically with |τ|.
        for off in range(1, 6):
            assert (K[0, off, 0, 0] < K[0, off - 1, 0, 0]).item() if off > 0 else True

    def test_rejects_bad_num_regions(self) -> None:
        with pytest.raises(ValueError, match="num_regions"):
            MOSEKernel(num_regions=0)

    def test_rejects_bad_init_sigma(self) -> None:
        with pytest.raises(ValueError, match="init_sigma"):
            MOSEKernel(num_regions=1, init_sigma=-1.0)


# ---------------------------------------------------------------------------
# TimeVaryingDelay
# ---------------------------------------------------------------------------


class TestTimeVaryingDelay:
    def test_shape_and_reference_zero(self) -> None:
        delay = TimeVaryingDelay(n_regions=3, n_latent=2, T=10)
        d = delay.as_tensor(10)
        assert d.shape == (10, 3, 2)
        torch.testing.assert_close(d[:, 0, :], torch.zeros(10, 2, dtype=d.dtype))

    def test_zero_init_returns_zero(self) -> None:
        delay = TimeVaryingDelay(n_regions=2, n_latent=1, T=8, init_scale=0.0)
        d = delay.as_tensor(8)
        torch.testing.assert_close(d, torch.zeros(8, 2, 1, dtype=d.dtype))

    def test_smoothing_preserves_endpoints_with_reflect(self) -> None:
        delay = TimeVaryingDelay(n_regions=2, n_latent=1, T=12, init_scale=0.0)
        with torch.no_grad():
            delay.raw_delay.fill_(1.0)  # constant delay everywhere
        d = delay.as_tensor(12)
        # A constant input should remain (approximately) constant after smoothing.
        torch.testing.assert_close(d[:, 1, 0], torch.ones(12, dtype=d.dtype), atol=1e-5, rtol=1e-5)

    def test_T_mismatch_raises(self) -> None:
        delay = TimeVaryingDelay(n_regions=2, n_latent=1, T=10)
        with pytest.raises(ValueError, match="T="):
            delay.as_tensor(11)

    def test_phase_shift_shape(self) -> None:
        delay = TimeVaryingDelay(n_regions=3, n_latent=1, T=8)
        freqs = torch.linspace(-0.5, 0.5, 5)
        phase = delay.phase_shift(freqs, T=8)
        assert phase.shape == (8, 5, 3, 1)


# ---------------------------------------------------------------------------
# kernel_to_lds
# ---------------------------------------------------------------------------


class TestKernelToLDS:
    def test_time_invariant_shape(self) -> None:
        k = MOSEKernel(num_regions=2)
        tau = lag_pair_grid(lag=3, dtype=torch.float32, device=torch.device("cpu"))
        K_lag = k.lagged_cov(tau)
        F, Q = kernel_to_lds(K_lag, lag=3, num_dim=2)
        assert F.shape == (6, 6)
        assert Q.shape == (6, 6)

    def test_time_varying_shape(self) -> None:
        k = MOSEKernel(num_regions=2)
        tau = lag_pair_grid(lag=4, dtype=torch.float32, device=torch.device("cpu"))
        delays = torch.zeros(11, 2)
        K_lag = k.lagged_cov(tau, delays)
        F, Q = kernel_to_lds(K_lag, lag=4, num_dim=2)
        assert F.shape == (11, 8, 8)
        assert Q.shape == (11, 8, 8)

    def test_F_lower_block_is_identity_shift(self) -> None:
        k = MOSEKernel(num_regions=2)
        tau = lag_pair_grid(lag=3, dtype=torch.float32, device=torch.device("cpu"))
        K_lag = k.lagged_cov(tau)
        F, _ = kernel_to_lds(K_lag, lag=3, num_dim=2)
        # F[2:, :] should be the identity-shift block [I_4, 0_{4×2}].
        expected = identity_shift_block(lag=3, num_dim=2, dtype=F.dtype, device=F.device)
        torch.testing.assert_close(F[2:, :], expected)

    def test_Q_offdiagonal_zero(self) -> None:
        k = MOSEKernel(num_regions=2)
        tau = lag_pair_grid(lag=3, dtype=torch.float32, device=torch.device("cpu"))
        K_lag = k.lagged_cov(tau)
        _, Q = kernel_to_lds(K_lag, lag=3, num_dim=2)
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                block = Q[i * 2 : (i + 1) * 2, j * 2 : (j + 1) * 2]
                torch.testing.assert_close(block, torch.zeros_like(block))

    def test_Q_lag_blocks_replicate_innovation(self) -> None:
        """``Q`` lag blocks are always ``Q_full`` (the only supported mode).

        Historical ``"scaled"`` and ``"fixed"`` modes were removed — every
        scenario in the repo's production demos uses what was named
        ``"adm_compat"`` and the codepath collapsed to that.
        """
        k = MOSEKernel(num_regions=2, init_sigma=0.1)
        tau = lag_pair_grid(lag=3, dtype=torch.float64, device=torch.device("cpu"))
        K_lag = k.lagged_cov(tau).to(torch.float64)
        _, Q = kernel_to_lds(K_lag, lag=3, num_dim=2)
        Q_full = Q[:2, :2]
        torch.testing.assert_close(Q[2:4, 2:4], Q_full, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(Q[4:6, 4:6], Q_full, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# Multi-region observation
# ---------------------------------------------------------------------------


class TestMultiRegionObservation:
    def test_block_diag_C_shape(self) -> None:
        obs = MultiRegionLinearObservation(y_dims=(3, 5, 4), n_obs_per_region=2)
        assert obs.block_diag_C().shape == (12, 6)

    def test_diag_R_strictly_positive(self) -> None:
        obs = MultiRegionLinearObservation(y_dims=(3,), n_obs_per_region=1, init_R=0.1)
        with torch.no_grad():
            obs.diag_R_param.fill_(-1.0)
        assert (obs.diag_R() > 0).all().item()


# ---------------------------------------------------------------------------
# Block-diagonal dynamics
# ---------------------------------------------------------------------------


class TestBlockDiagonalDynamics:
    def test_assembly_shape(self) -> None:
        T = 6
        b1 = MarkovianGPLatent(MOSEKernel(num_regions=2), lag=3, T=T)
        b2 = MarkovianGPLatent(MOSEKernel(num_regions=1), lag=2, T=T)
        dyn = BlockDiagonalDynamics(
            [b1, b2],
            n_observable=2,
            observable_to_state_indices=[(0, 0), (1, 6)],  # b1 region 0 + b2 current
        )
        A, Q = dyn.forward()
        assert A.shape == (T, 8, 8)
        assert Q.shape == (T, 8, 8)
        assert dyn.total_state_dim == 8
        assert dyn.H_select.shape == (2, 8)


# ---------------------------------------------------------------------------
# block_diag_time helper
# ---------------------------------------------------------------------------


class TestBlockDiagTime:
    def test_basic(self) -> None:
        T = 4
        a = torch.randn(T, 2, 2)
        b = torch.randn(T, 3, 3)
        out = block_diag_time([a, b])
        assert out.shape == (T, 5, 5)
        torch.testing.assert_close(out[:, :2, :2], a)
        torch.testing.assert_close(out[:, 2:, 2:], b)
        torch.testing.assert_close(out[:, :2, 2:], torch.zeros(T, 2, 3))


# ---------------------------------------------------------------------------
# ADM end-to-end (no training; just one forward pass)
# ---------------------------------------------------------------------------


def _mose_factories(R: int, sigma: float = 0.1) -> dict[str, object]:
    return {
        "kernel_factory_across": lambda: MOSEKernel(num_regions=R, init_sigma=sigma),
        "kernel_factory_within": lambda: MOSEKernel(num_regions=1, init_sigma=sigma),
    }


class TestADMSmoke:
    def test_construction_and_state_dims(self) -> None:
        spec = LatentSpec(n_across=2, n_within=(1, 1, 1))
        m = ADM(
            spec,
            y_dims=(3, 4, 5),
            T=12,
            lag_across=4,
            lag_within=2,
            device="cpu",
            **_mose_factories(3),  # type: ignore[arg-type]
        )
        # Total state = 2 across × 4 lag × 3 regions + 3 regions × 1 within × 2 lag = 24 + 6 = 30
        assert m.dynamics.total_state_dim == 30
        # Observable per region = 2 across + 1 within = 3; total = 9
        assert m.dynamics.H_select.shape == (9, 30)

    def test_forward_shapes(self) -> None:
        spec = LatentSpec(n_across=1, n_within=(1, 1))
        m = ADM(
            spec,
            y_dims=(2, 3),
            T=8,
            lag_across=3,
            lag_within=2,
            device="cpu",
            dtype=torch.float64,
            **_mose_factories(2),  # type: ignore[arg-type]
        )
        A, Q = m.dynamics.forward()
        assert A.shape == (8, 2 * 3 + 2 * 2, 2 * 3 + 2 * 2)  # 1×3×2 + 2×1×2 = 10
        assert Q.shape == A.shape

    def test_save_load_round_trip(self, tmp_path: object) -> None:
        from pathlib import Path

        assert isinstance(tmp_path, Path)
        spec = LatentSpec(n_across=1, n_within=(1, 1))
        m = ADM(
            spec,
            y_dims=(2, 3),
            T=6,
            device="cpu",
            dtype=torch.float64,
            **_mose_factories(2),  # type: ignore[arg-type]
        )
        with torch.no_grad():
            m.observation.diag_R_param.fill_(0.42)
        path = tmp_path / "adm.pt"
        m.save(path)
        loaded = ADM.load(
            path,
            device="cpu",
            dtype=torch.float64,
            **_mose_factories(2),  # type: ignore[arg-type]
        )
        torch.testing.assert_close(
            loaded.observation.diag_R_param, m.observation.diag_R_param, atol=1e-9, rtol=1e-9
        )

    def test_score_returns_finite(self) -> None:
        spec = LatentSpec(n_across=1, n_within=(1, 1))
        m = ADM(
            spec,
            y_dims=(2, 3),
            T=6,
            device="cpu",
            dtype=torch.float64,
            **_mose_factories(2),  # type: ignore[arg-type]
        )
        data = m.sample(n_trials=2, T=6, seed=0)
        ll = m.score(data)
        assert isinstance(ll, float)
        assert torch.isfinite(torch.tensor(ll)).item()

    def test_capabilities_advertise_to_lds(self) -> None:
        spec = LatentSpec(n_across=1, n_within=(1, 1))
        m = ADM(
            spec,
            y_dims=(2, 3),
            T=6,
            device="cpu",
            **_mose_factories(2),  # type: ignore[arg-type]
        )
        assert "to_lds" in m.capabilities()
