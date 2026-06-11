"""Smoke / shape tests for the DLAG model class.

These do not run a full training loop; see ``tests/recovery/`` for
parameter-recovery checks. The cases here verify the static plumbing
(construction, sample, score, save/load, capability advertisement,
init helpers).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mbrila import DLAG, LatentSpec, MOSEKernel


def _kf(R: int, sigma: float = 0.05) -> dict[str, object]:
    """Default MOSE kernel factories for the test fixtures."""
    return {
        "kernel_factory_across": lambda: MOSEKernel(num_regions=R, init_sigma=sigma),
        "kernel_factory_within": lambda: MOSEKernel(num_regions=1, init_sigma=sigma),
    }


class TestDLAGConstruction:
    def test_basic_dims(self) -> None:
        spec = LatentSpec(n_across=2, n_within=(1, 1, 1))
        m = DLAG(spec, y_dims=(4, 5, 6), T=10, device="cpu", **_kf(3))  # type: ignore[arg-type]
        # M per time = R · K_a + Σ K_w = 3·2 + 3 = 9
        assert m.dynamics.state_dim_per_time == 9

    def test_rejects_heterogeneous_within(self) -> None:
        with pytest.raises(ValueError, match="uniform n_within"):
            DLAG(
                LatentSpec(n_across=1, n_within=(1, 2)),
                y_dims=(4, 5),
                T=8,
                device="cpu",
                **_kf(2),  # type: ignore[arg-type]
            )

    def test_rejects_wrong_y_dims_length(self) -> None:
        with pytest.raises(ValueError, match="y_dims"):
            DLAG(
                LatentSpec(n_across=1, n_within=(1, 1)),
                y_dims=(4,),  # 1 region but spec has 2
                T=8,
                device="cpu",
                **_kf(2),  # type: ignore[arg-type]
            )

    def test_max_delay_default(self) -> None:
        m = DLAG(
            LatentSpec(n_across=1, n_within=(1, 1)),
            y_dims=(4, 5),
            T=20,
            device="cpu",
            **_kf(2),  # type: ignore[arg-type]
        )
        assert m._max_delay == 10  # floor(T/2)

    def test_capabilities_advertise_cov_full(self) -> None:
        m = DLAG(
            LatentSpec(n_across=1, n_within=(1, 1)),
            y_dims=(4, 5),
            T=8,
            device="cpu",
            **_kf(2),  # type: ignore[arg-type]
        )
        assert "cov_full" in m.capabilities()


class TestDLAGSample:
    def test_sample_shape(self) -> None:
        m = DLAG(
            LatentSpec(n_across=1, n_within=(1, 1)),
            y_dims=(3, 4),
            T=6,
            device="cpu",
            **_kf(2),  # type: ignore[arg-type]
        )
        data = m.sample(n_trials=5, T=6, seed=0)
        assert data.y.shape == (5, 6, 7)
        assert data.y.dtype == torch.float64

    def test_sample_T_mismatch_raises(self) -> None:
        m = DLAG(
            LatentSpec(n_across=1, n_within=(1, 1)),
            y_dims=(3, 4),
            T=6,
            device="cpu",
            **_kf(2),  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="must match model T"):
            m.sample(n_trials=2, T=10)


class TestDLAGScore:
    def test_score_finite(self) -> None:
        m = DLAG(
            LatentSpec(n_across=1, n_within=(1, 1)),
            y_dims=(3, 4),
            T=6,
            device="cpu",
            **_kf(2),  # type: ignore[arg-type]
        )
        data = m.sample(n_trials=2, T=6, seed=0)
        ll = m.score(data)
        assert isinstance(ll, float)
        assert torch.isfinite(torch.tensor(ll)).item()


class TestDLAGSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        m = DLAG(
            LatentSpec(n_across=1, n_within=(1, 1)),
            y_dims=(3, 4),
            T=6,
            device="cpu",
            **_kf(2),  # type: ignore[arg-type]
        )
        with torch.no_grad():
            m.observation.diag_R_param.fill_(0.5)
            m.dynamics.kernel_across[0].log_sigma.data.fill_(-2.0)
        path = tmp_path / "dlag.pt"
        m.save(path)
        loaded = DLAG.load(path, device="cpu", dtype=torch.float64, **_kf(2))
        torch.testing.assert_close(
            loaded.observation.diag_R_param,
            m.observation.diag_R_param,
            atol=1e-9,
            rtol=1e-9,
        )
        torch.testing.assert_close(
            loaded.dynamics.kernel_across[0].log_sigma,
            m.dynamics.kernel_across[0].log_sigma,
            atol=1e-9,
            rtol=1e-9,
        )


class TestDLAGInitialiseFromData:
    def test_fa_init_does_not_raise(self) -> None:
        m = DLAG(
            LatentSpec(n_across=1, n_within=(1, 1)),
            y_dims=(5, 6),
            T=8,
            device="cpu",
            **_kf(2),  # type: ignore[arg-type]
        )
        data = m.sample(n_trials=20, T=8, seed=0)
        C_before = m.observation.block_diag_C().clone()
        m.initialize_from_data(data, mode="fa", fa_max_iter=5)
        assert (m.observation.block_diag_C() - C_before).abs().max().item() > 1e-6

    def test_rejects_unknown_mode(self) -> None:
        m = DLAG(
            LatentSpec(n_across=1, n_within=(1, 1)),
            y_dims=(4, 5),
            T=6,
            device="cpu",
            **_kf(2),  # type: ignore[arg-type]
        )
        data = m.sample(n_trials=4, T=6, seed=0)
        with pytest.raises(ValueError, match="unknown init mode"):
            m.initialize_from_data(data, mode="bogus")  # type: ignore[arg-type]
