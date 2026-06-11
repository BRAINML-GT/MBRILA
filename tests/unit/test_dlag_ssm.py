"""Tests for CF4.5 — DLAG-SSM via ``engine="kalman"``.

The exact-GP path is unchanged from CF3 (existing tests cover it). These
tests focus on:

1. ``engine="kalman"`` builds the SSM stack with the right component types.
2. Both engines run ``sample`` / ``score`` end-to-end.
3. ``initialize_from_data`` (pCCA) works on both paths since they share
   :class:`MultiRegionLinearObservation`.
4. Save/load round-trips and preserves the engine choice.
5. ``build_model("dlag", engine="kalman", ...)`` dispatches correctly.
6. Type-safety: ``engine_override`` must match the chosen engine kind.

No recovery tests — those are the user's responsibility per CLAUDE.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mbrila import DLAG, LatentSpec, MOSEKernel, build_model
from mbrila.delays.fixed import FixedDelay
from mbrila.dynamics.exact_gp import ExactGPLatent
from mbrila.dynamics.markov_gp import BlockDiagonalDynamics, MarkovianGPLatent
from mbrila.inference.em_exact import ExactEMEngine
from mbrila.inference.kalman_em import KalmanEMEngine


def _make_spec(n_regions: int = 2, n_across: int = 2, n_within: int = 1) -> LatentSpec:
    return LatentSpec(n_across=n_across, n_within=tuple([n_within] * n_regions))


def _mose_factories(R: int, sigma: float = 0.1) -> dict[str, object]:
    return {
        "kernel_factory_across": lambda: MOSEKernel(num_regions=R, init_sigma=sigma),
        "kernel_factory_within": lambda: MOSEKernel(num_regions=1, init_sigma=sigma),
    }


# ---------------------------------------------------------------------------
# Engine dispatch on construction
# ---------------------------------------------------------------------------


class TestEngineDispatch:
    def test_default_engine_is_exact(self) -> None:
        model = DLAG(latent_spec=_make_spec(), y_dims=(4, 5), T=10, **_mose_factories(2))  # type: ignore[arg-type]
        assert isinstance(model.dynamics, ExactGPLatent)
        assert isinstance(model.inference, ExactEMEngine)

    def test_engine_kalman_builds_ssm_stack(self) -> None:
        model = DLAG(
            latent_spec=_make_spec(),
            y_dims=(4, 5),
            T=10,
            engine="kalman",
            lag_across=3,
            lag_within=2,
            **_mose_factories(2),  # type: ignore[arg-type]
        )
        assert isinstance(model.dynamics, BlockDiagonalDynamics)
        assert isinstance(model.inference, KalmanEMEngine)
        # Across blocks should each carry a FixedDelay; within blocks shouldn't.
        n_across = model.latent_spec.n_across
        for i in range(n_across):
            block = model.dynamics.blocks[i]
            assert isinstance(block, MarkovianGPLatent)
            assert isinstance(block.delay, FixedDelay)
            assert isinstance(block.kernel, MOSEKernel)
        # Within blocks: no delay.
        for j in range(n_across, len(model.dynamics.blocks)):
            block = model.dynamics.blocks[j]
            assert isinstance(block, MarkovianGPLatent)
            assert block.delay is None

    def test_kalman_state_dim_matches_layout(self) -> None:
        """Same H_select layout as ADM/GPFA — across first, then within."""
        spec = _make_spec(n_regions=3, n_across=2, n_within=1)
        model = DLAG(
            latent_spec=spec,
            y_dims=(4, 4, 4),
            T=8,
            engine="kalman",
            lag_across=3,
            lag_within=2,
            **_mose_factories(3),  # type: ignore[arg-type]
        )
        # Across: 2 * 3 * 3 = 18; Within: 3 * 1 * 2 = 6; total 24.
        assert isinstance(model.dynamics, BlockDiagonalDynamics)
        assert model.dynamics.total_state_dim == 24
        # n_observable = 3 * (2 + 1) = 9
        assert model.dynamics.H_select.shape == (9, 24)

    def test_unknown_engine_rejected(self) -> None:
        with pytest.raises(ValueError, match="engine must be"):
            DLAG(latent_spec=_make_spec(), y_dims=(4, 5), T=10, engine="bogus", **_mose_factories(2))  # type: ignore[arg-type]

    def test_kalman_requires_positive_lags(self) -> None:
        with pytest.raises(ValueError, match="lag_across and lag_within"):
            DLAG(
                latent_spec=_make_spec(),
                y_dims=(4, 5),
                T=10,
                engine="kalman",
                lag_across=0,
                lag_within=2,
                **_mose_factories(2),  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# engine_override type safety
# ---------------------------------------------------------------------------


class TestEngineOverride:
    def test_exact_override_with_exact_engine(self) -> None:
        engine = ExactEMEngine()
        model = DLAG(
            latent_spec=_make_spec(),
            y_dims=(4, 5),
            T=10,
            engine="exact",
            engine_override=engine,
            **_mose_factories(2),  # type: ignore[arg-type]
        )
        assert model.inference is engine

    def test_kalman_override_with_kalman_engine(self) -> None:
        engine = KalmanEMEngine()
        model = DLAG(
            latent_spec=_make_spec(),
            y_dims=(4, 5),
            T=10,
            engine="kalman",
            engine_override=engine,
            **_mose_factories(2),  # type: ignore[arg-type]
        )
        assert model.inference is engine

    def test_mismatched_override_kalman_with_exact_engine(self) -> None:
        with pytest.raises(TypeError, match="must be KalmanEMEngine"):
            DLAG(
                latent_spec=_make_spec(),
                y_dims=(4, 5),
                T=10,
                engine="kalman",
                engine_override=ExactEMEngine(),
                **_mose_factories(2),  # type: ignore[arg-type]
            )

    def test_mismatched_override_exact_with_kalman_engine(self) -> None:
        with pytest.raises(TypeError, match="must be ExactEMEngine"):
            DLAG(
                latent_spec=_make_spec(),
                y_dims=(4, 5),
                T=10,
                engine="exact",
                engine_override=KalmanEMEngine(),
                **_mose_factories(2),  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# Sample + score end-to-end on both engines
# ---------------------------------------------------------------------------


class TestSampleScoreBothEngines:
    def test_sample_exact(self) -> None:
        model = DLAG(latent_spec=_make_spec(), y_dims=(3, 4), T=8, **_mose_factories(2))  # type: ignore[arg-type]
        data = model.sample(n_trials=2, T=8, seed=0)
        assert data.y.shape == (2, 8, 3 + 4)
        assert torch.isfinite(data.y).all().item()

    def test_sample_kalman(self) -> None:
        model = DLAG(
            latent_spec=_make_spec(),
            y_dims=(3, 4),
            T=8,
            engine="kalman",
            lag_across=3,
            lag_within=2,
            **_mose_factories(2),  # type: ignore[arg-type]
        )
        data = model.sample(n_trials=2, T=8, seed=0)
        assert data.y.shape == (2, 8, 3 + 4)
        assert torch.isfinite(data.y).all().item()

    def test_score_both_engines(self) -> None:
        for engine in ("exact", "kalman"):
            model = DLAG(
                latent_spec=_make_spec(),
                y_dims=(3, 3),
                T=6,
                engine=engine,  # type: ignore[arg-type]
                lag_across=3,
                lag_within=2,
                **_mose_factories(2),  # type: ignore[arg-type]
            )
            data = model.sample(n_trials=2, T=6, seed=0)
            ll = model.score(data)
            assert torch.isfinite(torch.tensor(ll)).item()


# ---------------------------------------------------------------------------
# pCCA initialise_from_data on both engines (shared MultiRegionLinearObservation)
# ---------------------------------------------------------------------------


class TestInitializeFromDataBothEngines:
    def test_pcca_init_exact(self) -> None:
        model = DLAG(latent_spec=_make_spec(), y_dims=(4, 5), T=8, **_mose_factories(2))  # type: ignore[arg-type]
        data = model.sample(n_trials=8, T=8, seed=0)
        model.initialize_from_data(data, mode="pcca")
        # C should be non-degenerate after pCCA seeding.
        assert torch.isfinite(model.observation.Cs[0]).all().item()
        assert model.observation.Cs[0].abs().max().item() > 0.0

    def test_pcca_init_kalman(self) -> None:
        """Same pCCA seeding works on the Kalman path — they share the
        :class:`MultiRegionLinearObservation` emission."""
        model = DLAG(
            latent_spec=_make_spec(),
            y_dims=(4, 5),
            T=8,
            engine="kalman",
            lag_across=3,
            lag_within=2,
            **_mose_factories(2),  # type: ignore[arg-type]
        )
        # Sample using a fresh exact-path model with the same structural spec,
        # so the kalman model has realistic data to init against.
        sampler = DLAG(latent_spec=_make_spec(), y_dims=(4, 5), T=8, **_mose_factories(2))  # type: ignore[arg-type]
        data = sampler.sample(n_trials=8, T=8, seed=0)
        model.initialize_from_data(data, mode="pcca")
        assert torch.isfinite(model.observation.Cs[0]).all().item()
        assert model.observation.Cs[0].abs().max().item() > 0.0


# ---------------------------------------------------------------------------
# Save / load round-trip preserves engine choice
# ---------------------------------------------------------------------------


class TestSaveLoadEngineRoundTrip:
    def test_exact_round_trip(self, tmp_path: Path) -> None:
        model = DLAG(latent_spec=_make_spec(), y_dims=(3, 4), T=6, **_mose_factories(2))  # type: ignore[arg-type]
        path = tmp_path / "dlag_exact.pt"
        model.save(path)
        loaded = DLAG.load(path, **_mose_factories(2))
        assert isinstance(loaded.dynamics, ExactGPLatent)
        assert isinstance(loaded.inference, ExactEMEngine)

    def test_kalman_round_trip(self, tmp_path: Path) -> None:
        model = DLAG(
            latent_spec=_make_spec(),
            y_dims=(3, 4),
            T=6,
            engine="kalman",
            lag_across=3,
            lag_within=2,
            **_mose_factories(2),  # type: ignore[arg-type]
        )
        path = tmp_path / "dlag_kalman.pt"
        model.save(path)
        loaded = DLAG.load(path, **_mose_factories(2))  # type: ignore[arg-type]
        assert isinstance(loaded.dynamics, BlockDiagonalDynamics)
        assert isinstance(loaded.inference, KalmanEMEngine)
        # Lags survived the round-trip.
        assert loaded._lag_across == 3
        assert loaded._lag_within == 2

    def test_legacy_config_defaults_to_exact(self) -> None:
        """Old DLAG checkpoints (pre-CF4.5) don't have the ``engine`` key.
        ``from_config`` must default it to ``"exact"`` for backwards compat."""
        legacy_config = {
            "n_across": 2,
            "n_within": [1, 1],
            "y_dims": [3, 4],
            "T": 6,
            "init_gamma_across": 0.01,
            "init_gamma_within": 0.01,
            "eps_across": 1e-3,
            "eps_within": 1e-3,
            "max_delay": 3.0,
        }
        model = DLAG.from_config(legacy_config, **_mose_factories(2))
        assert isinstance(model.inference, ExactEMEngine)


# ---------------------------------------------------------------------------
# build_model dispatch
# ---------------------------------------------------------------------------


class TestBuildModelDLAGSSM:
    def test_build_model_dlag_kalman(self) -> None:
        model = build_model(
            "dlag",
            latent_spec=_make_spec(),
            y_dims=(4, 5),
            T=10,
            engine="kalman",
            lag_across=3,
            lag_within=2,
            **_mose_factories(2),
        )
        assert isinstance(model, DLAG)
        assert isinstance(model.dynamics, BlockDiagonalDynamics)
        assert isinstance(model.inference, KalmanEMEngine)

    def test_build_model_dlag_default_engine_unchanged(self) -> None:
        """Backwards compat: build_model('dlag', ...) without engine kwarg
        must still produce the historical exact-GP DLAG."""
        model = build_model(
            "dlag",
            latent_spec=_make_spec(),
            y_dims=(4, 5),
            T=10,
            **_mose_factories(2),
        )
        assert isinstance(model, DLAG)
        assert isinstance(model.dynamics, ExactGPLatent)
        assert isinstance(model.inference, ExactEMEngine)
