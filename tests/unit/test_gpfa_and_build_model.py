"""Tests for the CF4 deliverables.

1. :class:`GPFA` — the first preset that composes the framework's 4 axes
   end-to-end (NoDelay + arbitrary kernel + linear-Gaussian observation +
   Kalman engine). Verifies construction, sample/score smoke, and the
   kernel-pluggability that justifies CF3's BaseKernel work.

2. :func:`build_model` — the registry-dispatched entry point. Verifies
   it routes to the four built-in presets, propagates kwargs, and reports
   useful errors on unknown names.

No fitting / recovery tests here; those are the user's responsibility
per CLAUDE.md. Smoke tests only.
"""

from __future__ import annotations

import pytest
import torch

from mbrila import (
    ADM,
    DLAG,
    GPFA,
    MDLAG,
    LatentSpec,
    Matern32Kernel,
    Matern52Kernel,
    MOSEKernel,
    build_model,
    model_registry,
)
from mbrila.delays.none import NoDelay
from mbrila.dynamics.markov_gp import BlockDiagonalDynamics, MarkovianGPLatent
from mbrila.inference.kalman_em import KalmanEMEngine
from mbrila.kernels.base import BaseKernel

# ---------------------------------------------------------------------------
# GPFA: construction and structural invariants
# ---------------------------------------------------------------------------


def _make_spec(n_regions: int = 2, n_across: int = 2, n_within: int = 0) -> LatentSpec:
    return LatentSpec(n_across=n_across, n_within=tuple([n_within] * n_regions))


def _mose_factories(R: int, sigma: float = 0.1) -> dict[str, object]:
    """Both-axis factories — for ADM / DLAG / others that take within kernels."""
    return {
        "kernel_factory_across": lambda: MOSEKernel(num_regions=R, init_sigma=sigma),
        "kernel_factory_within": lambda: MOSEKernel(num_regions=1, init_sigma=sigma),
    }


def _mose_factory_across(R: int, sigma: float = 0.1) -> dict[str, object]:
    """GPFA / MDLAG-style — single across factory (no within latents)."""
    return {"kernel_factory_across": lambda: MOSEKernel(num_regions=R, init_sigma=sigma)}


class TestGPFAConstruction:
    def test_default_construction_uses_MOSE_NoDelay_Kalman(self) -> None:
        spec = _make_spec()  # n_within=0 by default — shared-only
        model = GPFA(latent_spec=spec, y_dims=(4, 5), T=10, **_mose_factory_across(2))  # type: ignore[arg-type]
        assert isinstance(model.inference, KalmanEMEngine)
        assert isinstance(model.delay, NoDelay)
        dynamics = model.dynamics
        assert isinstance(dynamics, BlockDiagonalDynamics)
        # GPFA = K_a across blocks, no within blocks.
        assert len(dynamics.blocks) == spec.n_across
        for block in dynamics.blocks:
            assert isinstance(block, MarkovianGPLatent)
            assert block.delay is not None
            assert block.delay.is_time_varying is False
            assert torch.all(block.delay.as_tensor() == 0)
            assert isinstance(block.kernel, MOSEKernel)

    def test_state_dim_matches_expected_layout(self) -> None:
        spec = _make_spec(n_regions=3, n_across=2)
        model = GPFA(
            latent_spec=spec,
            y_dims=(4, 4, 4),
            T=8,
            lag_across=3,
            **_mose_factory_across(3),  # type: ignore[arg-type]
        )
        # Across only: n_across * lag_a * n_regions = 2*3*3 = 18.
        assert isinstance(model.dynamics, BlockDiagonalDynamics)
        assert model.dynamics.total_state_dim == 18
        # H_select picks out n_observable = n_regions * n_across = 3*2 = 6 slots.
        assert model.dynamics.H_select.shape == (6, 18)

    def test_rejects_mismatched_y_dims(self) -> None:
        spec = _make_spec(n_regions=2)
        with pytest.raises(ValueError, match="y_dims has"):
            GPFA(latent_spec=spec, y_dims=(4, 5, 6), T=10, **_mose_factory_across(2))  # type: ignore[arg-type]

    def test_rejects_within_nonzero(self) -> None:
        """GPFA is shared-only; non-zero within count must error."""
        with pytest.raises(ValueError, match="shared-only"):
            GPFA(
                latent_spec=LatentSpec(n_across=1, n_within=(1, 1)),
                y_dims=(4, 5),
                T=10,
                **_mose_factory_across(2),  # type: ignore[arg-type]
            )

    def test_rejects_within_nonzero_single_region(self) -> None:
        with pytest.raises(ValueError, match="shared-only"):
            GPFA(
                latent_spec=LatentSpec(n_across=1, n_within=(0, 2)),
                y_dims=(4, 5),
                T=10,
                **_mose_factory_across(2),  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# GPFA: kernel pluggability — Matérn drops in via the factory hook
# ---------------------------------------------------------------------------


class TestGPFAKernelPluggable:
    def test_matern_32_as_across_kernel(self) -> None:
        spec = _make_spec(n_regions=2, n_across=2)
        model = GPFA(
            latent_spec=spec,
            y_dims=(4, 4),
            T=8,
            kernel_factory_across=lambda: Matern32Kernel(lengthscale=2.0),
        )
        assert isinstance(model.dynamics, BlockDiagonalDynamics)
        for block in model.dynamics.blocks:
            assert isinstance(block, MarkovianGPLatent)
            assert isinstance(block.kernel, Matern32Kernel)
            assert block.kernel.is_markovian is True

    def test_matern_52_across(self) -> None:
        spec = _make_spec(n_regions=2, n_across=1)
        model = GPFA(
            latent_spec=spec,
            y_dims=(3, 3),
            T=6,
            kernel_factory_across=lambda: Matern52Kernel(lengthscale=1.5),
        )
        assert isinstance(model.dynamics, BlockDiagonalDynamics)
        for block in model.dynamics.blocks:
            assert isinstance(block, MarkovianGPLatent)
            assert isinstance(block.kernel, Matern52Kernel)

    def test_custom_BaseKernel_subclass(self) -> None:
        """Load-bearing test: a user-defined ``BaseKernel`` subclass
        plumbs through the entire model assembly with no special hooks."""

        class _MyKernel(BaseKernel):
            def __init__(self) -> None:
                super().__init__()
                self.log_ell = torch.nn.Parameter(torch.log(torch.tensor(1.5, dtype=torch.float64)))

            def cov(self, tau: torch.Tensor) -> torch.Tensor:
                ell = torch.exp(self.log_ell)
                return torch.exp(-tau.to(ell.dtype).abs() / ell)

        model = GPFA(
            latent_spec=_make_spec(),
            y_dims=(3, 3),
            T=6,
            kernel_factory_across=_MyKernel,
        )
        assert isinstance(model.dynamics, BlockDiagonalDynamics)
        first_across = model.dynamics.blocks[0]
        assert isinstance(first_across, MarkovianGPLatent)
        assert isinstance(first_across.kernel, _MyKernel)


# ---------------------------------------------------------------------------
# GPFA: end-to-end sample / score smoke tests
# ---------------------------------------------------------------------------


class TestGPFASample:
    def test_sample_shapes_default_kernel(self) -> None:
        spec = _make_spec(n_regions=2, n_across=2)
        model = GPFA(latent_spec=spec, y_dims=(4, 5), T=8, **_mose_factory_across(2))  # type: ignore[arg-type]
        data = model.sample(n_trials=3, T=8, seed=0)
        assert data.y.shape == (3, 8, 4 + 5)

    def test_sample_with_matern(self) -> None:
        """End-to-end: Matern produces a usable model that samples cleanly."""
        spec = _make_spec(n_regions=2, n_across=1)
        model = GPFA(
            latent_spec=spec,
            y_dims=(3, 3),
            T=6,
            kernel_factory_across=lambda: Matern32Kernel(lengthscale=2.0),
        )
        data = model.sample(n_trials=2, T=6, seed=0)
        assert data.y.shape == (2, 6, 6)
        assert torch.isfinite(data.y).all().item()

    def test_sample_T_mismatch_raises(self) -> None:
        spec = _make_spec()
        model = GPFA(latent_spec=spec, y_dims=(4, 5), T=10, **_mose_factory_across(2))  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="sample T must match"):
            model.sample(n_trials=2, T=12)


class TestGPFAScore:
    def test_score_returns_finite(self) -> None:
        spec = _make_spec(n_regions=2, n_across=1)
        model = GPFA(latent_spec=spec, y_dims=(3, 3), T=6, **_mose_factory_across(2))  # type: ignore[arg-type]
        data = model.sample(n_trials=2, T=6, seed=0)
        ll = model.score(data)
        assert torch.isfinite(torch.tensor(ll)).item()


# ---------------------------------------------------------------------------
# GPFA: save / load round-trip
# ---------------------------------------------------------------------------


class TestGPFASaveLoad:
    def test_save_load_round_trip(self, tmp_path: object) -> None:
        from pathlib import Path

        assert isinstance(tmp_path, Path)
        spec = _make_spec(n_regions=2, n_across=1)
        model = GPFA(latent_spec=spec, y_dims=(3, 3), T=6, **_mose_factory_across(2))  # type: ignore[arg-type]
        path = tmp_path / "gpfa.pt"
        model.save(path)
        loaded = GPFA.load(path, **_mose_factory_across(2))  # type: ignore[arg-type]
        assert isinstance(loaded, GPFA)
        assert loaded.latent_spec.n_across == spec.n_across
        assert isinstance(loaded.dynamics, BlockDiagonalDynamics)


# ---------------------------------------------------------------------------
# build_model dispatch
# ---------------------------------------------------------------------------


class TestBuildModelDispatch:
    def test_registry_lists_all_presets(self) -> None:
        names = set(model_registry.names())
        assert {"adm", "dlag", "mdlag", "gpfa"} <= names

    def test_gpfa_via_build_model(self) -> None:
        model = build_model(
            "gpfa",
            latent_spec=_make_spec(),  # shared-only by default
            y_dims=(4, 5),
            T=10,
            **_mose_factory_across(2),
        )
        assert isinstance(model, GPFA)

    def test_adm_via_build_model(self) -> None:
        # ADM defaults need within latents — pass an n_within > 0 spec.
        model = build_model(
            "adm",
            latent_spec=_make_spec(n_within=1),
            y_dims=(4, 5),
            T=10,
            **_mose_factories(2),
        )
        assert isinstance(model, ADM)

    def test_dlag_via_build_model(self) -> None:
        model = build_model(
            "dlag",
            latent_spec=_make_spec(n_regions=2, n_across=1, n_within=1),
            y_dims=(4, 5),
            T=10,
            **_mose_factories(2),
        )
        assert isinstance(model, DLAG)

    def test_mdlag_via_build_model(self) -> None:
        # mDLAG requires n_within = 0 across regions and selection="ard".
        spec = LatentSpec(n_across=2, n_within=(0, 0), selection="ard")
        model = build_model(
            "mdlag",
            latent_spec=spec,
            y_dims=(4, 5),
            T=10,
            kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.1),
        )
        assert isinstance(model, MDLAG)

    def test_unknown_preset_errors(self) -> None:
        with pytest.raises(KeyError, match="unknown model"):
            build_model("not_a_real_model", latent_spec=_make_spec(), y_dims=(4, 5), T=10)

    def test_preset_name_case_insensitive(self) -> None:
        model = build_model(
            "GPFA",
            latent_spec=_make_spec(),  # shared-only by default
            y_dims=(4, 5),
            T=10,
            **_mose_factory_across(2),
        )
        assert isinstance(model, GPFA)

    def test_kwargs_propagate(self) -> None:
        # Pass a non-default `lag_across` and verify it lands on the model.
        model = build_model(
            "gpfa",
            latent_spec=_make_spec(),
            y_dims=(4, 5),
            T=10,
            lag_across=7,
            **_mose_factory_across(2),
        )
        assert isinstance(model, GPFA)
        assert model._lag_across == 7


# ---------------------------------------------------------------------------
# MarkovianGPLatent CF4 decoupling
# ---------------------------------------------------------------------------


class TestMarkovianGPLatentNumDimDecoupling:
    """num_dim now lives on MarkovianGPLatent, not on the kernel.

    This makes Matérn (which has no ``num_regions`` attribute) plumb
    through unmodified — the load-bearing point of the decoupling.
    """

    def test_matern_kernel_requires_explicit_num_dim(self) -> None:
        kernel = Matern32Kernel(lengthscale=1.0)
        with pytest.raises(ValueError, match="num_dim explicitly"):
            MarkovianGPLatent(kernel=kernel, lag=2, T=5)

    def test_matern_kernel_with_explicit_num_dim_works(self) -> None:
        kernel = Matern32Kernel(lengthscale=1.0)
        block = MarkovianGPLatent(kernel=kernel, lag=2, T=5, num_dim=3)
        assert block.num_dim == 3
        assert block.state_dim == 6
        A, Q = block.forward()
        assert A.shape == (5, 6, 6)
        assert Q.shape == (5, 6, 6)

    def test_mose_kernel_legacy_path_still_works(self) -> None:
        """Backward compat: MOSE without explicit num_dim still works."""
        kernel = MOSEKernel(num_regions=2)
        block = MarkovianGPLatent(kernel=kernel, lag=2, T=5)
        assert block.num_dim == 2  # inferred from kernel.num_regions

    def test_num_dim_mismatch_with_kernel_num_regions_raises(self) -> None:
        kernel = MOSEKernel(num_regions=2)
        with pytest.raises(ValueError, match="disagrees"):
            MarkovianGPLatent(kernel=kernel, lag=2, T=5, num_dim=3)
