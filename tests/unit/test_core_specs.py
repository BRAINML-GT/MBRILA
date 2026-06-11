"""Sanity checks for LatentSpec, ARDPriorConfig, DiscreteStateSpec, Delay, Observation."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from mbrila import (
    ARDPriorConfig,
    Delay,
    DiscreteStateSpec,
    LatentSpec,
    Observation,
)

# --- LatentSpec ---------------------------------------------------------


class TestLatentSpec:
    def test_basic_geometry(self) -> None:
        spec = LatentSpec(n_across=2, n_within=(3, 4))
        assert spec.n_regions == 2
        assert spec.n_latent_total == 9
        assert spec.selection == "fixed"

    def test_ard_default_prior(self) -> None:
        spec = LatentSpec(n_across=2, n_within=(3,), selection="ard")
        assert isinstance(spec.ard_prior, ARDPriorConfig)
        assert spec.ard_prior.shape == 1e-3

    def test_inducing_requires_count(self) -> None:
        with pytest.raises(ValueError, match="n_inducing"):
            LatentSpec(n_across=2, n_within=(3,), selection="inducing")

    def test_ard_rejects_inducing(self) -> None:
        with pytest.raises(ValueError, match="n_inducing"):
            LatentSpec(
                n_across=2,
                n_within=(3,),
                selection="ard",
                n_inducing=10,
            )

    def test_fixed_rejects_ard_prior(self) -> None:
        with pytest.raises(ValueError, match="ard_prior"):
            LatentSpec(
                n_across=2,
                n_within=(3,),
                selection="fixed",
                ard_prior=ARDPriorConfig(),
            )

    def test_rejects_empty_within(self) -> None:
        with pytest.raises(ValueError, match="at least one region"):
            LatentSpec(n_across=2, n_within=())

    def test_rejects_zero_total_latents(self) -> None:
        with pytest.raises(ValueError, match="at least one latent"):
            LatentSpec(n_across=0, n_within=(0, 0))

    def test_rejects_negative_within(self) -> None:
        with pytest.raises(ValueError, match=">= 0"):
            LatentSpec(n_across=2, n_within=(-1, 3))

    def test_discrete_state_carried(self) -> None:
        spec = LatentSpec(
            n_across=2,
            n_within=(2, 2),
            discrete=DiscreteStateSpec(n_states=3, sticky=0.5),
        )
        assert spec.discrete is not None
        assert spec.discrete.n_states == 3


class TestARDPrior:
    def test_rejects_nonpositive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ARDPriorConfig(shape=0.0, rate=1.0)


class TestDiscreteStateSpec:
    def test_rejects_zero_states(self) -> None:
        with pytest.raises(ValueError, match="n_states"):
            DiscreteStateSpec(n_states=0)

    def test_rejects_negative_sticky(self) -> None:
        with pytest.raises(ValueError, match="sticky"):
            DiscreteStateSpec(n_states=2, sticky=-0.1)


# --- Delay ABC ----------------------------------------------------------


class _ConstantDelay(Delay):
    """Test double: every (region, latent) shares the same constant delay."""

    def __init__(self, n_regions: int, n_latent: int, value: float) -> None:
        super().__init__(n_regions, n_latent)
        D = torch.full((n_regions, n_latent), value)
        D[0] = 0.0  # reference region
        self.register_buffer("_D", D)

    @property
    def is_time_varying(self) -> bool:
        return False

    def as_tensor(self, T: int | None = None) -> Tensor:
        del T
        return self._D


class TestDelay:
    def test_phase_shift_shape_and_unit(self) -> None:
        delay = _ConstantDelay(n_regions=3, n_latent=2, value=1.5)
        freqs = torch.linspace(-0.5, 0.5, 7)
        phase = delay.phase_shift(freqs)
        assert phase.shape == (7, 3, 2)
        # Reference region 0 has zero delay → phase factor is 1.
        assert torch.allclose(phase[:, 0, :].abs(), torch.ones(7, 2))
        # All other phases have unit modulus too.
        assert torch.allclose(phase.abs(), torch.ones_like(phase.abs()))

    def test_phase_shift_value_matches_definition(self) -> None:
        delay = _ConstantDelay(n_regions=2, n_latent=1, value=2.0)
        f = torch.tensor([0.25])
        phase = delay.phase_shift(f)
        # exp(-2 pi i * 0.25 * 2.0) = exp(-i pi) = -1
        assert torch.allclose(phase[0, 1, 0].real, torch.tensor(-1.0), atol=1e-6)
        assert phase[0, 1, 0].imag.abs() < 1e-6

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            Delay(n_regions=2, n_latent=1)  # type: ignore[abstract]


# --- Observation ABC ----------------------------------------------------


class _IdentityObservation(Observation):
    def __init__(self, y_dims: tuple[int, ...], n_latent_total: int) -> None:
        super().__init__(y_dims, n_latent_total)
        self.C = torch.nn.Parameter(
            torch.zeros(self.n_neurons, n_latent_total).index_fill_(
                1, torch.arange(min(self.n_neurons, n_latent_total)), 1.0
            )[: self.n_neurons, :n_latent_total]
        )
        self.R = torch.nn.Parameter(torch.ones(self.n_neurons))
        self.d = torch.nn.Parameter(torch.zeros(self.n_neurons))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.C.T + self.d

    def block_diag_C(self) -> Tensor:
        return self.C

    def diag_R(self) -> Tensor:
        return self.R

    def offset(self) -> Tensor:
        return self.d


class TestObservation:
    def test_dims(self) -> None:
        obs = _IdentityObservation(y_dims=(3, 5), n_latent_total=4)
        assert obs.n_regions == 2
        assert obs.n_neurons == 8
        assert obs.block_diag_C().shape == (8, 4)

    def test_rejects_empty_y_dims(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _IdentityObservation(y_dims=(), n_latent_total=4)

    def test_rejects_zero_latent(self) -> None:
        with pytest.raises(ValueError, match="n_latent_total"):
            _IdentityObservation(y_dims=(3,), n_latent_total=0)

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            Observation(y_dims=(3,), n_latent_total=2)  # type: ignore[abstract]
