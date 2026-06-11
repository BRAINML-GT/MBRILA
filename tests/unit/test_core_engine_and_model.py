"""Smoke tests for the BaseModel / InferenceEngine wiring.

We define minimal concrete subclasses of every ABC, hook them together, and
verify capability checking, save/load round-trip, and that the ABCs reject
direct instantiation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from mbrila import (
    BaseModel,
    Delay,
    FitResult,
    InferenceEngine,
    Kernel,
    LatentSpec,
    MultiRegionData,
    Observation,
    Posterior,
)
from mbrila.core.kernel_spec import SDECoefficients

# --- doubles ------------------------------------------------------------


class _NoOpKernel(nn.Module):
    """Bare-bones Kernel implementation for plumbing tests."""

    CAPABILITIES = frozenset({"cov_full"})

    @property
    def n_params(self) -> int:
        return 0

    def cov(self, tau: Tensor) -> Tensor:
        return torch.zeros_like(tau)

    def spectral_density(self, omega: Tensor) -> Tensor:
        return torch.zeros_like(omega)

    def sde_form(self) -> SDECoefficients | None:
        return None

    @property
    def is_markovian(self) -> bool:
        return False

    @property
    def is_complex(self) -> bool:
        return False


class _ZeroDelay(Delay):
    @property
    def is_time_varying(self) -> bool:
        return False

    def as_tensor(self, T: int | None = None) -> Tensor:
        del T
        return torch.zeros(self.n_regions, self.n_latent)


class _ZeroObservation(Observation):
    def __init__(self, y_dims: tuple[int, ...], n_latent_total: int) -> None:
        super().__init__(y_dims, n_latent_total)
        self.C = nn.Parameter(torch.zeros(self.n_neurons, n_latent_total))
        self.R = nn.Parameter(torch.ones(self.n_neurons))
        self.d = nn.Parameter(torch.zeros(self.n_neurons))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.C.T + self.d

    def block_diag_C(self) -> Tensor:
        return self.C

    def diag_R(self) -> Tensor:
        return self.R

    def offset(self) -> Tensor:
        return self.d


class _IdleEngine(InferenceEngine):
    name = "idle"
    required_capabilities = frozenset({"cov_full"})

    def fit(
        self,
        model: BaseModel,
        data: MultiRegionData,
        *,
        max_iter: int,
        tol: float,
        **kwargs: object,
    ) -> FitResult:
        del model, data, max_iter, tol, kwargs
        return FitResult(score_trace=[0.0], converged=True, n_iter=0, wall_time_s=0.0)

    def infer(self, model: BaseModel, data: MultiRegionData) -> Posterior:
        del model
        n_trials, T = data.y.shape[0], data.y.shape[1]
        latent = data.y.shape[-1]  # cheat: reuse obs dim for the smoke test
        return Posterior(
            mean=torch.zeros(n_trials, T, latent),
            cov=torch.zeros(n_trials, T, latent),
            cov_form="diagonal",
        )

    def score(self, model: BaseModel, data: MultiRegionData) -> float:
        del model, data
        return 0.0


class _PickyEngine(_IdleEngine):
    """Demands a capability the toy model doesn't have."""

    name = "picky"
    required_capabilities = frozenset({"cov_full", "to_lds"})


class _ToyModel(BaseModel):
    def __init__(
        self,
        latent_spec: LatentSpec,
        y_dims: tuple[int, ...],
        *,
        engine: InferenceEngine | None = None,
        device: str | torch.device | None = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self._y_dims = y_dims
        self._engine_override = engine
        super().__init__(latent_spec=latent_spec, device=device, dtype=dtype)

    def _init_components(self) -> None:
        self.kernel = _NoOpKernel()
        self.delay = _ZeroDelay(
            n_regions=self.latent_spec.n_regions,
            n_latent=max(self.latent_spec.n_across, 1),
        )
        self.observation = _ZeroObservation(
            y_dims=self._y_dims,
            n_latent_total=self.latent_spec.n_latent_total,
        )
        self.dynamics = nn.Identity()
        self.inference = self._engine_override or _IdleEngine()

    def sample(
        self,
        n_trials: int,
        T: int,
        *,
        seed: int | None = None,
    ) -> MultiRegionData:
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        y = torch.randn(n_trials, T, sum(self._y_dims), generator=gen)
        return MultiRegionData(y=y, y_dims=self._y_dims, bin_width=1.0)

    @classmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> _ToyModel:
        spec = LatentSpec(
            n_across=int(config["n_across"]),
            n_within=tuple(int(x) for x in config["n_within"]),
        )
        return cls(latent_spec=spec, y_dims=tuple(int(x) for x in config["y_dims"]), **kwargs)

    def to_config(self) -> dict[str, Any]:
        return {
            "n_across": self.latent_spec.n_across,
            "n_within": list(self.latent_spec.n_within),
            "y_dims": list(self._y_dims),
        }


# --- tests --------------------------------------------------------------


@pytest.fixture
def model() -> _ToyModel:
    return _ToyModel(
        latent_spec=LatentSpec(n_across=2, n_within=(1, 1)),
        y_dims=(3, 4),
    )


@pytest.fixture
def data() -> MultiRegionData:
    return MultiRegionData(y=torch.randn(2, 5, 7), y_dims=(3, 4))


def test_capabilities_collected_from_components(model: _ToyModel) -> None:
    assert "cov_full" in model.capabilities()


def test_fit_runs_when_capabilities_match(model: _ToyModel, data: MultiRegionData) -> None:
    result = model.fit(data, max_iter=1, tol=1e-3)
    assert result.converged
    assert result.n_iter == 0


def test_fit_rejects_incompatible_engine(data: MultiRegionData) -> None:
    bad = _ToyModel(
        latent_spec=LatentSpec(n_across=2, n_within=(1, 1)),
        y_dims=(3, 4),
        engine=_PickyEngine(),
    )
    with pytest.raises(ValueError, match=r"missing.*to_lds"):
        bad.fit(data, max_iter=1, tol=1e-3)


def test_infer_returns_posterior(model: _ToyModel, data: MultiRegionData) -> None:
    post = model.infer(data)
    assert post.mean.shape == (2, 5, 7)
    assert post.cov_form == "diagonal"


def test_score_returns_float(model: _ToyModel, data: MultiRegionData) -> None:
    assert isinstance(model.score(data), float)


def test_save_load_round_trip(model: _ToyModel, tmp_path: Path) -> None:
    # Wiggle a parameter so we can detect it's restored.
    with torch.no_grad():
        model.observation.C.fill_(0.7)
    p = tmp_path / "snap.pt"
    model.save(p)

    loaded = _ToyModel.load(p, device="cpu")
    assert torch.equal(loaded.observation.C, model.observation.C)
    assert loaded.latent_spec.n_across == 2
    assert loaded.latent_spec.n_within == (1, 1)


def test_load_rejects_wrong_format(tmp_path: Path) -> None:
    p = tmp_path / "bad.pt"
    torch.save({"_format_version": 999, "config": {}, "state_dict": {}}, str(p))
    with pytest.raises(ValueError, match="format v999"):
        _ToyModel.load(p)


def test_cannot_instantiate_base_classes() -> None:
    with pytest.raises(TypeError):
        BaseModel(latent_spec=LatentSpec(n_across=1, n_within=(1,)))  # type: ignore[abstract]


def test_kernel_protocol_runtime_check() -> None:
    # _NoOpKernel implements every method on the Protocol → isinstance accepts.
    assert isinstance(_NoOpKernel(), Kernel)
