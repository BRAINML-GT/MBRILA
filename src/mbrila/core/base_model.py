"""User-facing model base class.

Every concrete method in mbrila (``DLAG``, ``MDLAG``, ``MRMGP``, ``ADM``)
inherits from :class:`BaseModel` and wires up a kernel, a delay, an
observation, a latent dynamics, and an inference engine. The base class
handles the surrounding plumbing: capability advertisement, the
``fit``/``infer``/``sample`` facade, save/load, and device routing.

Concrete subclasses override :meth:`_init_components` (which builds the
plumbed components from the model's config) and :meth:`sample` (the model's
prior generator). :meth:`capabilities` is composed automatically from each
component's class-level ``CAPABILITIES`` attribute.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from mbrila.core.inference_engine import FitResult, InferenceEngine, Posterior
from mbrila.utils.device import resolve_device

if TYPE_CHECKING:
    from mbrila.core.data import MultiRegionData
    from mbrila.core.delay_spec import Delay
    from mbrila.core.latent_spec import LatentSpec
    from mbrila.core.observation_spec import Observation


# Bumped whenever the on-disk save format changes incompatibly.
SAVE_FORMAT_VERSION = 1


class BaseModel(nn.Module, ABC):
    """Abstract base class for every mbrila method.

    The concrete subclass is expected to populate ``self.kernel``,
    ``self.delay``, ``self.observation``, ``self.dynamics`` and
    ``self.inference`` in :meth:`_init_components` (called from
    :meth:`__init__`).

    Parameters
    ----------
    latent_spec:
        Structural latent description. The subclass typically derives the
        full configuration object from this plus its own kwargs.
    device:
        Target device. Defaults to CUDA if available, else CPU.
    dtype:
        Floating point dtype for parameters. Defaults to ``torch.float32``.
    """

    # Subclasses fill these in via ``_init_components``. The kernel slot is
    # typed as the nominal ``nn.Module`` rather than the structural
    # :class:`Kernel` protocol because PyTorch's ``nn.Module.__getattr__``
    # signature widens every attribute access to ``Tensor | Module``, which
    # makes it impossible for mypy to confirm Protocol conformance on
    # concrete kernel implementations. Engines that need the Protocol
    # contract should assert / cast at the use site.
    latent_spec: LatentSpec
    kernel: nn.Module
    delay: Delay
    observation: Observation
    inference: InferenceEngine
    dynamics: nn.Module

    def __init__(
        self,
        latent_spec: LatentSpec,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self.latent_spec = latent_spec
        self._device = resolve_device(device)
        self._dtype = dtype
        self._init_components()
        # Register components automatically as submodules so to(device) and
        # state_dict capture them. We assign as attributes; nn.Module's
        # __setattr__ does the right thing for nn.Module values.
        self.to(device=self._device, dtype=self._dtype)

    # --- subclass hooks -------------------------------------------------

    @abstractmethod
    def _init_components(self) -> None:
        """Build kernel / delay / observation / dynamics / inference.

        Called once from ``__init__``; subclasses must set every attribute
        listed in the class-level annotation block (kernel, delay, etc.).
        """

    @abstractmethod
    def sample(
        self,
        n_trials: int,
        T: int,
        *,
        seed: int | None = None,
    ) -> MultiRegionData:
        """Draw a synthetic dataset from the model's prior."""

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any], **kwargs: Any) -> BaseModel:
        """Re-build a model from a serialised configuration dict."""

    @abstractmethod
    def to_config(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the model's structure.

        Used by :meth:`save` together with ``state_dict`` to reconstruct the
        model. Should *not* include parameter values (those live in the
        state dict).
        """

    # --- capabilities ---------------------------------------------------

    def capabilities(self) -> frozenset[str]:
        """Union of every component's declared capabilities.

        Each component class may set a class attribute
        ``CAPABILITIES: frozenset[str] = frozenset(...)``; the union of all
        five components' attributes is what inference engines check
        against.
        """
        caps: set[str] = set()
        for component in (
            getattr(self, "kernel", None),
            getattr(self, "delay", None),
            getattr(self, "observation", None),
            getattr(self, "dynamics", None),
        ):
            if component is None:
                continue
            comp_caps = getattr(type(component), "CAPABILITIES", None)
            if comp_caps:
                caps.update(comp_caps)
        return frozenset(caps)

    # --- public facade --------------------------------------------------

    def fit(
        self,
        data: MultiRegionData,
        *,
        max_iter: int = 100,
        tol: float = 1e-5,
        **engine_kwargs: object,
    ) -> FitResult:
        """Run the model's inference engine on ``data``."""
        self.inference.check_compatible(self)
        data = data.to(device=self._device, dtype=self._dtype)
        return self.inference.fit(self, data, max_iter=max_iter, tol=tol, **engine_kwargs)

    def infer(self, data: MultiRegionData) -> Posterior:
        """Latent posterior for ``data`` at current parameters."""
        self.inference.check_compatible(self)
        data = data.to(device=self._device, dtype=self._dtype)
        return self.inference.infer(self, data)

    def score(self, data: MultiRegionData) -> float:
        """LL or ELBO at current parameters (for *intra-model* monitoring)."""
        self.inference.check_compatible(self)
        data = data.to(device=self._device, dtype=self._dtype)
        return self.inference.score(self, data)

    # --- I/O ------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model config + parameter state."""
        payload = {
            "_format_version": SAVE_FORMAT_VERSION,
            "_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "config": self.to_config(),
            "state_dict": self.state_dict(),
        }
        torch.save(payload, str(path))

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        **kwargs: Any,
    ) -> BaseModel:
        """Reconstruct a model previously saved with :meth:`save`.

        Extra ``**kwargs`` are forwarded to :meth:`from_config` — needed
        for subclasses whose constructor takes non-serialisable arguments
        (e.g. ``kernel_factory_*`` callables on Markov-GP models).
        """
        payload = torch.load(str(path), map_location=resolve_device(device), weights_only=False)
        version = int(payload.get("_format_version", 0))
        if version != SAVE_FORMAT_VERSION:
            raise ValueError(f"unsupported mbrila save format v{version}; expected v{SAVE_FORMAT_VERSION}")
        config = dict(payload["config"])
        model = cls.from_config(config, device=device, dtype=dtype, **kwargs)
        model.load_state_dict(payload["state_dict"])
        return model

    # --- introspection --------------------------------------------------

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype
