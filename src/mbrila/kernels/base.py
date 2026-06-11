"""User-facing kernel base class.

Subclasses of :class:`BaseKernel` are the canonical extension point for
custom kernels in mbrila. The contract is intentionally minimal:

- **Required**: implement :meth:`cov(tau)` — a pointwise scalar stationary
  kernel that accepts a real lag tensor of arbitrary shape and returns a
  tensor of the same shape. Multi-region structure and per-region delays
  are handled by the dynamics layer (see ``lagged_cov_grid``); the
  kernel itself is not delay-aware.
- **Optional**: override :meth:`sde_form` if the kernel has an exact
  continuous-time SDE realisation (Matérn-½/3/2/5/2 do; RBF/MOSE does
  not). Kernels without an exact SDE are still consumable by the Kalman
  engine via the AR(P) bridge ``cov → lagged_cov_grid → kernel_to_lds``.
- **Optional**: override :meth:`spectral_density` for frequency-domain
  engines (mDLAG-freq).

Capabilities are auto-advertised: :meth:`capabilities` inspects which
optional methods have been overridden / return non-``None`` and produces
a set of strings consumable by
:meth:`mbrila.core.inference_engine.InferenceEngine.check_compatible`.

Why a separate ABC and not the :class:`mbrila.core.kernel_spec.Kernel`
Protocol? The Protocol describes the *consumer-facing* contract used by
inference engines for structural typing. ``BaseKernel`` is the
*author-facing* contract: a concrete class with default implementations,
parameter registration, validation hooks, and ``capabilities``
auto-detection. Users subclass ``BaseKernel``; engines type against
``Kernel`` (or the narrower :class:`LaggedCovKernel`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import torch
from torch import Tensor, nn

from mbrila.core.kernel_spec import SDECoefficients


class BaseKernel(nn.Module, ABC):
    """Common base for scalar stationary kernels.

    Subclasses MUST implement :meth:`cov`. They MAY override
    :meth:`sde_form` and :meth:`spectral_density` if they have closed-form
    realisations.

    Class attributes
    ----------------
    is_markovian:
        ``True`` if the kernel has an exact finite-dimensional SDE form
        (Matérn family). ``False`` if the SSM path must rely on the
        AR(P) approximation bridge (RBF / MOSE). Subclasses override.
    is_complex:
        ``True`` for kernels that use complex spectral parameters
        (MRM-GP's ``ComplexSpectralKernel``, v2). Default ``False``.
    """

    is_markovian: ClassVar[bool] = False
    is_complex: ClassVar[bool] = False

    @abstractmethod
    def cov(self, tau: Tensor) -> Tensor:
        """Stationary covariance ``k(τ)``, evaluated pointwise.

        Parameters
        ----------
        tau:
            Real tensor of arbitrary shape.

        Returns
        -------
        Tensor of the same shape as ``tau``.
        """

    def sde_form(self) -> SDECoefficients | None:
        """Exact continuous-time SDE coefficients, if available.

        Default returns ``None`` — kernels without an exact finite-state
        SDE (RBF / MOSE) still consume the Kalman engine via the AR(P)
        bridge through :meth:`cov`. Override in subclasses that have an
        exact SDE realisation.
        """
        return None

    def spectral_density(self, omega: Tensor) -> Tensor | None:
        """Power spectral density ``S(ω)`` (angular frequency).

        Default returns ``None``. Override for frequency-domain engines.
        """
        del omega
        return None

    @property
    def n_params(self) -> int:
        """Total number of trainable scalar parameters."""
        return sum(int(p.numel()) for p in self.parameters() if p.requires_grad)

    def capabilities(self) -> frozenset[str]:
        """Set of strings ``InferenceEngine.check_compatible`` consumes.

        Auto-detected from which optional methods return non-``None``.
        Always includes ``"cov"`` (the required method).
        """
        caps: set[str] = {"cov"}
        if self.sde_form() is not None:
            caps.add("sde_form")
        # ``spectral_density`` needs a sample tensor to evaluate. Use a
        # 1-element zero tensor on the kernel's parameter dtype/device.
        # We just need to know whether the method is defined; calling it
        # is the cleanest way without introspecting method overrides.
        try:
            sample_omega = next(self.parameters()).new_zeros(1)
        except StopIteration:
            sample_omega = torch.zeros(1)
        if self.spectral_density(sample_omega) is not None:
            caps.add("spectral_density")
        return frozenset(caps)
