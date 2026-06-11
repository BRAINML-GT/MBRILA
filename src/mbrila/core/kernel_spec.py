"""Kernel interface used across mbrila.

Two kernel families coexist in v1: the multi-output squared-exponential
(MOSE) used by ADM / DLAG / mDLAG, and the complex spectral kernel used by
MRM-GP. Both expose the same minimal contract — every other piece of the
library queries kernels through this :class:`Kernel` runtime-checkable
Protocol so that injecting a third kernel does not require touching any
inference engine.

The contract is deliberately broader than what any single inference engine
needs: an exact-GP engine consumes ``cov`` (and possibly
``spectral_density``); a Markovian engine consumes ``sde_form``. Concrete
kernels return :data:`None` from the methods they cannot support and the
:class:`InferenceEngine.check_compatible` machinery rejects mismatched pairs
at fit time rather than failing inside a hot loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor


@dataclass(frozen=True, slots=True)
class SDECoefficients:
    """Linear-Gaussian SDE coefficients for a Markovian kernel.

    The continuous-time SDE is ``dx = F x dt + L dW`` with diffusion intensity
    ``Qc`` (``dW`` a standard Wiener process). The observation map ``H``
    extracts the latent value of interest from the augmented state used by
    higher-order Markov approximations (e.g. picking the position out of a
    position/velocity pair).

    Shapes
    ------
    All tensors live on the same device/dtype.

    - ``F``: ``(..., D, D)`` — drift matrix.
    - ``L``: ``(..., D, M)`` — noise gain.
    - ``Qc``: ``(..., M, M)`` — continuous-time diffusion covariance.
    - ``H``: ``(..., 1, D)`` — observation map (we always read off a single
      scalar latent per kernel; multi-output is handled at the dynamics
      layer).
    - ``stationary_cov``: ``(..., D, D)`` — solution of the Lyapunov
      equation ``F P + P F^T + L Qc L^T = 0`` (the steady-state covariance
      used to seed the Kalman recursion).

    Leading batch dimensions are arbitrary (e.g. ``(K, n_latent)`` for
    per-state per-latent kernels in MRM-GP).
    """

    F: Tensor
    L: Tensor
    Qc: Tensor
    H: Tensor
    stationary_cov: Tensor

    def state_dim(self) -> int:
        return int(self.F.shape[-1])


@runtime_checkable
class Kernel(Protocol):
    """The contract every mbrila kernel implements.

    Concrete kernels typically also subclass :class:`torch.nn.Module` to
    register their hyperparameters; that is orthogonal to this Protocol,
    which only describes the interface used by inference engines.
    """

    @property
    def n_params(self) -> int:
        """Number of free hyperparameters (for sanity checks / reporting)."""
        ...

    def cov(self, tau: Tensor) -> Tensor:
        """Stationary covariance ``k(tau)`` evaluated on a lag tensor.

        ``tau`` is a real tensor of arbitrary shape ``(...,)``; the output
        broadcasts to ``(..., n_outputs, n_outputs)`` for multi-output
        kernels (or ``(...,)`` for scalar kernels). Implementations that do
        not support an exact covariance (e.g. some highly-engineered SDE
        kernels) may raise :class:`NotImplementedError`.
        """
        ...

    def spectral_density(self, omega: Tensor) -> Tensor:
        """Power spectral density ``S(omega)`` (angular frequency)."""
        ...

    def sde_form(self) -> SDECoefficients | None:
        """Return SDE coefficients for Markovian kernels, else ``None``."""
        ...

    @property
    def is_markovian(self) -> bool:
        """Whether ``sde_form`` returns a finite-state SDE."""
        ...

    @property
    def is_complex(self) -> bool:
        """Whether the kernel uses complex-valued spectral parameters.

        Set by MRM-GP-style spectral kernels; the dynamics layer adds the
        complex-to-real conversion when it sees this flag.
        """
        ...
