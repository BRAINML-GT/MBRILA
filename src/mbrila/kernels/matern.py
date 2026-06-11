"""Matérn-½ / Matérn-3/2 / Matérn-5/2 kernels.

The Matérn family is the canonical *exact* GP↔SSM kernel: each
half-integer-smooth Matérn kernel has a closed-form finite-dimensional
linear SDE realisation (Hartikainen & Särkkä 2010, Särkkä & Solin 2019).
This is the qualitative difference vs RBF / MOSE: the Markovian-GP
inference path is **exact**, not an AR(``P``) approximation.

Beyond the GP-SSM honesty story, Matérn-3/2 / 5/2 are typically a better
default than RBF for biological signals: RBF is infinitely
differentiable, which is unphysical for many noise-floor-limited neural
recordings.

Parameterisation
----------------
All three kernels carry a single hyperparameter, the lengthscale ``ℓ``,
stored as ``log_lengthscale`` for unconstrained optimisation. Variance
is fixed at ``σ² = 1`` to match the MOSE convention — overall scale is
absorbed by the emission ``C``.

We use the convention ``λ = √(2ν) / ℓ`` for ν ∈ {½, 3⁄2, 5⁄2}, so:

- Matérn-½ (ν=½):   ``k(τ) = exp(-λ|τ|)``,
  ``F = [[-λ]]``, ``L = [[1]]``, ``Qc = 2λ``, ``H = [[1]]``, ``P∞ = [[1]]``.
- Matérn-3/2 (ν=3⁄2): ``k(τ) = (1 + λ|τ|) exp(-λ|τ|)``,
  ``F = [[0,1],[-λ²,-2λ]]``, ``L = [[0],[1]]``, ``Qc = 4λ³``, ``H = [[1,0]]``,
  ``P∞ = diag(1, λ²)``.
- Matérn-5/2 (ν=5⁄2): ``k(τ) = (1 + λ|τ| + (λ²/3) τ²) exp(-λ|τ|)``,
  ``F = [[0,1,0],[0,0,1],[-λ³,-3λ²,-3λ]]``, ``L = [[0],[0],[1]]``,
  ``Qc = 16λ⁵/3``, ``H = [[1,0,0]]``,
  ``P∞`` has the explicit block ``[[1, 0, -λ²/3],[0, λ²/3, 0],[-λ²/3, 0, λ⁴]]``.

All SDE coefficients above are verified by the Lyapunov equation
``F·P∞ + P∞·Fᵀ + L·Qc·Lᵀ = 0`` — :func:`mbrila.kernels.validate.check_kernel`
exercises this consistency on each instance.

Spectral density
----------------
The Matérn-(ν+½) PSD (angular frequency, σ²=1) is::

    S(ω) = 2π · Γ(ν+½) / Γ(ν) · λ^(2ν) / (λ² + ω²)^(ν+½)·... [standard form]

For Matérn-(½, 3⁄2, 5⁄2) the coefficient simplifies to ``Qc / (λ² + ω²)^p``
with ``p`` = state dim, which is the form we implement.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from mbrila.core.kernel_spec import SDECoefficients
from mbrila.kernels.base import BaseKernel


def _log_lengthscale_param(lengthscale: float) -> nn.Parameter:
    if lengthscale <= 0:
        raise ValueError(f"lengthscale must be positive; got {lengthscale}")
    return nn.Parameter(torch.log(torch.tensor(lengthscale, dtype=torch.float64)))


class _MaternBase(BaseKernel):
    """Common machinery for the three Matérn-(ν+½) kernels.

    Subclasses set ``_state_dim`` and implement :meth:`cov`,
    :meth:`_sde_F`, :meth:`_sde_Qc`, :meth:`_sde_Pinf`. ``L`` and ``H`` are
    fixed by the canonical state ``s = [x, x', …, x^(p)]``.
    """

    is_markovian = True
    _state_dim: int

    def __init__(self, *, lengthscale: float = 1.0) -> None:
        super().__init__()
        self.log_lengthscale = _log_lengthscale_param(lengthscale)

    @property
    def lengthscale(self) -> Tensor:
        return torch.exp(self.log_lengthscale)

    @property
    def _lambda(self) -> Tensor:
        """``λ = √(2ν) / ℓ`` — the SDE pole-scale parameter."""
        return self._sqrt_two_nu() / self.lengthscale

    def _sqrt_two_nu(self) -> Tensor:
        """``√(2ν)`` as a tensor on the kernel's dtype/device."""
        raise NotImplementedError

    # ---- SDE pieces shared by all three kernels ----------------------

    def _sde_F(self) -> Tensor:
        raise NotImplementedError

    def _sde_Qc(self) -> Tensor:
        raise NotImplementedError

    def _sde_Pinf(self) -> Tensor:
        raise NotImplementedError

    def sde_form(self) -> SDECoefficients:
        """Return exact continuous-time SDE coefficients.

        For Matérn-(ν+½) the canonical state is
        ``s = [x, x⁽¹⁾, …, x⁽ᵖ⁾]`` with ``p = ν``. ``L`` selects only the
        top-derivative innovation channel; ``H`` reads out the position.
        """
        F = self._sde_F()
        Qc = self._sde_Qc()  # scalar tensor or (1, 1)
        Pinf = self._sde_Pinf()
        # L: (state_dim, 1) — innovation only enters the bottom row.
        L = F.new_zeros(self._state_dim, 1)
        L[-1, 0] = 1.0
        # H: (1, state_dim) — read out the position component.
        H = F.new_zeros(1, self._state_dim)
        H[0, 0] = 1.0
        # Qc as (1, 1) per SDECoefficients contract.
        Qc_mat = Qc.view(1, 1) if Qc.ndim == 0 else Qc
        return SDECoefficients(F=F, L=L, Qc=Qc_mat, H=H, stationary_cov=Pinf)


class Matern12Kernel(_MaternBase):
    """Matérn-½: ``k(τ) = exp(-|τ| / ℓ)`` (the Ornstein–Uhlenbeck process).

    State dim 1. Discontinuous in derivative — appropriate for very rough
    signals; for smoother defaults prefer :class:`Matern32Kernel` or
    :class:`Matern52Kernel`.
    """

    _state_dim = 1

    def _sqrt_two_nu(self) -> Tensor:
        # ν = ½, √(2ν) = 1.
        return self.log_lengthscale.new_tensor(1.0)

    def cov(self, tau: Tensor) -> Tensor:
        lam = self._lambda
        return torch.exp(-lam * tau.to(lam.dtype).abs())

    def _sde_F(self) -> Tensor:
        lam = self._lambda
        return -lam.view(1, 1)

    def _sde_Qc(self) -> Tensor:
        return 2.0 * self._lambda

    def _sde_Pinf(self) -> Tensor:
        return self.log_lengthscale.new_ones(1, 1)


class Matern32Kernel(_MaternBase):
    """Matérn-3/2: ``k(τ) = (1 + √3|τ|/ℓ) exp(-√3|τ|/ℓ)``.

    State dim 2. ``C¹``-continuous; a standard "rough but differentiable"
    default for neural latents.
    """

    _state_dim = 2

    def _sqrt_two_nu(self) -> Tensor:
        # ν = 3⁄2, √(2ν) = √3.
        return self.log_lengthscale.new_tensor(math.sqrt(3.0))

    def cov(self, tau: Tensor) -> Tensor:
        lam = self._lambda
        abs_tau = tau.to(lam.dtype).abs()
        return (1.0 + lam * abs_tau) * torch.exp(-lam * abs_tau)

    def _sde_F(self) -> Tensor:
        lam = self._lambda
        zero = lam.new_zeros(())
        one = lam.new_ones(())
        return torch.stack(
            [
                torch.stack([zero, one]),
                torch.stack([-lam * lam, -2.0 * lam]),
            ]
        )

    def _sde_Qc(self) -> Tensor:
        return 4.0 * self._lambda.pow(3)

    def _sde_Pinf(self) -> Tensor:
        lam2 = self._lambda.pow(2)
        zero = lam2.new_zeros(())
        one = lam2.new_ones(())
        return torch.stack(
            [
                torch.stack([one, zero]),
                torch.stack([zero, lam2]),
            ]
        )


class Matern52Kernel(_MaternBase):
    """Matérn-5/2: ``k(τ) = (1 + √5|τ|/ℓ + 5τ²/(3ℓ²)) exp(-√5|τ|/ℓ)``.

    State dim 3. ``C²``-continuous; closer to RBF in smoothness but with
    an exact SDE form. A good default when one wants smoothness without
    accepting RBF's AR(P)-approximation hazard.
    """

    _state_dim = 3

    def _sqrt_two_nu(self) -> Tensor:
        # ν = 5⁄2, √(2ν) = √5.
        return self.log_lengthscale.new_tensor(math.sqrt(5.0))

    def cov(self, tau: Tensor) -> Tensor:
        lam = self._lambda
        abs_tau = tau.to(lam.dtype).abs()
        lam_t = lam * abs_tau
        poly = 1.0 + lam_t + lam_t.square() / 3.0
        return poly * torch.exp(-lam_t)

    def _sde_F(self) -> Tensor:
        lam = self._lambda
        lam2 = lam.pow(2)
        lam3 = lam.pow(3)
        zero = lam.new_zeros(())
        one = lam.new_ones(())
        return torch.stack(
            [
                torch.stack([zero, one, zero]),
                torch.stack([zero, zero, one]),
                torch.stack([-lam3, -3.0 * lam2, -3.0 * lam]),
            ]
        )

    def _sde_Qc(self) -> Tensor:
        return (16.0 / 3.0) * self._lambda.pow(5)

    def _sde_Pinf(self) -> Tensor:
        lam2 = self._lambda.pow(2)
        lam4 = lam2.pow(2)
        third = lam2.new_tensor(1.0 / 3.0)
        zero = lam2.new_zeros(())
        one = lam2.new_ones(())
        # P∞ = [[1,         0,           -λ²/3],
        #       [0,         λ²/3,        0    ],
        #       [-λ²/3,     0,           λ⁴   ]]
        return torch.stack(
            [
                torch.stack([one, zero, -lam2 * third]),
                torch.stack([zero, lam2 * third, zero]),
                torch.stack([-lam2 * third, zero, lam4]),
            ]
        )
