"""Validate that a kernel is internally consistent.

The :func:`check_kernel` utility is the "smoke detector" for custom
kernels: it sanity-checks that the kernel's pieces (``cov``, optional
``sde_form``, optional ``spectral_density``) agree with each other and
with the basic axioms of a valid stationary covariance. A buggy
hand-written kernel will typically fail one of these checks long before
it produces visibly wrong inference output, which makes this the right
first thing to run when porting a kernel into mbrila.

Checks
------
1. ``cov(τ) == cov(-τ)`` — stationary kernels are even.
2. ``cov`` is positive semi-definite on a small random grid.
3. ``cov(0)`` is positive (variance must be positive).
4. If ``sde_form()`` is non-``None``:

   a. Shapes are mutually consistent.
   b. ``stationary_cov`` (P∞) is symmetric and PSD.
   c. Lyapunov: ``F·P∞ + P∞·Fᵀ + L·Qc·Lᵀ ≈ 0``.
   d. ``H·P∞·Hᵀ`` matches ``cov(0)`` (marginal variance).
   e. ``H · expm(F·τ) · P∞ · Hᵀ`` matches ``cov(τ)`` for a few sample
      ``τ > 0``. This is the strictest test — it catches sign errors,
      missing factors, and wrong polynomial-coefficient mistakes.

5. If ``spectral_density(ω)`` is non-``None``: non-negative on a sample.

The thresholds are tight (``atol=1e-7`` by default on float64) so subtle
implementation bugs are caught.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mbrila.kernels.base import BaseKernel


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(f"check_kernel: {message}")


def _close(a: Tensor, b: Tensor, *, atol: float, rtol: float, label: str) -> None:
    diff = (a - b).abs().max().item()
    scale = b.abs().max().item()
    tol = atol + rtol * scale
    if diff > tol:
        raise AssertionError(f"check_kernel: {label} mismatch — max abs diff {diff:.3e} > tol {tol:.3e}")


def check_kernel(
    kernel: BaseKernel,
    *,
    n_grid: int = 8,
    tau_samples: tuple[float, ...] = (0.0, 0.5, 1.0, 2.5, 5.0),
    atol: float = 1e-7,
    rtol: float = 1e-7,
    spectral_omega_samples: tuple[float, ...] = (0.0, 0.5, 1.0, 5.0),
) -> None:
    """Validate ``kernel`` against the contract documented above.

    Raises :class:`AssertionError` with a descriptive message at the
    first failure. On success, returns ``None``.

    Parameters
    ----------
    kernel:
        Instance of :class:`BaseKernel`.
    n_grid:
        Size of the random τ grid used for the PSD-ness check.
    tau_samples:
        Real τ values at which to compare ``cov(τ)`` against the SDE
        prediction (for kernels that expose ``sde_form``).
    atol, rtol:
        Tolerances for the numerical comparisons. Defaults are tight
        (1e-7) and assume float64 parameters.
    spectral_omega_samples:
        Angular frequencies at which to spot-check
        ``spectral_density``.
    """
    # ------------------------------------------------------------------
    # 1. Evenness: cov(τ) == cov(-τ).
    # ------------------------------------------------------------------
    tau_pos = torch.tensor([t for t in tau_samples if t > 0.0], dtype=torch.float64)
    if tau_pos.numel() > 0:
        cov_pos = kernel.cov(tau_pos)
        cov_neg = kernel.cov(-tau_pos)
        _close(cov_pos, cov_neg, atol=atol, rtol=rtol, label="cov evenness")

    # ------------------------------------------------------------------
    # 2 & 3. Variance positivity + PSD on a random grid.
    # ------------------------------------------------------------------
    cov_0 = kernel.cov(torch.zeros((), dtype=torch.float64))
    _assert(
        bool(cov_0.item() > 0.0),
        f"cov(0) must be positive; got {cov_0.item():.3e}",
    )
    g = torch.Generator().manual_seed(0)
    grid = torch.randn(n_grid, generator=g, dtype=torch.float64).sort().values
    tau_grid = grid.unsqueeze(0) - grid.unsqueeze(1)  # (n, n)
    K_grid = kernel.cov(tau_grid)
    K_sym = 0.5 * (K_grid + K_grid.transpose(-1, -2))
    eigvals = torch.linalg.eigvalsh(K_sym)
    min_eig = eigvals.min().item()
    _assert(
        min_eig > -1e-9,
        f"cov is not PSD on a {n_grid}-point grid; min eigenvalue {min_eig:.3e}",
    )

    # ------------------------------------------------------------------
    # 4. sde_form consistency.
    # ------------------------------------------------------------------
    sde = kernel.sde_form()
    if sde is not None:
        F, L, Qc, H, Pinf = sde.F, sde.L, sde.Qc, sde.H, sde.stationary_cov
        D = sde.state_dim()

        # 4a. Shapes.
        _assert(F.shape == (D, D), f"F shape should be ({D},{D}); got {tuple(F.shape)}")
        _assert(L.shape[-2] == D, f"L should have {D} rows; got {tuple(L.shape)}")
        M = L.shape[-1]
        _assert(Qc.shape == (M, M), f"Qc shape should be ({M},{M}); got {tuple(Qc.shape)}")
        _assert(H.shape == (1, D), f"H shape should be (1,{D}); got {tuple(H.shape)}")
        _assert(
            Pinf.shape == (D, D),
            f"stationary_cov shape should be ({D},{D}); got {tuple(Pinf.shape)}",
        )

        # 4b. P∞ symmetry + PSD.
        _close(Pinf, Pinf.transpose(-1, -2), atol=atol, rtol=rtol, label="P∞ symmetry")
        p_eigvals = torch.linalg.eigvalsh(0.5 * (Pinf + Pinf.transpose(-1, -2)))
        _assert(
            p_eigvals.min().item() > -1e-9,
            f"stationary_cov is not PSD; min eigenvalue {p_eigvals.min().item():.3e}",
        )

        # 4c. Lyapunov: F·P + P·Fᵀ + L·Qc·Lᵀ == 0.
        lyap_residual = F @ Pinf + Pinf @ F.transpose(-1, -2) + L @ Qc @ L.transpose(-1, -2)
        _close(
            lyap_residual,
            torch.zeros_like(lyap_residual),
            atol=atol,
            rtol=rtol,
            label="Lyapunov F·P + P·Fᵀ + L·Qc·Lᵀ",
        )

        # 4d. Marginal variance: H·P∞·Hᵀ == cov(0).
        marginal = (H @ Pinf @ H.transpose(-1, -2)).reshape(())
        _close(
            marginal,
            cov_0.to(marginal.dtype),
            atol=atol,
            rtol=rtol,
            label="H·P∞·Hᵀ vs cov(0)",
        )

        # 4e. cov(τ) == H · expm(F·τ) · P∞ · Hᵀ for τ > 0.
        for tau_val in tau_samples:
            if tau_val <= 0.0:
                continue
            tau_t = torch.tensor(tau_val, dtype=F.dtype, device=F.device)
            transition = torch.linalg.matrix_exp(F * tau_t)
            pred = (H @ transition @ Pinf @ H.transpose(-1, -2)).reshape(())
            actual = kernel.cov(tau_t).reshape(())
            _close(
                pred,
                actual.to(pred.dtype),
                atol=atol,
                rtol=rtol,
                label=f"H·expm(F·τ)·P∞·Hᵀ vs cov(τ={tau_val})",
            )

    # ------------------------------------------------------------------
    # 5. Spectral density non-negativity (if implemented).
    # ------------------------------------------------------------------
    omega = torch.tensor(spectral_omega_samples, dtype=torch.float64)
    psd = kernel.spectral_density(omega)
    if psd is not None:
        _assert(
            bool((psd >= -1e-12).all().item()),
            f"spectral_density must be non-negative; got min {psd.min().item():.3e}",
        )
