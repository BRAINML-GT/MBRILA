"""Bayesian linear-Gaussian regression with conjugate Normal-Inverse-Wishart prior.

This is the closed-form M-step every method in mbrila uses to update its
emission parameters ``C`` (loading matrix), ``d`` (per-neuron offset), and
``R`` (observation noise). It also lives here as a standalone utility so
users can run it on any ``(X, Y)`` pair.

Model
-----
::

    y_n = W x_n + b + ε_n,      ε_n ∼ N(0, Σ),     n = 1, …, N

with prior
::

    W | Σ  ∼  MN(0, Σ ⊗ V_0^{-1})
    Σ      ∼  IW(ν_0, Ψ_0)

The ``Ψ_0`` prior is ``ψ_0 · I_d`` in this implementation (a single positive
scalar times the identity). The optional ``prior_ExxT`` / ``prior_ExyT``
arguments let callers inject ``V_0`` and ``V_0 W_0^T`` respectively, in
which case the prior mean of ``W`` is ``W_0 = (V_0)^{-1} prior_ExyT^T``.

EM use
------
In an EM M-step the posterior expectations replace the sample sufficient
statistics::

    E[xxᵀ] := Σ_n  (Cov_n[x] + μ_n μ_nᵀ)
    E[xyᵀ] := Σ_n  μ_n yᵀ_n
    E[yyᵀ] := Σ_n  y_n yᵀ_n          # y is observed; no Cov term

Pass ``expectations=(E[xxᵀ], E[xyᵀ], E[yyᵀ], weight_sum)`` to fit the M-step
without rebuilding sufficient statistics from raw samples.

Numerics
--------
The MAP estimate ``W_MAP = (E[xxᵀ])^{-1} E[xyᵀ]`` is computed via
Cholesky solves rather than a generic ``torch.linalg.solve``:
``E[xxᵀ]`` is symmetric PSD by construction, so the Cholesky path is
faster and slightly more accurate.

The diagonal noise estimate uses the **MAP** of an inverse-Wishart, i.e.
``Σ_diag = diag(Ψ_n) / (ν_n + d + 1)``. The posterior mean would use
``ν_n - d - 1`` instead; either is consistent in EM (they only differ
by a fixed rescaling that the next E-step absorbs).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True, frozen=True)
class LinearRegressionResult:
    """Outputs of :func:`bayesian_linear_regression`.

    Attributes
    ----------
    W:
        Regression weights, shape ``(d, p)``.
    b:
        Intercepts, shape ``(d,)`` (zeros if ``fit_intercept=False``).
    sigma_diag:
        Diagonal of the MAP observation-noise covariance, shape ``(d,)``.
    """

    W: Tensor
    b: Tensor
    sigma_diag: Tensor


def _accumulate_sufficient_stats(
    X: Tensor,
    Y: Tensor,
    weights: Tensor | None,
    *,
    fit_intercept: bool,
    prior_ExxT: Tensor | None,
    prior_ExyT: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build ``(ExxT, ExyT, EyyT, weight_sum)`` from raw samples.

    ``X`` has shape ``(*, p)`` (any leading batch shape; flattened), ``Y``
    has shape ``(*, d)``.
    """
    p = X.shape[-1]
    d = Y.shape[-1]
    device = X.device
    dtype = X.dtype

    X_flat = X.reshape(-1, p)
    Y_flat = Y.reshape(-1, d)
    n_samples = X_flat.shape[0]

    if weights is None:
        w = torch.ones(n_samples, 1, dtype=dtype, device=device)
    else:
        w = weights.reshape(-1, 1).to(dtype=dtype, device=device)
        if w.shape[0] != n_samples:
            raise ValueError(f"weights must have {n_samples} entries to match X/Y; got {w.shape[0]}")

    if fit_intercept:
        X_flat = torch.cat([X_flat, torch.ones_like(w)], dim=1)
    x_dim = X_flat.shape[-1]

    Xw = X_flat * w
    Yw = Y_flat * w

    ExxT = Xw.transpose(0, 1) @ X_flat  # (x_dim, x_dim)
    ExyT = Xw.transpose(0, 1) @ Y_flat  # (x_dim, d)
    EyyT = Yw.transpose(0, 1) @ Y_flat  # (d, d)
    weight_sum = w.sum()

    if prior_ExxT is not None:
        if prior_ExxT.shape != (x_dim, x_dim):
            raise ValueError(f"prior_ExxT must have shape ({x_dim}, {x_dim}); got {tuple(prior_ExxT.shape)}")
        ExxT = ExxT + prior_ExxT
    if prior_ExyT is not None:
        if prior_ExyT.shape != (x_dim, d):
            raise ValueError(f"prior_ExyT must have shape ({x_dim}, {d}); got {tuple(prior_ExyT.shape)}")
        ExyT = ExyT + prior_ExyT

    return ExxT, ExyT, EyyT, weight_sum


def bayesian_linear_regression(
    X: Tensor | None = None,
    Y: Tensor | None = None,
    *,
    weights: Tensor | None = None,
    fit_intercept: bool = True,
    expectations: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
    prior_ExxT: Tensor | None = None,
    prior_ExyT: Tensor | None = None,
    nu0: float = 1.0,
    psi0: float = 1.0,
    jitter: float = 1e-9,
) -> LinearRegressionResult:
    """Conjugate MAP fit for ``y = W x + b + ε``.

    Either pass raw ``X``, ``Y`` (and optional ``weights``), or precomputed
    ``expectations`` from an E-step. When ``fit_intercept`` is true and
    ``expectations`` is supplied, the sufficient statistics must already
    include the all-ones intercept column / row (``ExxT`` shape
    ``(p+1, p+1)`` and ``ExyT`` shape ``(p+1, d)``).
    """
    if expectations is not None:
        ExxT, ExyT, EyyT, weight_sum = expectations
        x_dim = ExxT.shape[-1]
        d = ExyT.shape[-1]
        if ExxT.shape != (x_dim, x_dim):
            raise ValueError(f"ExxT must be square; got {tuple(ExxT.shape)}")
        if ExyT.shape != (x_dim, d):
            raise ValueError(f"ExyT must have shape ({x_dim}, {d}); got {tuple(ExyT.shape)}")
        if EyyT.shape != (d, d):
            raise ValueError(f"EyyT must have shape ({d}, {d}); got {tuple(EyyT.shape)}")
        if weight_sum.ndim != 0:
            raise ValueError("weight_sum must be a scalar tensor")
    else:
        if X is None or Y is None:
            raise ValueError("either expectations or both X and Y must be provided")
        if X.shape[:-1] != Y.shape[:-1]:
            raise ValueError(f"X and Y must share leading shape; got X={tuple(X.shape)}, Y={tuple(Y.shape)}")
        ExxT, ExyT, EyyT, weight_sum = _accumulate_sufficient_stats(
            X,
            Y,
            weights,
            fit_intercept=fit_intercept,
            prior_ExxT=prior_ExxT,
            prior_ExyT=prior_ExyT,
        )

    x_dim = ExxT.shape[-1]
    d = ExyT.shape[-1]
    device = ExxT.device
    dtype = ExxT.dtype
    eye_x = torch.eye(x_dim, dtype=dtype, device=device)

    # Solve W_full = ExxT^{-1} ExyT via Cholesky (ExxT is symmetric PSD).
    # Add a tiny jitter for numerical stability when the design matrix is
    # rank-deficient (e.g. very small datasets).
    L_xx = torch.linalg.cholesky(ExxT + jitter * eye_x)
    W_full = torch.cholesky_solve(ExyT, L_xx)  # (x_dim, d)
    W_full_T = W_full.transpose(0, 1)  # (d, x_dim) — convenient for the residual

    # Posterior residual:
    # err = EyyT - W_full^T ExyT - ExyT^T W_full + W_full^T ExxT W_full
    #     = EyyT - W_full^T ExyT - (W_full^T ExyT)^T + W_full^T ExxT W_full
    # We compute W_full^T ExyT once and reuse.
    cross = W_full_T @ ExyT  # (d, d)
    err = EyyT - cross - cross.transpose(0, 1) + W_full_T @ ExxT @ W_full

    nu_n = nu0 + weight_sum.to(dtype=dtype)
    psi_n_diag = torch.diagonal(err, dim1=-2, dim2=-1) + psi0
    sigma_diag = psi_n_diag / (nu_n + d + 1)
    # Clamp away from zero to avoid degenerate likelihoods downstream.
    sigma_diag = torch.clamp(sigma_diag, min=jitter)

    if fit_intercept:
        W = W_full_T[:, :-1].contiguous()  # (d, p)
        b = W_full_T[:, -1].contiguous()  # (d,)
    else:
        W = W_full_T.contiguous()
        b = torch.zeros(d, dtype=dtype, device=device)

    return LinearRegressionResult(W=W, b=b, sigma_diag=sigma_diag)
