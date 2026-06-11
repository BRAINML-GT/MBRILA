"""Multi-view probabilistic CCA initialisation (DLAG / mDLAG warm start).

DLAG MATLAB's canonical init (``init_pCCA_dlag.m``) fits probabilistic
CCA jointly across all regions, then augments each region with within-
region directions orthogonal to the pCCA across-columns. mbrila
reproduces that recipe.

Why pCCA, not plain CCA
-----------------------
Probabilistic CCA models

    y_r = W_r · z + b_r + ε_r,   z ~ N(0, I_k),   ε_r ~ N(0, diag(ψ_r))

with a **shared** latent ``z`` across regions. The loading ``W_r`` is
returned in data-scaled units (``Var[y_r] ≈ W_r W_rᵀ + diag(ψ_r)``),
which is what the emission matrix needs to be on. This makes pCCA a
drop-in initialisation for engines that have no in-loop scale anchor
(notably :class:`ExactEMEngine`).

Implementation
--------------
Multi-view pCCA is mathematically equivalent to a single shared-z
factor analysis on the stacked data: stack ``(B·T, sum y_dims)``,
fit FA with ``k = n_across``, then split the loading matrix
``W ∈ (sum y_dims, n_across)`` row-block-wise into per-region
``W_r ∈ (y_dim_r, n_across)``. We reuse :func:`mbrila.init.factor_analysis.fa_em`
for the actual EM and only do the splitting / within-augmentation here.

Within-region augmentation matches the CCA-path SVD trick: for each
region, decompose ``C_across_rᵀ · cov(y_r)`` and pick the columns of
the right singular vectors past the first ``n_across`` slots — those
are orthogonal to what pCCA captured and span the dominant
within-region directions.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from mbrila.init.factor_analysis import fa_em


def _column_cov(X: Tensor) -> Tensor:
    """Sample covariance of columns of ``X`` (rows = samples).

    Equivalent to ``np.cov(X, rowvar=False)``: returns shape
    ``(X.shape[1], X.shape[1])`` with mean removed.
    """
    if X.shape[0] < 2:
        raise ValueError(f"need >= 2 samples for covariance; got {X.shape[0]}")
    centred = X - X.mean(dim=0, keepdim=True)
    n = X.shape[0]
    return centred.transpose(0, 1) @ centred / (n - 1)


def pcca_init_C(
    y: Tensor,
    *,
    y_dims: Sequence[int],
    n_across: int,
    n_within: int,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> tuple[list[Tensor], Tensor, Tensor]:
    """Multi-view pCCA emission init for DLAG / mDLAG.

    Mirrors ``init_pCCA_dlag.m``:

    1. Stack the data and fit a shared-z factor analysis with
       ``k = n_across`` factors. The fitted loading matrix
       ``W ∈ (sum y_dims, n_across)`` has block-row structure
       ``[W_0; W_1; …; W_{R-1}]`` because the stacked data layout
       puts each region's neurons in a contiguous slice.
    2. Split ``W`` per region to obtain ``C_across_r``.
    3. For each region with ``n_within > 0``, augment ``C_across_r``
       with directions orthogonal to it that capture the most
       remaining within-region variance (SVD of
       ``C_across_rᵀ · cov(y_r)``).
    4. Return per-region loadings + concatenated diagonal noise
       + concatenated offset.

    Parameters
    ----------
    y:
        Observed data, shape ``(n_trials, T, sum(y_dims))``.
    y_dims:
        Per-region neuron counts.
    n_across:
        Number of cross-region (shared-z) latents. Must be ``>= 1``;
        callers with ``n_across = 0`` should use
        :func:`mbrila.init.factor_analysis.fa_init_per_region` instead.
    n_within:
        Number of within-region latents to add per region.
    max_iter, tol:
        Forwarded to :func:`mbrila.init.factor_analysis.fa_em`.

    Returns
    -------
    Cs:
        List of ``(y_dim_r, n_across + n_within)`` tensors.
    diag_R:
        ``(sum y_dims,)`` per-neuron noise variance (pCCA's
        ``ψ``-estimate, concatenated across regions).
    mu:
        ``(sum y_dims,)`` per-neuron offset (column mean of the
        stacked data) — used to initialise the emission offset
        ``d`` when the caller wants pCCA's own mean rather than
        zero.
    """
    if y.ndim != 3:
        raise ValueError(f"y must have shape (n_trials, T, n_neurons); got {tuple(y.shape)}")
    if int(sum(y_dims)) != int(y.shape[-1]):
        raise ValueError(f"sum(y_dims)={int(sum(y_dims))} must equal y.shape[-1]={int(y.shape[-1])}")
    if n_across < 1:
        raise ValueError(
            f"pcca_init_C requires n_across >= 1; got {n_across}. "
            f"For within-only initialisation use fa_init_per_region."
        )
    if n_within < 0:
        raise ValueError(f"n_within must be >= 0; got {n_within}")
    min_y = min(y_dims)
    if n_across + n_within > min_y:
        raise ValueError(
            f"n_across + n_within = {n_across + n_within} exceeds the smallest "
            f"y_dim = {min_y}; the per-region SVD augmentation requires at "
            f"least n_across + n_within right singular vectors."
        )

    n_trials, T, _ = y.shape
    y_flat = y.reshape(n_trials * T, -1)

    # Shared-z FA on stacked data — multi-view pCCA's MLE.
    W_global, psi_global, mu_global = fa_em(y_flat, k=n_across, max_iter=max_iter, tol=tol)

    Cs: list[Tensor] = []
    offset = 0
    for r, d_r in enumerate(y_dims):
        C_across_r = W_global[offset : offset + d_r, :]  # (d_r, n_across)
        if n_within > 0:
            Y_r = y_flat[:, offset : offset + d_r]
            cov_r = _column_cov(Y_r)
            # SVD of C_across_rᵀ · cov(y_r) — same trick as ADM/DLAG CCA path.
            _, _, Vh = torch.linalg.svd(C_across_r.transpose(0, 1) @ cov_r, full_matrices=True)
            end = n_across + n_within
            if end > Vh.shape[1]:
                raise ValueError(
                    f"region {r}: y_dim_r = {d_r} cannot supply {n_across + n_within} orthogonal directions"
                )
            C_within_r = Vh[:, n_across:end]
            C_r = torch.cat([C_across_r, C_within_r], dim=1)
        else:
            C_r = C_across_r
        Cs.append(C_r.contiguous())
        offset += d_r

    return Cs, psi_global, mu_global
