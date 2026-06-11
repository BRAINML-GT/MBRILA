"""Factor-analysis based emission initialisation (DLAG / mDLAG warm start).

DLAG's MATLAB reference (``init_FA_dlag.m``) seeds each region's loading
matrix ``C_r`` by fitting a per-region factor analysis with
``k = n_across + n_within`` latent dimensions. The fitted FA loading
matrix ``W_r`` becomes ``C_r``; the diagonal noise estimate ``ψ_r`` seeds
``diag(R_r)``.

Two helpers are exported:

- :func:`fa_em` — vanilla FA EM for a single ``(N, d)`` data matrix.
  Returns ``(W, psi, mu)`` where ``W`` is ``(d, k)``, ``psi`` is ``(d,)``
  and ``mu`` is the column-mean offset.
- :func:`fa_init_per_region` — applies :func:`fa_em` to every region of
  a ``MultiRegionData``-shaped tensor and returns a list of per-region
  ``C_r`` matrices ready to be copied into
  :class:`mbrila.observations.MultiRegionLinearObservation`.

The FA route is preferred when ``n_across = 0`` (no cross-region
factor to extract) or when the per-region neuron counts vary so much that
the joint pCCA solve is dominated by the largest region; otherwise pCCA
typically gives a stronger across-region signal and should be the
default.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def _initial_W_psi(Y_centred: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Sensible non-degenerate FA initialisation: top-``k`` PCA loadings
    plus residual variance for ``ψ``.

    Using PCA seeds the EM in the basin of the global optimum so we can
    cap iterations to a small budget without losing recovery quality.
    """
    n, d = Y_centred.shape
    cov_full = Y_centred.T @ Y_centred / max(n - 1, 1)
    # Symmetrise to wipe floating-point drift.
    cov_full = 0.5 * (cov_full + cov_full.T)
    eigvals, eigvecs = torch.linalg.eigh(cov_full)
    # eigh returns ascending order; flip to descending.
    idx = torch.argsort(eigvals, descending=True)
    eigvals = eigvals[idx].clamp(min=0.0)
    eigvecs = eigvecs[:, idx]
    top_vals = eigvals[:k]
    top_vecs = eigvecs[:, :k]  # (d, k)
    W = top_vecs * top_vals.clamp(min=1e-8).sqrt().unsqueeze(0)  # (d, k)
    residual_var = (
        eigvals[k:].sum() / max(d - k, 1)
        if k < d
        else torch.tensor(1e-3, dtype=Y_centred.dtype, device=Y_centred.device)
    )
    psi = torch.full((d,), max(float(residual_var), 1e-6), dtype=Y_centred.dtype, device=Y_centred.device)
    return W, psi


def fa_em(
    Y: Tensor,
    *,
    k: int,
    max_iter: int = 50,
    tol: float = 1e-4,
    psi_floor: float = 1e-6,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fit a factor analysis model by EM.

    Model: ``y = W z + μ + ε`` with ``z ~ N(0, I_k)`` and
    ``ε ~ N(0, diag(ψ))``.

    Parameters
    ----------
    Y:
        Data tensor of shape ``(N, d)``. ``N`` is the number of samples
        and ``d`` is the observation dimensionality.
    k:
        Latent dimensionality. Must satisfy ``1 <= k <= d``.
    max_iter:
        Maximum EM iterations.
    tol:
        Relative-LL convergence tolerance.
    psi_floor:
        Lower bound on the diagonal noise ``ψ`` to avoid degenerate
        likelihoods (Tipping & Bishop 1999).

    Returns
    -------
    W: ``(d, k)`` loading matrix.
    psi: ``(d,)`` diagonal noise.
    mu: ``(d,)`` column mean of ``Y`` (the offset).
    """
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D (N, d); got shape {tuple(Y.shape)}")
    n, d = int(Y.shape[0]), int(Y.shape[1])
    if k < 1 or k > d:
        raise ValueError(f"k must satisfy 1 <= k <= d={d}; got k={k}")
    if n < 2:
        raise ValueError(f"need >= 2 samples; got N={n}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1; got {max_iter}")

    dtype = Y.dtype
    device = Y.device
    mu = Y.mean(dim=0)
    Y_centred = Y - mu

    W, psi = _initial_W_psi(Y_centred, k)
    eye_k = torch.eye(k, dtype=dtype, device=device)
    prev_ll = -float("inf")
    YYt_diag = (Y_centred.square()).sum(dim=0)  # (d,) - Σ_n y² diagonal of YYᵀ

    for _ in range(max_iter):
        # E-step: posterior over z given y (Gaussian).
        psi_inv = 1.0 / psi.clamp(min=psi_floor)  # (d,)
        WtPsiW = (W * psi_inv.unsqueeze(-1)).T @ W  # (k, k)
        # Posterior precision and covariance: Σ_z|y = (I + Wᵀ Ψ⁻¹ W)⁻¹
        post_prec = eye_k + WtPsiW
        L_post = torch.linalg.cholesky(post_prec)
        post_cov = torch.cholesky_solve(eye_k, L_post)
        # Posterior means: M_z = Σ_z|y · Wᵀ Ψ⁻¹ · y for each y
        Wt_PsiInv = (W * psi_inv.unsqueeze(-1)).T  # (k, d)
        Z_means = (Y_centred @ Wt_PsiInv.T) @ post_cov  # (N, k)
        # Sum of second moments: E[zzᵀ] summed over N
        Ezz_sum = n * post_cov + Z_means.T @ Z_means  # (k, k)

        # M-step: W and ψ.
        Yz = Y_centred.T @ Z_means  # (d, k) - Σ_n y_n z̄_nᵀ
        L_Ezz = torch.linalg.cholesky(Ezz_sum + 1e-12 * eye_k)
        # W_new = Yz · (Ezz_sum)⁻¹
        W_new = torch.cholesky_solve(Yz.T, L_Ezz).T  # (d, k)
        psi_new = (YYt_diag - (W_new * Yz).sum(dim=1)) / n  # (d,)
        psi_new = psi_new.clamp(min=psi_floor)

        # Approximate marginal LL (for convergence): log p(y) under FA.
        # Σ_y = W Wᵀ + Ψ. We use a cheap O(d·k²) form via Woodbury.
        # log det Σ_y = log det Ψ + log det(I + Wᵀ Ψ⁻¹ W)
        log_det_y = (
            torch.log(psi_new.clamp(min=psi_floor)).sum()
            + 2.0
            * torch.diagonal(torch.linalg.cholesky(eye_k + (W_new * (1.0 / psi_new).unsqueeze(-1)).T @ W_new))
            .log()
            .sum()
        )
        # Quadratic via Woodbury: yᵀ Σ_y⁻¹ y = yᵀ Ψ⁻¹ y - yᵀ Ψ⁻¹ W (I + Wᵀ Ψ⁻¹ W)⁻¹ Wᵀ Ψ⁻¹ y
        psi_inv_new = 1.0 / psi_new
        Y_psi = Y_centred * psi_inv_new.unsqueeze(0)
        quad_a = (Y_psi * Y_centred).sum()
        WtPsiY = (W_new * psi_inv_new.unsqueeze(-1)).T @ Y_centred.T  # (k, N)
        L_post2 = torch.linalg.cholesky(eye_k + (W_new * psi_inv_new.unsqueeze(-1)).T @ W_new)
        sol = torch.cholesky_solve(WtPsiY, L_post2)
        quad_b = (WtPsiY * sol).sum()
        quad = quad_a - quad_b
        ll = float(
            -0.5
            * (
                n * log_det_y
                + quad
                + n * d * torch.log(torch.tensor(2.0 * torch.pi, dtype=dtype, device=device))
            )
        )

        W, psi = W_new, psi_new
        if abs(ll - prev_ll) < tol * max(abs(ll), 1.0):
            break
        prev_ll = ll

    return W, psi, mu


def fa_init_per_region(
    y: Tensor,
    *,
    y_dims: Sequence[int],
    n_per_region: int,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> tuple[list[Tensor], Tensor]:
    """Per-region factor analysis to seed DLAG's emission matrices.

    Reshapes ``y`` from ``(n_trials, T, sum(y_dims))`` to
    ``(n_trials · T, sum(y_dims))`` then runs an independent FA fit on
    each region's columns. Returns the per-region loading matrices and a
    concatenated diagonal noise estimate suitable for
    :class:`mbrila.observations.MultiRegionLinearObservation`.

    Parameters
    ----------
    y:
        Observed data, shape ``(n_trials, T, sum(y_dims))``.
    y_dims:
        Per-region neuron counts.
    n_per_region:
        Latent dimensionality per region (``= n_across + n_within`` in
        DLAG). Must satisfy ``1 <= n_per_region <= min(y_dims)``.
    max_iter:
        Maximum FA-EM iterations per region.
    tol:
        Convergence tolerance per region.

    Returns
    -------
    Cs:
        List of length ``len(y_dims)``; entry ``r`` has shape
        ``(y_dims[r], n_per_region)``.
    diag_R:
        Concatenated diagonal noise of shape ``(sum(y_dims),)``.
    """
    if y.ndim != 3:
        raise ValueError(f"y must have shape (n_trials, T, n_neurons); got {tuple(y.shape)}")
    if int(sum(y_dims)) != int(y.shape[-1]):
        raise ValueError(f"sum(y_dims)={int(sum(y_dims))} must equal y.shape[-1]={int(y.shape[-1])}")
    if n_per_region < 1:
        raise ValueError(f"n_per_region must be >= 1; got {n_per_region}")
    min_y = min(y_dims)
    if n_per_region > min_y:
        raise ValueError(
            f"n_per_region={n_per_region} exceeds the smallest y_dim={min_y}; "
            "factor analysis requires k <= d per region"
        )

    n_trials, T, _ = y.shape
    y_flat = y.reshape(n_trials * T, -1)

    Cs: list[Tensor] = []
    psi_parts: list[Tensor] = []
    offset = 0
    for d_r in y_dims:
        Y_r = y_flat[:, offset : offset + d_r]
        W_r, psi_r, _ = fa_em(Y_r, k=n_per_region, max_iter=max_iter, tol=tol)
        Cs.append(W_r.contiguous())
        psi_parts.append(psi_r)
        offset += d_r
    diag_R = torch.cat(psi_parts, dim=0)
    return Cs, diag_R
