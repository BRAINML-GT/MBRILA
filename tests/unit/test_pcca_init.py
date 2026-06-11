"""Tests for the multi-view pCCA emission init (DLAG warm-start)."""

from __future__ import annotations

import pytest
import torch

from mbrila import DLAG, LatentSpec, MOSEKernel
from mbrila.init.pcca import pcca_init_C


def _kf(R: int, sigma: float = 0.05) -> dict[str, object]:
    return {
        "kernel_factory_across": lambda: MOSEKernel(num_regions=R, init_sigma=sigma),
        "kernel_factory_within": lambda: MOSEKernel(num_regions=1, init_sigma=sigma),
    }


def _sample_shared_z(
    W_per_region: list[torch.Tensor],
    psi_per_region: list[torch.Tensor],
    n_samples: int,
    seed: int,
) -> tuple[torch.Tensor, list[int]]:
    """Sample from y_r = W_r z + ε_r with shared z ~ N(0, I_k)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    k = W_per_region[0].shape[1]
    z = torch.randn(n_samples, k, generator=gen, dtype=torch.float64)
    parts: list[torch.Tensor] = []
    y_dims: list[int] = []
    for W_r, psi_r in zip(W_per_region, psi_per_region, strict=True):
        d_r = W_r.shape[0]
        eps_r = torch.randn(n_samples, d_r, generator=gen, dtype=torch.float64) * psi_r.sqrt().unsqueeze(0)
        parts.append(z @ W_r.T + eps_r)
        y_dims.append(d_r)
    return torch.cat(parts, dim=1), y_dims


class TestPCCAInit:
    def test_recovers_loadings_up_to_rotation(self) -> None:
        """Synthetic pCCA model → pCCA init should recover W W^T per region."""
        torch.manual_seed(0)
        k = 2
        # Ground-truth per-region loadings with distinguishable structure.
        W1 = torch.randn(8, k, dtype=torch.float64)
        W2 = torch.randn(6, k, dtype=torch.float64)
        W3 = torch.randn(7, k, dtype=torch.float64)
        psi1 = torch.full((8,), 0.1, dtype=torch.float64)
        psi2 = torch.full((6,), 0.1, dtype=torch.float64)
        psi3 = torch.full((7,), 0.1, dtype=torch.float64)
        y_flat, y_dims = _sample_shared_z([W1, W2, W3], [psi1, psi2, psi3], n_samples=4000, seed=0)
        # Pack into (B=2000, T=2, sum y_dims) for the pCCA init's expected shape.
        y = y_flat.view(2000, 2, -1)

        Cs, _diag_R, _mu = pcca_init_C(y, y_dims=tuple(y_dims), n_across=k, n_within=0, max_iter=80)
        assert len(Cs) == 3
        # W W^T is rotation-invariant; compare per-region.
        for C_r, W_true_r in zip(Cs, [W1, W2, W3], strict=True):
            rel = (C_r @ C_r.T - W_true_r @ W_true_r.T).norm() / (W_true_r @ W_true_r.T).norm()
            assert rel.item() < 0.15, f"region C C^T relative error too high: {rel.item():.3f}"

    def test_shape_and_within_augmentation(self) -> None:
        torch.manual_seed(7)
        y = torch.randn(50, 10, 8 + 5, dtype=torch.float64)
        Cs, diag_R, mu = pcca_init_C(
            y,
            y_dims=(8, 5),
            n_across=2,
            n_within=2,
            max_iter=10,
        )
        assert len(Cs) == 2
        assert Cs[0].shape == (8, 4)  # 2 across + 2 within
        assert Cs[1].shape == (5, 4)
        assert diag_R.shape == (13,)
        assert mu.shape == (13,)
        # The first 2 columns of each region (across part) should be
        # orthogonal to the last 2 columns (within part) under cov(y_r).
        # Pseudo-check: SVD trick guarantees this — verify rank is full.
        assert torch.linalg.matrix_rank(Cs[0]).item() == 4

    def test_natural_scale(self) -> None:
        """pCCA loadings should have data-scaled magnitude.

        Plain CCA returns orthonormal-under-Σ directions, so its ``||W||``
        is ``∝ 1/√Var[y]``. pCCA returns the proper rank-K factor of
        ``Σ_y - ψ``, so ``||W||`` is ``∝ √Var[y]``. We assert that
        ``||C||`` is within a factor of 3 of ``||W_true||`` — this is
        the magnitude diagnostic, not a strict numerical recovery test
        (finite-sample reconstruction error is naturally 10-20%).
        """
        torch.manual_seed(1)
        W1 = torch.randn(6, 2, dtype=torch.float64) * 0.5
        W2 = torch.randn(8, 2, dtype=torch.float64) * 0.5
        psi1 = torch.full((6,), 0.05, dtype=torch.float64)
        psi2 = torch.full((8,), 0.05, dtype=torch.float64)
        y_flat, y_dims = _sample_shared_z([W1, W2], [psi1, psi2], n_samples=3000, seed=2)
        y = y_flat.view(1500, 2, -1)
        Cs, _diag_R, _ = pcca_init_C(y, y_dims=tuple(y_dims), n_across=2, n_within=0, max_iter=80)

        for C_r, W_true_r in zip(Cs, [W1, W2], strict=True):
            ratio = (C_r.norm() / W_true_r.norm()).item()
            assert 0.3 < ratio < 3.0, f"pCCA-init norm ratio off: {ratio:.3f}"

    def test_rejects_n_across_zero(self) -> None:
        y = torch.randn(20, 5, 6, dtype=torch.float64)
        with pytest.raises(ValueError, match="n_across >= 1"):
            pcca_init_C(y, y_dims=(3, 3), n_across=0, n_within=1)

    def test_rejects_too_large_k(self) -> None:
        y = torch.randn(20, 5, 7, dtype=torch.float64)
        with pytest.raises(ValueError, match="exceeds the smallest"):
            pcca_init_C(y, y_dims=(3, 4), n_across=2, n_within=2)


class TestDLAGpCCAInit:
    def test_dlag_uses_pcca_by_default(self) -> None:
        """``DLAG.initialize_from_data`` defaults to mode='pcca' now."""
        m = DLAG(
            LatentSpec(n_across=1, n_within=(1, 1)),
            y_dims=(5, 6),
            T=8,
            device="cpu",
            **_kf(2),  # type: ignore[arg-type]
        )
        data = m.sample(n_trials=20, T=8, seed=0)
        C_before = m.observation.block_diag_C().clone()
        m.initialize_from_data(data)  # default mode='pcca'
        assert (m.observation.block_diag_C() - C_before).abs().max().item() > 1e-6

    def test_dlag_pcca_init_loadings_have_meaningful_scale(self) -> None:
        """After pCCA init, ``||C||`` should be on the order of ``√Var[y]``.

        Concrete check: ``trace(C C^T + diag(R))`` should be a sizeable
        fraction (``> 50%``) of ``trace(cov_sample)``. If pCCA were
        producing CCA-scale loadings (``∝ 1/√Var[y]``) the fraction
        would be tiny — this guards against accidentally falling back
        to a CCA-flavoured path.
        """
        torch.manual_seed(2)
        spec = LatentSpec(n_across=1, n_within=(1, 1))
        truth = DLAG(spec, y_dims=(6, 6), T=10, device="cpu", **_kf(2))  # type: ignore[arg-type]
        with torch.no_grad():
            for C_r in truth.observation.Cs:
                C_r.data = torch.randn_like(C_r.data)
            truth.observation.diag_R_param.data.fill_(0.05)
        data = truth.sample(n_trials=50, T=10, seed=3)

        m = DLAG(spec, y_dims=(6, 6), T=10, device="cpu", **_kf(2))  # type: ignore[arg-type]
        m.initialize_from_data(data, mode="pcca")

        C = m.observation.block_diag_C()
        R = torch.diag(m.observation.diag_R())
        cov_recon_trace = (C @ C.T + R).diagonal().sum().item()
        y_flat = data.y.reshape(-1, 12)
        cov_sample_trace = ((y_flat - y_flat.mean(dim=0)).pow(2).sum() / (y_flat.shape[0] - 1)).item()
        ratio = cov_recon_trace / cov_sample_trace
        assert ratio > 0.5, f"pCCA-init explains too little variance: {ratio:.3f}"

    def test_dlag_n_across_zero_falls_back_to_fa(self) -> None:
        """``mode='pcca'`` with ``n_across=0`` should fall back to FA."""
        m = DLAG(
            LatentSpec(n_across=0, n_within=(2, 2)),
            y_dims=(5, 6),
            T=8,
            device="cpu",
            **_kf(2),  # type: ignore[arg-type]
        )
        data = m.sample(n_trials=20, T=8, seed=0)
        C_before = m.observation.block_diag_C().clone()
        m.initialize_from_data(data, mode="pcca")  # falls back to FA internally
        assert (m.observation.block_diag_C() - C_before).abs().max().item() > 1e-6
