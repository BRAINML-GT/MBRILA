"""Tests for ADM's ``--init-mode pcca`` branch.

The cross-region multi-view pCCA init was introduced to remove the
scale-anchor wobble we saw on the V1V2 real-data demo (ADM's per-region
``C`` ended up on a ``1/√Var[y]`` scale under CCA + closed-form M-step,
then the per-iteration anchor had to repeatedly rescale it). pCCA fits
loadings on the data scale directly, so these tests assert exactly that
shape-invariant: ``‖C‖`` after pCCA-init should be on the order of
``√Var[y]``, not ``1/√Var[y]``.
"""

from __future__ import annotations

import pytest
import torch

from mbrila import ADM, LatentSpec, MOSEKernel


def _mose_factories(R: int, sigma: float = 0.1) -> dict[str, object]:
    return {
        "kernel_factory_across": lambda: MOSEKernel(num_regions=R, init_sigma=sigma),
        "kernel_factory_within": lambda: MOSEKernel(num_regions=1, init_sigma=sigma),
    }


def _two_region_model(*, n_across: int = 1, n_within: int = 1) -> ADM:
    return ADM(
        LatentSpec(n_across=n_across, n_within=(n_within, n_within)),
        y_dims=(8, 10),
        T=6,
        device="cpu",
        **_mose_factories(2),  # type: ignore[arg-type]
    )


class TestADMPCCAInit:
    def test_pcca_mode_runs_and_changes_C(self) -> None:
        m = _two_region_model()
        data = m.sample(n_trials=20, T=6, seed=0)
        C_before = m.observation.block_diag_C().clone()
        m.initialize_from_data(data, mode="pcca")
        assert (m.observation.block_diag_C() - C_before).abs().max().item() > 1e-6

    def test_pcca_mode_is_default(self) -> None:
        """``initialize_from_data()`` defaults to ``mode='pcca'``."""
        m_default = _two_region_model()
        m_pcca = _two_region_model()
        data = m_default.sample(n_trials=20, T=6, seed=0)
        m_default.initialize_from_data(data)
        m_pcca.initialize_from_data(data, mode="pcca")
        torch.testing.assert_close(
            m_default.observation.block_diag_C(),
            m_pcca.observation.block_diag_C(),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_pcca_init_loads_on_data_scale(self) -> None:
        """pCCA loadings explain a meaningful fraction of ``cov(y)``.

        ``trace(C Cᵀ + diag(R))`` should be on the same order as
        ``trace(cov(y))``.
        """
        torch.manual_seed(0)
        spec = LatentSpec(n_across=1, n_within=(1, 1))
        truth = ADM(spec, y_dims=(8, 10), T=8, device="cpu", **_mose_factories(2))  # type: ignore[arg-type]
        with torch.no_grad():
            for C_r in truth.observation.Cs:
                C_r.data = torch.randn_like(C_r.data)
            truth.observation.diag_R_param.data.fill_(0.05)
        data = truth.sample(n_trials=40, T=8, seed=1)

        m_pcca = ADM(spec, y_dims=(8, 10), T=8, device="cpu", **_mose_factories(2))  # type: ignore[arg-type]
        m_pcca.initialize_from_data(data, mode="pcca")
        C_pcca = m_pcca.observation.block_diag_C()
        R_pcca = torch.diag(m_pcca.observation.diag_R())
        recon_trace = (C_pcca @ C_pcca.T + R_pcca).diagonal().sum().item()

        y_flat = data.y.reshape(-1, 18)
        cov_sample_trace = ((y_flat - y_flat.mean(dim=0)).pow(2).sum() / (y_flat.shape[0] - 1)).item()

        ratio_pcca = recon_trace / cov_sample_trace
        assert ratio_pcca > 0.5, f"pCCA-init explained too little: {ratio_pcca:.3f}"

    def test_pcca_works_with_many_regions(self) -> None:
        """pCCA generalises to arbitrary R — no pairwise loop needed.

        The implementation runs a single shared-z FA on the stacked
        ``(B·T, sum y_dims)`` matrix; the block-row structure of the
        loading matrix is preserved by data layout. This is the
        scalability the user cares about for 5+ region recordings.
        """
        spec = LatentSpec(n_across=1, n_within=(1, 1, 1, 1, 1))
        m = ADM(spec, y_dims=(6, 8, 5, 7, 9), T=6, device="cpu", **_mose_factories(5))  # type: ignore[arg-type]
        data = m.sample(n_trials=30, T=6, seed=0)
        m.initialize_from_data(data, mode="pcca")
        # Per-region C shapes match (y_dim_r, n_across + n_within).
        for r, C_r in enumerate(m.observation.Cs):
            assert C_r.shape == (m._y_dims[r], 2)
        # Block-diag emission has the right total shape.
        assert m.observation.block_diag_C().shape == (sum(m._y_dims), 5 * 2)

    def test_pcca_with_n_across_zero_falls_back_to_fa(self) -> None:
        """``mode='pcca'`` with ``n_across=0`` should not raise.

        pCCA requires ``n_across >= 1`` (no shared latent otherwise);
        ADM's init delegates to per-region FA in this degenerate case,
        matching DLAG's behaviour.
        """
        spec = LatentSpec(n_across=0, n_within=(2, 2))
        m = ADM(spec, y_dims=(6, 8), T=6, device="cpu", **_mose_factories(2))  # type: ignore[arg-type]
        data = m.sample(n_trials=20, T=6, seed=0)
        C_before = m.observation.block_diag_C().clone()
        m.initialize_from_data(data, mode="pcca")
        assert (m.observation.block_diag_C() - C_before).abs().max().item() > 1e-6

    def test_pcca_rejects_unknown_mode(self) -> None:
        m = _two_region_model()
        data = m.sample(n_trials=10, T=6, seed=0)
        with pytest.raises(ValueError, match="unknown init mode"):
            m.initialize_from_data(data, mode="bogus")  # type: ignore[arg-type]
