"""Tests for CF6b — :func:`build_variational_kalman_inputs`.

Three responsibilities:

1. Algebraic identities — the synthetic ``(H_eff, R_eff, y_pseudo)``
   reproduce the variational precision and info-vector contributions
   to the latent posterior, by construction. Verified directly.
2. Point-limit parity — when the variational ``q(C)`` collapses to a
   point (``C_cov → 0``), the synthetic inputs reproduce what a standard
   :class:`MultiRegionLinearObservation` would feed into the Kalman
   filter. This is the load-bearing test that the variational engine
   smoothly degenerates to the point case.
3. End-to-end smoke — feeding the synthetic inputs into the existing
   sequential :func:`kalman_filter` runs to completion and produces
   finite means/covs of the expected shapes. (Bit-identical to a
   variational filter; this is what CF6c will consume.)
"""

from __future__ import annotations

import pytest
import torch

from mbrila import MDLAG, LatentSpec, MOSEKernel
from mbrila.inference.ard_helpers import (
    build_variational_kalman_inputs,
    compute_CPhi,
    compute_CPhiC_block,
)
from mbrila.inference.kalman.sequential import kalman_filter
from mbrila.observations.ard import ARDObservation


def _make_mdlag(K: int = 2, R: int = 2, T: int = 6, neuron_per_region: int = 4) -> MDLAG:
    spec = LatentSpec(n_across=K, n_within=(0,) * R, selection="ard")
    return MDLAG(
        latent_spec=spec,
        y_dims=tuple(neuron_per_region for _ in range(R)),
        T=T,
        kernel_factory_across=lambda: MOSEKernel(num_regions=R, init_sigma=0.1),
        dtype=torch.float64,
        device="cpu",
    )


def _identity_H_select(M: int) -> torch.Tensor:
    """For mDLAG-without-within, H_select picks every slot directly."""
    return torch.eye(M, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Algebraic identities — the very point of the construction
# ---------------------------------------------------------------------------


class TestAlgebraicIdentities:
    """The whole CF6b trick rests on two equalities holding to machine precision:

    H_effᵀ · R_eff⁻¹ · H_eff = H_selectᵀ · CPhiC_block · H_select
    H_effᵀ · R_eff⁻¹ · y_pseudo[t] = H_selectᵀ · CPhi · (y[t] - d_mean)

    These are what license calling the standard Kalman filter on the
    synthetic inputs.
    """

    def test_precision_identity(self) -> None:
        model = _make_mdlag(K=2, R=2, T=5, neuron_per_region=3)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        M = obs.n_regions * obs.n_obs_per_region
        H_select = _identity_H_select(M)
        y = model.sample(n_trials=4, T=5, seed=0).y

        inputs = build_variational_kalman_inputs(obs, H_select, y, jitter=1e-12)
        H_eff = inputs["H_eff"]
        R_eff = inputs["R_eff"]
        R_inv = torch.linalg.inv(R_eff)
        # H_effᵀ · R⁻¹ · H_eff
        prec_synthetic = H_eff.transpose(-1, -2) @ R_inv @ H_eff
        # H_selectᵀ · CPhiC_block · H_select
        CPhiC_block = compute_CPhiC_block(obs)
        prec_direct = H_select.transpose(-1, -2) @ CPhiC_block @ H_select
        torch.testing.assert_close(prec_synthetic, prec_direct, atol=1e-10, rtol=1e-10)

    def test_info_vector_identity(self) -> None:
        model = _make_mdlag(K=2, R=2, T=4, neuron_per_region=3)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        M = obs.n_regions * obs.n_obs_per_region
        H_select = _identity_H_select(M)
        y = model.sample(n_trials=3, T=4, seed=0).y

        inputs = build_variational_kalman_inputs(obs, H_select, y, jitter=1e-12)
        H_eff = inputs["H_eff"]
        R_eff = inputs["R_eff"]
        y_pseudo = inputs["y_pseudo"]
        R_inv = torch.linalg.inv(R_eff)
        # H_effᵀ · R⁻¹ · y_pseudo[t] for all (b, t).
        info_synthetic = torch.einsum("md,mn,btn->btd", H_eff, R_inv, y_pseudo)
        # Direct: H_selectᵀ · CPhi · (y - d_mean)
        CPhi = compute_CPhi(obs)
        y_centred = y - obs.d_mean
        info_direct = torch.einsum("md,mi,bti->btd", H_select, CPhi, y_centred)
        torch.testing.assert_close(info_synthetic, info_direct, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# Point-limit parity — variational degenerates to point as C_cov → 0
# ---------------------------------------------------------------------------


class TestPointLimitParity:
    """In the limit ``q(C) → δ(C - C_mean)`` (i.e. ``C_cov = 0``),
    ``⟨CᵀΦC⟩ = C_meanᵀ · diag(φ) · C_mean``. The variational Kalman
    inputs should then reproduce what a standard linear-Gaussian
    observation with ``H = block_diag(C_mean) · H_select`` and
    ``R = diag(1/φ_mean)`` would feed into the Kalman filter (i.e.
    the same H^T R^{-1} H and the same info vector)."""

    def test_C_cov_zero_matches_point_emission(self) -> None:
        model = _make_mdlag(K=2, R=2, T=5, neuron_per_region=3)
        obs = model.observation
        assert isinstance(obs, ARDObservation)

        # Spike C_cov to zero so C_moment = outer(C_mean).
        with torch.no_grad():
            for r in range(obs.n_regions):
                # C_moment_{r} buffer is what compute_CPhiC_block reads;
                # set it to the rank-1 outer product of C_means.
                C_r = obs.C_means[r]  # (y_r, k)
                outer = C_r.unsqueeze(-1) * C_r.unsqueeze(-2)  # (y_r, k, k)
                obs.C_moments[r].copy_(outer)

        M = obs.n_regions * obs.n_obs_per_region
        H_select = _identity_H_select(M)
        y = model.sample(n_trials=3, T=5, seed=0).y

        inputs = build_variational_kalman_inputs(obs, H_select, y, jitter=1e-12)
        prec_variational = (
            inputs["H_eff"].transpose(-1, -2) @ torch.linalg.inv(inputs["R_eff"]) @ inputs["H_eff"]
        )

        # Point: H = block_diag(C_mean) · H_select; precision = Hᵀ · diag(φ) · H.
        C_blk = obs.block_diag_C()  # (n_y, M)
        H_point = C_blk @ H_select  # (n_y, D)
        phi = obs.phi_mean
        prec_point = H_point.transpose(-1, -2) @ torch.diag(phi) @ H_point

        torch.testing.assert_close(prec_variational, prec_point, atol=1e-9, rtol=1e-9)

    def test_C_cov_zero_info_vector_matches_point_emission(self) -> None:
        model = _make_mdlag(K=2, R=2, T=4, neuron_per_region=3)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        # Zero the C_cov contribution to C_moment.
        with torch.no_grad():
            for r in range(obs.n_regions):
                C_r = obs.C_means[r]
                outer = C_r.unsqueeze(-1) * C_r.unsqueeze(-2)
                obs.C_moments[r].copy_(outer)

        M = obs.n_regions * obs.n_obs_per_region
        H_select = _identity_H_select(M)
        y = model.sample(n_trials=3, T=4, seed=0).y

        inputs = build_variational_kalman_inputs(obs, H_select, y, jitter=1e-12)
        info_variational = torch.einsum(
            "md,mn,btn->btd",
            inputs["H_eff"],
            torch.linalg.inv(inputs["R_eff"]),
            inputs["y_pseudo"],
        )

        # Point: info = Hᵀ · diag(φ) · (y - d).
        C_blk = obs.block_diag_C()
        H_point = C_blk @ H_select
        phi = obs.phi_mean
        y_centred = y - obs.d_mean
        info_point = torch.einsum("nd,n,btn->btd", H_point, phi, y_centred)

        torch.testing.assert_close(info_variational, info_point, atol=1e-9, rtol=1e-9)


# ---------------------------------------------------------------------------
# End-to-end smoke: synthetic inputs run through the existing Kalman filter
# ---------------------------------------------------------------------------


class TestKalmanFilterSmoke:
    """The synthetic inputs are designed to plug straight into the
    existing :func:`kalman_filter`. This is the load-bearing check that
    CF6c can rely on the trick."""

    def test_filter_runs_and_produces_finite_output(self) -> None:
        model = _make_mdlag(K=2, R=2, T=6, neuron_per_region=3)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        M = obs.n_regions * obs.n_obs_per_region
        D = M  # for this no-within mDLAG case, state dim equals M.
        H_select = _identity_H_select(M)
        y = model.sample(n_trials=4, T=6, seed=0).y
        inputs = build_variational_kalman_inputs(obs, H_select, y)

        B, T = y.shape[:2]
        # Use trivial dynamics: identity transition, identity innovation cov.
        # Kalman filter should still run and produce sensible posteriors.
        F = torch.eye(D, dtype=torch.float64).unsqueeze(0).expand(T, -1, -1).contiguous()
        Q = torch.eye(D, dtype=torch.float64).unsqueeze(0).expand(T, -1, -1).contiguous()
        m0 = torch.zeros(D, dtype=torch.float64)
        P0 = torch.eye(D, dtype=torch.float64)

        filt_means, filt_covs, log_ml = kalman_filter(
            y=inputs["y_pseudo"],
            F=F,
            Q=Q,
            H=inputs["H_eff"],
            R=inputs["R_eff"],
            m0=m0,
            P0=P0,
            return_log_marginal=True,
        )
        assert filt_means.shape == (B, T, D)
        assert filt_covs.shape == (B, T, D, D)
        assert log_ml.shape == (B,)
        assert torch.isfinite(filt_means).all().item()
        assert torch.isfinite(filt_covs).all().item()
        # Filtered covariances are PSD per trial per time.
        eigvals = torch.linalg.eigvalsh(filt_covs)
        assert eigvals.min().item() > -1e-9


# ---------------------------------------------------------------------------
# Shape / error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_rejects_2d_y(self) -> None:
        model = _make_mdlag()
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        M = obs.n_regions * obs.n_obs_per_region
        H_select = _identity_H_select(M)
        bad_y = torch.zeros(3, 5, dtype=torch.float64)  # missing trial dim
        with pytest.raises(ValueError, match="y must be"):
            build_variational_kalman_inputs(obs, H_select, bad_y)

    def test_rejects_non_2d_H_select(self) -> None:
        model = _make_mdlag()
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        with pytest.raises(ValueError, match="H_select must be 2-D"):
            build_variational_kalman_inputs(
                obs, torch.zeros(2, 3, 4, dtype=torch.float64), torch.zeros(1, 2, 3)
            )

    def test_rejects_H_select_row_mismatch(self) -> None:
        model = _make_mdlag(K=2, R=2, neuron_per_region=3)
        obs = model.observation
        assert isinstance(obs, ARDObservation)
        # H_select has wrong number of rows (M expected = 4).
        bad_H = torch.eye(5, dtype=torch.float64)
        y = torch.zeros(2, 5, sum(model._y_dims), dtype=torch.float64)
        with pytest.raises(ValueError, match="CPhiC_block shape"):
            build_variational_kalman_inputs(obs, bad_H, y)
