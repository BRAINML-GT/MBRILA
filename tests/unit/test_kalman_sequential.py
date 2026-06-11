"""Sequential Kalman filter and RTS smoother tests."""

from __future__ import annotations

import math

import pytest
import torch

from mbrila.inference.kalman import kalman_filter, rts_smoother


def _random_lds(
    *, B: int, T: int, D: int, N: int, seed: int = 0, dtype: torch.dtype = torch.float64
) -> dict[str, torch.Tensor]:
    """Generate a small random linear-Gaussian state-space system + observations."""
    g = torch.Generator().manual_seed(seed)
    F = torch.eye(D, dtype=dtype) + 0.05 * torch.randn(D, D, generator=g, dtype=dtype)
    Q = 0.1 * torch.eye(D, dtype=dtype)
    H = torch.randn(N, D, generator=g, dtype=dtype)
    R = 0.2 * torch.eye(N, dtype=dtype)
    m0 = torch.zeros(D, dtype=dtype)
    P0 = torch.eye(D, dtype=dtype)

    # Forward simulate. x_0 ~ N(F m0, F P0 F^T + Q); we just initialise from
    # m0 + L_P0 ε to keep the helper concise.
    L_P0 = torch.linalg.cholesky(P0)
    L_Q = torch.linalg.cholesky(Q)
    L_R = torch.linalg.cholesky(R)
    x_prev = m0.expand(B, D) + torch.randn(B, D, generator=g, dtype=dtype) @ L_P0.T
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for _t in range(T):  # trial-loop: ok  (this is a *time* loop in a test fixture; not over trials)
        x_t = x_prev @ F.T + torch.randn(B, D, generator=g, dtype=dtype) @ L_Q.T
        y_t = x_t @ H.T + torch.randn(B, N, generator=g, dtype=dtype) @ L_R.T
        xs.append(x_t)
        ys.append(y_t)
        x_prev = x_t
    return {
        "F": F,
        "Q": Q,
        "H": H,
        "R": R,
        "m0": m0,
        "P0": P0,
        "x_true": torch.stack(xs, dim=1),
        "y": torch.stack(ys, dim=1),
    }


class TestFilterNumeric:
    def test_steady_state_1d(self) -> None:
        # 1D LDS: x_{t+1} = x_t + N(0, q),  y_t = x_t + N(0, r).
        # Steady-state filtered variance solves p = (1/((p+q)^{-1} + r^{-1}))
        # which for q = r = 1 gives p = (sqrt(5)-1)/2.
        T = 200
        q, r = 1.0, 1.0
        F = torch.tensor([[1.0]], dtype=torch.float64)
        Q = q * torch.eye(1, dtype=torch.float64)
        H = torch.tensor([[1.0]], dtype=torch.float64)
        R = r * torch.eye(1, dtype=torch.float64)
        m0 = torch.zeros(1, dtype=torch.float64)
        P0 = torch.eye(1, dtype=torch.float64) * 1e6  # diffuse prior

        torch.manual_seed(0)
        y = torch.randn(1, T, 1, dtype=torch.float64)

        _means, covs, _ = kalman_filter(y, F, Q, H, R, m0, P0)
        ss = covs[0, -1, 0, 0].item()
        expected = (math.sqrt(5.0) - 1.0) / 2.0
        assert abs(ss - expected) < 1e-3

    def test_filter_converges_to_truth_in_low_noise(self) -> None:
        sys = _random_lds(B=4, T=30, D=2, N=3, seed=7)
        # Drop observation noise to near-zero: the filter should track the truth.
        sys["R"] = 1e-6 * torch.eye(3, dtype=torch.float64)
        means, _covs, _log_ml = kalman_filter(
            sys["y"], sys["F"], sys["Q"], sys["H"], sys["R"], sys["m0"], sys["P0"]
        )
        # Last few time steps should be very close to truth (modulo small Q).
        err = (means[:, -5:] - sys["x_true"][:, -5:]).abs().mean().item()
        assert err < 0.5  # generous because Q noise is non-trivial

    def test_log_marginal_matches_manual_sum(self) -> None:
        # Manually compute log p(y_t | y_{<t}) and compare to filter's output.
        sys = _random_lds(B=2, T=8, D=2, N=2, seed=3)
        _, _, log_ml = kalman_filter(sys["y"], sys["F"], sys["Q"], sys["H"], sys["R"], sys["m0"], sys["P0"])

        # Reference: re-implement predict-update with explicit log-density per step.
        F, Q, H, R = sys["F"], sys["Q"], sys["H"], sys["R"]
        y = sys["y"]
        B, T, N = y.shape
        m_prev = sys["m0"].expand(B, sys["m0"].shape[0]).clone()
        P_prev = sys["P0"].expand(B, *sys["P0"].shape).clone()
        ref = torch.zeros(B, dtype=torch.float64)
        for t in range(T):  # trial-loop: ok  (time loop; not over trials)
            m_pred = m_prev @ F.T
            P_pred = F @ P_prev @ F.T + Q
            S = H @ P_pred @ H.T + R
            innov = y[:, t] - m_pred @ H.T
            L = torch.linalg.cholesky(S)
            z = torch.linalg.solve_triangular(L, innov.unsqueeze(-1), upper=False).squeeze(-1)
            log_det = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(dim=-1)
            ref += -0.5 * (z.pow(2).sum(-1) + log_det + N * math.log(2 * math.pi))
            # update — HP has shape (B, N, D), L has (B, N, N)
            HP = H @ P_pred
            K = torch.cholesky_solve(HP, L).transpose(-1, -2)  # (B, D, N)
            m_prev = m_pred + (K @ innov.unsqueeze(-1)).squeeze(-1)
            I_KH = torch.eye(F.shape[0], dtype=torch.float64) - K @ H
            P_prev = I_KH @ P_pred @ I_KH.transpose(-1, -2) + K @ R @ K.transpose(-1, -2)

        torch.testing.assert_close(log_ml, ref, atol=1e-8, rtol=1e-8)


class TestSmootherNumeric:
    def test_last_step_unchanged(self) -> None:
        sys = _random_lds(B=3, T=12, D=2, N=2, seed=11)
        means, covs, _ = kalman_filter(sys["y"], sys["F"], sys["Q"], sys["H"], sys["R"], sys["m0"], sys["P0"])
        sm_means, sm_covs, _ = rts_smoother(means, covs, sys["F"], sys["Q"])
        torch.testing.assert_close(sm_means[:, -1], means[:, -1])
        torch.testing.assert_close(sm_covs[:, -1], covs[:, -1])

    def test_smoothed_cov_is_psd(self) -> None:
        sys = _random_lds(B=3, T=15, D=3, N=4, seed=23)
        means, covs, _ = kalman_filter(sys["y"], sys["F"], sys["Q"], sys["H"], sys["R"], sys["m0"], sys["P0"])
        _sm_means, sm_covs, _ = rts_smoother(means, covs, sys["F"], sys["Q"])
        # Each smoothed covariance must be symmetric and have non-negative eigenvalues.
        assert torch.allclose(sm_covs, sm_covs.transpose(-1, -2), atol=1e-8)
        eigvals = torch.linalg.eigvalsh(sm_covs)
        assert eigvals.min().item() > -1e-8

    def test_smoothed_cov_has_smaller_trace_than_filtered(self) -> None:
        # Smoothing uses future observations, so it should not increase
        # the marginal variance (averaged over the trajectory).
        sys = _random_lds(B=2, T=20, D=3, N=3, seed=42)
        means, covs, _ = kalman_filter(sys["y"], sys["F"], sys["Q"], sys["H"], sys["R"], sys["m0"], sys["P0"])
        _, sm_covs, _ = rts_smoother(means, covs, sys["F"], sys["Q"])
        tr_f = torch.diagonal(covs, dim1=-2, dim2=-1).sum(dim=-1)
        tr_s = torch.diagonal(sm_covs, dim1=-2, dim2=-1).sum(dim=-1)
        # Allow tiny tolerance for numerical noise; the average trace must
        # be strictly smaller than the filtered version.
        assert (tr_s.mean() < tr_f.mean()).item()

    def test_pairwise_cov_shape(self) -> None:
        sys = _random_lds(B=2, T=10, D=3, N=2, seed=5)
        means, covs, _ = kalman_filter(sys["y"], sys["F"], sys["Q"], sys["H"], sys["R"], sys["m0"], sys["P0"])
        _, _, pair = rts_smoother(means, covs, sys["F"], sys["Q"])
        assert pair.shape == (2, 9, 3, 3)


class TestValidation:
    def test_filter_rejects_wrong_y_shape(self) -> None:
        with pytest.raises(ValueError, match="y must have shape"):
            kalman_filter(
                torch.randn(3, 4),
                torch.eye(2),
                torch.eye(2),
                torch.eye(2),
                torch.eye(2),
                torch.zeros(2),
                torch.eye(2),
            )

    def test_filter_rejects_R_shape(self) -> None:
        with pytest.raises(ValueError, match="R must have shape"):
            kalman_filter(
                torch.randn(2, 5, 3),
                torch.eye(2),
                torch.eye(2),
                torch.randn(3, 2),
                torch.eye(2),
                torch.zeros(2),
                torch.eye(2),
            )
