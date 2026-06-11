"""Tests for CF5 — :class:`FreeLDSLatent` + :class:`LDS` naive SSM preset.

Focus areas:

1. ``FreeLDSLatent`` produces valid ``(A, Q)`` with PSD ``Q`` from its
   Cholesky parameterisation, and the H_select replicates the shared
   state to per-region observable slots.
2. The :class:`KalmanEMEngine` accepts ``FreeLDSLatent`` (Protocol
   relaxation in CF5) — same engine ADM / GPFA / DLAG-SSM use.
3. ``LDS`` model preset: construction, sample, score, save/load, and
   ``build_model("lds", ...)`` dispatch.
4. Single-region degenerate case (``n_regions=1``) behaves correctly.

No fitting or recovery tests — those are the user's responsibility per
CLAUDE.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mbrila import LDS, FreeLDSLatent, build_model
from mbrila.delays.none import NoDelay
from mbrila.inference.kalman_em import KalmanEMEngine
from mbrila.observations.multi_region import MultiRegionLinearObservation

# ---------------------------------------------------------------------------
# FreeLDSLatent: parameterisation, shapes, PSD invariants
# ---------------------------------------------------------------------------


class TestFreeLDSLatentBasics:
    def test_forward_shapes(self) -> None:
        dyn = FreeLDSLatent(n_latent=4, n_regions=2, T=10)
        A, Q = dyn.forward()
        assert A.shape == (10, 4, 4)
        assert Q.shape == (10, 4, 4)

    def test_H_select_replicates_state(self) -> None:
        """``H_select[r * K + k, k] = 1`` so each region sees the full latent."""
        K, R = 3, 4
        dyn = FreeLDSLatent(n_latent=K, n_regions=R, T=5)
        H = dyn.H_select
        assert H.shape == (R * K, K)
        # Replication: slot (r, k) picks state[k] regardless of r.
        for r in range(R):
            for k in range(K):
                assert H[r * K + k, k].item() == 1.0
        # Off-target entries are zero.
        assert int((H != 0).sum().item()) == R * K

    def test_Q_is_psd_at_init(self) -> None:
        dyn = FreeLDSLatent(n_latent=5, n_regions=2, T=4, init_Q_diag=0.07)
        Q = dyn.Q()
        eigvals = torch.linalg.eigvalsh(Q)
        assert eigvals.min().item() > 0.0
        # Diagonal Q at init: should match init_Q_diag exactly on the diagonal.
        torch.testing.assert_close(
            torch.diagonal(Q),
            torch.full((5,), 0.07, dtype=Q.dtype),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_Q_is_psd_after_perturbation(self) -> None:
        """Random perturbations to L parameters must still give PSD Q."""
        dyn = FreeLDSLatent(n_latent=4, n_regions=1, T=3)
        torch.manual_seed(42)
        with torch.no_grad():
            dyn.L_log_diag.copy_(torch.randn_like(dyn.L_log_diag))
            dyn.L_off_diag.copy_(torch.randn_like(dyn.L_off_diag))
        Q = dyn.Q()
        eigvals = torch.linalg.eigvalsh(Q)
        assert eigvals.min().item() > -1e-12

    def test_A_default_is_contractive_identity(self) -> None:
        """Default ``A`` should be ``0.95 · I`` (stable, near-identity)."""
        dyn = FreeLDSLatent(n_latent=3, n_regions=1, T=5)
        torch.testing.assert_close(
            dyn.A_param,
            0.95 * torch.eye(3, dtype=dyn.A_param.dtype),
            atol=1e-12,
            rtol=1e-12,
        )

    def test_init_A_custom(self) -> None:
        custom_A = torch.tensor([[0.5, 0.1], [-0.2, 0.7]], dtype=torch.float64)
        dyn = FreeLDSLatent(n_latent=2, n_regions=1, T=3, init_A=custom_A)
        torch.testing.assert_close(dyn.A_param, custom_A)

    def test_init_A_wrong_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="init_A must have shape"):
            FreeLDSLatent(
                n_latent=3,
                n_regions=1,
                T=5,
                init_A=torch.zeros(4, 4, dtype=torch.float64),
            )

    def test_rejects_non_positive_Q_diag(self) -> None:
        with pytest.raises(ValueError, match="init_Q_diag must be positive"):
            FreeLDSLatent(n_latent=2, n_regions=1, T=3, init_Q_diag=0.0)

    def test_rejects_bad_dims(self) -> None:
        for kw in [
            {"n_latent": 0, "n_regions": 2, "T": 5},
            {"n_latent": 2, "n_regions": 0, "T": 5},
            {"n_latent": 2, "n_regions": 1, "T": 0},
        ]:
            with pytest.raises(ValueError):
                FreeLDSLatent(**kw)  # type: ignore[arg-type]

    def test_forward_stationary_broadcast(self) -> None:
        """Every time slice of (A, Q) must be identical — LDS is stationary in v1."""
        dyn = FreeLDSLatent(n_latent=3, n_regions=2, T=5)
        A, Q = dyn.forward()
        for t in range(1, 5):
            torch.testing.assert_close(A[t], A[0])
            torch.testing.assert_close(Q[t], Q[0])

    def test_parameters_registered(self) -> None:
        """A_param, L_log_diag, L_off_diag must all be learnable."""
        dyn = FreeLDSLatent(n_latent=4, n_regions=2, T=5)
        param_names = {name for name, _ in dyn.named_parameters()}
        assert {"A_param", "L_log_diag", "L_off_diag"} <= param_names


# ---------------------------------------------------------------------------
# LDS preset model
# ---------------------------------------------------------------------------


def _build_lds(n_regions: int = 2, n_latent: int = 3, T: int = 8) -> LDS:
    y_dims = tuple(4 + i for i in range(n_regions))  # (4,), (4, 5), (4, 5, 6), ...
    return LDS(n_latent=n_latent, y_dims=y_dims, T=T)


class TestLDSConstruction:
    def test_default_components(self) -> None:
        model = _build_lds()
        assert isinstance(model.dynamics, FreeLDSLatent)
        assert isinstance(model.observation, MultiRegionLinearObservation)
        assert isinstance(model.inference, KalmanEMEngine)
        # Placeholders: LDS has no kernel and a degenerate (zero) delay slot.
        assert isinstance(model.delay, NoDelay)
        # Sanity: latent_spec encodes the flat shared latent via n_across.
        assert model.latent_spec.n_across == 3
        assert model.latent_spec.n_within == (0, 0)

    def test_single_region(self) -> None:
        # Pin to CPU so this test is portable across GPU / CPU runners — the
        # H_select buffer lives on the model's device and ``assert_close``
        # is strict about it.
        model = LDS(n_latent=3, y_dims=(8,), T=5, device="cpu")
        assert isinstance(model.dynamics, FreeLDSLatent)
        assert model.latent_spec.n_regions == 1
        # H_select for single region is the identity.
        assert isinstance(model.dynamics, FreeLDSLatent)
        eye = torch.eye(3, dtype=torch.float64)
        torch.testing.assert_close(model.dynamics.H_select, eye)

    def test_rejects_bad_n_latent(self) -> None:
        with pytest.raises(ValueError, match="n_latent must be >= 1"):
            LDS(n_latent=0, y_dims=(4,), T=5)

    def test_rejects_empty_y_dims(self) -> None:
        with pytest.raises(ValueError, match="y_dims must be a non-empty"):
            LDS(n_latent=2, y_dims=(), T=5)


class TestLDSSample:
    def test_sample_shapes(self) -> None:
        model = _build_lds(n_regions=3, n_latent=2, T=6)
        data = model.sample(n_trials=4, T=6, seed=0)
        # y has shape (B, T, sum y_dims) = (4, 6, 4+5+6).
        assert data.y.shape == (4, 6, 15)
        assert torch.isfinite(data.y).all().item()

    def test_sample_T_mismatch_raises(self) -> None:
        model = _build_lds()
        with pytest.raises(ValueError, match="sample T must match"):
            model.sample(n_trials=2, T=99)


class TestLDSScore:
    def test_score_finite(self) -> None:
        model = _build_lds(n_regions=2, n_latent=2, T=6)
        data = model.sample(n_trials=2, T=6, seed=0)
        # KalmanEMEngine.score on LDS validates the Protocol-relaxed dispatch.
        ll = model.score(data)
        assert torch.isfinite(torch.tensor(ll)).item()


class TestLDSSaveLoad:
    def test_round_trip(self, tmp_path: Path) -> None:
        model = _build_lds(n_regions=2, n_latent=3, T=5)
        # Perturb parameters so the round-trip is meaningful.
        with torch.no_grad():
            assert isinstance(model.dynamics, FreeLDSLatent)
            model.dynamics.A_param.add_(torch.randn_like(model.dynamics.A_param) * 0.01)
            model.dynamics.L_log_diag.add_(torch.randn_like(model.dynamics.L_log_diag) * 0.1)
        path = tmp_path / "lds.pt"
        model.save(path)
        loaded = LDS.load(path)
        assert isinstance(loaded.dynamics, FreeLDSLatent)
        # Parameters survived the round-trip.
        assert isinstance(model.dynamics, FreeLDSLatent)
        torch.testing.assert_close(loaded.dynamics.A_param, model.dynamics.A_param)
        torch.testing.assert_close(loaded.dynamics.L_log_diag, model.dynamics.L_log_diag)


# ---------------------------------------------------------------------------
# build_model dispatch
# ---------------------------------------------------------------------------


class TestBuildModelLDS:
    def test_build_model_lds(self) -> None:
        model = build_model("lds", n_latent=3, y_dims=(4, 5), T=8)
        assert isinstance(model, LDS)
        assert isinstance(model.dynamics, FreeLDSLatent)
        assert isinstance(model.inference, KalmanEMEngine)

    def test_kwargs_propagate(self) -> None:
        model = build_model("lds", n_latent=2, y_dims=(3,), T=5, init_Q_diag=0.25)
        assert isinstance(model, LDS)
        assert model._init_Q_diag == 0.25


# ---------------------------------------------------------------------------
# Engine compatibility: FreeLDSLatent satisfies the relaxed Protocol
# ---------------------------------------------------------------------------


class TestEngineProtocolRelaxation:
    """KalmanEMEngine's CF5 relaxation accepts any dynamics with the
    structural ``H_select + forward()`` interface — not just
    ``BlockDiagonalDynamics``. This test exercises that with
    ``FreeLDSLatent``.
    """

    def test_engine_accepts_free_lds(self) -> None:
        model = _build_lds()
        # If _assemble's isinstance check still required BlockDiagonalDynamics,
        # this would raise TypeError. score() exercises _assemble end-to-end.
        data = model.sample(n_trials=2, T=8, seed=0)
        model.score(data)
