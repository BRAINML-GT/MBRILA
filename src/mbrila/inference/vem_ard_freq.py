"""Frequency-domain variational EM engine for mDLAG.

Mirrors :class:`mbrila.inference.vem_ard.VEMARDEngine` but performs the
E-step in the (unitary-FFT) frequency domain. The circulant
approximation diagonalises the GP prior into a per-frequency
``diag(S_x(f))`` so the latent posterior splits into ``T`` independent
``(K_a, K_a)`` Cholesky factorisations instead of one ``(M·T, M·T)``
solve. Total inference cost drops from ``O(M³ T³)`` to
``O(T · M·log T + T · K_a³)``.

Latent layout
-------------
mDLAG (as set up in :class:`mbrila.models.MDLAG`) has ``K_a`` across
latents and **zero** within latents. The per-time state ``M = R · K_a``
collapses in frequency: the latent ``x ∈ ℝ^{T·K_a}`` (real, K_a per
bin) maps to ``xfft ∈ ℂ^{T·K_a}`` (complex with conjugate symmetry),
and each region's "view" applies a complex phase shift
``Q_m(f) = exp(-i·2π·f·δ_{m,:})`` to the same K_a-vector. So the
latent dim per frequency is **K_a**, not ``R · K_a``.

E-step (per frequency f)
------------------------
::

    Λ(f) = diag(1/S_x(f))
         + Σ_m diag(Q_m(f))^H · ⟨C_m^T diag(φ_m) C_m⟩ · diag(Q_m(f))
    Σ_X(f) = Λ(f)⁻¹    (complex Hermitian, K_a × K_a)
    μ_X(b, f) = Σ_X(f) · Σ_m diag(Q_m(f))^H · ⟨C_m⟩^T diag(φ_m) · y0fft_m[b, f]

where ``⟨C_m^T diag(φ_m) C_m⟩`` is the same "CPhiC_m" used by
:class:`VEMARDEngine`, computed from the per-row C second moments.

Sufficient stats for the emission M-step (via Parseval)
-------------------------------------------------------
For each region ``m``:

* ``XX_m[k1, k2] = Re(Σ_f Q_m(f, k1) · A_f[k1, k2] · Q_m(f, k2)^*)``
  with ``A_f = Σ_b μ_X μ_X^H + B · Σ_X(f)``  (the per-freq second moment).
* ``XY_m[k, i] = Re(Σ_f Σ_b Q_m(f, k) · μ_X[b, f, k] · yfft[b, f, m, i]^*)``
  (raw ``yfft``, un-centred — :meth:`ARDObservation.update_C` subtracts
  the d-contribution internally).
* ``sum_x_per_region[m, k] = √T · Re(Σ_b μ_X[b, f=0, k])`` (same across
  regions because ``Q_m(f=0) = 1``).

``sum_y / sum_y2`` are computed from the time-domain ``y`` directly.

ELBO (matches ``em_mdlag_freq.m:582-583``)
------------------------------------------
::

    ELBO = elbo_emission(NT)
         + 0.5 · B · K_a · T
         + lb_gp
         + 0.5 · B · Σ_f log|Σ_X(f)|

    lb_gp = -0.5 · B · Σ_f Σ_k log S_x(f, k) - 0.5 · Σ_f Σ_k A_f[k, k] / S_x(f, k)

The leading ``½ · K_a · B · T`` plus the ``½`` factors on the spectral
terms come from the real⇔complex unitary FFT bijection (T·K_a real
DoFs maps to T·K_a complex DoFs with conjugate symmetry — see
``make_S_mdlag.m`` derivation).

GP M-step
---------
LBFGS jointly on ``(log γ_a, delay.β)`` using the freq-domain Q-function

::

    Q(γ, δ) = lb_gp(γ) + Q_δ_lik(γ, δ)
    Q_δ_lik = -0.5 · Σ_f,m tr(diag(Q_m^H) · CPhiC_m · diag(Q_m) · A_f)
              + Σ_f,m Re(Σ_k Q_m(f, k) · yX[f, m, k])

with ``yX[f, m, k] = Σ_b,i φ_i · ⟨C_m[i, k]⟩ · y0fft_m[b, f, i]^* ·
μ_X[b, f, k]``. Autograd flows through :func:`rbf_psd` and
:meth:`FixedDelay.phase_at_freq`; no separate analytical-gradient
wiring is needed — at ``T·K_a`` ≲ a few hundred the cost is dominated
by the per-freq Cholesky in the E-step, not the closure backward.

Trial-batched
-------------
The freq E-step batches trials inside the ``(T, K_a, K_a)`` Cholesky
via PyTorch's batched ``linalg.cholesky``; the trial axis enters as a
right-hand-side dimension in the linear-solve step. Sufficient-stat
aggregation is a sum over ``(0, 1)`` axes (batch, freq) inside
einsums. Loops over regions only.

Float64 / complex128
--------------------
All complex algebra runs in complex128 — at long ``T`` the per-freq
``(K_a, K_a)`` matrices come from products of unitary-FFT'd data that
accumulates round-off cubically in ``T`` without double precision.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from mbrila.core.inference_engine import FitResult, InferenceEngine, Posterior

if TYPE_CHECKING:
    from mbrila.core.base_model import BaseModel
    from mbrila.core.data import MultiRegionData


DEFAULT_JITTER: float = 1e-10


class VEMARDFreqEngine(InferenceEngine):
    """Frequency-domain mDLAG VEM engine."""

    name = "vem_ard_freq"
    required_capabilities = frozenset({"cov_freq"})

    def __init__(
        self,
        *,
        learn_gp: bool = True,
        learn_emission: bool = True,
        max_lbfgs_iter: int = 15,
        lbfgs_history: int = 10,
        jitter: float = DEFAULT_JITTER,
        log_every: int = 0,
        alpha_prune_ratio: float = 10.0,
    ) -> None:
        """
        Parameters
        ----------
        alpha_prune_ratio:
            ARD-aware gating threshold for the GP M-step. A latent
            column ``k`` is considered "pruned" if its max-over-regions
            ``α_mean[r, k]`` exceeds
            ``alpha_prune_ratio · min_k α_mean.max(dim=0).values``.
            Pruned columns' ``(log γ_a, β_{:, k})`` gradients are
            detached during the GP M-step's LBFGS to prevent the
            data-disconnected δ parameter from drifting on numerical
            noise. Set to ``inf`` to disable (back to legacy behaviour).
            Default ``10.0`` (one order of magnitude gap) — heuristic;
            matches the spirit of fast-mDLAG's ``pruneX`` flag but
            without requiring shape mutation.
        """
        if max_lbfgs_iter < 1:
            raise ValueError(f"max_lbfgs_iter must be >= 1; got {max_lbfgs_iter}")
        if lbfgs_history < 1:
            raise ValueError(f"lbfgs_history must be >= 1; got {lbfgs_history}")
        if jitter < 0:
            raise ValueError(f"jitter must be >= 0; got {jitter}")
        if log_every < 0:
            raise ValueError(f"log_every must be >= 0; got {log_every}")
        if alpha_prune_ratio <= 1.0:
            raise ValueError(f"alpha_prune_ratio must be > 1.0 (or inf to disable); got {alpha_prune_ratio}")
        self.learn_gp = learn_gp
        self.learn_emission = learn_emission
        self.max_lbfgs_iter = max_lbfgs_iter
        self.lbfgs_history = lbfgs_history
        self.jitter = jitter
        self.log_every = log_every
        self.alpha_prune_ratio = alpha_prune_ratio

    # ------------------------------------------------------------------
    # Component access
    # ------------------------------------------------------------------

    @staticmethod
    def _components(model: BaseModel) -> tuple[object, object]:
        from mbrila.dynamics.exact_gp import ExactGPLatent
        from mbrila.observations.ard import ARDObservation

        dyn = model.dynamics
        obs = model.observation
        if not isinstance(dyn, ExactGPLatent):
            raise TypeError(f"VEMARDFreqEngine requires ExactGPLatent dynamics; got {type(dyn).__name__}")
        if not isinstance(obs, ARDObservation):
            raise TypeError(f"VEMARDFreqEngine requires ARDObservation emission; got {type(obs).__name__}")
        # mDLAG layout — engine assumes no within latents (M = R · K_a).
        if any(w != 0 for w in dyn.n_within):
            raise ValueError(
                "VEMARDFreqEngine assumes n_within = (0, …, 0) (the MDLAG layout); "
                f"got n_within = {dyn.n_within}"
            )
        return dyn, obs

    # ------------------------------------------------------------------
    # Per-region CPhiC under q(C) (uses C row second moments)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_CPhiC(obs: object) -> Tensor:
        """``CPhiC[m, k1, k2] = Σ_i φ_i · ⟨C_m[i, k1] C_m[i, k2]⟩`` per region.

        Returns a real ``(R, K_a, K_a)`` tensor.
        """
        from mbrila.observations.ard import ARDObservation

        assert isinstance(obs, ARDObservation)
        R = obs.n_regions
        K_a = obs.n_obs_per_region
        dtype = obs.phi_mean.dtype
        device = obs.phi_mean.device
        CPhiC = torch.zeros(R, K_a, K_a, dtype=dtype, device=device)
        cum = 0
        for r, y_r in enumerate(obs.y_dims):
            phi_r = obs.phi_mean[cum : cum + y_r]  # (y_r,)
            moment_r = obs.C_moments[r]  # (y_r, K_a, K_a)
            CPhiC[r] = (phi_r.view(-1, 1, 1) * moment_r).sum(dim=0)
            CPhiC[r] = 0.5 * (CPhiC[r] + CPhiC[r].T)
            cum += y_r
        return CPhiC

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------

    def _e_step(self, model: BaseModel, data: MultiRegionData) -> dict[str, Tensor]:
        """Frequency-domain VEM E-step.

        Returns detached tensors:

        - ``mu_X``           : ``(B, T, K_a)`` complex posterior means.
        - ``Sigma_X``        : ``(T, K_a, K_a)`` complex Hermitian per-freq cov.
        - ``A_f``            : ``(T, K_a, K_a)`` Σ_b μμ^H + B·Σ_X.
        - ``yfft``           : ``(B, T, Y)`` complex unitary FFT of ``y``.
        - ``y0fft``          : ``yfft`` minus √T·d_mean at the DC bin.
        - ``Sx``             : ``(T, K_a)`` real prior PSD.
        - ``Q``              : ``(T, R, K_a)`` complex delay phase.
        - ``CPhiC``          : ``(R, K_a, K_a)`` real.
        - ``CPhi``           : list of ``(K_a, y_r)`` real (per region).
        - ``yX``             : ``(T, R, K_a)`` complex (used by GP M-step).
        - ``logdet_Sigma_X`` : scalar real = Σ_f log|Σ_X(f)|.
        - ``lb_gp``          : scalar real (see module docstring).
        - ``B``              : scalar trial count.
        """
        from mbrila.dynamics.exact_gp import ExactGPLatent
        from mbrila.frequency import centered_freqs, unitary_fft, zero_freq_index
        from mbrila.observations.ard import ARDObservation

        dyn, obs = self._components(model)
        assert isinstance(dyn, ExactGPLatent)
        assert isinstance(obs, ARDObservation)

        B, T, _ = data.y.shape
        K_a = dyn.n_across
        R = obs.n_regions
        dtype = data.y.dtype
        cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        device = data.y.device

        with torch.no_grad():
            # --- Prior + delays ---------------------------------------
            Sx_TK = dyn.cov_freq(T)  # (T, K_a) real
            freqs = centered_freqs(T, dtype=dtype, device=device)  # (T,)
            Q_TRK = dyn.delay.phase_at_freq(freqs)  # (T, R, K_a) complex

            # --- CPhiC + CPhi (mean-field q(C)) -----------------------
            CPhiC_RKK = self._build_CPhiC(obs)  # (R, K_a, K_a) real
            # CPhi_m = ⟨C_m⟩^T · diag(φ_m) — shape (K_a, y_r) per region.
            CPhi_list: list[Tensor] = []
            cum = 0
            for r, y_r in enumerate(obs.y_dims):
                phi_r = obs.phi_mean[cum : cum + y_r]
                CPhi_list.append(obs.C_means[r].transpose(0, 1) * phi_r.unsqueeze(0))
                cum += y_r

            # --- Λ(f) precision per frequency -------------------------
            # Λ(f, k1, k2) = δ_{k1=k2}/Sx_k(f)
            #              + Σ_m Q_m(f, k1)^* · CPhiC_m[k1, k2] · Q_m(f, k2)
            # Vectorised over (T, K_a, K_a) via outer of Q in (k1, k2).
            Q_conj_outer = Q_TRK.conj().unsqueeze(-1) * Q_TRK.unsqueeze(-2)  # (T, R, K_a, K_a)
            # element-wise CPhiC_m: same (K_a, K_a) broadcasted across T.
            Lambda_TKK = (Q_conj_outer * CPhiC_RKK.to(cdtype).unsqueeze(0)).sum(dim=1)
            # Add diag(1/Sx) per freq.
            inv_Sx_TK = (1.0 / Sx_TK).to(cdtype)  # (T, K_a) complex
            eye_KK = torch.eye(K_a, dtype=cdtype, device=device)
            Lambda_TKK = Lambda_TKK + inv_Sx_TK.unsqueeze(-1) * eye_KK.unsqueeze(0)
            # Symmetrise to absorb floating-point asymmetry, add jitter.
            Lambda_TKK = 0.5 * (Lambda_TKK + Lambda_TKK.conj().transpose(-2, -1))
            jitter_eye = self.jitter * eye_KK.unsqueeze(0)
            L_TKK = torch.linalg.cholesky(Lambda_TKK + jitter_eye)
            # Σ_X(f) = Λ(f)^{-1}
            Sigma_X_TKK = torch.cholesky_solve(eye_KK.unsqueeze(0).expand(T, K_a, K_a), L_TKK)
            Sigma_X_TKK = 0.5 * (Sigma_X_TKK + Sigma_X_TKK.conj().transpose(-2, -1))
            # log|Σ_X(f)| = -log|Λ(f)| = -2·log(det(L)) = -2·Σ log diag(L); take real
            logdet_Lambda = 2.0 * torch.diagonal(L_TKK, dim1=-2, dim2=-1).real.log().sum(dim=-1)  # (T,)
            logdet_Sigma_X_total = -logdet_Lambda.sum()  # scalar real

            # --- yfft, y0fft ------------------------------------------
            yfft = unitary_fft(data.y.to(cdtype), dim=1)  # (B, T, Y) complex
            # y0fft = yfft minus √T · d_mean at DC bin (other bins unchanged).
            zero_idx = zero_freq_index(T)
            d_mean_complex = obs.d_mean.to(cdtype)
            y0fft = yfft.clone()
            y0fft[:, zero_idx, :] = y0fft[:, zero_idx, :] - math.sqrt(T) * d_mean_complex

            # --- Per-trial means μ_X(b, f) -----------------------------
            # term1_m[b, f, k] = Q_m(f, k)^* · Σ_i CPhi_m[k, i] · y0fft_m[b, f, i]
            aux_TBK = torch.zeros(T, B, K_a, dtype=cdtype, device=device)
            cum = 0
            for r, y_r in enumerate(obs.y_dims):
                y0fft_m = y0fft[:, :, cum : cum + y_r]  # (B, T, y_m)
                CPhi_m_c = CPhi_list[r].to(cdtype)  # (K_a, y_m)
                # Σ_i CPhi_m[k, i] · y0fft_m[b, t, i] → (B, T, K_a)
                temp_m_BTK = torch.einsum("ki,bti->btk", CPhi_m_c, y0fft_m)
                # Multiply by Q_m(:, k)^* — phase per (T, K_a).
                aux_TBK = aux_TBK + Q_TRK[:, r, :].conj().unsqueeze(1) * temp_m_BTK.transpose(0, 1)
                cum += y_r
            # μ_X = Σ_X · aux: solve Λ · μ = aux for μ.
            # cholesky_solve expects (..., n, k) rhs. Reshape aux to (T, K_a, B).
            mu_solve_TKB = torch.cholesky_solve(aux_TBK.transpose(-2, -1).contiguous(), L_TKK)
            mu_X_BTK = mu_solve_TKB.permute(2, 0, 1).contiguous()  # (B, T, K_a)

            # --- A_f = Σ_b μ μ^H + B · Σ_X ---------------------------
            # μ has shape (B, T, K_a); outer per-(b, t): (T, B, K_a, K_a) summed over b.
            mu_per_freq = mu_X_BTK.transpose(0, 1)  # (T, B, K_a)
            A_f = torch.einsum("tbk,tbl->tkl", mu_per_freq, mu_per_freq.conj()) + B * Sigma_X_TKK
            A_f = 0.5 * (A_f + A_f.conj().transpose(-2, -1))

            # --- yX[f, m, k] = Σ_{b, i} φ_i · ⟨C_m[i, k]⟩ · y0fft_m[b, f, i]^* · μ_X[b, f, k]
            yX_TRK = torch.zeros(T, R, K_a, dtype=cdtype, device=device)
            cum = 0
            for r, y_r in enumerate(obs.y_dims):
                y0fft_m = y0fft[:, :, cum : cum + y_r]  # (B, T, y_m)
                CPhi_m_c = CPhi_list[r].to(cdtype)  # (K_a, y_m)
                # mu_per_freq: (T, B, K_a); y0fft_m_T: (T, B, y_m)
                # BYM[t, i, k] = Σ_b μ_X[b, t, k] · y0fft_m[b, t, i]^*
                y0fft_m_T = y0fft_m.conj().transpose(0, 1)  # (T, B, y_m)
                BYM = torch.einsum("tbk,tbi->tik", mu_per_freq, y0fft_m_T)  # (T, y_m, K_a)
                # Σ_i CPhi_m[k, i] · BYM[t, i, k]  (k matches across factors, i summed)
                yX_TRK[:, r, :] = (CPhi_m_c.transpose(0, 1).unsqueeze(0) * BYM).sum(dim=1)
                cum += y_r

            # --- lb_gp (for ELBO) -------------------------------------
            # lb_gp = -0.5·B·Σ_f log Sx_k - 0.5·Σ_f A_f[k,k] / Sx_k
            log_Sx = torch.log(Sx_TK)  # (T, K_a) real
            A_diag = torch.diagonal(A_f, dim1=-2, dim2=-1).real  # (T, K_a) real
            lb_gp = -0.5 * B * log_Sx.sum() - 0.5 * (A_diag / Sx_TK).sum()

        return {
            "mu_X": mu_X_BTK,
            "Sigma_X": Sigma_X_TKK,
            "A_f": A_f,
            "yfft": yfft,
            "y0fft": y0fft,
            "Sx": Sx_TK,
            "Q": Q_TRK,
            "CPhiC": CPhiC_RKK,
            "CPhi_per_region": _PerRegionList(CPhi_list),  # type: ignore[dict-item]
            "yX": yX_TRK,
            "logdet_Sigma_X": logdet_Sigma_X_total.detach(),
            "lb_gp": lb_gp.detach(),
            "B": torch.tensor(B, dtype=dtype, device=device),
        }

    # ------------------------------------------------------------------
    # Sufficient-statistic aggregation for emission M-step
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_emission_stats(
        data: MultiRegionData,
        posterior: dict[str, Tensor],
        y_dims: tuple[int, ...],
        K_a: int,
    ) -> dict[str, Tensor]:
        """Build the (R, K_a, K_a) XX, list-of-(K_a, y_r) XY, etc.

        Uses Parseval to derive the time-domain second-moment tensors
        from the freq-domain posterior so the existing
        :class:`ARDObservation` update API works unchanged.
        """
        from mbrila.frequency import zero_freq_index

        R = len(y_dims)
        mu_X_BTK: Tensor = posterior["mu_X"]
        A_f: Tensor = posterior["A_f"]
        yfft: Tensor = posterior["yfft"]
        Q_TRK: Tensor = posterior["Q"]
        _, T, _ = mu_X_BTK.shape

        sum_y = data.y.sum(dim=(0, 1))  # (Y,) real
        sum_y2 = (data.y * data.y).sum(dim=(0, 1))  # (Y,) real

        # sum_x_per_region = √T · Re(Σ_b μ_X(f=0, b)) (same across regions).
        zero_idx = zero_freq_index(T)
        mu_dc = mu_X_BTK[:, zero_idx, :]  # (B, K_a) complex
        sum_x_K = math.sqrt(T) * mu_dc.sum(dim=0).real  # (K_a,) real
        sum_x_per_region = sum_x_K.unsqueeze(0).expand(R, K_a).contiguous()  # (R, K_a) real

        # XX_per_region[m, k1, k2] = Re(Σ_f Q_m(f, k1) · A_f[k1, k2] · Q_m(f, k2)^*)
        # Vectorised: outer of Q over (k1, k2) → (T, R, K_a, K_a), · A_f, sum over T.
        Q_outer = Q_TRK.unsqueeze(-1) * Q_TRK.conj().unsqueeze(-2)  # (T, R, K_a, K_a)
        XX_RKK_complex = (Q_outer * A_f.unsqueeze(1)).sum(dim=0)  # (R, K_a, K_a)
        XX_RKK = XX_RKK_complex.real
        XX_RKK = 0.5 * (XX_RKK + XX_RKK.transpose(-2, -1))

        # XY_per_region: list per region of (K_a, y_r) real.
        # XY_m[k, i] = Re(Σ_f Σ_b Q_m(f, k) · μ_X[b, f, k] · yfft[b, f, m, i]^*)
        XY_list: list[Tensor] = []
        cum = 0
        mu_per_freq = mu_X_BTK.transpose(0, 1)  # (T, B, K_a)
        for r, y_r in enumerate(y_dims):
            yfft_m = yfft[:, :, cum : cum + y_r]  # (B, T, y_r)
            # mu_per_freq · yfft_m^* summed over b: einsum (T, B, K_a) × (T, B, y_r) → (T, K_a, y_r) complex.
            MY = torch.einsum("tbk,tbi->tki", mu_per_freq, yfft_m.conj().transpose(0, 1))
            # Multiply by Q_m(f, k) (broadcast over i), sum over T, take real.
            XY_m_complex = (Q_TRK[:, r, :].unsqueeze(-1) * MY).sum(dim=0)  # (K_a, y_r) complex
            XY_list.append(XY_m_complex.real)
            cum += y_r

        return {
            "sum_y": sum_y,
            "sum_y2": sum_y2,
            "sum_x_per_region": sum_x_per_region,
            "XX_per_region": XX_RKK,
            "XY_per_region": _PerRegionList(XY_list),  # type: ignore[dict-item]
        }

    # ------------------------------------------------------------------
    # GP M-step (LBFGS through autograd; freq-domain Q-function)
    # ------------------------------------------------------------------

    def _m_step_gp(
        self,
        model: BaseModel,
        data: MultiRegionData,
        posterior: dict[str, Tensor],
    ) -> None:
        from mbrila.dynamics.exact_gp import ExactGPLatent
        from mbrila.frequency import centered_freqs
        from mbrila.observations.ard import ARDObservation

        dyn, obs = self._components(model)
        assert isinstance(dyn, ExactGPLatent)
        assert isinstance(obs, ARDObservation)
        if dyn.n_across == 0:
            return

        gp_params = [p for p in dyn.parameters() if p.requires_grad]
        if not gp_params:
            return

        A_f: Tensor = posterior["A_f"]
        CPhiC_RKK: Tensor = posterior["CPhiC"]
        yX_TRK: Tensor = posterior["yX"]
        B = int(data.y.shape[0])
        T = int(data.y.shape[1])
        K_a = dyn.n_across

        cdtype = A_f.dtype
        dtype = A_f.real.dtype
        device = A_f.device
        freqs = centered_freqs(T, dtype=dtype, device=device)
        A_diag_real = torch.diagonal(A_f, dim1=-2, dim2=-1).real  # (T, K_a)

        # ARD-aware pruning mask. Columns with max-over-regions α much
        # larger than the minimum (i.e. soft-pruned by ARD) get their
        # ``(γ_k, δ_{:, k})`` gradients detached so LBFGS doesn't push
        # the data-disconnected δ parameter on numerical noise. The mask
        # is recomputed each iter so the engine can recover an active
        # column if ARD changes its mind.
        with torch.no_grad():
            max_alpha = obs.alpha_mean.max(dim=0).values  # (K_a,)
            min_alpha = max_alpha.min().clamp(min=1e-12)
            prune_mask_K = max_alpha > (self.alpha_prune_ratio * min_alpha)  # (K_a,) bool
        # Helper that gates a tensor's gradient on the last axis (K_a):
        # for pruned columns we substitute ``.detach()``, so backward
        # carries no gradient through to (γ_k, β_{:, k}).
        prune_mask_TK = prune_mask_K.view(1, -1).expand(T, K_a)  # (T, K_a)
        prune_mask_TRK = prune_mask_K.view(1, 1, -1).expand(T, dyn.n_regions, K_a)  # (T, R, K_a)

        optimiser = torch.optim.LBFGS(
            gp_params,
            max_iter=self.max_lbfgs_iter,
            history_size=self.lbfgs_history,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
        )

        def closure() -> Tensor:
            optimiser.zero_grad()
            # Across-block PSD with eps blend, via each kernel's
            # ``spectral_density`` method. ``cov_freq`` returns
            # ``(T, K_a + Σ K_w)``; slice the across columns.
            Sx_TK = dyn.cov_freq(T)[:, :K_a]  # (T, K_a) real
            Q_TRK = dyn.delay.phase_at_freq(freqs)  # (T, R, K_a) complex

            # Apply ARD pruning gates: pruned columns see ``.detach()`` so
            # backward doesn't write gradients into the underlying kernel
            # parameters or delay.beta[:, k]. Active columns see the
            # original autograd-tracking tensors.
            if prune_mask_K.any():
                Sx_gated = torch.where(prune_mask_TK, Sx_TK.detach(), Sx_TK)
                Q_gated = torch.where(prune_mask_TRK, Q_TRK.detach(), Q_TRK)
            else:
                Sx_gated = Sx_TK
                Q_gated = Q_TRK

            # lb_gp(γ) — prior contribution.
            log_Sx = torch.log(Sx_gated)
            lb_gp = -0.5 * B * log_Sx.sum() - 0.5 * (A_diag_real / Sx_gated).sum()

            # Q_δ_lik = -0.5·Σ_f,m tr(diag(Q_m^H)·CPhiC_m·diag(Q_m)·A_f)
            #         + Σ_f,m Re(Σ_k Q_m(f, k) · yX[f, m, k])
            #
            # The emission-precision matrix is
            #   M[k1, k2] = conj(Q_m[k1]) · CPhiC_m[k1, k2] · Q_m[k2]
            # and the quadratic is the trace tr(M · A_f). By definition
            #   tr(M · A_f) = Σ_{k1,k2} M[k1, k2] · A_f[k2, k1]
            # so M must be contracted with A_f **transposed** — element
            # [k1, k2] of M pairs with A_f[k2, k1]. Using A_f[k1, k2]
            # directly (A_f is Hermitian, so that flips the sign of the
            # cross-latent off-diagonal phase) silently corrupts the δ
            # gradient whenever latents are coupled — invisible at δ = 0
            # since then every Q ≡ 1.
            Q_conj_outer = Q_gated.conj().unsqueeze(-1) * Q_gated.unsqueeze(-2)  # (T, R, K_a, K_a)
            A_f_T = A_f.transpose(-2, -1)  # so element [k1, k2] holds A_f[k2, k1]
            quad = (Q_conj_outer * CPhiC_RKK.to(cdtype).unsqueeze(0) * A_f_T.unsqueeze(1)).sum().real
            cross = (Q_gated * yX_TRK).sum().real
            Q_delta_lik = cross - 0.5 * quad

            # Maximise Q = lb_gp + Q_δ_lik → minimise -Q.
            loss: Tensor = -(lb_gp + Q_delta_lik)
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        optimiser.step(closure)  # type: ignore[no-untyped-call]

    # ------------------------------------------------------------------
    # Emission M-step
    # ------------------------------------------------------------------

    def _m_step_emission(
        self,
        model: BaseModel,
        data: MultiRegionData,
        posterior: dict[str, Tensor],
    ) -> None:
        from mbrila.observations.ard import ARDObservation

        _, obs = self._components(model)
        assert isinstance(obs, ARDObservation)

        B, T = int(data.y.shape[0]), int(data.y.shape[1])
        NT = B * T
        stats = self._aggregate_emission_stats(
            data=data,
            posterior=posterior,
            y_dims=obs.y_dims,
            K_a=obs.n_obs_per_region,
        )
        XY_list = stats["XY_per_region"]
        assert isinstance(XY_list, _PerRegionList)
        obs.update_d(sum_y=stats["sum_y"], sum_x_per_region=stats["sum_x_per_region"], NT=NT)
        obs.update_C(
            XX=stats["XX_per_region"],
            XY=XY_list.tensors,
            sum_x_per_region=stats["sum_x_per_region"],
        )
        obs.update_alpha()
        obs.update_phi(
            NT=NT,
            sum_y=stats["sum_y"],
            sum_y2=stats["sum_y2"],
            XX=stats["XX_per_region"],
            XY=XY_list.tensors,
            sum_x_per_region=stats["sum_x_per_region"],
        )

    # ------------------------------------------------------------------
    # ELBO
    # ------------------------------------------------------------------

    def _compute_elbo(
        self,
        model: BaseModel,
        data: MultiRegionData,
        posterior: dict[str, Tensor],
    ) -> Tensor:
        from mbrila.dynamics.exact_gp import ExactGPLatent
        from mbrila.observations.ard import ARDObservation

        dyn, obs = self._components(model)
        assert isinstance(dyn, ExactGPLatent)
        assert isinstance(obs, ARDObservation)

        B = int(data.y.shape[0])
        T = int(data.y.shape[1])
        NT = B * T
        K_a = dyn.n_across

        emission = obs.elbo_emission(NT)
        gp_term = 0.5 * B * K_a * T + posterior["lb_gp"] + 0.5 * B * posterior["logdet_Sigma_X"]
        return emission + gp_term

    # ------------------------------------------------------------------
    # Fit loop
    # ------------------------------------------------------------------

    def fit(
        self,
        model: BaseModel,
        data: MultiRegionData,
        *,
        max_iter: int,
        tol: float,
        **kwargs: object,
    ) -> FitResult:
        del kwargs
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1; got {max_iter}")

        from mbrila.observations.ard import ARDObservation

        _, obs = self._components(model)
        assert isinstance(obs, ARDObservation)

        B, T, n_y = data.y.shape
        NT = B * T
        obs.set_phi_shape_from_NT(NT)
        y_var = data.y.reshape(-1, n_y).var(dim=0, unbiased=False)
        obs.set_variance_floor(y_var)
        # Prime α_mean from the current ⟨C C^T⟩ so iter-0's GP M-step
        # sees data-driven α (otherwise α stays at the constructor default
        # and the ARD prune gate cannot fire for iter 0 — letting LBFGS
        # drive δ on data-disconnected columns before ARD's emission
        # update catches up). Skipped when ``learn_emission`` is off so
        # the emission posterior (which includes α) stays frozen.
        if self.learn_emission:
            obs.update_alpha()

        score_trace: list[float] = []
        wall_start = time.perf_counter()
        prev_elbo = -math.inf
        converged = False

        for iteration in range(max_iter):
            posterior = self._e_step(model, data)
            if self.learn_gp:
                self._m_step_gp(model, data, posterior)
            if self.learn_emission:
                self._m_step_emission(model, data, posterior)
            posterior_end = self._e_step(model, data)
            elbo_value = float(self._compute_elbo(model, data, posterior_end).item())
            score_trace.append(elbo_value)

            if self.log_every > 0 and (iteration + 1) % self.log_every == 0:
                print(f"[vem_ard_freq] iter {iteration + 1}/{max_iter}  ELBO = {elbo_value:.3f}")

            if iteration > 0 and abs(elbo_value - prev_elbo) < tol * max(abs(elbo_value), 1.0):
                converged = True
                break
            prev_elbo = elbo_value

        wall = time.perf_counter() - wall_start
        return FitResult(
            score_trace=score_trace,
            converged=converged,
            n_iter=len(score_trace),
            wall_time_s=wall,
            reason="converged" if converged else "completed max_iter",
        )

    # ------------------------------------------------------------------
    # Inference / scoring
    # ------------------------------------------------------------------

    def infer(self, model: BaseModel, data: MultiRegionData) -> Posterior:
        """Return the per-region time-domain posterior reconstructed from
        the freq-domain estimate.

        ``mean`` has shape ``(B, T, R·K_a)`` and matches
        :class:`VEMARDEngine.infer`: each region's "view" of the shared
        latent is the delayed version, obtained by applying
        ``Q_m(f) = exp(-i·2π·f·δ_m)`` per region in frequency before
        IFFT'ing. Without this step, the freq engine would expose only
        the un-delayed K_a-dim underlying signal — which doesn't match
        :meth:`ARDObservation.forward`'s ``(B, T, R·K_a)`` contract.

        Extras
        ------
        ``mu_X_freq``  : ``(B, T, K_a)`` complex — shared latent in freq.
        ``Sigma_X_freq``: ``(T, K_a, K_a)`` complex Hermitian per-freq cov.
        ``x_shared``   : ``(B, T, K_a)`` real — shared (un-delayed) signal
                         after IFFT. Useful when the consumer wants the
                         "before delay" representation.
        """
        from mbrila.dynamics.exact_gp import ExactGPLatent
        from mbrila.frequency import centered_freqs, unitary_ifft

        dyn, _ = self._components(model)
        assert isinstance(dyn, ExactGPLatent)

        info = self._e_step(model, data)
        mu_X: Tensor = info["mu_X"]  # (B, T, K_a) complex
        B, T, K_a = mu_X.shape
        R = dyn.n_regions

        # Apply per-region delay phase, then IFFT each region's view.
        freqs = centered_freqs(T, dtype=mu_X.real.dtype, device=mu_X.device)
        Q_TRK = dyn.delay.phase_at_freq(freqs)  # (T, R, K_a) complex
        # mu_X_per_region[b, t, r, k] = Q_r(t, k) · mu_X[b, t, k]
        mu_X_per_region_freq = Q_TRK.unsqueeze(0) * mu_X.unsqueeze(2)  # (B, T, R, K_a)
        x_per_region_time = unitary_ifft(mu_X_per_region_freq, dim=1).real  # (B, T, R, K_a)
        x_hat = x_per_region_time.reshape(B, T, R * K_a)  # match time-engine layout

        # Shared (un-delayed) signal for callers who want it.
        x_shared = unitary_ifft(mu_X, dim=1).real  # (B, T, K_a)

        # Reuse freq-domain posterior cov as a per-time approximation.
        Sigma_X: Tensor = info["Sigma_X"]
        cov_real = Sigma_X.real
        cov_for_posterior = cov_real.unsqueeze(0).expand(B, -1, -1, -1)
        return Posterior(
            mean=x_hat,
            cov=cov_for_posterior,
            cov_form="per_time_block",
            extras={
                "Sigma_X_freq": Sigma_X,
                "mu_X_freq": mu_X,
                "x_shared": x_shared,
            },
        )

    def score(self, model: BaseModel, data: MultiRegionData) -> float:
        posterior = self._e_step(model, data)
        return float(self._compute_elbo(model, data, posterior).item())


# A tiny helper that lets us store a list-of-tensors as a "value" in a
# ``dict[str, Tensor]`` without confusing mypy. The list itself isn't a
# Tensor but the engine's posterior dict is otherwise tensor-typed.
class _PerRegionList:
    __slots__ = ("tensors",)

    def __init__(self, tensors: list[Tensor]) -> None:
        self.tensors = tensors
