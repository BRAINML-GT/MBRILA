"""Shared helpers for the ``examples/synthetic/demo_*.py`` model-zoo scripts.

Each ``demo_<method>.py`` script samples a synthetic multi-region dataset,
fits one model, and dumps a fixed set of plots plus a JSON summary. This
module factors out the parts that are the same across methods:

- :func:`sample_scenario`: wrap :func:`generate_multiregion_synthetic`
  and pull out the ground-truth quantities the plots need.
- :func:`extract_delay` / :func:`extract_observable` /
  :func:`extract_y_recon` / :func:`extract_ard_alpha`: read fitted
  quantities back out of a model in a layout the plot helpers
  understand. Dispatch on the concrete model class.
- :func:`pair_rmse`, :func:`best_permutation_corr`,
  :func:`latent_sign_to_truth`: numerical recovery metrics.
- ``plot_*`` helpers: convergence, delay-pairs over time, per-region
  latent traces, y-reconstruction, ARD alpha bar. Saved as PNGs
  alongside a ``summary.json`` so each script's output directory is
  self-contained.
"""

from __future__ import annotations

import json
import math
from itertools import combinations, permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from mbrila import ADM, DLAG, GPFA, LDS, MDLAG
from mbrila.core.data import MultiRegionData
from mbrila.dynamics.markov_gp import BlockDiagonalDynamics, MarkovianGPLatent
from mbrila.init.pcca import pcca_init_C
from mbrila.observations.ard import ARDObservation
from mbrila.observations.multi_region import MultiRegionLinearObservation
from mbrila.synthetic.multiregion import (
    MultiRegionScenario,
    SyntheticDataset,
    generate_multiregion_synthetic,
)

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------


def sample_scenario(
    scenario: MultiRegionScenario,
    *,
    device: str = "cpu",
) -> tuple[MultiRegionData, dict]:
    """Sample synthetic data, return ``(data, truth)``.

    ``truth`` has the layout the plotting helpers expect:

    - ``delay``        : ``(T, R-1, n_across)`` — region 0 reference dropped
    - ``observable``   : ``(B, T, n_obs_total)`` — ground-truth latent in
      observable layout (each region's ``g_r(t)`` stacked)
    - ``y``            : ``(B, T, sum y_dims)`` — observed
    - ``n_regions``, ``n_across``, ``n_within``, ``y_dims``, ``T``,
      ``sigma_across``, ``sigma_within`` — scenario metadata
    """
    sd: SyntheticDataset = generate_multiregion_synthetic(scenario)
    dtype = scenario.dtype
    n_regions = len(scenario.y_dims)
    data = MultiRegionData(
        y=sd.data.y.to(device=device, dtype=dtype),
        y_dims=scenario.y_dims,
        bin_width=1.0,
    )
    truth = {
        "delay": sd.true_delay[:, 1:, :].detach().cpu().numpy(),
        "observable": sd.true_latents.detach().cpu().numpy(),
        "y": sd.data.y.detach().cpu().numpy(),
        "n_regions": n_regions,
        "n_across": scenario.n_across,
        "n_within": scenario.n_within,
        "y_dims": scenario.y_dims,
        "T": scenario.T,
        "sigma_across": scenario.sigma_across,
        "sigma_within": scenario.sigma_within,
    }
    return data, truth


# ---------------------------------------------------------------------------
# Emission initialisation — GPFA / LDS gap
# ---------------------------------------------------------------------------


def init_linear_observation_pcca(
    model: object,
    data: MultiRegionData,
    *,
    n_across: int,
    n_within: int,
) -> None:
    """pCCA-seed a :class:`MultiRegionLinearObservation` in-place.

    :class:`ADM`, :class:`DLAG`, :class:`MDLAG` all expose
    ``initialize_from_data`` on their model classes. :class:`GPFA` and
    :class:`LDS` don't — they would normally start from the random C
    that the observation constructor builds. For the demo scripts we
    want every method to converge from a comparable starting point, so
    this helper does the equivalent pCCA seed and copies it onto the
    model's :class:`MultiRegionLinearObservation`.
    """
    assert isinstance(model, GPFA | LDS)
    obs = model.observation
    assert isinstance(obs, MultiRegionLinearObservation)
    y_dims = tuple(int(d) for d in obs.y_dims)
    Cs, diag_R, mu = pcca_init_C(
        data.y.detach().cpu(),
        y_dims=y_dims,
        n_across=n_across,
        n_within=n_within,
    )
    with torch.no_grad():
        for r, C_r in enumerate(Cs):
            obs.Cs[r].data.copy_(C_r.to(dtype=obs.Cs[r].dtype, device=obs.Cs[r].device))
        obs.diag_R_param.data.copy_(diag_R.to(dtype=obs.diag_R_param.dtype, device=obs.diag_R_param.device))
        obs.d_param.data.copy_(mu.to(dtype=obs.d_param.dtype, device=obs.d_param.device))


# Re-exported from the library so demo scripts and the deflation helper
# below share the same recipe with :class:`mbrila.KalmanEMEngine.fit`.
from mbrila.inference import build_grouped_adamw as build_deflation_optimizer  # noqa: E402


def _detect_deflation_kind(model: ADM | GPFA | DLAG) -> str:
    """Return ``"adm"`` / ``"gpfa"`` / ``"dlag_ssm"`` or raise."""
    if isinstance(model, ADM):
        return "adm"
    if isinstance(model, GPFA):
        return "gpfa"
    if isinstance(model, DLAG):
        if model._engine_kind != "kalman":
            raise ValueError(
                f"rank1_deflation_init: DLAG only supported with engine='kalman'; got {model._engine_kind!r}"
            )
        return "dlag_ssm"
    raise TypeError(
        f"rank1_deflation_init: unsupported model type {type(model).__name__}; "
        "supported: ADM, GPFA, DLAG(engine='kalman')"
    )


def _build_rank1_model(
    model_kind: str,
    full_model: ADM | GPFA | DLAG,
    *,
    K_w: int,
    engine: object,
    device: torch.device,
    dtype: torch.dtype,
) -> ADM | GPFA | DLAG:
    """Construct a rank-1 sibling of ``full_model`` (K_a=1, full's within)."""
    from mbrila import KalmanEMEngine, LatentSpec

    n_regions = len(full_model._y_dims)
    rank1_spec = LatentSpec(n_across=1, n_within=(K_w,) * n_regions)
    assert isinstance(engine, KalmanEMEngine)
    if model_kind == "adm":
        assert isinstance(full_model, ADM)
        return ADM(
            latent_spec=rank1_spec,
            y_dims=full_model._y_dims,
            T=full_model._T,
            kernel_factory_across=full_model._kernel_factory_across,
            kernel_factory_within=full_model._kernel_factory_within,
            lag_across=full_model._lag_across,
            lag_within=full_model._lag_within,
            delay_smoothing_sigma_across=full_model._delay_smoothing_sigma_across,
            eps=full_model._eps,
            engine=engine,
            device=device,
            dtype=dtype,
        ).to(device)
    if model_kind == "dlag_ssm":
        assert isinstance(full_model, DLAG)
        return DLAG(
            latent_spec=rank1_spec,
            y_dims=full_model._y_dims,
            T=full_model._T,
            kernel_factory_across=full_model._kernel_factory_across,
            kernel_factory_within=full_model._kernel_factory_within,
            eps_across=full_model._eps_across,
            eps_within=full_model._eps_within,
            max_delay=full_model._max_delay,
            engine="kalman",
            engine_override=engine,
            lag_across=full_model._lag_across,
            lag_within=full_model._lag_within,
            cov_jitter=full_model._cov_jitter,
            device=device,
            dtype=dtype,
        ).to(device)
    assert model_kind == "gpfa"
    assert isinstance(full_model, GPFA)
    return GPFA(
        latent_spec=rank1_spec,
        y_dims=full_model._y_dims,
        T=full_model._T,
        lag_across=full_model._lag_across,
        kernel_factory_across=full_model._kernel_factory_across,
        eps=full_model._eps,
        init_R=full_model._init_R,
        engine=engine,
        device=device,
        dtype=dtype,
    ).to(device)


def _delay_param(block: MarkovianGPLatent) -> torch.nn.Parameter | None:
    """Return the learnable delay tensor of ``block``, or ``None`` for NoDelay."""
    delay = block.delay
    if delay is None:
        return None
    # TimeVaryingDelay → raw_delay; FixedDelay → beta; NoDelay → neither.
    for attr in ("raw_delay", "beta"):
        if hasattr(delay, attr):
            target = getattr(delay, attr)
            if isinstance(target, torch.nn.Parameter):
                return target
    return None


def _perturb_delay(rank1_model: ADM | GPFA | DLAG, *, perturb_std: float, seed: int) -> None:
    """Add zero-mean Gaussian noise to the across-block's delay parameter.

    Used by restart > 0 to break delay-symmetry between restart rounds.
    No-op for :class:`NoDelay` (GPFA) — perturbation is meaningless there.
    """
    if perturb_std <= 0:
        return
    block = rank1_model.dynamics.blocks[0]
    assert isinstance(block, MarkovianGPLatent)
    target = _delay_param(block)
    if target is None:
        return
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = (
        torch.randn(target.shape, generator=gen, dtype=target.dtype).to(device=target.device) * perturb_std
    )
    with torch.no_grad():
        target.data.add_(noise)


def rank1_deflation_init(  # noqa: PLR0912 — dispatch over (model kind × restart × stage) is naturally branchy
    model: ADM | GPFA | DLAG,
    data: MultiRegionData,
    *,
    n_iters_per_round: int = 200,
    n_restarts_per_round: int = 1,
    restart_perturb_std: float = 0.5,
    lr: float = 1e-2,
    lr_min: float = 1e-3,
    weight_decay: float = 0.0,
    verbose: bool = False,
) -> dict:
    """Sequential rank-1 deflation init for ADM / DLAG-SSM / GPFA.

    Per round ``k = 0 … K_a - 1``:

    1. Fit ``n_restarts_per_round`` rank-1 siblings on the current
       residual (restart 0 unperturbed; restart 1+ adds Gaussian noise
       to the delay parameter — :attr:`TimeVaryingDelay.raw_delay` for
       ADM, :attr:`FixedDelay.beta` for DLAG-SSM, no-op for GPFA's
       :class:`NoDelay`).
    2. Keep the restart with the lowest training loss.
    3. Extract the across-block kernel state + delay state + per-region
       C column.
    4. Subtract the rank-1 across-only reconstruction from the residual.

    All rounds' extracts are aggregated into ``model`` in-place.

    Kernel-agnostic: kernel parameters are copied by name via
    ``named_parameters()`` so any :class:`BaseKernel` subclass works
    (MOSE / Matérn / user-defined). The optimiser is
    :func:`build_deflation_optimizer` (grouped AdamW); with
    ``weight_decay=0`` it reduces to plain Adam.

    For DLAG-SSM / ADM with within blocks (``K_w > 0``): the residual
    subtraction masks the across observable slots only, so within
    contributions stay in the residual.

    Parameters
    ----------
    model:
        Full :class:`ADM`, :class:`GPFA`, or :class:`DLAG(engine="kalman")`
        instance. Constructed but not yet fit; ``model.observation`` is
        overwritten.
    data:
        Observed data to fit against.
    n_iters_per_round:
        Adam(W) steps per restart.
    n_restarts_per_round, restart_perturb_std:
        Number of restarts per round (restart 0 unperturbed) and the
        Gaussian noise std added to the delay parameter on restarts
        ``>= 1``. ADM's canonical recipe uses ``n_restarts=1`` by default
        (single unperturbed run) — set higher only if the delay landscape
        is suspected to be multi-modal.
    lr, lr_min, weight_decay:
        Optimiser hyperparameters for the rank-1 inner fits. ``ADM``
        callers typically use ``weight_decay=0.01``; ``GPFA``/``DLAG-SSM``
        callers use the default ``0.0``.
    verbose:
        Print per-round diagnostics (residual energy, kernel params,
        delay magnitude where applicable).

    Returns
    -------
    dict with keys ``"n_rounds"`` (= K_a) and ``"rank1_losses"`` (the
    best-of-restarts loss for each round).
    """
    from mbrila import KalmanEMEngine

    obs_full = model.observation
    assert isinstance(obs_full, MultiRegionLinearObservation)

    K_a = model.latent_spec.n_across
    n_regions = len(model._y_dims)
    if K_a == 0 or n_regions < 2:
        return {"n_rounds": 0, "rank1_losses": []}

    model_kind = _detect_deflation_kind(model)
    K_w = 0 if model_kind == "gpfa" else int(model.latent_spec.n_within[0])
    n_obs_per_region = 1 + K_w  # 1 across + K_w within per region in the rank-1 model

    device = data.y.device
    dtype = data.y.dtype
    residual_y = data.y.clone()
    extracted: list[dict] = []
    rank1_losses: list[float] = []

    for k in range(K_a):
        if verbose:
            print(f"  [{model_kind}-deflation round {k + 1}/{K_a}]")

        residual_data = MultiRegionData(y=residual_y, y_dims=data.y_dims, bin_width=data.bin_width)

        best_model: ADM | GPFA | DLAG | None = None
        best_loss = float("inf")
        per_restart_losses: list[float] = []
        for r_idx in range(n_restarts_per_round):
            engine_k = KalmanEMEngine(
                lr=lr,
                lr_min=lr_min,
                weight_decay=0.0,
                update_obs_every=0,
                log_every=0,
            )
            rank1_model = _build_rank1_model(
                model_kind, model, K_w=K_w, engine=engine_k, device=device, dtype=dtype
            )
            # pCCA-init emission on residual. Both ADM and DLAG/GPFA expose
            # the same call surface here (model-level for ADM/DLAG; helper
            # for GPFA which has no `initialize_from_data`).
            if model_kind == "gpfa":
                init_linear_observation_pcca(rank1_model, residual_data, n_across=1, n_within=0)
            else:
                assert isinstance(rank1_model, ADM | DLAG)
                rank1_model.initialize_from_data(residual_data, mode="pcca")

            _perturb_delay(
                rank1_model,
                perturb_std=(0.0 if r_idx == 0 else restart_perturb_std),
                seed=model._T * (k + 1) + r_idx,
            )

            opt = build_deflation_optimizer(rank1_model, lr=lr, weight_decay=weight_decay)
            last_loss = float("inf")
            for _ in range(n_iters_per_round):
                opt.zero_grad()
                ll = engine_k._loss_value(rank1_model, residual_data)
                (-ll).backward()
                opt.step()
                last_loss = float((-ll).detach().item())
            per_restart_losses.append(last_loss)
            if last_loss < best_loss:
                best_loss = last_loss
                best_model = rank1_model
        assert best_model is not None
        rank1_losses.append(best_loss)

        # Extract across-block kernel state + delay state + C columns.
        across_block = best_model.dynamics.blocks[0]
        assert isinstance(across_block, MarkovianGPLatent)
        kernel_state = {name: p.detach().clone() for name, p in across_block.kernel.named_parameters()}
        delay_target = _delay_param(across_block)
        delay_state: torch.Tensor | None = delay_target.detach().clone() if delay_target is not None else None
        rank1_obs = best_model.observation
        assert isinstance(rank1_obs, MultiRegionLinearObservation)
        C_per_region_k: list[torch.Tensor] = [
            rank1_obs.Cs[r][:, 0:1].detach().clone() for r in range(n_regions)
        ]
        extracted.append(
            {
                "kernel_state": kernel_state,
                "delay_state": delay_state,
                "C_per_region": C_per_region_k,
                "d_offset": rank1_obs.d_param.detach().clone(),
                "diag_R": rank1_obs.diag_R_param.detach().clone(),
            }
        )

        # Subtract only the across reconstruction from the residual; leave
        # within contribution in play for the next across round.
        with torch.no_grad():
            best_engine = best_model.inference
            assert isinstance(best_engine, KalmanEMEngine)
            info = best_engine._smoother_posterior(best_model, residual_data)
            s_means = info["means"]
            H_select = best_model.dynamics.H_select
            obs_observable = torch.einsum("ij,btj->bti", H_select, s_means)
            n_obs_total = int(H_select.shape[0])
            mask = torch.zeros(n_obs_total, dtype=dtype, device=device)
            for r in range(n_regions):
                # Across slot is at the start of each region's observable block.
                mask[r * n_obs_per_region] = 1.0
            obs_across_only = obs_observable * mask
            block_C = rank1_obs.block_diag_C()
            recon_across = torch.einsum("ij,btj->bti", block_C, obs_across_only)
            residual_y = residual_y - recon_across

        if verbose:
            res_energy = float(residual_y.pow(2).mean().item())
            kernel_str = ", ".join(
                f"{name}={(p.exp() if 'log_' in name else p).item():.4f}" for name, p in kernel_state.items()
            )
            extra = ""
            if delay_state is not None:
                d_abs = delay_state.abs()
                extra = f"  |delta| mean={float(d_abs.mean().item()):.3f} max={float(d_abs.max().item()):.3f}"
            restart_str = (
                f"  per-restart losses = {[f'{x:.0f}' for x in per_restart_losses]}"
                if n_restarts_per_round > 1
                else ""
            )
            print(
                f"    rank-1 final loss = {best_loss:.3f}{restart_str}  "
                f"residual_energy = {res_energy:.4f}{extra}  kernel({kernel_str})"
            )

    # Aggregate into the full model.
    with torch.no_grad():
        for k, ext in enumerate(extracted):
            blk = model.dynamics.blocks[k]
            assert isinstance(blk, MarkovianGPLatent)
            dst_named = dict(blk.kernel.named_parameters())
            for name, p_src in ext["kernel_state"].items():
                if name not in dst_named:
                    raise RuntimeError(
                        f"deflation: kernel param {name!r} present in rank-1 model "
                        f"but missing from full model's block {k}"
                    )
                dst_named[name].data.copy_(p_src)
            if ext["delay_state"] is not None:
                dst_delay = _delay_param(blk)
                assert dst_delay is not None, "delay state extracted but full model has NoDelay?"
                dst_delay.data.copy_(ext["delay_state"])
            for r, C_r in enumerate(ext["C_per_region"]):
                obs_full.Cs[r][:, k : k + 1].copy_(C_r)
        obs_full.d_param.data.copy_(extracted[0]["d_offset"])
        obs_full.diag_R_param.data.copy_(extracted[-1]["diag_R"])

    return {"n_rounds": K_a, "rank1_losses": rank1_losses}


# ---------------------------------------------------------------------------
# Extract fitted quantities from the model
# ---------------------------------------------------------------------------


def extract_delay(model: object, T: int) -> np.ndarray:
    """Return fitted delay as ``(T, R-1, n_across)`` numpy, region 0 dropped.

    Dispatch by dynamics layout, not by model class:

    - :class:`BlockDiagonalDynamics` (ADM, DLAG-SSM, mDLAG-SSM, GPFA,
      LDS): per-across-block ``Delay``. ADM's blocks have
      :class:`TimeVaryingDelay`; the others have :class:`FixedDelay`
      (or :class:`NoDelay` for the no-delay models). Stack per-block
      ``(T, R, 1)`` (or ``(R, 1)`` broadcast along T) into
      ``(T, R, n_across)``.
    - Otherwise (:class:`DLAG` / :class:`MDLAG` dense-GP path):
      ``dynamics.delay`` is a single :class:`FixedDelay` of shape
      ``(R, n_across)``; broadcast along time.
    """
    if isinstance(model, GPFA | LDS):
        n_regions = model.latent_spec.n_regions
        n_across = model.latent_spec.n_across
        return np.zeros((T, max(0, n_regions - 1), n_across))

    assert isinstance(model, ADM | DLAG | MDLAG)
    n_across = model.latent_spec.n_across
    n_regions = model.latent_spec.n_regions
    if n_across == 0:
        return np.zeros((T, max(0, n_regions - 1), 0))

    with torch.no_grad():
        dyn = model.dynamics
        if isinstance(dyn, BlockDiagonalDynamics):
            # Per-across-block delays — covers ADM (TimeVaryingDelay),
            # DLAG-SSM and mDLAG-SSM (FixedDelay) uniformly.
            per_latent: list[np.ndarray] = []
            for k in range(n_across):
                block = dyn.blocks[k]
                assert isinstance(block, MarkovianGPLatent)
                block_delay = block.delay
                assert block_delay is not None
                d = block_delay.as_tensor(T).detach().cpu().numpy()
                if d.ndim == 3:
                    # (T, R, 1) — TimeVaryingDelay
                    per_latent.append(d[:, :, 0])
                else:
                    # (R, 1) — FixedDelay; broadcast along time.
                    per_latent.append(np.broadcast_to(d[None, :, 0], (T, d.shape[0])).copy())
            stacked = np.stack(per_latent, axis=-1)  # (T, R, K)
            return stacked[:, 1:, :].copy()

        # Dense-GP path: single FixedDelay (R, K) on dynamics.
        delta_static = dyn.delay.as_tensor().detach().cpu().numpy()  # (R, K)
    drop_ref = delta_static[1:, :]
    return np.broadcast_to(drop_ref[None, :, :], (T, drop_ref.shape[0], drop_ref.shape[1])).copy()


def _observable_from_posterior(model: object, data: MultiRegionData) -> torch.Tensor:
    """Return ``g(t)`` (observable layout, ``(B, T, n_total_obs)``).

    Engines disagree on what ``Posterior.mean`` is:

    - :class:`ExactEMEngine` / :class:`VEMARDEngine` /
      :class:`VEMARDFreqEngine` already return ``x_hat`` in observable
      layout (their ``n_total_obs`` axis).
    - :class:`VEMKalmanARDEngine` (mDLAG-SSM) projects to observable
      layout inside its own ``_e_step``, so its ``mean`` is observable.
    - :class:`KalmanEMEngine` (ADM, DLAG-SSM, GPFA, LDS) returns the
      **state-space** smoother mean instead — last dim is the lifted
      state dim ``D = lag·R·K_a + R·K_w·lag_w``, not ``n_total_obs``.
      We project through ``model.dynamics.H_select`` to recover the
      observable layout.

    We branch on dynamics layout: :class:`BlockDiagonalDynamics`
    exposes ``H_select`` and (under :class:`KalmanEMEngine`) needs the
    projection; everything else is already observable.
    """
    assert isinstance(model, ADM | DLAG | MDLAG | GPFA | LDS)
    with torch.no_grad():
        post = model.infer(data)
        mean = post.mean
        dyn = model.dynamics
        # Duck-typing on ``H_select`` rather than ``isinstance(dyn,
        # BlockDiagonalDynamics)`` so the same projection works for
        # :class:`FreeLDSLatent` too — it carries an ``H_select`` buffer
        # but isn't a :class:`BlockDiagonalDynamics`. Matches what
        # :func:`KalmanEMEngine._is_kalman_compatible_dynamics` does
        # internally.
        H_select = getattr(dyn, "H_select", None)
        if isinstance(H_select, torch.Tensor):
            n_total_obs = int(H_select.shape[0])
            if mean.shape[-1] == n_total_obs:
                # mDLAG-SSM (VEMKalmanARDEngine): infer already projected.
                return mean
            return torch.einsum("ij,btj->bti", H_select, mean)
        # Dense-GP path (DLAG-exact, mDLAG-time, mDLAG-freq): infer
        # returns observable directly.
        return mean


def extract_observable(model: object, data: MultiRegionData) -> np.ndarray:
    """Return per-region observable latent ``g(t)`` as ``(B, T, M)`` numpy."""
    return _observable_from_posterior(model, data).detach().cpu().numpy()


def extract_y_recon(model: object, data: MultiRegionData) -> np.ndarray:
    """Return reconstructed observations ``(B, T, sum y_dims)`` numpy."""
    with torch.no_grad():
        observable = _observable_from_posterior(model, data)
        assert isinstance(model, ADM | DLAG | MDLAG | GPFA | LDS)
        y_recon = model.observation.forward(observable)
    return y_recon.detach().cpu().numpy()


def extract_ard_alpha(model: object) -> np.ndarray | None:
    """Return ``alpha_mean`` of shape ``(R, K_a)`` for ARD models, else ``None``.

    Only :class:`MDLAG` (any engine) uses :class:`ARDObservation`. ARD α
    values > 1 indicate latent columns the variational posterior has
    pulled toward zero (effectively pruned).
    """
    if not isinstance(model, MDLAG):
        return None
    obs = model.observation
    if not isinstance(obs, ARDObservation):
        return None
    with torch.no_grad():
        return obs.alpha_mean.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def ard_aware_delay_rmse(
    truth_delay: np.ndarray,
    fitted_delay: np.ndarray,
    alpha_mean: np.ndarray,
    n_regions: int,
    *,
    alpha_prune_ratio: float = 10.0,
) -> dict[str, object]:
    """Pairwise-delay RMSE restricted to ARD-active columns only.

    The naive :func:`pair_rmse` averages over every K_init column of
    the fitted model, including columns ARD has effectively pruned.
    Pruned columns have weak (or no) data signal constraining their
    ``δ``, so their delay can drift far from any truth — and the
    naive RMSE picks that up as if it were the model's actual
    recovery error.

    This helper restricts to the **active columns only** (those
    whose ``max-over-regions α`` is within ``alpha_prune_ratio×`` of
    the minimum across columns — matches :class:`VEMARDEngine`'s
    own pruning gate). Two regimes:

    1. ``n_active >= K_true``: model found at least as many latents
       as truth. Best-permutation pick ``K_true`` active columns and
       match them to all ``K_true`` truth columns.
    2. ``n_active < K_true``: model under-found. Match the
       ``n_active`` available columns to **the best subset of truth
       columns** (best subset + best permutation). The remaining
       truth columns are not in the RMSE — ``n_active`` and
       ``matched_truth`` are returned separately so the caller can
       see the model only found ``n_active`` of ``K_true`` latents.

    Parameters
    ----------
    truth_delay:
        Ground-truth pairwise delay, shape ``(T, R-1, K_true)``
        (region 0 reference dropped).
    fitted_delay:
        Model pairwise delay, shape ``(T, R-1, K_init)``.
    alpha_mean:
        ARD posterior mean, shape ``(R, K_init)``. From
        :func:`extract_ard_alpha`.
    n_regions:
        Number of regions ``R`` (must be ``>= 2``).
    alpha_prune_ratio:
        Threshold for "active" — a column ``k`` is active iff
        ``max-over-r α[r, k] <= alpha_prune_ratio × min_k max-over-r α``.

    Returns
    -------
    dict with:
        - ``rmse``: pairwise RMSE in bins on the best matched
          ``K_match = min(n_active, K_true)`` columns. ``nan`` if no
          active columns (defensive — the gate normally guarantees
          ``n_active >= 1``).
        - ``active_cols``: indices of fitted columns ARD considers active.
        - ``matched_active``: ``active_cols`` reordered to match
          ``matched_truth`` element-wise.
        - ``matched_truth``: the truth column indices that ended up in
          the RMSE. If ``n_active < K_true`` this is a **subset** of
          ``range(K_true)``; the missing indices are "not recovered".
        - ``n_active``: ``len(active_cols)``.
        - ``K_true``: truth's K.
    """
    K_true = int(truth_delay.shape[-1])
    max_per_col = alpha_mean.max(axis=0)
    min_alpha = float(max_per_col.min())
    threshold = alpha_prune_ratio * min_alpha
    active_cols = np.flatnonzero(max_per_col <= threshold)
    n_active = int(active_cols.size)
    K_match = min(n_active, K_true)

    out: dict[str, object] = {
        "rmse": float("nan"),
        "active_cols": active_cols,
        "matched_active": np.array([], dtype=int),
        "matched_truth": np.array([], dtype=int),
        "n_active": n_active,
        "K_true": K_true,
    }
    if K_match == 0:
        return out

    # Build pairwise (j > i) delay differences per latent column.
    T = int(truth_delay.shape[0])
    truth_full = np.concatenate([np.zeros((T, 1, K_true)), truth_delay], axis=1)
    fitted_active = fitted_delay[:, :, active_cols]
    fitted_full = np.concatenate([np.zeros((T, 1, n_active)), fitted_active], axis=1)

    truth_pairs_list: list[np.ndarray] = []
    fitted_pairs_list: list[np.ndarray] = []
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            truth_pairs_list.append(truth_full[:, j] - truth_full[:, i])
            fitted_pairs_list.append(fitted_full[:, j] - fitted_full[:, i])
    truth_pairs = np.stack(truth_pairs_list, axis=0)  # (n_pairs, T, K_true)
    fitted_pairs = np.stack(fitted_pairs_list, axis=0)  # (n_pairs, T, n_active)

    best_rmse = float("inf")
    best_match_active: tuple[int, ...] = ()
    best_match_truth: tuple[int, ...] = ()
    # Choose which K_match truth cols get matched; permute the K_match
    # active cols against that subset.
    for truth_subset in combinations(range(K_true), K_match):
        truth_sel = truth_pairs[:, :, list(truth_subset)]
        for active_perm in permutations(range(n_active), K_match):
            fitted_sel = fitted_pairs[:, :, list(active_perm)]
            rmse = float(np.sqrt(((fitted_sel - truth_sel) ** 2).mean()))
            if rmse < best_rmse:
                best_rmse = rmse
                best_match_active = active_perm
                best_match_truth = truth_subset

    out["rmse"] = best_rmse
    out["matched_active"] = active_cols[list(best_match_active)]
    out["matched_truth"] = np.array(list(best_match_truth))
    return out


def pair_rmse(delay_a: np.ndarray, delay_b: np.ndarray, n_regions: int) -> float:
    """Pairwise inter-region delay RMSE in bins, averaged over (T, pairs, n_across).

    Both inputs have shape ``(T, R-1, n_across)`` — region 0 reference
    dropped. We expand ``δ`` to ``(T, R, n_across)`` with a zero column
    at index 0 and then compare every (i, j) pair.
    """
    if delay_a.shape != delay_b.shape:
        raise ValueError(f"delay shapes differ: {delay_a.shape} vs {delay_b.shape}")
    T = delay_a.shape[0]
    n_across = delay_a.shape[-1]
    if n_across == 0 or T == 0:
        return float("nan")
    a_full = np.concatenate([np.zeros((T, 1, n_across)), delay_a], axis=1)
    b_full = np.concatenate([np.zeros((T, 1, n_across)), delay_b], axis=1)
    sq_err: list[float] = []
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            diff = (a_full[:, j] - a_full[:, i]) - (b_full[:, j] - b_full[:, i])  # (T, K)
            sq_err.append(float((diff**2).mean()))
    return float(np.sqrt(np.mean(sq_err)))


def best_permutation_corr(
    truth_C: np.ndarray,
    learned_C: np.ndarray,
    active_cols: np.ndarray,
) -> tuple[float, tuple[int, ...]]:
    """Best mean ``|corr|`` between truth and learned C columns under
    sign-flip × permutation. ``active_cols`` selects which learned
    columns to match against truth (length must be ``>= K_truth``).
    """
    K_truth = truth_C.shape[1]
    truth_z = truth_C - truth_C.mean(axis=0, keepdims=True)
    learned_z = learned_C - learned_C.mean(axis=0, keepdims=True)
    truth_norm = truth_z / np.maximum(truth_z.std(axis=0, keepdims=True), 1e-8)
    learned_norm = learned_z / np.maximum(learned_z.std(axis=0, keepdims=True), 1e-8)
    corr = (learned_norm[:, active_cols].T @ truth_norm) / truth_C.shape[0]
    corr_abs = np.abs(corr)
    best_score = -1.0
    best_perm: tuple[int, ...] = tuple(range(K_truth))
    for perm in permutations(range(int(active_cols.shape[0])), K_truth):
        score = float(corr_abs[list(perm), range(K_truth)].mean())
        if score > best_score:
            best_score = score
            best_perm = perm
    return best_score, best_perm


def align_across_permutation(
    fitted_obs: np.ndarray,
    truth_obs: np.ndarray,
    *,
    n_regions: int,
    n_across: int,
    n_within: int,
) -> tuple[int, ...]:
    """Find the best across-latent permutation aligning fitted to truth.

    Multi-region SSM-GP models identify across latents only up to a
    permutation (and a per-latent sign flip — handled separately by
    :func:`latent_sign_to_truth` in the trace plots). Without permutation
    alignment, "fitted latent 1" in the plot may correspond to "truth
    latent 2", making the figure look broken when the fit is actually
    fine.

    For each ``(truth_k, fitted_k)`` pair this computes the mean absolute
    correlation across regions / trials / time of the corresponding
    observable traces, then picks the permutation maximising the
    diagonal sum. K_a! permutations are enumerated; cheap for K_a ≤ 6.

    Returns ``perm`` such that ``fitted slot perm[k_truth]`` best matches
    ``truth slot k_truth``. Identity (``tuple(range(K_a))``) when
    ``n_across <= 1`` or shapes don't match.

    Parameters
    ----------
    fitted_obs, truth_obs:
        Shape ``(B, T, R * (n_across + n_within))``. Region-major layout
        with across latents first inside each region's block (the
        :func:`build_observable_to_state` convention).
    n_regions, n_across, n_within:
        Layout constants.
    """
    if n_across <= 1:
        return tuple(range(n_across))
    if fitted_obs.shape != truth_obs.shape:
        return tuple(range(n_across))
    npr = n_across + n_within
    # Extract per-region across slices: shape (B, T, R, K_a).
    fit_across = np.stack(
        [fitted_obs[..., r * npr : r * npr + n_across] for r in range(n_regions)],
        axis=2,
    )
    truth_across = np.stack(
        [truth_obs[..., r * npr : r * npr + n_across] for r in range(n_regions)],
        axis=2,
    )

    # Score matrix: mean |corr| over (B, T, R) between truth_k and fit_k.
    K = n_across
    score = np.zeros((K, K))
    for tk in range(K):
        t_flat = truth_across[..., tk].ravel()
        t_z = t_flat - t_flat.mean()
        t_n = max(float(np.linalg.norm(t_z)), 1e-12)
        for fk in range(K):
            f_flat = fit_across[..., fk].ravel()
            f_z = f_flat - f_flat.mean()
            f_n = max(float(np.linalg.norm(f_z)), 1e-12)
            score[tk, fk] = abs(float((t_z @ f_z) / (t_n * f_n)))

    # Pick perm maximising the diagonal.
    best_perm = tuple(range(K))
    best_score = -1.0
    for perm in permutations(range(K)):
        s = float(score[range(K), list(perm)].mean())
        if s > best_score:
            best_score = s
            best_perm = perm
    return best_perm


def apply_across_permutation(
    fitted_obs: np.ndarray,
    fitted_delay: np.ndarray,
    perm: tuple[int, ...],
    *,
    n_regions: int,
    n_across: int,
    n_within: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorder fitted across-latent columns to align with truth.

    Returns ``(obs_aligned, delay_aligned)``.

    ``obs_aligned[..., r * npr + k_truth] = fitted_obs[..., r * npr + perm[k_truth]]``
    for the across slots; within slots are passed through unchanged.

    ``delay_aligned[..., k_truth] = fitted_delay[..., perm[k_truth]]``.

    No sign flips applied here — sign ambiguity on the latent trace is
    handled per-column at plot time by :func:`latent_sign_to_truth`;
    delay itself is invariant to latent sign.
    """
    if perm == tuple(range(n_across)):
        return fitted_obs, fitted_delay
    npr = n_across + n_within
    obs_aligned = fitted_obs.copy()
    for r in range(n_regions):
        base = r * npr
        for k_truth in range(n_across):
            obs_aligned[..., base + k_truth] = fitted_obs[..., base + perm[k_truth]]
    delay_aligned = fitted_delay[..., list(perm)]
    return obs_aligned, delay_aligned


def latent_sign_to_truth(est: np.ndarray, truth: np.ndarray) -> float:
    """Return +1 / -1 — the sign that best aligns ``est`` to ``truth``.

    Latent models identify ``g`` only up to a per-latent sign flip, so
    the smoother estimate may come out flipped. The flip is applied
    only to the *plotted* trace.
    """
    return -1.0 if float(np.sum(est * truth)) < 0.0 else 1.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_convergence(
    trace: np.ndarray,
    out_path: Path,
    *,
    ylabel: str,
    title: str,
) -> None:
    """One clean monotone trace of the per-iter score."""
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = np.arange(1, len(trace) + 1)
    ax.plot(epochs, trace, "o-", color="darkgreen", lw=1.2, markersize=3)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _pairwise_curve(delay: np.ndarray, n_regions: int, pair_idx: int, latent: int = 0) -> np.ndarray:
    """One pair's δ_j(t) − δ_i(t) curve for the given latent column."""
    full = np.concatenate([np.zeros((delay.shape[0], 1, delay.shape[-1])), delay], axis=1)
    c = 0
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            if c == pair_idx:
                return full[:, j, latent] - full[:, i, latent]
            c += 1
    raise ValueError(f"pair_idx {pair_idx} out of range")


def plot_delay_pairs(
    truth_delay: np.ndarray,
    fitted_delay: np.ndarray,
    n_regions: int,
    n_across: int,
    out_path_template: str,
    *,
    model_label: str,
    rmse: float,
) -> None:
    """One figure per across-latent: fitted vs true δ(t) for every region pair.

    ``out_path_template`` should contain ``{lat}`` — one PNG is saved per
    across latent column. Skipped silently if there is no delay structure.
    """
    if n_across == 0 or n_regions < 2:
        return
    pair_ij = [(i, j) for i in range(n_regions) for j in range(i + 1, n_regions)]
    n_pairs = len(pair_ij)
    for lat in range(n_across):
        curves = [
            (
                _pairwise_curve(truth_delay, n_regions, p, lat),
                _pairwise_curve(fitted_delay, n_regions, p, lat),
            )
            for p in range(n_pairs)
        ]
        all_vals = np.concatenate([np.concatenate([t, f]) for t, f in curves])
        lo = min(float(all_vals.min()), 0.0)
        hi = max(float(all_vals.max()), 0.0)
        pad = max(0.1 * (hi - lo), 0.5)
        ylim = (lo - pad, hi + pad)

        ncols = max(1, n_pairs // 2 + 1)
        fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 5), squeeze=False, sharex=True, sharey=True)
        for p, (i, j) in enumerate(pair_ij):
            ax = axes[p // ncols, p % ncols]
            truth, final = curves[p]
            ax.plot(truth, "--", color="purple", lw=1.0, label="true delay")
            ax.plot(final, color="darkred", lw=1.1, label=f"{model_label} delay")
            ax.set_ylim(*ylim)
            ax.set_title(f"region {j} vs region {i}", fontsize=8)
            ax.set_xlabel("time (bins)", fontsize=7)
            ax.set_ylabel("delay (bins)", fontsize=7)
            ax.legend(loc=1, fontsize=6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        for idx in range(n_pairs, 2 * ncols):
            axes[idx // ncols, idx % ncols].axis("off")
        fig.suptitle(
            f"{model_label} — inter-region delay vs time, across-latent {lat + 1}  (RMSE = {rmse:.3f} bins)",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(out_path_template.format(lat=lat + 1), dpi=200)
        plt.close(fig)


def plot_latent_traces(
    truth_obs: np.ndarray,
    fitted_obs: np.ndarray,
    *,
    n_regions: int,
    n_across: int,
    n_within: int,
    out_dir: Path,
    prefix: str,
    model_label: str,
    trial: int = 0,
) -> None:
    """Per-region observable-latent traces — one figure per across-latent
    column (and a stacked within figure if ``n_within > 0``).
    """
    n_obs_per_region = n_across + n_within
    T = truth_obs.shape[1]

    for k in range(n_across):
        ncols = max(1, (n_regions + 1) // 2)
        fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 5), squeeze=False)
        for r_idx in range(n_regions):
            ax = axes[r_idx // ncols, r_idx % ncols]
            col = r_idx * n_obs_per_region + k
            sign = latent_sign_to_truth(fitted_obs[:, :, col], truth_obs[:, :, col])
            t_axis = np.arange(T)
            ax.plot(t_axis, truth_obs[trial, :, col], "--", color="purple", lw=1.0, label="truth")
            ax.plot(
                t_axis,
                sign * fitted_obs[trial, :, col],
                color="darkred",
                lw=1.0,
                label=model_label,
            )
            ax.set_title(f"region {r_idx}", fontsize=8)
            ax.legend(loc=1, fontsize=6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        for idx in range(n_regions, 2 * ncols):
            axes[idx // ncols, idx % ncols].axis("off")
        fig.suptitle(
            f"{model_label} — across-latent g_{k + 1}(t) per region (trial {trial}; fit sign-aligned)",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}latent_across_{k + 1}.png", dpi=200)
        plt.close(fig)

    if n_within == 0:
        return
    fig, axes = plt.subplots(
        n_regions, n_within, figsize=(4 * n_within, 2 * n_regions), squeeze=False, sharex=True
    )
    for r_idx in range(n_regions):
        for w_idx in range(n_within):
            ax = axes[r_idx, w_idx]
            col = r_idx * n_obs_per_region + n_across + w_idx
            sign = latent_sign_to_truth(fitted_obs[:, :, col], truth_obs[:, :, col])
            t_axis = np.arange(T)
            ax.plot(t_axis, truth_obs[trial, :, col], "--", color="purple", lw=1.0)
            ax.plot(t_axis, sign * fitted_obs[trial, :, col], color="darkred", lw=1.0)
            ax.set_title(f"region {r_idx} within {w_idx + 1}", fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle(f"{model_label} — within-latents (trial {trial}; fit sign-aligned)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}latent_within.png", dpi=200)
    plt.close(fig)


def y_recon_rmse(truth_y: np.ndarray, recon_y: np.ndarray, y_dims: tuple[int, ...]) -> dict:
    """Reconstruction RMSE on the observation tensor.

    Returns
    -------
    dict with:
        - ``overall``: scalar RMSE over the entire ``(B, T, sum y_dims)`` tensor.
        - ``per_region``: list of length ``len(y_dims)`` — RMSE per region.
    """
    overall = float(np.sqrt(((truth_y - recon_y) ** 2).mean()))
    per_region: list[float] = []
    cum = 0
    for ydim in y_dims:
        diff = truth_y[..., cum : cum + ydim] - recon_y[..., cum : cum + ydim]
        per_region.append(float(np.sqrt((diff**2).mean())))
        cum += ydim
    return {"overall": overall, "per_region": per_region}


def plot_y_reconstruction(
    truth_y: np.ndarray,
    recon_y: np.ndarray,
    y_dims: tuple[int, ...],
    out_path: Path,
    *,
    model_label: str,
    trial: int = 0,
    n_per_region: int = 4,
) -> None:
    """Per-region y reconstruction (a few neurons per region).

    The figure title shows the overall reconstruction RMSE
    ``sqrt(mean((y_obs - y_recon)^2))`` over the entire ``(B, T, sum y_dims)``
    tensor; each row's first panel additionally reports that region's
    own per-neuron-pooled RMSE so users can spot region-specific
    under-fitting.
    """
    rmse = y_recon_rmse(truth_y, recon_y, y_dims)
    n_regions = len(y_dims)
    T = truth_y.shape[1]
    ncols = max(1, n_per_region)
    fig, axes = plt.subplots(n_regions, ncols, figsize=(3 * ncols, 2 * n_regions), squeeze=False, sharex=True)
    cum = 0
    for r_idx, ydim in enumerate(y_dims):
        n_show = min(n_per_region, ydim)
        for n_i in range(n_show):
            ax = axes[r_idx, n_i]
            col = cum + n_i
            t_axis = np.arange(T)
            ax.plot(t_axis, truth_y[trial, :, col], color="black", lw=0.6, alpha=0.6, label="y_obs")
            ax.plot(t_axis, recon_y[trial, :, col], color="darkred", lw=0.9, label=model_label)
            if r_idx == 0 and n_i == 0:
                ax.legend(loc=1, fontsize=6)
            if n_i == 0:
                ax.set_title(
                    f"region {r_idx} neuron {n_i} | region RMSE = {rmse['per_region'][r_idx]:.4f}",
                    fontsize=7,
                )
            else:
                ax.set_title(f"region {r_idx} neuron {n_i}", fontsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        cum += ydim
        for n_i in range(n_show, ncols):
            axes[r_idx, n_i].axis("off")
    fig.suptitle(
        f"{model_label} — y reconstruction (trial {trial};  overall RMSE = {rmse['overall']:.4f})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_ard_alpha(
    alpha_mean: np.ndarray,
    out_path: Path,
    *,
    model_label: str,
) -> None:
    """ARD α per latent column — high bars are pruned latents.

    ``alpha_mean`` has shape ``(R, K_a)``. We plot ``max α per column``
    (the most-pruned region's α), which is the criterion the mDLAG
    recovery tests use for an active-vs-pruned decision.
    """
    max_per_col = alpha_mean.max(axis=0)
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(max_per_col)), 4))
    x = np.arange(len(max_per_col))
    ax.bar(x, max_per_col, 0.55, color="steelblue")
    for xi, v in zip(x, max_per_col, strict=True):
        ax.text(xi, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"col {k}" for k in x])
    ax.set_ylabel("max α across regions")
    ax.set_title(f"{model_label} — ARD α per across-latent column (large = pruned)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary save
# ---------------------------------------------------------------------------


def save_summary(result: dict, out_dir: Path, method_name: str) -> Path:
    """Dump scalar metrics to ``summary.json`` and convergence trace to
    ``trace.npy``. Non-JSON-serialisable fields (arrays, tensors) are
    skipped — they live in the plot PNGs.
    """
    json_safe: dict = {}
    for k, v in result.items():
        if isinstance(v, int | float | str | bool | tuple | list):
            json_safe[k] = v
        elif isinstance(v, np.ndarray) and v.ndim == 0:
            json_safe[k] = float(v.item())
    json_safe["method_name"] = method_name
    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(json_safe, indent=2, default=str))

    if "score_trace" in result and result["score_trace"] is not None:
        np.save(out_dir / "trace.npy", np.asarray(result["score_trace"], dtype=float))
    return json_path


def print_summary(result: dict, method_name: str) -> None:
    """One-line summary of the headline metrics for SLURM logs."""
    delay_rmse = result.get("delay_rmse", float("nan"))
    y_rmse = result.get("y_rmse", float("nan"))
    wall = result.get("wall_s", float("nan"))
    line = f"[{method_name}] delay_rmse={delay_rmse:.3f} bins  y_rmse={y_rmse:.4f}  wall={wall:.1f}s"
    perm = result.get("across_perm")
    if isinstance(perm, list) and perm != list(range(len(perm))):
        # Only mention the perm when it deviates from identity — keeps logs
        # quiet for the common case and loud when latents got swapped.
        line += f"  across_perm={tuple(perm)}"
    extra = result.get("extra_summary")
    if isinstance(extra, str) and extra:
        line += f"  {extra}"
    print(line)


# ---------------------------------------------------------------------------
# Convenience: build a one-shot record and write all outputs
# ---------------------------------------------------------------------------


def write_method_outputs(
    *,
    method_name: str,
    model_label: str,
    truth: dict,
    fitted_delay: np.ndarray,
    fitted_obs: np.ndarray,
    fitted_y: np.ndarray,
    score_trace: np.ndarray,
    score_ylabel: str,
    wall_s: float,
    out_dir: Path,
    alpha_mean: np.ndarray | None = None,
    extra_summary: str | None = None,
) -> dict:
    """End-of-fit: compute the standard metrics, save the standard plots,
    write ``summary.json``. Returns the metrics dict so the caller can
    print custom summaries.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    T = truth["T"]
    n_regions = truth["n_regions"]
    n_across = truth["n_across"]
    n_within = truth["n_within"]
    y_rmse_info = y_recon_rmse(truth["y"], fitted_y, truth["y_dims"])

    # Delay and latent-trace plots overlay truth on fit, which only
    # makes sense when truth and fit share the same latent layout.
    # Methods whose fitted latent doesn't match the truth's
    # ``(n_across + n_within)`` structure (e.g. LDS, which has a flat
    # K-dim latent) skip these overlays automatically.
    n_obs_per_region = n_across + n_within
    expected_obs_dim = n_regions * n_obs_per_region
    delay_layouts_match = fitted_delay.shape == truth["delay"].shape
    latent_layouts_match = fitted_obs.shape[-1] == expected_obs_dim

    # Align fitted across-latent slot ordering to truth via permutation
    # search. SSM-GP models identify across latents only up to a
    # permutation; without this, the delay / latent-trace plots can
    # appear "swapped" even when the fit is fine, and the naive
    # ``pair_rmse`` reports a misleadingly large delay error.
    across_perm: tuple[int, ...] = tuple(range(n_across))
    if latent_layouts_match and n_across > 1:
        across_perm = align_across_permutation(
            fitted_obs,
            truth["observable"],
            n_regions=n_regions,
            n_across=n_across,
            n_within=n_within,
        )
        fitted_obs, fitted_delay = apply_across_permutation(
            fitted_obs,
            fitted_delay,
            across_perm,
            n_regions=n_regions,
            n_across=n_across,
            n_within=n_within,
        )

    delay_rmse = pair_rmse(fitted_delay, truth["delay"], n_regions) if delay_layouts_match else float("nan")

    plot_convergence(
        score_trace,
        out_dir / "convergence.png",
        ylabel=score_ylabel,
        title=f"{model_label} — {score_ylabel} convergence",
    )
    if delay_layouts_match:
        plot_delay_pairs(
            truth["delay"],
            fitted_delay,
            n_regions,
            n_across,
            str(out_dir / "delay_lat{lat}.png"),
            model_label=model_label,
            rmse=delay_rmse,
        )
    if latent_layouts_match:
        plot_latent_traces(
            truth["observable"],
            fitted_obs,
            n_regions=n_regions,
            n_across=n_across,
            n_within=n_within,
            out_dir=out_dir,
            prefix="",
            model_label=model_label,
        )
    plot_y_reconstruction(
        truth["y"],
        fitted_y,
        truth["y_dims"],
        out_dir / "y_recon.png",
        model_label=model_label,
    )
    if alpha_mean is not None:
        plot_ard_alpha(alpha_mean, out_dir / "ard_alpha.png", model_label=model_label)

    record = {
        "method_name": method_name,
        "model_label": model_label,
        "delay_rmse": delay_rmse,
        "across_perm": list(across_perm),
        "y_rmse": y_rmse_info["overall"],
        "y_rmse_per_region": y_rmse_info["per_region"],
        "wall_s": wall_s,
        "n_regions": n_regions,
        "n_across": n_across,
        "n_within": n_within,
        "T": T,
        "score_trace": score_trace.tolist() if isinstance(score_trace, np.ndarray) else list(score_trace),
        "max_alpha_per_col": (alpha_mean.max(axis=0).tolist() if alpha_mean is not None else None),
        "extra_summary": extra_summary,
    }
    save_summary(record, out_dir, method_name)
    return record


# Re-export math.inf so scripts can use `tol=demo_common.INF` without import noise.
INF = math.inf
