"""Shared helpers for the ``examples/v1v2/demo_*.py`` scripts.

Real-data V1/V2 visual-cortex counterpart of
:mod:`examples.synthetic.demo_common`. Real recordings have no
ground-truth delay or latent, so the demo plots are self-consistency
diagnostics (no truth overlay) and the headline metric is held-out-
neuron co-smoothing RMSE on the held-out test split.

This module re-exports the synthetic helpers (``extract_delay`` /
``extract_observable`` / ``extract_y_recon`` etc.) since the layout
conventions are identical, and adds:

- :func:`load_v1v2`              load the V1/V2 pickle into a tensor.
- :func:`split_indices`          deterministic train/val/test shuffle.
- :func:`co_smoothing_rmse`      held-out-neuron prediction RMSE.
- ``plot_*``                     fitted-only convergence / delay /
                                 latent / PSTH / trial-0 figures.
- :func:`write_v1v2_outputs`     end-of-fit orchestrator (mirrors
                                 :func:`demo_common.write_method_outputs`).
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Re-use the synthetic-side extraction helpers. They dispatch on model
# class and don't care whether the data is synthetic or real.
_SYNTHETIC_DIR = Path(__file__).resolve().parent.parent / "synthetic"
if str(_SYNTHETIC_DIR) not in sys.path:
    sys.path.insert(0, str(_SYNTHETIC_DIR))

import demo_common as _demo  # noqa: E402

from mbrila import ADM, DLAG, MDLAG  # noqa: E402
from mbrila.core.data import MultiRegionData  # noqa: E402
from mbrila.observations.ard import ARDObservation  # noqa: E402
from mbrila.observations.multi_region import MultiRegionLinearObservation  # noqa: E402

# Re-exports — same surface as ``demo_common`` for consistency.
extract_delay = _demo.extract_delay
extract_observable = _demo.extract_observable
extract_y_recon = _demo.extract_y_recon
extract_ard_alpha = _demo.extract_ard_alpha
rank1_deflation_init = _demo.rank1_deflation_init
init_linear_observation_pcca = _demo.init_linear_observation_pcca
y_recon_rmse = _demo.y_recon_rmse


# ---------------------------------------------------------------------
# Data loading + train/val/test split
# ---------------------------------------------------------------------
def load_v1v2(
    path: Path, *, device: torch.device | str, dtype: torch.dtype
) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load the V1/V2 pickle. Returns ``(y, (n_v1, n_v2))``.

    Pickle layout: dict with arrays ``"V1": (n_trials, T, n_v1)`` and
    ``"V2": (n_trials, T, n_v2)``. The two are concatenated on the
    neuron axis to a single ``(B, T, n_v1 + n_v2)`` tensor.
    """
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    V1 = np.asarray(data["V1"], dtype=np.float32)
    V2 = np.asarray(data["V2"], dtype=np.float32)
    if V1.shape[:2] != V2.shape[:2]:
        raise ValueError(f"V1 shape {V1.shape} and V2 shape {V2.shape} disagree on (n_trials, T)")
    y = np.concatenate([V1, V2], axis=2)
    y_t = torch.tensor(y, dtype=dtype, device=device)
    return y_t, (int(V1.shape[-1]), int(V2.shape[-1]))


def split_indices(
    n_trials: int, n_train: int, n_val: int, n_test: int, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic shuffle then split into (train, val, test) index tensors."""
    if n_train + n_val + n_test > n_trials:
        raise ValueError(f"split sizes ({n_train}+{n_val}+{n_test}) exceed n_trials={n_trials}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_trials)
    return (
        torch.from_numpy(perm[:n_train]).long(),
        torch.from_numpy(perm[n_train : n_train + n_val]).long(),
        torch.from_numpy(perm[n_train + n_val : n_train + n_val + n_test]).long(),
    )


def make_v1v2_splits(
    y_all: torch.Tensor,
    y_dims: tuple[int, int],
    *,
    num_train: int,
    num_val: int,
    num_test: int,
    split_seed: int,
) -> tuple[MultiRegionData, MultiRegionData, MultiRegionData]:
    """One-stop: shuffle + slice + wrap into :class:`MultiRegionData`."""
    n_trials = int(y_all.shape[0])
    train_idx, val_idx, test_idx = split_indices(n_trials, num_train, num_val, num_test, seed=split_seed)
    train_data = MultiRegionData(y=y_all[train_idx], y_dims=y_dims, bin_width=1.0)
    val_data = MultiRegionData(y=y_all[val_idx], y_dims=y_dims, bin_width=1.0)
    test_data = MultiRegionData(y=y_all[test_idx], y_dims=y_dims, bin_width=1.0)
    return train_data, val_data, test_data


def _region_slices(y_dims: tuple[int, ...]) -> list[slice]:
    out: list[slice] = []
    cum = 0
    for y_r in y_dims:
        out.append(slice(cum, cum + y_r))
        cum += y_r
    return out


# ---------------------------------------------------------------------
# Co-smoothing (held-out neuron RMSE) — the V1V2 headline metric
# ---------------------------------------------------------------------
def co_smoothing_rmse(
    *,
    model: ADM | DLAG | MDLAG,
    data: MultiRegionData,
    y_dims: tuple[int, ...],
    holdout_frac: float,
    holdout_seed: int,
) -> tuple[list[tuple[float, float, int]], list[np.ndarray], np.ndarray]:
    """Held-out-neuron prediction RMSE (co-smoothing) on ``data``.

    For each region randomly designate ``holdout_frac`` of neurons as
    held-out, run inference with those neurons treated as effectively
    unobserved, then predict every neuron via ``E[y | y_context]`` and
    score RMSE on the held-out subset only. This is blind to the
    spike-noise floor that full-set per-trial RMSE saturates against,
    and so actually discriminates model classes.

    The "unobserved neuron" trick depends on the emission class:

    - :class:`MultiRegionLinearObservation` (ADM, DLAG, DLAG-SSM, custom
      kernel): inflate ``diag_R_param`` so ``R^{-1} ≈ 0`` for the
      held-out neurons; the filter / E-step gives them near-zero Kalman
      gain.
    - :class:`ARDObservation` (mDLAG, all engines): there is no
      ``diag_R_param`` — instead drive ``phi_mean[holdout] ≈ 0`` so the
      held-out neurons drop out of the variational E-step's CPhiC.

    The held-out neuron mask is deterministic in ``holdout_seed`` and
    decoupled from any other RNG knob. The parameter is restored after
    inference so the model is untouched.

    Returns ``(per_region, holdout_indices, y_recon)`` where
    ``per_region[r] = (per_trial_rmse, psth_rmse, n_held)``,
    ``holdout_indices[r]`` lists the held-out neuron columns of region
    ``r``, and ``y_recon`` is the full ``(B, T, sum y_dims)`` model
    reconstruction with held-out neurons masked during inference (i.e.
    the reconstruction the co-smoothing metric was scored on).
    """
    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in (0, 1); got {holdout_frac}")

    rng = np.random.default_rng(holdout_seed)
    slices = _region_slices(y_dims)
    holdout_indices: list[np.ndarray] = []
    for sl, n_r in zip(slices, y_dims, strict=True):
        n_hold = max(1, round(holdout_frac * n_r))
        local = rng.choice(int(n_r), size=n_hold, replace=False)
        holdout_indices.append(local + sl.start)
    holdout_flat = np.concatenate(holdout_indices)

    obs = model.observation
    if isinstance(obs, ARDObservation):
        holdout_t = torch.from_numpy(holdout_flat).to(device=obs.phi_mean.device).long()
        orig_phi = obs.phi_mean.data.clone()
        try:
            with torch.no_grad():
                obs.phi_mean.data[holdout_t] = 1e-10
            y_recon = extract_y_recon(model, data)
        finally:
            with torch.no_grad():
                obs.phi_mean.data.copy_(orig_phi)
    elif isinstance(obs, MultiRegionLinearObservation):
        holdout_t = torch.from_numpy(holdout_flat).to(device=obs.diag_R_param.device).long()
        orig_R = obs.diag_R_param.data.clone()
        try:
            with torch.no_grad():
                obs.diag_R_param.data[holdout_t] = 1e10
            y_recon = extract_y_recon(model, data)
        finally:
            with torch.no_grad():
                obs.diag_R_param.data.copy_(orig_R)
    else:
        raise TypeError(f"co_smoothing_rmse: unsupported observation type {type(obs).__name__}")

    y_true = data.y.detach().cpu().numpy()
    per_region: list[tuple[float, float, int]] = []
    for cols in holdout_indices:
        diff = y_true[:, :, cols] - y_recon[:, :, cols]
        per_trial = float(np.sqrt(np.mean(diff**2)))
        psth_diff = y_true[:, :, cols].mean(axis=0) - y_recon[:, :, cols].mean(axis=0)
        psth = float(np.sqrt(np.mean(psth_diff**2)))
        per_region.append((per_trial, psth, int(cols.shape[0])))
    return per_region, holdout_indices, y_recon


def co_smoothing_eval_multiseed(
    *,
    model: ADM | DLAG | MDLAG,
    data: MultiRegionData,
    y_dims: tuple[int, ...],
    holdout_frac: float,
    holdout_seeds: list[int],
) -> dict[str, object]:
    """Run co-smoothing for several neuron-holdout splits, aggregate.

    A single neuron split is a noisy estimate — which ~20% of neurons
    are held out can swing the per-region RMSE by O(10%). Averaging
    across several splits gives a smoother number. We use the FIRST
    seed's reconstruction for the figures (a representative example;
    averaging y_recon across seeds is not meaningful because each
    seed conditions on a different neuron set).

    Returns a dict with keys:

    - ``per_region_mean[r]``: ``(per_trial_mean, psth_mean, n_held_mean)``
    - ``per_region_std[r]``:  ``(per_trial_std, psth_std)``
    - ``per_seed[i]``: same shape as :func:`co_smoothing_rmse` output for
      seed ``holdout_seeds[i]`` (for downstream inspection)
    - ``y_recon_plot``: ``(B, T, sum y_dims)`` reconstruction for plotting
      (first seed)
    - ``holdout_indices_plot``: ``list[np.ndarray]`` for first seed
    - ``y_true``: ``data.y`` as numpy
    - ``holdout_seeds``: the seeds used
    """
    if not holdout_seeds:
        raise ValueError("holdout_seeds must contain at least one seed")
    per_seed: list[list[tuple[float, float, int]]] = []
    y_recon_plot: np.ndarray | None = None
    holdout_indices_plot: list[np.ndarray] | None = None
    for i, seed in enumerate(holdout_seeds):
        per_region, holdout_indices, y_recon = co_smoothing_rmse(
            model=model,
            data=data,
            y_dims=y_dims,
            holdout_frac=holdout_frac,
            holdout_seed=seed,
        )
        per_seed.append(per_region)
        if i == 0:
            y_recon_plot = y_recon
            holdout_indices_plot = holdout_indices

    assert y_recon_plot is not None and holdout_indices_plot is not None
    n_regions = len(y_dims)
    per_region_mean: list[tuple[float, float, float]] = []
    per_region_std: list[tuple[float, float]] = []
    for r in range(n_regions):
        per_trial_vals = [per_seed[i][r][0] for i in range(len(holdout_seeds))]
        psth_vals = [per_seed[i][r][1] for i in range(len(holdout_seeds))]
        n_held_vals = [per_seed[i][r][2] for i in range(len(holdout_seeds))]
        per_region_mean.append(
            (
                float(np.mean(per_trial_vals)),
                float(np.mean(psth_vals)),
                float(np.mean(n_held_vals)),
            )
        )
        per_region_std.append(
            (
                float(np.std(per_trial_vals, ddof=1) if len(holdout_seeds) > 1 else 0.0),
                float(np.std(psth_vals, ddof=1) if len(holdout_seeds) > 1 else 0.0),
            )
        )
    return {
        "per_region_mean": per_region_mean,
        "per_region_std": per_region_std,
        "per_seed": per_seed,
        "y_recon_plot": y_recon_plot,
        "holdout_indices_plot": holdout_indices_plot,
        "y_true": data.y.detach().cpu().numpy(),
        "holdout_seeds": list(holdout_seeds),
    }


# ---------------------------------------------------------------------
# ELBO tracking for KalmanEMEngine — opt-in convergence diagnostic
# ---------------------------------------------------------------------
def _entropy_q_from_smoother(engine: object, model: object, data: MultiRegionData) -> float:
    """``H[q]`` for the Gaussian smoother posterior, summed over (B, T).

    ``KalmanEMEngine._smoother_posterior`` returns the smoothed covariances
    ``Σ_{q,t}`` of shape ``(B, T, D, D)`` from one filter+smoother pass.
    For each (b, t) the differential entropy of the Gaussian posterior is

        H = 0.5 · ( D · (1 + log 2π) + log det Σ_{q,t}^{(b)} )

    Sum over (B, T). Computed via Cholesky for numerical stability.
    """
    import math

    import torch

    post = engine._smoother_posterior(model, data)
    sigma = post["covs"]  # shape (B, T, D, D)
    chol = torch.linalg.cholesky(sigma)
    log_det = 2.0 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(-1)  # (B, T)
    D = int(sigma.shape[-1])
    entropy = 0.5 * (D * (1.0 + math.log(2.0 * math.pi)) + log_det).sum()
    return float(entropy.item())


def fit_with_elbo_tracking(
    *,
    model: object,
    train_data: MultiRegionData,
    engine: object,
    num_iters: int,
    check_every: int = 10,
    tol: float = 1e-8,
    keep_scale_anchor: bool = False,
) -> dict[str, object]:
    """Chunked ``model.fit()`` that records the **true ELBO** at each chunk
    boundary.

    Why this exists: ``KalmanEMEngine`` reports ``joint_ll_em`` =
    ``E_q[log p(x, y)]`` in its ``score_trace``. That quantity lacks the
    ``H[q]`` entropy term, so it can drift non-monotonically late in
    training. The true ELBO = ``joint_ll + H[q]`` is far better-behaved
    (monotone for classical EM; gradient-EM with Adam + weight decay can
    still wobble but much less than joint_ll alone).

    This helper splits ``model.fit()`` into chunks of ``check_every``
    iterations. After each chunk it runs ``engine._smoother_posterior``
    once (cheap: ~50ms on V1V2 sizes), computes ``H[q]`` via Cholesky on
    the smoothed covariances, and records ``(iter, joint_ll, H[q], ELBO)``.

    Engine requirements (validated on construction):
    - ``engine.cosine_anneal = False``: chunked ``fit()`` would otherwise
      restart the cosine schedule each chunk, producing a stair-step LR.

    Engine state mutations (auto-restored in ``finally``):
    - ``engine.closed_form_obs_refit`` is set to ``False`` after chunk 0
      so the Tier-1A refit only runs once.
    - ``engine.scale_anchor`` is set to ``False`` after chunk 0 by
      default — UNLESS ``keep_scale_anchor=True``, in which case the
      engine re-anchors latent scales at the start and end of every
      chunk (extra ~2x smoother passes per chunk, ~4ms/iter overhead on
      V1V2). Use ``keep_scale_anchor=True`` to keep C-norm and latent
      variance bounded across the full training run, which empirically
      helps make ELBO closer to monotone.

    Returns a dict:
    - ``joint_ll_trace``: ``list[float]`` of joint-LL per iter (raw,
      possibly non-monotone). Length ≈ ``num_iters``.
    - ``elbo_checkpoints``: ``list[tuple[int, float, float, float]]``,
      one entry per chunk boundary, ``(iter, joint_ll, entropy, elbo)``.
    - ``n_iter``: ``int``, total iterations actually run.
    - ``wall_time_s``: ``float``.
    """
    import time

    if getattr(engine, "cosine_anneal", False):
        raise ValueError(
            "fit_with_elbo_tracking requires engine.cosine_anneal=False to "
            "avoid stair-step LR across chunks. Construct the engine with "
            "cosine_anneal=False (recommend lr=5e-3 constant for V1V2 ADM/DLAG-SSM)."
        )

    saved_refit = getattr(engine, "closed_form_obs_refit", True)
    saved_anchor = getattr(engine, "scale_anchor", True)

    joint_ll_trace: list[float] = []
    elbo_checkpoints: list[tuple[int, float, float, float]] = []

    t0 = time.perf_counter()

    # iter=0 baseline: ELBO at the post-init / pre-training state. Useful
    # for visualising how much Tier-1A refit + initial scale anchor bump
    # the LL on chunk 0 (gap between this point and the next checkpoint).
    import torch

    with torch.no_grad():
        joint_ll_init = float(engine._loss_value(model, train_data).item())
    entropy_init = _entropy_q_from_smoother(engine, model, train_data)
    elbo_checkpoints.append((0, joint_ll_init, entropy_init, joint_ll_init + entropy_init))

    cum_iter = 0
    remaining = int(num_iters)
    try:
        chunk_idx = 0
        while remaining > 0:
            chunk = min(int(check_every), remaining)
            result = model.fit(train_data, max_iter=chunk, tol=tol)  # type: ignore[attr-defined]
            joint_ll_trace.extend(float(x) for x in result.score_trace)
            cum_iter += int(result.n_iter)

            joint_ll_now = float(result.score_trace[-1])
            entropy = _entropy_q_from_smoother(engine, model, train_data)
            elbo = joint_ll_now + entropy
            elbo_checkpoints.append((cum_iter, joint_ll_now, entropy, elbo))

            if chunk_idx == 0:
                engine.closed_form_obs_refit = False  # type: ignore[attr-defined]
                if not keep_scale_anchor:
                    engine.scale_anchor = False  # type: ignore[attr-defined]

            remaining -= chunk
            chunk_idx += 1
            if getattr(result, "converged", False):
                break
    finally:
        engine.closed_form_obs_refit = saved_refit  # type: ignore[attr-defined]
        engine.scale_anchor = saved_anchor  # type: ignore[attr-defined]

    wall_s = time.perf_counter() - t0
    return {
        "joint_ll_trace": joint_ll_trace,
        "elbo_checkpoints": elbo_checkpoints,
        "n_iter": cum_iter,
        "wall_time_s": wall_s,
    }


# ---------------------------------------------------------------------
# Plots (no ground truth — all plots are self-consistency diagnostics)
# ---------------------------------------------------------------------
def plot_convergence(
    train_trace: np.ndarray,
    val_trace: np.ndarray,
    out_path: Path | None,
    *,
    model_label: str,
    ylabel: str,
) -> None:
    """Training-objective convergence with optional held-out (val) curve.

    ``train_trace`` / ``val_trace`` are ``(n, 2)`` arrays of
    ``(iter, score_per_trial)``. ``val_trace`` may be empty.

    The "train" line is always green-circle ``label='train'``. What's
    actually on the y-axis depends on the engine the caller used:

    - ``KalmanEMEngine`` paths via :func:`fit_with_elbo_tracking`: ELBO
      per checkpoint (per-trial), sparse points joined by line.
    - ``ExactEMEngine`` / ``VEMARDEngine`` / ``VEMARDFreqEngine`` /
      ``VEMKalmanARDEngine``: marginal LL or proxy-ELBO per iter
      (whatever the engine put in ``result.score_trace``).

    The caller is responsible for picking the right source data and
    pre-normalising by ``NUM_TRAIN``. Legend label stays ``train`` to
    keep the plot uniform across methods.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    if train_trace.size:
        ax.plot(
            train_trace[:, 0],
            train_trace[:, 1],
            "o-",
            color="darkgreen",
            lw=1.2,
            markersize=3,
            label="train",
        )
    if val_trace.size:
        ax.plot(
            val_trace[:, 0],
            val_trace[:, 1],
            "s--",
            color="darkorange",
            lw=1.0,
            markersize=3,
            label="held-out (val)",
        )
    ax.legend(fontsize=8)
    ax.set_xlabel("EM / VEM / AdamW iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{model_label} — {ylabel} convergence", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return None
    return fig


def plot_delay(
    fitted_delay: np.ndarray,
    n_across: int,
    out_path: Path | None,
    *,
    model_label: str,
    region_names: list[str],
) -> None:
    """Per-across-latent inter-region delay over time, fitted only.

    ``fitted_delay`` has shape ``(T, R-1, n_across)``. V1V2 has R=2 so
    there is exactly one pair (V2 vs V1) per latent.
    """
    if n_across == 0 or fitted_delay.shape[1] == 0:
        return
    T = fitted_delay.shape[0]
    n_pairs = fitted_delay.shape[1]
    ts = np.arange(T)
    fig, axes = plt.subplots(n_pairs, n_across, figsize=(4 * n_across, 2.5 * n_pairs), squeeze=False)
    # Pair index ordering matches truth: pair i = (region 0, region i+1).
    pair_labels = [f"{region_names[i]} - {region_names[0]}" for i in range(1, len(region_names))]
    for k in range(n_across):
        for p in range(n_pairs):
            ax = axes[p, k]
            ax.plot(ts, fitted_delay[:, p, k], lw=1.4, color="darkred", label=f"{model_label}")
            ax.axhline(0.0, color="grey", lw=0.5, ls="--")
            ax.set_xlabel("time (bins)", fontsize=8)
            ax.set_ylabel(f"{pair_labels[p]} delay (bins)", fontsize=8)
            ax.set_title(f"across-latent {k + 1}", fontsize=9)
            ax.legend(fontsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle(f"{model_label} — recovered inter-region delay", fontsize=11)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return None
    return fig


def plot_latents(
    fitted_obs: np.ndarray,
    *,
    n_regions: int,
    n_across: int,
    n_within: int,
    out_path: Path | None,
    model_label: str,
    region_names: list[str],
    trial: int = 0,
) -> None:
    """Per-region smoother-latent traces (fitted only) for one trial."""
    n_per_region = n_across + n_within
    if n_per_region == 0:
        return
    fig, axes = plt.subplots(
        n_regions, n_per_region, figsize=(3.0 * n_per_region, 1.6 * n_regions), squeeze=False
    )
    for r in range(n_regions):
        for s in range(n_per_region):
            col = r * n_per_region + s
            ax = axes[r, s]
            ax.plot(fitted_obs[trial, :, col], lw=1.2, color="darkred")
            kind = f"across k={s + 1}" if s < n_across else f"within w={s - n_across + 1}"
            ax.set_title(f"{region_names[r]} — {kind}", fontsize=8)
            ax.set_xticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle(f"{model_label} — smoother latents (trial {trial})", fontsize=10)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return None
    return fig


def plot_y_recon_psth(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_dims: tuple[int, ...],
    out_path: Path | None,
    *,
    model_label: str,
    region_names: list[str],
    holdout_indices: list[np.ndarray] | None = None,
    holdout_seed: int | None = None,
) -> None:
    """Trial-mean PSTH heatmap: 3 rows (truth / pred / residual) × R columns.

    When ``holdout_indices`` is provided, the held-out neuron rows are
    sorted to the bottom of each region's panel and a horizontal divider
    + side annotation marks them. The reported per-panel ``RMSE``
    numbers split into ``held-out`` / ``context`` so the two regimes can
    be compared directly — held-out RMSE is the model's true predictive
    error, context RMSE is the inference-only error on neurons it had
    access to.
    """
    n_regions = len(y_dims)
    slices = _region_slices(y_dims)
    psth_true = y_true.mean(axis=0)
    psth_pred = y_pred.mean(axis=0)

    fig, axes = plt.subplots(3, n_regions, figsize=(4.5 * n_regions, 7.8), squeeze=False)
    row_titles = ["true (trial-mean)", "pred (trial-mean)", "residual (true - pred)"]
    for c, sl in enumerate(slices):
        n_r = sl.stop - sl.start
        # Reorder: held-out neurons first (top), context second (bottom).
        # Default = identity.
        if holdout_indices is not None and holdout_indices[c].size:
            local_held = holdout_indices[c] - sl.start
            local_ctx = np.setdiff1d(np.arange(n_r), local_held)
            order = np.concatenate([local_held, local_ctx])
            n_held = int(local_held.size)
        else:
            order = np.arange(n_r)
            n_held = 0

        v_true_full = psth_true[:, sl][:, order]
        v_pred_full = psth_pred[:, sl][:, order]
        v_resid_full = v_true_full - v_pred_full

        # RMSE split: held-out (top n_held rows) vs context (rest).
        if n_held > 0:
            held_resid = v_resid_full[:, :n_held]
            ctx_resid = v_resid_full[:, n_held:]
            held_full = (y_true[:, :, sl][..., order] - y_pred[:, :, sl][..., order])[:, :, :n_held]
            ctx_full = (y_true[:, :, sl][..., order] - y_pred[:, :, sl][..., order])[:, :, n_held:]
            held_psth_rmse = float(np.sqrt(np.mean(held_resid**2))) if held_resid.size else float("nan")
            held_pt_rmse = float(np.sqrt(np.mean(held_full**2))) if held_full.size else float("nan")
            ctx_psth_rmse = float(np.sqrt(np.mean(ctx_resid**2))) if ctx_resid.size else float("nan")
            ctx_pt_rmse = float(np.sqrt(np.mean(ctx_full**2))) if ctx_full.size else float("nan")
            rmse_line = (
                f"held-out PSTH={held_psth_rmse:.4f} per-trial={held_pt_rmse:.4f}\n"
                f"context  PSTH={ctx_psth_rmse:.4f} per-trial={ctx_pt_rmse:.4f}"
            )
        else:
            all_psth_rmse = float(np.sqrt(np.mean(v_resid_full**2)))
            all_pt_rmse = float(np.sqrt(np.mean((y_true[:, :, sl] - y_pred[:, :, sl]) ** 2)))
            rmse_line = f"PSTH RMSE = {all_psth_rmse:.4f}   per-trial RMSE = {all_pt_rmse:.4f}"

        vmax = float(np.abs(np.concatenate([v_true_full.ravel(), v_pred_full.ravel()])).max())
        for r, mat in enumerate([v_true_full, v_pred_full]):
            ax = axes[r, c]
            im = ax.imshow(mat.T, aspect="auto", origin="lower", vmin=-vmax, vmax=vmax, cmap="RdBu_r")
            ax.set_title(f"{region_names[c]} - {row_titles[r]}", fontsize=9)
            ax.set_xlabel("time (bins)")
            ax.set_ylabel("neuron")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            if n_held > 0:
                ax.axhline(n_held - 0.5, color="black", lw=0.8, ls="--")
                ax.text(
                    1.01,
                    n_held / 2 / n_r,
                    "held-out",
                    transform=ax.transAxes,
                    rotation=270,
                    va="center",
                    ha="left",
                    fontsize=7,
                    color="black",
                )
                ax.text(
                    1.01,
                    (n_held + n_r) / 2 / n_r,
                    "context",
                    transform=ax.transAxes,
                    rotation=270,
                    va="center",
                    ha="left",
                    fontsize=7,
                    color="grey",
                )

        vmax_r = float(np.abs(v_resid_full).max() + 1e-9)
        ax = axes[2, c]
        im = ax.imshow(
            v_resid_full.T, aspect="auto", origin="lower", vmin=-vmax_r, vmax=vmax_r, cmap="RdBu_r"
        )
        ax.set_title(f"{region_names[c]} - {row_titles[2]}\n{rmse_line}", fontsize=8)
        ax.set_xlabel("time (bins)")
        ax.set_ylabel("neuron")
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        if n_held > 0:
            ax.axhline(n_held - 0.5, color="black", lw=0.8, ls="--")

    sub = "co-smoothing reconstruction (test set"
    if holdout_seed is not None:
        sub += f", holdout-seed={holdout_seed}"
    sub += ")"
    fig.suptitle(f"{model_label} — {sub}", fontsize=11, y=1.0)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return None
    return fig


def plot_y_recon_trial(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_dims: tuple[int, ...],
    out_path: Path | None,
    *,
    model_label: str,
    region_names: list[str],
    trial: int = 0,
    holdout_indices: list[np.ndarray] | None = None,
    holdout_seed: int | None = None,
) -> None:
    """Single-trial full heatmap: 2 rows (truth / pred) × R columns.

    When ``holdout_indices`` is given, held-out neurons are sorted to
    the top of each region's panel and divided from context neurons by
    a horizontal line (same convention as :func:`plot_y_recon_psth`).
    """
    n_regions = len(y_dims)
    slices = _region_slices(y_dims)
    fig, axes = plt.subplots(2, n_regions, figsize=(4.5 * n_regions, 5.2), squeeze=False)
    row_titles = [f"true (trial {trial})", f"pred (trial {trial})"]
    for c, sl in enumerate(slices):
        n_r = sl.stop - sl.start
        if holdout_indices is not None and holdout_indices[c].size:
            local_held = holdout_indices[c] - sl.start
            local_ctx = np.setdiff1d(np.arange(n_r), local_held)
            order = np.concatenate([local_held, local_ctx])
            n_held = int(local_held.size)
        else:
            order = np.arange(n_r)
            n_held = 0
        v_true = y_true[trial, :, sl][:, order]
        v_pred = y_pred[trial, :, sl][:, order]
        vmax = float(np.abs(np.concatenate([v_true.ravel(), v_pred.ravel()])).max())
        for r, mat in enumerate([v_true, v_pred]):
            ax = axes[r, c]
            im = ax.imshow(mat.T, aspect="auto", origin="lower", vmin=-vmax, vmax=vmax, cmap="RdBu_r")
            ax.set_title(f"{region_names[c]} - {row_titles[r]}", fontsize=9)
            ax.set_xlabel("time (bins)")
            ax.set_ylabel("neuron")
            fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
            if n_held > 0:
                ax.axhline(n_held - 0.5, color="black", lw=0.8, ls="--")
    sub = "co-smoothing reconstruction (test set"
    if holdout_seed is not None:
        sub += f", holdout-seed={holdout_seed}"
    sub += ")"
    fig.suptitle(f"{model_label} — single-trial: {sub}", fontsize=11)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return None
    return fig


def plot_ard_alpha(alpha_mean: np.ndarray, out_path: Path | None, *, model_label: str) -> object:
    """ARD α per latent column. When ``out_path`` is None, return the
    Figure for inline notebook display; otherwise save + close.
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
    ax.set_title(f"{model_label} - ARD α per across-latent column (large = pruned)", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return None
    return fig


# ---------------------------------------------------------------------
# End-of-fit writer
# ---------------------------------------------------------------------
def write_v1v2_outputs(
    *,
    method_name: str,
    model_label: str,
    train_data: MultiRegionData,
    fitted_delay: np.ndarray,
    fitted_obs: np.ndarray,
    y_dims: tuple[int, ...],
    n_across: int,
    n_within: int,
    train_trace: np.ndarray,
    val_trace: np.ndarray,
    score_ylabel: str,
    cosmoothing: dict[str, object],
    holdout_frac: float,
    seed: int,
    split_seed: int,
    wall_s: float,
    out_dir: Path,
    alpha_mean: np.ndarray | None = None,
    region_names: list[str] | None = None,
    extra_summary: str | None = None,
) -> dict[str, object]:
    """End-of-fit orchestrator: write all plots + ``eval_metrics.txt`` +
    ``summary.json`` + ``snapshots.npz``.

    ``cosmoothing`` is the dict returned by
    :func:`co_smoothing_eval_multiseed`. ``eval_metrics.txt`` reports
    mean ± std across holdout seeds; the headline mean values use the
    same key names ``holdout_{psth,per_trial}_rmse_{REGION}`` that
    :mod:`examples.v1v2.compare_v1v2_runs` parses, so the existing aggregator
    works unchanged. PSTH / trial-0 plots show the first holdout seed's
    co-smoothing reconstruction on the TEST set, with held-out neurons
    visually separated from context.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_regions = len(y_dims)
    if region_names is None:
        region_names = ["V1", "V2"] if n_regions == 2 else [f"region_{r}" for r in range(n_regions)]
    T = int(train_data.y.shape[1])

    per_region_mean = cosmoothing["per_region_mean"]
    per_region_std = cosmoothing["per_region_std"]
    holdout_seeds = cosmoothing["holdout_seeds"]
    y_recon_plot = cosmoothing["y_recon_plot"]
    holdout_indices_plot = cosmoothing["holdout_indices_plot"]
    y_test = cosmoothing["y_true"]
    assert isinstance(per_region_mean, list)
    assert isinstance(per_region_std, list)
    assert isinstance(holdout_seeds, list)
    assert isinstance(y_recon_plot, np.ndarray)
    assert isinstance(holdout_indices_plot, list)
    assert isinstance(y_test, np.ndarray)

    # ---- ARD-aware subset for plotting (mDLAG only) ----
    # When ``alpha_mean`` is given (mDLAG paths), restrict the across
    # latents shown in delay.png / latents.png to ARD-active columns.
    # Active = ``max-over-region α[:, k] <= alpha_prune_ratio · min_k α``
    # (same threshold as ard_aware_delay_rmse). Columns are reordered
    # by α ascending so the strongest latent is "latent 1" in the plot.
    # Non-mDLAG paths pass alpha_mean=None and the plot uses all
    # n_across columns unchanged.
    plot_obs = fitted_obs
    plot_delay_arr = fitted_delay
    plot_n_across = n_across
    if alpha_mean is not None and n_across > 0:
        max_per_col = alpha_mean.max(axis=0)
        threshold = 10.0 * float(max_per_col.min())
        active_cols = np.flatnonzero(max_per_col <= threshold)
        # Reorder active by ascending α (smallest α = "strongest latent").
        active_cols = active_cols[np.argsort(max_per_col[active_cols])]
        plot_n_across = int(active_cols.size)
        if plot_n_across != n_across or list(active_cols) != list(range(n_across)):
            plot_delay_arr = fitted_delay[..., active_cols]
            npr_old = n_across + n_within
            npr_new = plot_n_across + n_within
            B, T_obs = fitted_obs.shape[:2]
            plot_obs = np.zeros((B, T_obs, n_regions * npr_new), dtype=fitted_obs.dtype)
            for r in range(n_regions):
                for i, k_src in enumerate(active_cols):
                    plot_obs[..., r * npr_new + i] = fitted_obs[..., r * npr_old + int(k_src)]
                for w in range(n_within):
                    plot_obs[..., r * npr_new + plot_n_across + w] = fitted_obs[
                        ..., r * npr_old + n_across + w
                    ]

    # ---- plots ----
    plot_convergence(
        train_trace,
        val_trace,
        out_dir / "convergence.png",
        model_label=model_label,
        ylabel=score_ylabel,
    )
    plot_delay(
        plot_delay_arr,
        plot_n_across,
        out_dir / "delay.png",
        model_label=model_label,
        region_names=region_names,
    )
    plot_latents(
        plot_obs,
        n_regions=n_regions,
        n_across=plot_n_across,
        n_within=n_within,
        out_path=out_dir / "latents.png",
        model_label=model_label,
        region_names=region_names,
    )
    plot_y_recon_psth(
        y_test,
        y_recon_plot,
        y_dims,
        out_dir / "y_recon_psth.png",
        model_label=model_label,
        region_names=region_names,
        holdout_indices=holdout_indices_plot,
        holdout_seed=int(holdout_seeds[0]),
    )
    plot_y_recon_trial(
        y_test,
        y_recon_plot,
        y_dims,
        out_dir / "y_recon_trial0.png",
        model_label=model_label,
        region_names=region_names,
        holdout_indices=holdout_indices_plot,
        holdout_seed=int(holdout_seeds[0]),
    )
    if alpha_mean is not None:
        plot_ard_alpha(alpha_mean, out_dir / "ard_alpha.png", model_label=model_label)

    # ---- eval_metrics.txt (compare_v1v2_runs.py parses this) ----
    metrics_lines: list[str] = [
        f"model: {method_name}",
        f"seed: {seed}",
        f"split_seed: {split_seed}",
        f"holdout_seeds: {','.join(str(s) for s in holdout_seeds)}",
        f"holdout_frac: {holdout_frac}",
        f"n_trials_train: {int(train_data.y.shape[0])}",
        f"n_trials_test: {int(y_test.shape[0])}",
        f"T: {T}",
        f"y_dims: {','.join(str(d) for d in y_dims)}",
        f"n_across: {n_across}",
        f"n_within: {n_within}",
        "",
        "# Co-smoothing (held-out neuron prediction) RMSE on the TEST set,",
        f"# averaged across {len(holdout_seeds)} holdout-neuron seeds. Lower is better.",
        "# The plain key holds the MEAN; *_std holds the cross-seed std.",
    ]
    for r, name in enumerate(region_names):
        pt_mean, psth_mean, n_held = per_region_mean[r]
        pt_std, psth_std = per_region_std[r]
        metrics_lines.append(f"holdout_n_neurons_{name}: {round(n_held)}")
        metrics_lines.append(f"holdout_per_trial_rmse_{name}: {pt_mean:.6f}")
        metrics_lines.append(f"holdout_per_trial_rmse_{name}_std: {pt_std:.6f}")
        metrics_lines.append(f"holdout_psth_rmse_{name}: {psth_mean:.6f}")
        metrics_lines.append(f"holdout_psth_rmse_{name}_std: {psth_std:.6f}")
    (out_dir / "eval_metrics.txt").write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")

    # ---- summary.json ----
    summary = {
        "method_name": method_name,
        "model_label": model_label,
        "n_regions": n_regions,
        "n_across": n_across,
        "n_within": n_within,
        "T": T,
        "wall_s": wall_s,
        "holdout_frac": holdout_frac,
        "holdout_seeds": [int(s) for s in holdout_seeds],
        "seed": seed,
        "split_seed": split_seed,
        "holdout_per_trial_rmse_mean": [m[0] for m in per_region_mean],
        "holdout_per_trial_rmse_std": [s[0] for s in per_region_std],
        "holdout_psth_rmse_mean": [m[1] for m in per_region_mean],
        "holdout_psth_rmse_std": [s[1] for s in per_region_std],
        "holdout_n_neurons": [round(m[2]) for m in per_region_mean],
        "region_names": list(region_names),
        "max_alpha_per_col": alpha_mean.max(axis=0).tolist() if alpha_mean is not None else None,
        "extra_summary": extra_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # ---- snapshots.npz ----
    np.savez(
        out_dir / "snapshots.npz",
        method_name=method_name,
        delay=fitted_delay,
        observable=fitted_obs,
        y_recon_cosmoothing=y_recon_plot,
        y_test=y_test,
        train_trace=train_trace,
        val_trace=val_trace,
        holdout_per_trial_rmse_mean=np.array([m[0] for m in per_region_mean]),
        holdout_per_trial_rmse_std=np.array([s[0] for s in per_region_std]),
        holdout_psth_rmse_mean=np.array([m[1] for m in per_region_mean]),
        holdout_psth_rmse_std=np.array([s[1] for s in per_region_std]),
        holdout_n_neurons=np.array([round(m[2]) for m in per_region_mean], dtype=np.int64),
        holdout_indices=np.array(list(holdout_indices_plot), dtype=object),
        holdout_frac=float(holdout_frac),
        holdout_seeds=np.array(holdout_seeds, dtype=np.int64),
    )

    return summary


def print_latent_diagnostics(
    *,
    method_name: str,
    fitted_obs: np.ndarray,
    fitted_delay: np.ndarray,
    train_y: np.ndarray,
    n_regions: int,
    n_across: int,
    n_within: int,
    region_names: list[str] | None = None,
    alpha_mean: np.ndarray | None = None,
) -> None:
    """Per-across-latent strength + delay diagnostic.

    On V1V2 we routinely build a model with more latents than the data
    can support — when a latent is weakly constrained (low explained
    variance), its inferred delay δ is essentially noise and may flip
    sign between methods or seeds. This print helps the reader decide
    which latent's δ to trust.

    For each across-latent ``k`` we report:

    - ``var_frac_per_region[r]``: trial+time variance of the observable
      trace ``g_k^{(r)}(t)`` divided by the total observed-variance of
      region ``r`` (uses the trial+time variance of ``train_y``). Higher
      = latent has more energy in that region. A value near zero means
      the latent is silent there and any associated δ is unconstrained.
    - ``delta_pair[p]``: time-mean inter-region delay δ_j − δ_i for
      every region pair ``(i, j)`` with ``j > i``, in bins. Positive →
      region ``j`` lags region ``i``; negative → ``j`` leads.
    - For mDLAG, prints the per-region ARD ``max α`` per column too
      (column with much larger α than the smallest is ARD-pruned).

    The two-latent symmetry (latent identifiability) is up to a sign
    flip and column permutation. Sign flips are absorbed into both C[:,
    k] and g_k(t) together, so the variance fraction is sign-invariant
    — but δ is not, so two methods that fit ``+g_k`` vs ``-g_k`` will
    report ``+δ_k`` vs ``-δ_k``. When a latent is weakly constrained,
    even the sign convention can flip across methods.
    """
    if region_names is None:
        region_names = ["V1", "V2"] if n_regions == 2 else [f"region_{r}" for r in range(n_regions)]
    n_per_region = n_across + n_within
    if n_per_region == 0:
        return

    # Per-neuron observed variance pooled over (trial, time); average is
    # the denominator used to normalize each latent's energy. We don't
    # know per-region y_dims here, so use the whole-tensor average — fair
    # across regions and across latents.
    y_centered = train_y - train_y.mean(axis=(0, 1), keepdims=True)
    total_var = float(y_centered.var(ddof=0))

    print(f"[{method_name}] per-across-latent diagnostics:")
    if alpha_mean is not None:
        max_alpha_per_col = alpha_mean.max(axis=0)
        print(f"  ARD max α per across-column: {np.array2string(max_alpha_per_col, precision=2)}")

    pair_labels: list[str] = []
    pair_idx: list[tuple[int, int]] = []
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            pair_labels.append(f"{region_names[j]}-{region_names[i]}")
            pair_idx.append((i, j))

    # Pad fitted_delay (T, R-1, K_a) into (T, R, K_a) with zero ref.
    T = int(fitted_delay.shape[0])
    delay_full = np.concatenate([np.zeros((T, 1, n_across)), fitted_delay], axis=1)

    for k in range(n_across):
        # Per-region: variance of g_k^{(r)}(t) over trials+time.
        var_per_region: list[float] = []
        for r in range(n_regions):
            col = r * n_per_region + k
            var_per_region.append(float(np.var(fitted_obs[..., col], ddof=0)))
        var_frac_total = sum(var_per_region) / max(total_var, 1e-12)
        var_per_region_str = (
            "{" + ", ".join(f"{region_names[r]}:{var_per_region[r]:.3g}" for r in range(n_regions)) + "}"
        )

        # Per-pair time-mean delay.
        delta_pairs: list[float] = []
        for i, j in pair_idx:
            delta_pairs.append(float((delay_full[:, j, k] - delay_full[:, i, k]).mean()))
        pair_strs = [f"{label}:{d:+.3f}" for label, d in zip(pair_labels, delta_pairs, strict=True)]
        weak = "  ← weak (low variance)" if var_frac_total < 1e-2 else ""
        print(
            f"  latent {k + 1}:  var_per_region={var_per_region_str}  "
            f"var_frac_total={var_frac_total:.3g}  "
            f"δ_pair={{{', '.join(pair_strs)}}}{weak}"
        )


def print_summary(summary: dict[str, object], method_name: str, region_names: list[str]) -> None:
    """One-line co-smoothing summary (mean ± std across holdout seeds)."""
    pt_mean = summary["holdout_per_trial_rmse_mean"]
    pt_std = summary["holdout_per_trial_rmse_std"]
    psth_mean = summary["holdout_psth_rmse_mean"]
    psth_std = summary["holdout_psth_rmse_std"]
    wall = summary["wall_s"]
    n_seeds = len(summary.get("holdout_seeds", []) or [])  # type: ignore[arg-type]
    pieces = [f"[{method_name}]"]
    for r, name in enumerate(region_names):
        pieces.append(
            f"{name}: psth={psth_mean[r]:.4f}±{psth_std[r]:.4f}  per_trial={pt_mean[r]:.4f}±{pt_std[r]:.4f}"
        )
    pieces.append(f"(n_seeds={n_seeds})")
    pieces.append(f"wall={wall:.1f}s")
    print("  ".join(pieces))
