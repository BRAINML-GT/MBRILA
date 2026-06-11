"""Plotting helpers for the mbrila CF7 model-zoo notebooks.

Each notebook (``notebooks/demo_*.ipynb``) builds one model on the
canonical demo synthetic scenario, fits it, then calls these helpers
to produce a fixed set of five figures:

1. :func:`plot_convergence` — log-likelihood trace
2. :func:`plot_delay_comparison` — fitted vs true δ(t) per latent
3. :func:`plot_latent_comparison` — fitted vs true latent traces
4. :func:`plot_psth_matrix` — trial-averaged neuron-by-time heatmaps
5. :func:`plot_trial0` — single-trial reconstruction overlay

All five take ``truth`` (the dict returned by
:func:`examples.demo_common.sample_scenario`) plus the fitted-side
quantities extracted by ``demo_common.extract_*`` helpers, so a
notebook only has to thread the model through.

Style choices:

* ``seaborn-v0_8-whitegrid`` base with a ``tab10`` palette.
* Truth is always dashed-purple, fit is always solid-darkred —
  matches the ``examples.demo_common`` PNG outputs so figures line up
  between notebook and CLI runs.
* Figures use ``constrained_layout`` so multi-panel grids fit cleanly
  without overlap.
* All text in English (latents are 1-indexed in titles to match the
  ``delay_lat{lat}.png`` filename convention).

The helpers accept an optional ``axes`` / ``ax`` parameter so a
notebook can compose them into a larger grid; if omitted a fresh
figure is created.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Standard style — applied on import so notebooks pick it up just by
# ``from nb_helpers import *``.
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 180,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

TRUTH_COLOR = "#7E57C2"  # purple, dashed
FIT_COLOR = "#C62828"  # dark red, solid
PALETTE = plt.get_cmap("tab10")


# ---------------------------------------------------------------------------
# 1. Convergence
# ---------------------------------------------------------------------------


def plot_convergence(
    score_trace: np.ndarray,
    *,
    ax: plt.Axes | None = None,
    ylabel: str = "joint log-likelihood",
    title: str | None = None,
) -> plt.Figure:
    """Plot the per-iter log-likelihood trace.

    A vertical dotted line marks the iter of peak LL so it's easy to
    spot late drift.
    """
    arr = np.asarray(score_trace, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.6))
    else:
        fig = ax.figure
    iters = np.arange(1, len(arr) + 1)
    ax.plot(iters, arr, color=FIT_COLOR, lw=1.4)
    if arr.size:
        peak = int(np.argmax(arr))
        ax.axvline(peak + 1, color="grey", ls=":", lw=0.8, alpha=0.6)
        ax.scatter([peak + 1], [arr[peak]], color="grey", s=18, zorder=5)
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Training convergence")
    return fig


# ---------------------------------------------------------------------------
# 2. Delay comparison
# ---------------------------------------------------------------------------


def _pairwise_delay_curves(delay: np.ndarray, n_regions: int, latent: int) -> list[np.ndarray]:
    """Return T-long curves for every (i, j) region pair, given delay tensor
    of shape ``(T, R-1, K)`` (region 0 reference dropped)."""
    full = np.concatenate([np.zeros((delay.shape[0], 1, delay.shape[-1])), delay], axis=1)
    return [
        full[:, j, latent] - full[:, i, latent] for i in range(n_regions) for j in range(i + 1, n_regions)
    ]


def plot_delay_comparison(
    truth_delay: np.ndarray,
    fitted_delay: np.ndarray,
    *,
    n_regions: int,
    n_across: int,
    rmse: float | None = None,
) -> plt.Figure:
    """Grid of (across-latent × region-pair) panels: truth vs fitted δ(t).

    One panel per ``(latent k, region pair (i, j))`` — matches the
    ``demo_*`` CLI's ``delay_lat{k}.png`` layout. Truth dashed-purple,
    fit solid-darkred, shared y-axis across all panels for a given
    latent so panels are visually comparable.

    ``truth_delay`` / ``fitted_delay`` must share shape
    ``(T, n_regions - 1, n_across)``.
    """
    if n_across == 0 or n_regions < 2:
        return plt.figure(figsize=(4, 2))

    pair_ij = [(i, j) for i in range(n_regions) for j in range(i + 1, n_regions)]
    n_pairs = len(pair_ij)
    fig, axes = plt.subplots(
        n_across,
        n_pairs,
        figsize=(2.4 * n_pairs, 2.2 * n_across),
        sharex=True,
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )

    for lat in range(n_across):
        truth_curves = _pairwise_delay_curves(truth_delay, n_regions, lat)
        fit_curves = _pairwise_delay_curves(fitted_delay, n_regions, lat)
        # Per-latent y-limit covering both truth and fit, with a small pad.
        all_vals = np.concatenate(
            [np.concatenate([t, f]) for t, f in zip(truth_curves, fit_curves, strict=True)]
        )
        lo = min(float(all_vals.min()), 0.0)
        hi = max(float(all_vals.max()), 0.0)
        pad = max(0.1 * (hi - lo), 0.5)
        for p, (i, j) in enumerate(pair_ij):
            ax = axes[lat, p]
            ax.plot(truth_curves[p], "--", color=TRUTH_COLOR, lw=1.2, alpha=0.9)
            ax.plot(fit_curves[p], color=FIT_COLOR, lw=1.3, alpha=0.95)
            ax.axhline(0.0, color="grey", lw=0.5, alpha=0.4)
            ax.set_ylim(lo - pad, hi + pad)
            if lat == 0:
                ax.set_title(f"region {j} – region {i}", fontsize=8)
            if lat == n_across - 1:
                ax.set_xlabel("time (bins)", fontsize=8)
            if p == 0:
                ax.set_ylabel(f"Latent {lat + 1}\nδ (bins)", fontsize=9)

    # Legend on the top-right panel.
    legend_handles = [
        plt.Line2D([], [], color=TRUTH_COLOR, ls="--", lw=1.2, label="truth"),
        plt.Line2D([], [], color=FIT_COLOR, lw=1.3, label="fitted"),
    ]
    axes[0, -1].legend(handles=legend_handles, loc="best", fontsize=7)
    title_suffix = f"  (RMSE = {rmse:.3f} bins)" if rmse is not None else ""
    fig.suptitle(f"Inter-region delay per pair{title_suffix}")
    return fig


# ---------------------------------------------------------------------------
# 3. Latent comparison
# ---------------------------------------------------------------------------


def _flip_sign_to_truth(est: np.ndarray, truth: np.ndarray) -> float:
    return -1.0 if float(np.sum(est * truth)) < 0 else 1.0


def plot_latent_comparison(
    truth_obs: np.ndarray,
    fitted_obs: np.ndarray,
    *,
    n_regions: int,
    n_across: int,
    n_within: int,
    trial: int = 0,
) -> plt.Figure:
    """Grid of (latent × region) panels — truth vs fitted g(t) per region.

    One panel per ``(latent k, region r)`` — matches the ``demo_*`` CLI's
    ``latent_across_{k}.png`` / ``latent_within.png`` layout (here we
    fuse them into one figure so the notebook produces a single image).
    Truth dashed-purple, fit solid-darkred, fit sign-flipped per panel
    to align with truth (latents are identified only up to sign).

    Observables have shape ``(B, T, R * (n_across + n_within))`` in
    region-major layout: ``[reg0_lat0, reg0_lat1, ..., reg1_lat0, ...]``.
    """
    npr = n_across + n_within
    if npr == 0:
        return plt.figure(figsize=(4, 2))

    fig, axes = plt.subplots(
        npr,
        n_regions,
        figsize=(3.2 * n_regions, 1.8 * npr),
        sharex=True,
        sharey="row",
        constrained_layout=True,
        squeeze=False,
    )

    for lat in range(npr):
        kind = "across" if lat < n_across else f"within (slot {lat - n_across})"
        for r in range(n_regions):
            ax = axes[lat, r]
            col = r * npr + lat
            t_curve = truth_obs[trial, :, col]
            f_curve = fitted_obs[trial, :, col]
            sign = _flip_sign_to_truth(f_curve, t_curve)
            ax.plot(t_curve, "--", color=TRUTH_COLOR, lw=1.1, alpha=0.9)
            ax.plot(sign * f_curve, color=FIT_COLOR, lw=1.2, alpha=0.95)
            if lat == 0:
                ax.set_title(f"region {r}", fontsize=8)
            if lat == npr - 1:
                ax.set_xlabel("time (bins)", fontsize=8)
            if r == 0:
                ax.set_ylabel(f"Latent {lat + 1}\n({kind})", fontsize=8)

    legend_handles = [
        plt.Line2D([], [], color=TRUTH_COLOR, ls="--", lw=1.1, label="truth"),
        plt.Line2D([], [], color=FIT_COLOR, lw=1.2, label="fitted (sign-aligned)"),
    ]
    axes[0, -1].legend(handles=legend_handles, loc="best", fontsize=7)
    fig.suptitle(f"Per-latent traces (trial {trial})")
    return fig


# ---------------------------------------------------------------------------
# 4. PSTH matrix comparison (trial-averaged neuron-by-time heatmap)
# ---------------------------------------------------------------------------


def plot_psth_matrix(
    truth_y: np.ndarray,
    fitted_y: np.ndarray,
    *,
    y_dims: tuple[int, ...],
) -> plt.Figure:
    """Side-by-side PSTH heatmaps (truth | fitted | residual).

    PSTH = trial-average of ``y`` → shape ``(neurons, T)``. Neurons are
    kept in their natural (region, within-region index) order so the
    visualization reflects the data's layout rather than a sorting
    artifact. Horizontal lines mark region boundaries. Truth and fitted
    panels share a divergent color scale centered at zero.
    """
    truth_psth = truth_y.mean(axis=0).T  # (N, T)
    fitted_psth = fitted_y.mean(axis=0).T
    residual = fitted_psth - truth_psth

    # Robust symmetric color limits — use the 99th percentile of |truth|
    # so a few high-variance neurons don't blow out the scale.
    abs_max = float(np.percentile(np.abs(truth_psth), 99))
    abs_max = max(abs_max, float(np.percentile(np.abs(fitted_psth), 99)))
    res_max = max(float(np.percentile(np.abs(residual), 99)), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)

    panels = [
        ("Truth PSTH", truth_psth, "RdBu_r", {"vmin": -abs_max, "vmax": abs_max}),
        ("Fitted PSTH", fitted_psth, "RdBu_r", {"vmin": -abs_max, "vmax": abs_max}),
        ("Fitted - truth", residual, "RdBu_r", {"vmin": -res_max, "vmax": res_max}),
    ]
    region_boundaries = np.cumsum(y_dims)[:-1] - 0.5
    for ax, (title, mat, cmap, kw) in zip(axes, panels, strict=True):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest", **kw)
        ax.set_title(title)
        ax.set_xlabel("time (bins)")
        ax.set_ylabel("neuron (region-grouped)")
        # Horizontal lines at each region boundary.
        for b in region_boundaries:
            ax.axhline(b, color="black", lw=0.6, alpha=0.4)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)

    fig.suptitle("Per-neuron PSTH comparison (neurons in natural region-grouped order)")
    return fig


# ---------------------------------------------------------------------------
# 5. Trial-0 reconstruction overlay
# ---------------------------------------------------------------------------


def plot_trial0(
    truth_y: np.ndarray,
    fitted_y: np.ndarray,
    *,
    y_dims: tuple[int, ...],
    trial: int = 0,
    n_per_region: int = 4,
    fig: plt.Figure | None = None,
) -> plt.Figure:
    """Overlay truth and fit on a single trial for a sample of neurons.

    One row per region; ``n_per_region`` representative neurons are
    chosen by truth-variance rank so the visible signal is meaningful.
    """
    R = len(y_dims)
    if fig is None:
        fig, axes = plt.subplots(
            R,
            n_per_region,
            figsize=(2.8 * n_per_region, 1.8 * R),
            sharex=True,
            constrained_layout=True,
        )
    else:
        axes = np.array(fig.axes).reshape(R, n_per_region)
    axes = np.atleast_2d(axes)

    cum = [0, *list(np.cumsum(y_dims))]
    for r in range(R):
        start, stop = cum[r], cum[r + 1]
        var = truth_y[:, :, start:stop].var(axis=(0, 1))
        # Pick top-variance neurons within the region for visual signal.
        chosen = np.argsort(var)[-n_per_region:][::-1]
        for slot, n_idx in enumerate(chosen):
            ax = axes[r, slot] if R > 1 else axes[slot]
            t = truth_y[trial, :, start + n_idx]
            f = fitted_y[trial, :, start + n_idx]
            ax.plot(t, "--", color=TRUTH_COLOR, lw=1.2, alpha=0.85)
            ax.plot(f, color=FIT_COLOR, lw=1.3, alpha=0.95)
            if slot == 0:
                ax.set_ylabel(f"region {r}", fontsize=9)
            if r == R - 1:
                ax.set_xlabel("time (bins)")
            ax.set_title(f"neuron {n_idx}", fontsize=8)

    # Single legend at the top.
    legend_handles = [
        plt.Line2D([], [], color=TRUTH_COLOR, ls="--", lw=1.2, label="truth"),
        plt.Line2D([], [], color=FIT_COLOR, lw=1.3, label="fitted"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", ncol=2, fontsize=9)
    fig.suptitle(f"Trial {trial}: per-neuron reconstruction (top-{n_per_region} variance per region)")
    return fig


# ---------------------------------------------------------------------------
# Convenience: align fitted latents to truth (handles latent permutation)
# ---------------------------------------------------------------------------


def restrict_to_ard_active(
    truth_obs: np.ndarray,
    truth_delay: np.ndarray,
    fitted_obs: np.ndarray,
    fitted_delay: np.ndarray,
    alpha_mean: np.ndarray,
    *,
    n_regions: int,
    k_true: int,
    k_init: int,
    n_within: int = 0,
    alpha_prune_ratio: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, int, dict]:
    """For ARD models (mDLAG): subset fitted to the active columns matched
    to truth, then permute so fitted column ``i`` corresponds to truth
    column ``i``. The truth-side subset (in ``info["truth_obs_subset"]``
    / ``info["truth_delay_subset"]``) is what you should overlay against.

    Thin wrapper around :func:`examples.synthetic.demo_common.restrict_to_ard_active`
    so notebooks and CLI demos share one implementation. The notebook
    return signature (``fitted_obs_sub, fitted_delay_sub, K_match,
    info``) is kept for backward compatibility with existing
    notebook cells.
    """
    import sys
    from pathlib import Path

    examples_dir = Path(__file__).resolve().parent.parent / "examples" / "synthetic"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    import demo_common as demo

    (
        _truth_obs_sub,
        _truth_delay_sub,
        fitted_obs_sub,
        fitted_delay_sub,
        K_match,
        info_out,
    ) = demo.restrict_to_ard_active(
        truth_obs,
        truth_delay,
        fitted_obs,
        fitted_delay,
        alpha_mean,
        n_regions=n_regions,
        k_true=k_true,
        k_init=k_init,
        n_within=n_within,
        alpha_prune_ratio=alpha_prune_ratio,
    )
    return fitted_obs_sub, fitted_delay_sub, K_match, info_out


def align_and_repermute(
    truth_obs: np.ndarray,
    fitted_obs: np.ndarray,
    fitted_delay: np.ndarray,
    *,
    n_regions: int,
    n_across: int,
    n_within: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Convenience wrapper around the alignment in :mod:`examples.demo_common`.

    Returns ``(fitted_obs_aligned, fitted_delay_aligned, perm)``.
    Identity permutation when ``n_across <= 1``.
    """
    import sys
    from pathlib import Path

    examples_dir = Path(__file__).resolve().parent.parent / "examples" / "synthetic"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    import demo_common as demo

    perm = demo.align_across_permutation(
        fitted_obs, truth_obs, n_regions=n_regions, n_across=n_across, n_within=n_within
    )
    aligned_obs, aligned_delay = demo.apply_across_permutation(
        fitted_obs,
        fitted_delay,
        perm,
        n_regions=n_regions,
        n_across=n_across,
        n_within=n_within,
    )
    return aligned_obs, aligned_delay, perm


# ---------------------------------------------------------------------------
# Path helper: make examples.demo_common importable from any notebook
# ---------------------------------------------------------------------------


def add_examples_to_path() -> Any:
    """Insert ``examples/synthetic/`` into ``sys.path``.

    Call this near the top of any notebook so ``import demo_common as demo``
    works regardless of where Jupyter was launched from. Synthetic notebooks
    consume the helpers from :mod:`examples.synthetic.demo_common`.
    """
    import sys
    from pathlib import Path

    examples_dir = Path(__file__).resolve().parent.parent / "examples" / "synthetic"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))
    return examples_dir
