"""Compare ADM / DLAG / mDLAG runs on V1V2 data across seeds.

Reads every ``eval_metrics.txt`` produced by :mod:`demo_v1v2` underneath
each supplied run directory (one per seed per model) and produces:

  - ``summary.txt``   table with mean ± std per model per metric
  - ``bars.png``      grouped bar chart; the y-axis of each panel is
                      zoomed to the data range so small but consistent
                      differences between models are visible

What this compares
------------------
Real V1V2 data has no ground-truth latent / delay, so the comparison is
on held-out-neuron co-smoothing RMSE, per region (lower is better):

  holdout_psth_rmse_*       ‖trial-mean(y) − trial-mean(ŷ)‖ on held-out
                            neurons — stimulus-locked signal recovery.
  holdout_per_trial_rmse_*  trial-by-trial ‖y − ŷ‖ on held-out neurons.

For each metric we report mean ± std across seeds. Seeds vary the model
init only (the data split is fixed by ``--split-seed`` in
:mod:`demo_v1v2`), so the spread is purely optimisation variance.

Usage
-----
::

    uv run python scratch/compare_v1v2_runs.py \\
        --label adm          --runs "scratch/v1v2_adm/seed_*" \\
        --label dlag         --runs "scratch/v1v2_dlag/seed_*" \\
        --label mdlag_time   --runs "scratch/v1v2_mdlag_time/seed_*" \\
        --label mdlag_freq   --runs "scratch/v1v2_mdlag_freq/seed_*" \\
        --label mdlag_kalman --runs "scratch/v1v2_mdlag_kalman/seed_*" \\
        --out-dir scratch/v1v2_compare

Each ``--label/--runs`` pair declares one model. The 5-way variant above
is the CF6d post-mDLAG-SSM comparison; the 4-way (without
``mdlag_kalman``) is the historical baseline (CLAUDE.md §B.6).
"""

from __future__ import annotations

import argparse
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_metrics_file(path: Path) -> dict[str, str]:
    """Parse a ``key: value`` text file written by :mod:`demo_v1v2`."""
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _collect_runs(label: str, pattern: str) -> list[dict[str, str]]:
    """Glob ``pattern`` and parse every ``eval_metrics.txt`` underneath."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise SystemExit(f"[{label}] glob {pattern!r} matched no directories")
    rows: list[dict[str, str]] = []
    for run_dir in matches:
        metrics_path = Path(run_dir) / "eval_metrics.txt"
        if not metrics_path.exists():
            print(f"[{label}] skipping {run_dir}: no eval_metrics.txt")
            continue
        info = _parse_metrics_file(metrics_path)
        info["__path__"] = run_dir
        info["__label__"] = label
        rows.append(info)
    if not rows:
        raise SystemExit(f"[{label}] no eval_metrics.txt found under {pattern!r}")
    return rows


def _numeric_keys(rows: list[dict[str, str]]) -> list[str]:
    """Return keys whose values are floats in every row."""
    if not rows:
        return []
    sample = rows[0]
    candidates: list[str] = []
    for k, v in sample.items():
        if k.startswith("_"):
            continue
        try:
            float(v)
        except ValueError:
            continue
        if all(_can_parse_float(r.get(k)) for r in rows):
            candidates.append(k)
    return candidates


def _can_parse_float(s: str | None) -> bool:
    if s is None:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def _agg(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0), int(arr.size)


def _write_summary(
    grouped: dict[str, list[dict[str, str]]],
    metric_keys: list[str],
    out_path: Path,
) -> None:
    lines: list[str] = ["# V1V2 comparison summary (mean ± std across seeds)"]
    lines.append("")
    lines.append(f"models: {', '.join(grouped.keys())}")
    for label, rows in grouped.items():
        seeds = [r.get("seed", "?") for r in rows]
        lines.append(f"  [{label}]  n_runs={len(rows)}  seeds={seeds}")
    lines.append("")
    width_metric = max(len("metric"), *(len(k) for k in metric_keys))
    header = f"{'metric':<{width_metric}}"
    for label in grouped:
        header += f"  {label:>26}"
    lines.append(header)
    lines.append("-" * len(header))
    for key in metric_keys:
        row = f"{key:<{width_metric}}"
        for rows in grouped.values():
            vals = [float(r[key]) for r in rows if r.get(key) is not None]
            mean, std, n = _agg(vals)
            row += f"  {mean:>+12.4f} ± {std:>8.4f}  (n={n})"
        lines.append(row)
    out_path.write_text("\n".join(lines) + "\n")


def _plot_bars(
    grouped: dict[str, list[dict[str, str]]],
    metric_keys: list[str],
    out_path: Path,
) -> None:
    """Grouped bar chart, one panel per metric, one bar per model.

    The y-axis of each panel is zoomed to the data range (rather than
    anchored at 0): co-smoothing RMSE values are close across models, so
    a 0-anchored axis makes the bars look identical. Each panel is
    clipped to ``[min − pad, max + pad]`` over the model mean ± std so
    the (small but consistent) differences are visible.
    """
    n_panels = len(metric_keys)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    labels = list(grouped.keys())
    x = np.arange(len(labels))

    for i, key in enumerate(metric_keys):
        ax = axes[i // ncols, i % ncols]
        means: list[float] = []
        stds: list[float] = []
        for label in labels:
            vals = [float(r[key]) for r in grouped[label] if r.get(key) is not None]
            mean, std, _ = _agg(vals)
            means.append(mean)
            stds.append(std)
        ax.bar(
            x,
            means,
            yerr=stds,
            capsize=4,
            color=["steelblue", "darkred", "darkgreen", "darkorange"][: len(labels)],
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(key, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Zoom the y-axis to the data range so close values are separable.
        lo = min(m - s for m, s in zip(means, stds, strict=True))
        hi = max(m + s for m, s in zip(means, stds, strict=True))
        span = hi - lo
        if span < 1e-9:  # all models identical — fall back to a small window
            span = max(abs(hi), 1e-3)
        pad = 0.25 * span
        ax.set_ylim(lo - pad, hi + pad)

    # Hide unused panels.
    for j in range(len(metric_keys), nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle("Held-out co-smoothing comparison", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--label",
        action="append",
        required=True,
        help="Model label for the following --runs (e.g. 'adm', 'dlag'). "
        "Provide one --label / --runs pair per model.",
    )
    p.add_argument(
        "--runs",
        action="append",
        required=True,
        help="Glob pattern matching run directories for the most recent "
        "--label. Each match must contain an eval_metrics.txt.",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    if len(args.label) != len(args.runs):
        raise SystemExit(
            f"--label given {len(args.label)} times but --runs given "
            f"{len(args.runs)} times; each --label needs a matching --runs"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Collect rows per label.
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for label, pattern in zip(args.label, args.runs, strict=True):
        rows = _collect_runs(label, pattern)
        grouped[label].extend(rows)
        print(f"  [{label}] loaded {len(rows)} runs from {pattern!r}")

    # Decide which numeric keys to compare — intersection across models.
    keys_per_model = [set(_numeric_keys(rows)) for rows in grouped.values()]
    metric_keys = sorted(set.intersection(*keys_per_model))
    if not metric_keys:
        raise SystemExit("no common numeric metrics across models")

    # Plot the co-smoothing (held-out neuron) RMSE metrics — the primary
    # discriminator on real data. PSTH RMSE first, then per-trial RMSE.
    priority = [k for k in metric_keys if k.startswith("holdout_psth_rmse_")]
    priority += [k for k in metric_keys if k.startswith("holdout_per_trial_rmse_")]
    plot_keys = priority if priority else metric_keys[:6]

    _write_summary(grouped, metric_keys, args.out_dir / "summary.txt")
    _plot_bars(grouped, plot_keys, args.out_dir / "bars.png")

    print(f"  → wrote summary.txt + bars.png to {args.out_dir}/")
    print()
    print(args.out_dir.joinpath("summary.txt").read_text())


if __name__ == "__main__":
    main()
