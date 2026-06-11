"""ADM on real V1/V2 visual-cortex recordings.

Real-data counterpart of :mod:`examples.synthetic.demo_adm`. Two regions
(V1 / V2) of binned spike counts. There is no ground-truth delay or
latent, so the headline metric is held-out-neuron co-smoothing RMSE
on the test split — see :func:`v1v2_common.co_smoothing_rmse`.

Pipeline:

1. Load the V1/V2 pickle and split into train / val / test.
2. pCCA emission init via ``model.initialize_from_data(mode="pcca")``.
3. Rank-1 deflation init (one round per across-latent). ADM's
   time-varying-delay parameter space is high-dim and pCCA alone
   tends to leave latents merged.
4. Main fit via ``model.fit()`` — :class:`KalmanEMEngine` runs the
   closed-form ``(C, d, R)`` LSE refit, initial scale anchor,
   grouped-AdamW with cosine LR, and a final scale anchor.
5. Co-smoothing on the held-out test split.

Outputs: convergence trace (train + val), fitted delay over time,
smoother latents (trial 0), PSTH and trial-0 reconstruction heatmaps,
``eval_metrics.txt`` (co-smoothing RMSE), ``summary.json``, ``snapshots.npz``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import v1v2_common as v1v2

from mbrila import ADM, KalmanEMEngine, LatentSpec, MOSEKernel


def main() -> None:
    parser = argparse.ArgumentParser(description="ADM on real V1/V2 data.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "demo_v1v2_data.pkl",
        help="Pickle of the V1/V2 recordings (dict with 'V1' / 'V2' arrays).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("examples/v1v2/demo_outputs/adm"))
    # Data split
    parser.add_argument("--num-train", type=int, default=300)
    parser.add_argument("--num-val", type=int, default=50)
    parser.add_argument("--num-test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0, help="Model-init / optimizer seed.")
    parser.add_argument("--split-seed", type=int, default=0, help="Train/val/test shuffle seed.")
    # Latent structure
    parser.add_argument("--n-across", type=int, default=2)
    parser.add_argument("--n-within", type=int, default=2)
    parser.add_argument("--lag-across", type=int, default=2)
    parser.add_argument("--lag-within", type=int, default=2)
    parser.add_argument("--init-sigma-across", type=float, default=0.01)
    parser.add_argument("--init-sigma-within", type=float, default=0.1)
    # Fit
    parser.add_argument(
        "--deflation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run rank-1 deflation init before fitting (on by default for ADM).",
    )
    parser.add_argument("--deflation-iters-per-round", type=int, default=200)
    parser.add_argument("--num-iters", type=int, default=500)
    parser.add_argument("--lr-deflation", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-min", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    # Co-smoothing
    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=0,
        help="Base seed for holdout-neuron splits; used as base of --n-holdout-seeds.",
    )
    parser.add_argument(
        "--n-holdout-seeds", type=int, default=3, help="Number of holdout-neuron splits to average over."
    )
    parser.add_argument(
        "--track-elbo",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run model.fit() in chunks and record true ELBO = "
        "joint_ll + H[q] at each chunk boundary (monotone diagnostic). "
        "Requires cosine_anneal=False on the engine (auto-applied when on). "
        "Adds ~5-10s overhead per fit. Default ON to match the notebook "
        "convergence figure. Pass --no-track-elbo to fall back to the "
        "original cosine-LR fit() with raw joint_ll trace.",
    )
    parser.add_argument(
        "--elbo-check-every",
        type=int,
        default=10,
        help="(only with --track-elbo) ELBO checkpoint cadence in iters.",
    )
    parser.add_argument(
        "--keep-scale-anchor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="(only with --track-elbo) Re-anchor latent scales at every chunk boundary. "
        "Adds ~2x smoother per chunk (small overhead) but keeps the C-norm and "
        "latent variance bounded across training, making the ELBO curve closer "
        "to monotone. Default OFF (empirically no effect on V1V2).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64

    print(f"[demo_adm] device={args.device}  out_dir={args.out_dir}")
    y_all, y_dims = v1v2.load_v1v2(args.data_path, device=args.device, dtype=dtype)
    _n_trials, T, _ = y_all.shape
    print(f"[demo_adm] data: y={tuple(y_all.shape)}  y_dims={y_dims}")
    train_data, val_data, test_data = v1v2.make_v1v2_splits(
        y_all,
        y_dims,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        split_seed=args.split_seed,
    )
    print(
        f"[demo_adm] split: train={train_data.y.shape[0]}  val={val_data.y.shape[0]}  test={test_data.y.shape[0]}"
    )

    n_regions = len(y_dims)
    spec = LatentSpec(n_across=args.n_across, n_within=(args.n_within,) * n_regions)
    # When --track-elbo is on, the fit is chunked so cosine_anneal must be
    # off (otherwise LR restarts on every chunk → stair-step schedule).
    # Default behavior unchanged otherwise.
    engine = KalmanEMEngine(
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        cosine_anneal=(not args.track_elbo),
        log_every=args.log_every,
    )
    _sigma_a = float(args.init_sigma_across)
    _sigma_w = float(args.init_sigma_within)
    model = ADM(
        latent_spec=spec,
        y_dims=y_dims,
        T=T,
        kernel_factory_across=lambda: MOSEKernel(num_regions=n_regions, init_sigma=_sigma_a),
        kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=_sigma_w),
        lag_across=args.lag_across,
        lag_within=args.lag_within,
        delay_smoothing_sigma_across=_sigma_a,
        engine=engine,
        device=args.device,
        dtype=dtype,
    ).to(args.device)

    print("[demo_adm] phase 2: pCCA emission init ...")
    t_init = time.perf_counter()
    model.initialize_from_data(train_data, mode="pcca")

    if args.deflation and args.deflation_iters_per_round > 0:
        print(f"[demo_adm] phase 3: rank-1 deflation ({args.deflation_iters_per_round} iter per round) ...")
        info = v1v2.rank1_deflation_init(
            model,
            train_data,
            n_iters_per_round=args.deflation_iters_per_round,
            lr=args.lr_deflation,
            lr_min=args.lr_min,
            weight_decay=args.weight_decay,
            verbose=True,
        )
        print(
            f"[demo_adm]   deflation: {info['n_rounds']} rounds; "
            f"per-round losses = {[f'{x:.0f}' for x in info['rank1_losses']]}"
        )
    else:
        print("[demo_adm] phase 3: deflation skipped (--no-deflation)")

    print(f"[demo_adm] phase 4: main fit ({args.num_iters} iters) ...")
    t_fit = time.perf_counter()
    elbo_checkpoints: list[tuple[int, float, float, float]] | None = None
    if args.track_elbo:
        elbo_info = v1v2.fit_with_elbo_tracking(
            model=model,
            train_data=train_data,
            engine=engine,
            num_iters=args.num_iters,
            check_every=args.elbo_check_every,
            tol=1e-8,
            keep_scale_anchor=args.keep_scale_anchor,
        )
        elbo_checkpoints = elbo_info["elbo_checkpoints"]
        score_iters = np.asarray([c[0] for c in elbo_checkpoints], dtype=float)
        score_values = np.asarray([c[3] for c in elbo_checkpoints], dtype=float)
        n_iter = int(elbo_info["n_iter"])
        for it, jll, ent, elbo in elbo_checkpoints:
            print(f"  [track_elbo] iter {it:4d}  joint_ll={jll:.1f}  H[q]={ent:.1f}  ELBO={elbo:.1f}")
    else:
        result = model.fit(train_data, max_iter=args.num_iters, tol=1e-8)
        score_values = np.asarray(result.score_trace, dtype=float)
        score_iters = np.arange(1, len(score_values) + 1, dtype=float)
        n_iter = int(result.n_iter)
    wall_s = time.perf_counter() - t_fit
    print(f"[demo_adm] fit done in {wall_s:.1f}s  ({n_iter} iters)  (init wall = {t_fit - t_init:.1f}s)")

    # Train trace = per-iter joint_ll_em (raw). When --track-elbo is on,
    # we additionally have the true ELBO at sparse checkpoints.
    train_trace = np.column_stack([score_iters, score_values / max(int(train_data.y.shape[0]), 1)])
    with torch.no_grad():
        val_ll = float(engine._marginal_ll(model, val_data).item())
    val_trace = np.asarray([[n_iter, val_ll / max(int(val_data.y.shape[0]), 1)]], dtype=float)

    fitted_delay = v1v2.extract_delay(model, T)
    fitted_obs = v1v2.extract_observable(model, train_data)

    region_names = ["V1", "V2"] if n_regions == 2 else [f"region_{r}" for r in range(n_regions)]
    v1v2.print_latent_diagnostics(
        method_name="demo_adm",
        fitted_obs=fitted_obs,
        fitted_delay=fitted_delay,
        train_y=train_data.y.detach().cpu().numpy(),
        n_regions=n_regions,
        n_across=args.n_across,
        n_within=args.n_within,
        region_names=region_names,
    )

    print(
        f"[demo_adm] phase 5: co-smoothing on TEST  "
        f"(frac={args.holdout_frac}, seeds={args.holdout_seed}..+{args.n_holdout_seeds - 1}) ..."
    )
    holdout_seeds = [args.holdout_seed + i for i in range(args.n_holdout_seeds)]
    cosmoothing = v1v2.co_smoothing_eval_multiseed(
        model=model,
        data=test_data,
        y_dims=y_dims,
        holdout_frac=args.holdout_frac,
        holdout_seeds=holdout_seeds,
    )

    summary = v1v2.write_v1v2_outputs(
        method_name="adm",
        model_label="ADM",
        train_data=train_data,
        fitted_delay=fitted_delay,
        fitted_obs=fitted_obs,
        y_dims=y_dims,
        n_across=args.n_across,
        n_within=args.n_within,
        train_trace=train_trace,
        val_trace=val_trace,
        score_ylabel="joint log-likelihood / trial",
        cosmoothing=cosmoothing,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
        split_seed=args.split_seed,
        wall_s=wall_s,
        out_dir=args.out_dir,
        region_names=region_names,
    )
    v1v2.print_summary(summary, "adm", region_names)


if __name__ == "__main__":
    main()
