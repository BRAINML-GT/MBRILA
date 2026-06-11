"""DLAG-SSM (Kalman engine) on real V1/V2 visual-cortex recordings.

Real-data counterpart of :mod:`examples.synthetic.demo_dlag_ssm`.
DLAG-SSM lifts the dense GP into an AR(P) state-space model and runs
:class:`KalmanEMEngine` (grouped AdamW + cosine LR on joint LL). The
per-iter cost is O(T) (parallel filter+smoother) rather than the
O(T^3) of the dense path.

Pipeline:

1. Load V1/V2 → train/val/test split.
2. pCCA emission init.
3. Optional rank-1 deflation init (off by default — the engine's
   closed-form ``(C, d, R)`` refit + scale anchor in ``fit()`` usually
   handle init mismatch for constant-delay models).
4. Main fit via ``model.fit()``.
5. Co-smoothing on the held-out test split.

Outputs: standard V1V2 layout.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import v1v2_common as v1v2

from mbrila import DLAG, KalmanEMEngine, LatentSpec, MOSEKernel


def main() -> None:
    parser = argparse.ArgumentParser(description="DLAG-SSM (Kalman engine) on real V1/V2 data.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "demo_v1v2_data.pkl",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("examples/v1v2/demo_outputs/dlag_ssm"))
    parser.add_argument("--num-train", type=int, default=300)
    parser.add_argument("--num-val", type=int, default=50)
    parser.add_argument("--num-test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--n-across", type=int, default=2)
    parser.add_argument("--n-within", type=int, default=2)
    parser.add_argument("--lag-across", type=int, default=2)
    parser.add_argument("--lag-within", type=int, default=2)
    parser.add_argument("--init-gamma-across", type=float, default=0.1)
    parser.add_argument("--init-gamma-within", type=float, default=0.1)
    parser.add_argument(
        "--deflation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run rank-1 deflation init before fitting (off by default for const-delay models).",
    )
    parser.add_argument("--deflation-iters-per-round", type=int, default=200)
    parser.add_argument("--num-iters", type=int, default=500)
    parser.add_argument("--lr-deflation", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-min", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
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
        "Default ON to match the notebook convergence figure. Pass --no-track-elbo to fall back to the original cosine-LR fit().",
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

    print(f"[demo_dlag_ssm] device={args.device}  out_dir={args.out_dir}")
    y_all, y_dims = v1v2.load_v1v2(args.data_path, device=args.device, dtype=dtype)
    _n_trials, T, _ = y_all.shape
    print(f"[demo_dlag_ssm] data: y={tuple(y_all.shape)}  y_dims={y_dims}")
    train_data, val_data, test_data = v1v2.make_v1v2_splits(
        y_all,
        y_dims,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        split_seed=args.split_seed,
    )
    print(
        f"[demo_dlag_ssm] split: train={train_data.y.shape[0]}  "
        f"val={val_data.y.shape[0]}  test={test_data.y.shape[0]}"
    )

    n_regions = len(y_dims)
    spec = LatentSpec(n_across=args.n_across, n_within=(args.n_within,) * n_regions)
    engine = KalmanEMEngine(
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
        cosine_anneal=(not args.track_elbo),
    )
    _gamma_a = float(args.init_gamma_across)
    _gamma_w = float(args.init_gamma_within)
    model = DLAG(
        latent_spec=spec,
        y_dims=y_dims,
        T=T,
        kernel_factory_across=lambda: MOSEKernel(num_regions=n_regions, init_sigma=_gamma_a),
        kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=_gamma_w),
        engine="kalman",
        engine_override=engine,
        lag_across=args.lag_across,
        lag_within=args.lag_within,
        device=args.device,
        dtype=dtype,
    ).to(args.device)

    print("[demo_dlag_ssm] initializing from data (pCCA) ...")
    model.initialize_from_data(train_data, mode="pcca")

    if args.deflation and args.deflation_iters_per_round > 0:
        print(f"[demo_dlag_ssm] rank-1 deflation ({args.deflation_iters_per_round} iter per round) ...")
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
            f"[demo_dlag_ssm]   deflation: {info['n_rounds']} rounds; "
            f"per-round losses = {[f'{x:.0f}' for x in info['rank1_losses']]}"
        )

    print(f"[demo_dlag_ssm] fitting ({args.num_iters} iters) ...")
    t0 = time.perf_counter()
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
    wall_s = time.perf_counter() - t0
    print(f"[demo_dlag_ssm] fit done in {wall_s:.1f}s  ({n_iter} iters)")

    train_trace = np.column_stack([score_iters, score_values / max(int(train_data.y.shape[0]), 1)])
    with torch.no_grad():
        val_ll = float(engine._marginal_ll(model, val_data).item())
    val_trace = np.asarray([[n_iter, val_ll / max(int(val_data.y.shape[0]), 1)]], dtype=float)

    fitted_delay = v1v2.extract_delay(model, T)
    fitted_obs = v1v2.extract_observable(model, train_data)

    region_names = ["V1", "V2"] if n_regions == 2 else [f"region_{r}" for r in range(n_regions)]
    v1v2.print_latent_diagnostics(
        method_name="demo_dlag_ssm",
        fitted_obs=fitted_obs,
        fitted_delay=fitted_delay,
        train_y=train_data.y.detach().cpu().numpy(),
        n_regions=n_regions,
        n_across=args.n_across,
        n_within=args.n_within,
        region_names=region_names,
    )

    print(
        f"[demo_dlag_ssm] co-smoothing on TEST  "
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
        method_name="dlag_ssm",
        model_label="DLAG-SSM",
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
    v1v2.print_summary(summary, "dlag_ssm", region_names)


if __name__ == "__main__":
    main()
