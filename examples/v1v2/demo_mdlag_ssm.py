"""mDLAG-SSM (Kalman + ARD) on real V1/V2 visual-cortex recordings.

Real-data counterpart of :mod:`examples.synthetic.demo_mdlag_ssm`.
mDLAG-SSM lifts the dense GP into an AR(P) state-space model and runs
:class:`VEMKalmanARDEngine`: ARD on C-columns (as in the dense mDLAG
paths) plus parallel Kalman filter+smoother for the latent E-step,
with Adam-based M-step on the GP / delay hyperparameters.

Pipeline:

1. Load V1/V2 → train/val/test split.
2. pCCA emission init via ``model.initialize_from_data()``.
3. Fit via ``model.fit()``.
4. Co-smoothing on the held-out test split.

Outputs: standard V1V2 layout plus ``ard_alpha.png``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import v1v2_common as v1v2

from mbrila import MDLAG, ARDPriorConfig, LatentSpec, MOSEKernel, VEMKalmanARDEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="mDLAG-SSM (Kalman + ARD) on real V1/V2 data.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "demo_v1v2_data.pkl",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("examples/v1v2/demo_outputs/mdlag_ssm"))
    parser.add_argument("--num-train", type=int, default=300)
    parser.add_argument("--num-val", type=int, default=50)
    parser.add_argument("--num-test", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--n-across", type=int, default=3, help="model K_a (ARD soft-prunes spurious)")
    parser.add_argument("--init-gamma-across", type=float, default=0.1)
    parser.add_argument("--num-iters", type=int, default=500)
    parser.add_argument("--lag-across", type=int, default=2, help="Markov order P for the AR(P) lift.")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-min", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gp-steps", type=int, default=1, help="Adam steps per outer EM iter on GP/delay.")
    parser.add_argument("--ard-prior-shape", type=float, default=1e-3)
    parser.add_argument("--ard-prior-rate", type=float, default=1e-3)
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
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float64

    print(f"[demo_mdlag_ssm] device={args.device}  out_dir={args.out_dir}")
    y_all, y_dims = v1v2.load_v1v2(args.data_path, device=args.device, dtype=dtype)
    _n_trials, T, _ = y_all.shape
    print(f"[demo_mdlag_ssm] data: y={tuple(y_all.shape)}  y_dims={y_dims}")
    train_data, val_data, test_data = v1v2.make_v1v2_splits(
        y_all,
        y_dims,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        split_seed=args.split_seed,
    )
    print(
        f"[demo_mdlag_ssm] split: train={train_data.y.shape[0]}  "
        f"val={val_data.y.shape[0]}  test={test_data.y.shape[0]}"
    )

    n_regions = len(y_dims)
    spec = LatentSpec(
        n_across=args.n_across,
        n_within=(0,) * n_regions,
        selection="ard",
        ard_prior=ARDPriorConfig(shape=args.ard_prior_shape, rate=args.ard_prior_rate),
    )
    engine = VEMKalmanARDEngine(
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        gp_steps_per_em=args.gp_steps,
        log_every=args.log_every,
    )
    _gamma_a = float(args.init_gamma_across)
    model = MDLAG(
        latent_spec=spec,
        y_dims=y_dims,
        T=T,
        kernel_factory_across=lambda: MOSEKernel(num_regions=n_regions, init_sigma=_gamma_a),
        engine="kalman",
        engine_override=engine,
        lag_across=args.lag_across,
        device=args.device,
        dtype=dtype,
    ).to(args.device)

    print("[demo_mdlag_ssm] initializing from data (pCCA) ...")
    model.initialize_from_data(train_data)

    print(f"[demo_mdlag_ssm] fitting ({args.num_iters} iters) ...")
    t0 = time.perf_counter()
    result = model.fit(train_data, max_iter=args.num_iters, tol=1e-8)
    wall_s = time.perf_counter() - t0
    print(f"[demo_mdlag_ssm] fit done in {wall_s:.1f}s  ({result.n_iter} iters)")

    train_trace = np.column_stack(
        [
            np.arange(1, len(result.score_trace) + 1, dtype=float),
            np.asarray(result.score_trace, dtype=float) / max(int(train_data.y.shape[0]), 1),
        ]
    )
    with torch.no_grad():
        val_elbo = float(engine.score(model, val_data))
    val_trace = np.asarray([[result.n_iter, val_elbo / max(int(val_data.y.shape[0]), 1)]], dtype=float)

    fitted_delay = v1v2.extract_delay(model, T)
    fitted_obs = v1v2.extract_observable(model, train_data)
    alpha_mean = v1v2.extract_ard_alpha(model)
    max_alpha = alpha_mean.max(axis=0).tolist() if alpha_mean is not None else None
    print(f"[demo_mdlag_ssm] max α per latent column: {max_alpha}")

    region_names = ["V1", "V2"] if n_regions == 2 else [f"region_{r}" for r in range(n_regions)]
    v1v2.print_latent_diagnostics(
        method_name="demo_mdlag_ssm",
        fitted_obs=fitted_obs,
        fitted_delay=fitted_delay,
        train_y=train_data.y.detach().cpu().numpy(),
        n_regions=n_regions,
        n_across=args.n_across,
        n_within=0,
        region_names=region_names,
        alpha_mean=alpha_mean,
    )

    print(
        f"[demo_mdlag_ssm] co-smoothing on TEST  "
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
        method_name="mdlag_ssm",
        model_label="mDLAG-SSM",
        train_data=train_data,
        fitted_delay=fitted_delay,
        fitted_obs=fitted_obs,
        y_dims=y_dims,
        n_across=args.n_across,
        n_within=0,
        train_trace=train_trace,
        val_trace=val_trace,
        score_ylabel="proxy ELBO / trial",
        cosmoothing=cosmoothing,
        holdout_frac=args.holdout_frac,
        seed=args.seed,
        split_seed=args.split_seed,
        wall_s=wall_s,
        out_dir=args.out_dir,
        alpha_mean=alpha_mean,
        region_names=region_names,
        extra_summary=f"max_alpha_per_col={max_alpha}",
    )
    v1v2.print_summary(summary, "mdlag_ssm", region_names)


if __name__ == "__main__":
    main()
