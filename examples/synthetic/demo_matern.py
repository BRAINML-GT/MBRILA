"""DLAG-SSM with Matérn-5/2 kernel.

Companion to :mod:`examples.demo_dlag_ssm` (which uses MOSE/RBF) —
same scenario and same engine, only the kernel changes. Demonstrates
that any :class:`~mbrila.kernels.base.BaseKernel` subclass plugs into
the Markov-GP SSM path via ``kernel_factory_*``.

:class:`mbrila.Matern52Kernel` has an **exact finite-dimensional SDE**
(state dim = 3), so the Kalman engine consumes it directly via
``sde_form()`` with no AR(P) approximation. The kernel is light-tailed
(``(1 + √5τ/ℓ + ...) · exp(-√5τ/ℓ)``), giving sharp lengthscale
discrimination similar to RBF.

Outputs: standard layout (convergence trace, per-pair delay overlays,
per-region latent traces, y reconstruction, summary.json).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import demo_common as demo
import numpy as np
import torch

from mbrila import DLAG, KalmanEMEngine, LatentSpec, Matern52Kernel
from mbrila.synthetic.multiregion import MultiRegionScenario


def build_scenario(args: argparse.Namespace) -> MultiRegionScenario:
    """Constant-delay multi-region scenario."""
    return MultiRegionScenario(
        n_trials=args.n_trials,
        T=args.T,
        y_dims=tuple([args.y_dim_per_region] * args.n_regions),
        n_across=args.n_across,
        n_within=args.n_within,
        lag_across=args.lag_across,
        lag_within=args.lag_within,
        sigma_across=args.sigma_across,
        sigma_within=args.sigma_within,
        delay_shape="constant",
        delay_amplitude=args.delay_amplitude,
        per_latent_amplitudes=tuple(
            args.delay_amplitude * (1.0 / args.per_latent_amp_ratio ** (k / max(args.n_across - 1, 1)))
            for k in range(args.n_across)
        ),
        per_latent_sigma_across=tuple(
            args.sigma_across * (args.per_latent_sigma_ratio ** (k / max(args.n_across - 1, 1)))
            for k in range(args.n_across)
        ),
        per_latent_shapes=tuple(["constant"] * args.n_across),
        region_heterogeneity=args.region_heterogeneity,
        snr=args.snr,
        seed=args.data_seed,
        dtype=torch.float64,
        device="cpu",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DLAG-SSM with Matérn-5/2 kernel.")
    parser.add_argument("--out-dir", type=Path, default=Path("examples/synthetic/demo_outputs/matern"))
    # Scenario
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--n-regions", type=int, default=5)
    parser.add_argument("--y-dim-per-region", type=int, default=100)
    parser.add_argument("--n-across", type=int, default=2)
    parser.add_argument("--n-within", type=int, default=1)
    parser.add_argument("--lag-across", type=int, default=2)
    parser.add_argument("--lag-within", type=int, default=2)
    parser.add_argument("--sigma-across", type=float, default=0.05)
    parser.add_argument("--sigma-within", type=float, default=0.05)
    parser.add_argument("--delay-amplitude", type=float, default=3.0)
    parser.add_argument("--per-latent-sigma-ratio", type=float, default=10.0)
    parser.add_argument("--per-latent-amp-ratio", type=float, default=3.0)
    parser.add_argument("--region-heterogeneity", type=float, default=1.0)
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--data-seed", type=int, default=0)
    # Fit
    parser.add_argument("--num-iters", type=int, default=300)
    # Matérn-5/2 lengthscale init. Truth (RBF) per-latent timescale ≈ 1/√σ:
    # σ=0.05 → ℓ_eff ≈ 4.5; σ=0.5 → ℓ_eff ≈ 1.4. Init at 2 sits between.
    parser.add_argument("--init-lengthscale-across", type=float, default=2.0)
    parser.add_argument("--init-lengthscale-within", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-min", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    # Rank-1 deflation init. Data-driven per-latent kernel + δ + C seed,
    # bypassing the "init lengthscale must match a truth latent" trap.
    # On by default; --no-deflation for A/B comparison.
    parser.add_argument(
        "--deflation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run rank-1 deflation init before fitting (off by default). "
        "For const-delay models the engine's closed-form (C, d, R) refit and "
        "scale anchor in fit() already handle init mismatch; pass --deflation "
        "as belt-and-suspenders if needed.",
    )
    parser.add_argument("--deflation-iters-per-round", type=int, default=200)
    parser.add_argument("--lr-deflation", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demo_matern] device={args.device}  out_dir={args.out_dir}")
    scenario = build_scenario(args)
    print(
        f"[demo_matern] scenario: T={scenario.T} B={scenario.n_trials} "
        f"R={len(scenario.y_dims)} K_a={scenario.n_across} K_w={scenario.n_within} "
        f"shape=constant amp={scenario.delay_amplitude} snr={scenario.snr}"
    )

    data, truth = demo.sample_scenario(scenario, device=args.device)

    spec = LatentSpec(
        n_across=scenario.n_across,
        n_within=(scenario.n_within,) * len(scenario.y_dims),
    )
    engine = KalmanEMEngine(
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        cosine_anneal=True,
        log_every=args.log_every,
    )
    _ell_a = float(args.init_lengthscale_across)
    _ell_w = float(args.init_lengthscale_within)
    model = DLAG(
        latent_spec=spec,
        y_dims=scenario.y_dims,
        T=scenario.T,
        kernel_factory_across=lambda: Matern52Kernel(lengthscale=_ell_a),
        kernel_factory_within=lambda: Matern52Kernel(lengthscale=_ell_w),
        engine="kalman",
        engine_override=engine,
        device=args.device,
        dtype=torch.float64,
    ).to(args.device)

    print("[demo_matern] initializing from data (pCCA) ...")
    model.initialize_from_data(data, mode="pcca")

    if args.deflation and args.deflation_iters_per_round > 0:
        print(
            f"[demo_matern] rank-1 deflation init "
            f"({args.deflation_iters_per_round} iter × {scenario.n_across} rounds) ..."
        )
        info = demo.rank1_deflation_init(
            model,
            data,
            n_iters_per_round=args.deflation_iters_per_round,
            lr=args.lr_deflation,
            lr_min=args.lr_min,
            verbose=True,
        )
        print(f"[demo_matern]   per-round losses = {[f'{x:.0f}' for x in info['rank1_losses']]}")
    else:
        print("[demo_matern] deflation skipped (--no-deflation)")

    print(f"[demo_matern] fitting ({args.num_iters} iters) ...")
    t_fit = time.perf_counter()
    result = model.fit(data, max_iter=args.num_iters, tol=1e-8)
    wall_s = time.perf_counter() - t_fit
    print(f"[demo_matern] fit done in {wall_s:.1f}s  ({result.n_iter} iters)")

    fitted_delay = demo.extract_delay(model, scenario.T)
    fitted_obs = demo.extract_observable(model, data)
    fitted_y = demo.extract_y_recon(model, data)

    record = demo.write_method_outputs(
        method_name="matern",
        model_label="DLAG-SSM + Matérn-5/2",
        truth=truth,
        fitted_delay=fitted_delay,
        fitted_obs=fitted_obs,
        fitted_y=fitted_y,
        score_trace=np.asarray(result.score_trace, dtype=float),
        score_ylabel="joint log-likelihood",
        wall_s=wall_s,
        out_dir=args.out_dir,
    )
    demo.print_summary(record, "matern")


if __name__ == "__main__":
    main()
