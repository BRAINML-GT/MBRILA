"""ADM on a multi-region scenario with time-varying inter-region delay.

Scenario: 5-region synthetic data with K_a=2 across-latents, Gaussian
time-varying delays δ(t), heterogeneous per-latent timescales and
amplitudes. ADM's :class:`TimeVaryingDelay` tracks δ(t) over time —
the ``delay_lat*.png`` figure shows that tracking against the truth.

Pipeline:

1. Sample data from the multi-region scenario.
2. pCCA emission init via ``model.initialize_from_data(mode="pcca")``.
3. Rank-1 deflation init (one round per across-latent) to seed each
   block's ``(σ_k, δ_k, C_{:,k})`` from the residual. Without this,
   the joint M-step easily gets stuck in a "merged-latents" basin
   under heterogeneous σ.
4. Main fit via ``model.fit()`` — :class:`KalmanEMEngine` runs a
   closed-form ``(C, d, R)`` LSE refit, a latent-scale anchor, then
   grouped-AdamW with cosine LR on the joint log-likelihood, and a
   final scale anchor.

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

from mbrila import ADM, KalmanEMEngine, LatentSpec, MOSEKernel
from mbrila.synthetic.multiregion import MultiRegionScenario


def build_scenario(args: argparse.Namespace) -> MultiRegionScenario:
    """Multi-region scenario with Gaussian time-varying delay."""
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
        delay_shape="gaussian",
        delay_amplitude=args.delay_amplitude,
        per_latent_amplitudes=tuple(
            args.delay_amplitude * (1.0 / args.per_latent_amp_ratio ** (k / max(args.n_across - 1, 1)))
            for k in range(args.n_across)
        ),
        per_latent_sigma_across=tuple(
            args.sigma_across * (args.per_latent_sigma_ratio ** (k / max(args.n_across - 1, 1)))
            for k in range(args.n_across)
        ),
        per_latent_shapes=tuple(["gaussian"] * args.n_across),
        region_heterogeneity=args.region_heterogeneity,
        snr=args.snr,
        seed=args.data_seed,
        dtype=torch.float64,
        device="cpu",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ADM on multi-region synthetic data with time-varying delay."
    )
    parser.add_argument("--out-dir", type=Path, default=Path("examples/synthetic/demo_outputs/adm"))
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
    parser.add_argument("--init-sigma-across", type=float, default=0.1)
    parser.add_argument("--init-sigma-within", type=float, default=0.1)
    parser.add_argument(
        "--deflation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run rank-1 deflation init before fitting (on by default for ADM). "
        "The time-varying-delay parameter space is high-dim and pCCA alone "
        "tends to leave latents merged; deflation seeds each latent's δ(t) "
        "and σ separately. Pass --no-deflation only as a diagnostic.",
    )
    parser.add_argument("--deflation-iters-per-round", type=int, default=200)
    parser.add_argument("--num-iters", type=int, default=300)
    parser.add_argument("--lr-deflation", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--lr-min", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demo_adm] device={args.device}  out_dir={args.out_dir}")
    scenario = build_scenario(args)
    print(
        f"[demo_adm] scenario: T={scenario.T} B={scenario.n_trials} "
        f"R={len(scenario.y_dims)} K_a={scenario.n_across} K_w={scenario.n_within} "
        f"shape=gaussian amp={scenario.delay_amplitude} snr={scenario.snr}"
    )

    data, truth = demo.sample_scenario(scenario, device=args.device)

    spec = LatentSpec(
        n_across=scenario.n_across,
        n_within=(scenario.n_within,) * len(scenario.y_dims),
    )
    # Engine defaults give grouped AdamW (excludes raw_delay/beta/d_param
    # from weight decay) + closed-form (C, d, R) LSE refit + scale anchor
    # at start/end + cosine LR.
    engine = KalmanEMEngine(
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        log_every=args.log_every,
    )
    _sigma_a = float(args.init_sigma_across)
    _sigma_w = float(args.init_sigma_within)
    _n_regions = len(scenario.y_dims)
    model = ADM(
        latent_spec=spec,
        y_dims=scenario.y_dims,
        T=scenario.T,
        kernel_factory_across=lambda: MOSEKernel(num_regions=_n_regions, init_sigma=_sigma_a),
        kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=_sigma_w),
        lag_across=args.lag_across,
        lag_within=args.lag_within,
        delay_smoothing_sigma_across=_sigma_a,
        engine=engine,
        device=args.device,
        dtype=torch.float64,
    ).to(args.device)

    # --- Phase 2: pCCA emission init ----------------------------------
    print("[demo_adm] phase 2: pCCA emission init ...")
    t_init = time.perf_counter()
    model.initialize_from_data(data, mode="pcca")

    # --- Phase 3: rank-1 deflation per across-latent ------------------
    if args.deflation and args.deflation_iters_per_round > 0:
        print(f"[demo_adm] phase 3: rank-1 deflation ({args.deflation_iters_per_round} iter per round) ...")
        info = demo.rank1_deflation_init(
            model,
            data,
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

    # --- Phase 4: main fit -------------------------------------------
    # ``model.fit()`` runs KalmanEMEngine.fit:
    #   (a) one-shot closed-form (C, d, R) LSE refit
    #   (b) initial scale anchor
    #   (c) max_iter steps of grouped-AdamW + cosine LR on joint-LL
    #   (d) final scale anchor
    # All four are toggleable via engine kwargs (closed_form_obs_refit /
    # scale_anchor / grouped_weight_decay / cosine_anneal).
    print(f"[demo_adm] phase 4: main fit ({args.num_iters} iters) ...")
    t_fit = time.perf_counter()
    result = model.fit(data, max_iter=args.num_iters, tol=1e-8)
    wall_s = time.perf_counter() - t_fit
    print(
        f"[demo_adm] fit done in {wall_s:.1f}s  ({result.n_iter} iters)  (init wall = {t_fit - t_init:.1f}s)"
    )

    fitted_delay = demo.extract_delay(model, scenario.T)
    fitted_obs = demo.extract_observable(model, data)
    fitted_y = demo.extract_y_recon(model, data)

    record = demo.write_method_outputs(
        method_name="adm",
        model_label="ADM",
        truth=truth,
        fitted_delay=fitted_delay,
        fitted_obs=fitted_obs,
        fitted_y=fitted_y,
        score_trace=np.asarray(result.score_trace, dtype=float),
        score_ylabel="joint log-likelihood",
        wall_s=wall_s,
        out_dir=args.out_dir,
    )
    demo.print_summary(record, "adm")


if __name__ == "__main__":
    main()
