"""Original DLAG (exact-GP) on a two-region constant-delay scenario.

Scenario: 2-region synthetic data (matching the original DLAG paper's
benchmark setup) with K_a=2 across-latents and a constant inter-region
delay. With one region pair the delay panel is a single subplot — the
cleanest demonstration of "constant per-region delay recovered exactly".

Pipeline: pCCA emission init, then :class:`ExactEMEngine` outer EM
(E-step + closed-form ``(C, d, R)`` + LBFGS on the GP hyperparameters).
The exact dense-GP path is ``O(T³)`` per Cholesky.

Outputs: standard layout (convergence trace, per-pair delay overlays,
per-region latent traces, y reconstruction, summary.json).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import demo_common as demo
import torch

from mbrila import DLAG, ExactEMEngine, LatentSpec, MOSEKernel
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
    parser = argparse.ArgumentParser(description="DLAG (exact GP) on 2-region synthetic data.")
    parser.add_argument("--out-dir", type=Path, default=Path("examples/synthetic/demo_outputs/dlag"))
    # 2-region per the original DLAG paper.
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--n-regions", type=int, default=2)
    parser.add_argument("--y-dim-per-region", type=int, default=100)
    parser.add_argument("--n-across", type=int, default=2)
    parser.add_argument("--n-within", type=int, default=1)
    parser.add_argument("--lag-across", type=int, default=3)
    parser.add_argument("--lag-within", type=int, default=2)
    parser.add_argument("--sigma-across", type=float, default=0.05)
    parser.add_argument("--sigma-within", type=float, default=0.05)
    parser.add_argument("--delay-amplitude", type=float, default=3.0)
    parser.add_argument("--per-latent-sigma-ratio", type=float, default=10.0)
    parser.add_argument("--per-latent-amp-ratio", type=float, default=3.0)
    parser.add_argument("--region-heterogeneity", type=float, default=1.0)
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--data-seed", type=int, default=0)
    # Fit. ExactEMEngine uses LBFGS on the GP hyperparameters inside
    # each outer EM iteration.
    parser.add_argument("--num-iters", type=int, default=300)
    parser.add_argument("--init-gamma-across", type=float, default=0.1)
    parser.add_argument("--init-gamma-within", type=float, default=0.1)
    parser.add_argument("--max-lbfgs-iter", type=int, default=40)
    parser.add_argument("--lbfgs-history", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demo_dlag] device={args.device}  out_dir={args.out_dir}")
    scenario = build_scenario(args)
    print(
        f"[demo_dlag] scenario: T={scenario.T} B={scenario.n_trials} "
        f"R={len(scenario.y_dims)} K_a={scenario.n_across} K_w={scenario.n_within} "
        f"shape=constant amp={scenario.delay_amplitude} snr={scenario.snr}"
    )

    data, truth = demo.sample_scenario(scenario, device=args.device)

    spec = LatentSpec(
        n_across=scenario.n_across,
        n_within=(scenario.n_within,) * len(scenario.y_dims),
    )
    engine = ExactEMEngine(
        max_lbfgs_iter=args.max_lbfgs_iter,
        lbfgs_history=args.lbfgs_history,
        log_every=args.log_every,
    )
    R = len(scenario.y_dims)
    init_sigma_across = args.init_gamma_across
    init_sigma_within = args.init_gamma_within
    model = DLAG(
        latent_spec=spec,
        y_dims=scenario.y_dims,
        T=scenario.T,
        kernel_factory_across=lambda: MOSEKernel(num_regions=R, init_sigma=init_sigma_across),
        kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=init_sigma_within),
        engine="exact",
        engine_override=engine,
        device=args.device,
        dtype=torch.float64,
    ).to(args.device)

    print("[demo_dlag] initializing from data (pCCA) ...")
    model.initialize_from_data(data, mode="pcca")

    print(f"[demo_dlag] fitting ({args.num_iters} iters) ...")
    t0 = time.perf_counter()
    result = model.fit(data, max_iter=args.num_iters, tol=1e-8)
    wall_s = time.perf_counter() - t0
    print(f"[demo_dlag] fit done in {wall_s:.1f}s  ({result.n_iter} iters)")

    fitted_delay = demo.extract_delay(model, scenario.T)
    fitted_obs = demo.extract_observable(model, data)
    fitted_y = demo.extract_y_recon(model, data)

    record = demo.write_method_outputs(
        method_name="dlag",
        model_label="DLAG (exact GP)",
        truth=truth,
        fitted_delay=fitted_delay,
        fitted_obs=fitted_obs,
        fitted_y=fitted_y,
        score_trace=result.score_trace,
        score_ylabel="marginal log-likelihood",
        wall_s=wall_s,
        out_dir=args.out_dir,
    )
    demo.print_summary(record, "dlag")


if __name__ == "__main__":
    main()
