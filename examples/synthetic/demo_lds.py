"""Naive linear dynamical system baseline (no GP prior).

Scenario: same no-delay 5-region synthetic data as :mod:`examples.demo_gpfa`.
:class:`mbrila.LDS` fits a free transition matrix ``A`` — no kernel,
no smoothness constraint, no delay parametrisation — providing the
counter-baseline to the GP-prior methods (GPFA / DLAG / mDLAG / ADM).

LDS has a flat ``n_latent``-dim state with no across / within
distinction; the truth's within latents are absorbed into it.
``delay_lat*.png`` is flat zero (LDS has no delay concept) and the
latent-trace overlays are skipped automatically when the fitted
latent shape does not match the truth layout.

Outputs: convergence trace, y reconstruction, summary.json.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import demo_common as demo
import torch

from mbrila import LDS, KalmanEMEngine
from mbrila.synthetic.multiregion import MultiRegionScenario


def build_scenario(args: argparse.Namespace) -> MultiRegionScenario:
    """No-delay multi-region scenario."""
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
        delay_amplitude=0.0,
        per_latent_sigma_across=tuple(
            args.sigma_across * (args.per_latent_sigma_ratio ** (k / max(args.n_across - 1, 1)))
            for k in range(args.n_across)
        ),
        snr=args.snr,
        seed=args.data_seed,
        dtype=torch.float64,
        device="cpu",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="naive LDS on no-delay multi-region synthetic data.")
    parser.add_argument("--out-dir", type=Path, default=Path("examples/synthetic/demo_outputs/lds"))
    # Scenario
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--n-regions", type=int, default=5)
    parser.add_argument("--y-dim-per-region", type=int, default=100)
    parser.add_argument("--n-across", type=int, default=2)
    parser.add_argument("--n-within", type=int, default=1)
    parser.add_argument("--lag-across", type=int, default=3)
    parser.add_argument("--lag-within", type=int, default=2)
    parser.add_argument("--sigma-across", type=float, default=0.05)
    parser.add_argument("--sigma-within", type=float, default=0.05)
    parser.add_argument("--per-latent-sigma-ratio", type=float, default=10.0)
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--data-seed", type=int, default=0)
    # Fit. n_latent absorbs both across and within ground-truth
    # latents (LDS has no across/within distinction).
    parser.add_argument(
        "--n-latent",
        type=int,
        default=7,
        help="LDS latent dim (absorbs across + within: 2 across + 5 region × 1 within)",
    )
    parser.add_argument("--num-iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-min", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demo_lds] device={args.device}  out_dir={args.out_dir}")
    scenario = build_scenario(args)
    print(
        f"[demo_lds] scenario: T={scenario.T} B={scenario.n_trials} "
        f"R={len(scenario.y_dims)} K_true(a+w)={scenario.n_across + scenario.n_within} "
        f"K_lds={args.n_latent}  snr={scenario.snr}"
    )

    data, truth = demo.sample_scenario(scenario, device=args.device)

    engine = KalmanEMEngine(
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        cosine_anneal=True,
        log_every=args.log_every,
    )
    model = LDS(
        n_latent=args.n_latent,
        y_dims=scenario.y_dims,
        T=scenario.T,
        engine=engine,
        device=args.device,
        dtype=torch.float64,
    ).to(args.device)

    print("[demo_lds] initializing emission from data (pCCA, flat latent) ...")
    demo.init_linear_observation_pcca(model, data, n_across=args.n_latent, n_within=0)

    print(f"[demo_lds] fitting ({args.num_iters} iters) ...")
    t0 = time.perf_counter()
    result = model.fit(data, max_iter=args.num_iters, tol=1e-8)
    wall_s = time.perf_counter() - t0
    print(f"[demo_lds] fit done in {wall_s:.1f}s  ({result.n_iter} iters)")

    fitted_delay = demo.extract_delay(model, scenario.T)
    fitted_obs = demo.extract_observable(model, data)
    fitted_y = demo.extract_y_recon(model, data)

    # LDS has a flat K-dim latent with no across/within structure, so its
    # fitted shape does not match the truth's ``(n_across + n_within)``
    # layout. :func:`demo.write_method_outputs` detects the shape mismatch
    # and skips the latent-overlay / delay-overlay figures automatically.
    # ``y_recon.png`` and ``convergence.png`` are still produced.
    record = demo.write_method_outputs(
        method_name="lds",
        model_label="LDS",
        truth=truth,
        fitted_delay=fitted_delay,
        fitted_obs=fitted_obs,
        fitted_y=fitted_y,
        score_trace=result.score_trace,
        score_ylabel="marginal log-likelihood",
        wall_s=wall_s,
        out_dir=args.out_dir,
    )
    demo.print_summary(record, "lds")


if __name__ == "__main__":
    main()
