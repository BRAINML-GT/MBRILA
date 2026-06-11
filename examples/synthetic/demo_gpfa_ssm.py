"""GPFA on a no-delay multi-region scenario (Yu et al. 2009).

Scenario: 5-region synthetic data with K_a=2 shared across-latents and
``delay_amplitude = 0`` (no inter-region delay). GPFA's
:class:`NoDelay` is the right inductive bias; ``delay_lat*.png`` is
flat zero for both truth and fit.

GPFA is the "across-only" baseline — shared latents across regions,
no within-region structure (``n_within`` forced to 0). Useful as a
warm-up before introducing DLAG / mDLAG's delay layer.

Pipeline: pCCA emission init, rank-1 deflation init (per across-latent),
then :class:`KalmanEMEngine` SSM fit, with a final latent-scale anchor.

Outputs: standard layout (convergence trace, per-region latent traces,
y reconstruction, summary.json). The delay-overlay figure is flat zero.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import demo_common as demo
import torch

from mbrila import GPFA, KalmanEMEngine, LatentSpec, MOSEKernel, normalize_latent_scales
from mbrila.synthetic.multiregion import MultiRegionScenario


def build_scenario(args: argparse.Namespace) -> MultiRegionScenario:
    """No-delay multi-region scenario."""
    return MultiRegionScenario(
        n_trials=args.n_trials,
        T=args.T,
        y_dims=tuple([args.y_dim_per_region] * args.n_regions),
        n_across=args.n_across,
        n_within=0,
        lag_across=args.lag_across,
        lag_within=1,  # unused (no within latents); scenario API requires a value.
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
    parser = argparse.ArgumentParser(description="GPFA on no-delay multi-region synthetic data.")
    parser.add_argument("--out-dir", type=Path, default=Path("examples/synthetic/demo_outputs/gpfa_ssm"))
    # Scenario
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--n-regions", type=int, default=5)
    parser.add_argument("--y-dim-per-region", type=int, default=100)
    parser.add_argument("--n-across", type=int, default=2)
    parser.add_argument("--lag-across", type=int, default=2)
    parser.add_argument("--sigma-across", type=float, default=0.05)
    parser.add_argument("--sigma-within", type=float, default=0.05)
    parser.add_argument("--per-latent-sigma-ratio", type=float, default=10.0)
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--data-seed", type=int, default=0)
    # Fit
    parser.add_argument("--num-iters", type=int, default=300)
    parser.add_argument("--init-sigma-across", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-min", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    # Rank-1 deflation init. Each round fits a rank-1 GPFA on the
    # residual to seed per-latent ``(kernel state, C column)`` before
    # the main fit. Without it both blocks start at the same
    # ``init_sigma_across`` and have to climb to per-block σ from the
    # same starting point.
    parser.add_argument("--deflation-iters-per-round", type=int, default=200)
    parser.add_argument("--lr-deflation", type=float, default=1e-2)
    # Closed-form (C, d, R) emission M-step every N iterations. GPFA
    # without delay relies on this to break the σ degeneracy when two
    # latents would otherwise get stuck at the same σ. Set to 0 to
    # disable.
    parser.add_argument("--update-obs-every", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demo_gpfa_ssm] device={args.device}  out_dir={args.out_dir}")
    scenario = build_scenario(args)
    print(
        f"[demo_gpfa_ssm] scenario: T={scenario.T} B={scenario.n_trials} "
        f"R={len(scenario.y_dims)} K_a={scenario.n_across} K_w={scenario.n_within} "
        f"delay=0 (no-delay regime)  snr={scenario.snr}"
    )

    data, truth = demo.sample_scenario(scenario, device=args.device)

    spec = LatentSpec(
        n_across=scenario.n_across,
        n_within=(0,) * len(scenario.y_dims),
    )
    engine = KalmanEMEngine(
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        cosine_anneal=True,
        update_obs_every=args.update_obs_every,
        log_every=args.log_every,
    )
    _sigma_a = float(args.init_sigma_across)
    _n_regions = len(scenario.y_dims)
    model = GPFA(
        latent_spec=spec,
        y_dims=scenario.y_dims,
        T=scenario.T,
        kernel_factory_across=lambda: MOSEKernel(num_regions=_n_regions, init_sigma=_sigma_a),
        engine=engine,
        device=args.device,
        dtype=torch.float64,
    ).to(args.device)

    print("[demo_gpfa_ssm] initializing emission from data (pCCA) ...")
    # GPFA does not ship its own ``initialize_from_data``; use the
    # demo helper, which mirrors the pCCA step ADM / DLAG / MDLAG
    # do internally.
    demo.init_linear_observation_pcca(model, data, n_across=scenario.n_across, n_within=0)

    # Rank-1 deflation init — fit each across latent individually on
    # the residual so each block's kernel σ converges to its own data
    # signal before the joint main fit starts.
    print(
        f"[demo_gpfa_ssm] rank-1 deflation init ({args.deflation_iters_per_round} iter × {scenario.n_across} rounds) ..."
    )
    info = demo.rank1_deflation_init(
        model,
        data,
        n_iters_per_round=args.deflation_iters_per_round,
        lr=args.lr_deflation,
        lr_min=args.lr_min,
        verbose=True,
    )
    print(f"[demo_gpfa_ssm]   per-round losses = {[f'{x:.0f}' for x in info['rank1_losses']]}")

    print(f"[demo_gpfa_ssm] fitting ({args.num_iters} iters) ...")
    t0 = time.perf_counter()
    result = model.fit(data, max_iter=args.num_iters, tol=1e-8)
    wall_s = time.perf_counter() - t0
    print(f"[demo_gpfa_ssm] fit done in {wall_s:.1f}s  ({result.n_iter} iters)")

    # Scale anchor pins each fitted latent's rms to its kernel's prior
    # ``√K(0)``. Pure ``(C, g)`` rebalancing; ``y = C·g`` unchanged.
    # Joint-LL EM is invariant to this rescaling, so without the anchor
    # Adam tends to drift to non-canonical magnitudes.
    norm_info = normalize_latent_scales(model, data)
    print(f"[demo_gpfa_ssm] scale anchor: alpha_k = {[f'{a:.3f}' for a in norm_info['alphas']]}")

    fitted_delay = demo.extract_delay(model, scenario.T)
    fitted_obs = demo.extract_observable(model, data)
    fitted_y = demo.extract_y_recon(model, data)

    record = demo.write_method_outputs(
        method_name="gpfa",
        model_label="GPFA",
        truth=truth,
        fitted_delay=fitted_delay,
        fitted_obs=fitted_obs,
        fitted_y=fitted_y,
        score_trace=result.score_trace,
        score_ylabel="marginal log-likelihood",
        wall_s=wall_s,
        out_dir=args.out_dir,
    )
    demo.print_summary(record, "gpfa")


if __name__ == "__main__":
    main()
