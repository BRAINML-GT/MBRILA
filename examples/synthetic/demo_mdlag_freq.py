"""fast-mDLAG (frequency-domain, circulant) on a 3-region scenario.

Scenario: identical to :mod:`examples.demo_mdlag_time` — same
``K_true / K_init`` and the same data seed — so the two outputs can
be diffed directly.

Pipeline: :class:`VEMARDFreqEngine` runs the variational EM in the
frequency domain. Delays become circular shifts
``Q(f) = exp(-i2πfδ)``; the circulant approximation is fast (much
cheaper than the time-domain ``O(T³)`` Cholesky) but introduces a
boundary error that shrinks as ``1/T``. Designed for ``T ≥ 200``.

Expected differences vs the time-domain engine: similar ELBO faster,
slightly noisier δ recovery, and qualitatively-agreeing ARD α
selection (the same columns are pruned, but absolute α values can
differ).

Outputs: standard layout plus ``ard_alpha.png``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import demo_common as demo
import torch

from mbrila import MDLAG, LatentSpec, MOSEKernel, VEMARDFreqEngine
from mbrila.core.latent_spec import ARDPriorConfig
from mbrila.synthetic.multiregion import MultiRegionScenario


def build_scenario(args: argparse.Namespace) -> MultiRegionScenario:
    """Constant-delay multi-region scenario."""
    return MultiRegionScenario(
        n_trials=args.n_trials,
        T=args.T,
        y_dims=tuple([args.y_dim_per_region] * args.n_regions),
        n_across=args.k_true,
        n_within=0,
        lag_across=args.lag_across,
        lag_within=2,
        sigma_across=args.sigma_across,
        sigma_within=args.sigma_across,
        delay_shape="constant",
        delay_amplitude=args.delay_amplitude,
        per_latent_amplitudes=tuple(
            args.delay_amplitude * (1.0 / args.per_latent_amp_ratio ** (k / max(args.k_true - 1, 1)))
            for k in range(args.k_true)
        ),
        per_latent_sigma_across=tuple(
            args.sigma_across * (args.per_latent_sigma_ratio ** (k / max(args.k_true - 1, 1)))
            for k in range(args.k_true)
        ),
        per_latent_shapes=tuple(["constant"] * args.k_true),
        region_heterogeneity=args.region_heterogeneity,
        snr=args.snr,
        seed=args.data_seed,
        dtype=torch.float64,
        device="cpu",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="fast-mDLAG (frequency-domain, circulant) on 3-region data.")
    parser.add_argument("--out-dir", type=Path, default=Path("examples/synthetic/demo_outputs/mdlag_freq"))
    # 3-region per the original mDLAG paper.
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--n-regions", type=int, default=3)
    parser.add_argument("--y-dim-per-region", type=int, default=100)
    parser.add_argument("--k-true", type=int, default=2, help="ground-truth K_a")
    parser.add_argument("--k-init", type=int, default=4, help="model K_a (more = ARD soft-prunes spurious)")
    parser.add_argument("--lag-across", type=int, default=3)
    parser.add_argument("--sigma-across", type=float, default=0.05)
    parser.add_argument("--delay-amplitude", type=float, default=3.0)
    parser.add_argument("--per-latent-sigma-ratio", type=float, default=10.0)
    parser.add_argument("--per-latent-amp-ratio", type=float, default=3.0)
    parser.add_argument("--region-heterogeneity", type=float, default=1.0)
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-iters", type=int, default=300)
    parser.add_argument("--init-gamma-across", type=float, default=0.1)
    parser.add_argument("--max-lbfgs-iter", type=int, default=40)
    parser.add_argument("--lbfgs-history", type=int, default=15)
    parser.add_argument("--ard-prior-shape", type=float, default=1e-3)
    parser.add_argument("--ard-prior-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demo_mdlag_freq] device={args.device}  out_dir={args.out_dir}")
    scenario = build_scenario(args)
    print(
        f"[demo_mdlag_freq] scenario: T={scenario.T} B={scenario.n_trials} "
        f"R={len(scenario.y_dims)} K_true={scenario.n_across} K_init={args.k_init} "
        f"amp={scenario.delay_amplitude} snr={scenario.snr}"
    )

    data, truth = demo.sample_scenario(scenario, device=args.device)

    spec = LatentSpec(
        n_across=args.k_init,
        n_within=(0,) * len(scenario.y_dims),
        selection="ard",
        ard_prior=ARDPriorConfig(shape=args.ard_prior_shape, rate=args.ard_prior_rate),
    )
    engine = VEMARDFreqEngine(
        max_lbfgs_iter=args.max_lbfgs_iter,
        lbfgs_history=args.lbfgs_history,
        log_every=args.log_every,
    )
    _n_regions = len(scenario.y_dims)
    _gamma_a = float(args.init_gamma_across)
    model = MDLAG(
        latent_spec=spec,
        y_dims=scenario.y_dims,
        T=scenario.T,
        kernel_factory_across=lambda: MOSEKernel(num_regions=_n_regions, init_sigma=_gamma_a),
        engine="freq",
        engine_override=engine,
        device=args.device,
        dtype=torch.float64,
    ).to(args.device)

    print("[demo_mdlag_freq] initializing from data (pCCA) ...")
    model.initialize_from_data(data)

    print(f"[demo_mdlag_freq] fitting ({args.num_iters} iters) ...")
    t0 = time.perf_counter()
    result = model.fit(data, max_iter=args.num_iters, tol=1e-8)
    wall_s = time.perf_counter() - t0
    print(f"[demo_mdlag_freq] fit done in {wall_s:.1f}s  ({result.n_iter} iters)")

    fitted_delay = demo.extract_delay(model, scenario.T)
    fitted_obs = demo.extract_observable(model, data)
    fitted_y = demo.extract_y_recon(model, data)
    alpha_mean = demo.extract_ard_alpha(model)

    # Default ``delay_rmse`` averages over all K_init columns including
    # ARD-pruned ones (whose δ is noise). The ARD-aware metric below
    # restricts to active columns and best-permutation matches them —
    # see :func:`demo.ard_aware_delay_rmse` for details.
    max_alpha = alpha_mean.max(axis=0) if alpha_mean is not None else None
    if alpha_mean is not None:
        ard_info = demo.ard_aware_delay_rmse(truth["delay"], fitted_delay, alpha_mean, len(scenario.y_dims))
        extra = (
            f"max_alpha_per_col={max_alpha.tolist() if max_alpha is not None else 'N/A'}  "
            f"n_active={ard_info['n_active']}/{ard_info['K_true']}  "
            f"matched_active={ard_info['matched_active'].tolist()}  "
            f"matched_truth={ard_info['matched_truth'].tolist()}  "
            f"ard_aware_delay_rmse={ard_info['rmse']:.3f} bins"
        )
    else:
        extra = f"max_alpha_per_col={max_alpha.tolist() if max_alpha is not None else 'N/A'}"


    record = demo.write_method_outputs(
        method_name="mdlag_freq",
        model_label="fast-mDLAG (frequency)",
        truth=truth,
        fitted_delay=fitted_delay,
        fitted_obs=fitted_obs,
        fitted_y=fitted_y,
        score_trace=result.score_trace,
        score_ylabel="ELBO",
        wall_s=wall_s,
        out_dir=args.out_dir,
        alpha_mean=alpha_mean,
        extra_summary=extra,
    )
    demo.print_summary(record, "mdlag_freq")


if __name__ == "__main__":
    main()
