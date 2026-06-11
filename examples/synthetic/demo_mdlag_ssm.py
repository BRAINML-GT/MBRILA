"""mDLAG-SSM (Kalman + ARD) on a 3-region scenario.

Scenario: 3-region synthetic data with ``K_true = 2`` shared latents.
The model is built with ``K_init = 4``; the ARD prior on the loading
matrix should soft-prune the two spurious columns. ``ard_alpha.png``
is the headline figure — two short (active) bars and two tall (pruned)
bars.

This is mDLAG's distinguishing feature vs ADM / DLAG: the number of
latents does not have to be pre-specified.

Pipeline: pCCA emission init, then :class:`VEMKalmanARDEngine` —
mDLAG's variational ARD updates wrap a Markov-GP Kalman E-step, with
joint-LL EM for the kernel / delay parameters via grouped-AdamW.

Outputs: standard layout plus ``ard_alpha.png``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import demo_common as demo
import torch

from mbrila import MDLAG, LatentSpec, MOSEKernel, VEMKalmanARDEngine
from mbrila.core.latent_spec import ARDPriorConfig
from mbrila.synthetic.multiregion import MultiRegionScenario


def build_scenario(args: argparse.Namespace) -> MultiRegionScenario:
    """Constant-delay scenario with K_truth shared across-latents."""
    return MultiRegionScenario(
        n_trials=args.n_trials,
        T=args.T,
        y_dims=tuple([args.y_dim_per_region] * args.n_regions),
        n_across=args.k_true,
        n_within=0,
        lag_across=args.lag_across,
        lag_within=2,  # unused (n_within = 0) but the scenario API requires a value
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
    parser = argparse.ArgumentParser(description="mDLAG-SSM with ARD pruning.")
    parser.add_argument("--out-dir", type=Path, default=Path("examples/synthetic/demo_outputs/mdlag_ssm"))
    # Scenario
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--n-regions", type=int, default=3)
    parser.add_argument("--y-dim-per-region", type=int, default=100)
    parser.add_argument("--k-true", type=int, default=2, help="ground-truth K_a")
    parser.add_argument("--k-init", type=int, default=4, help="model K_a (more = ARD soft-prunes spurious)")
    parser.add_argument("--lag-across", type=int, default=2)
    parser.add_argument("--sigma-across", type=float, default=0.05)
    parser.add_argument("--delay-amplitude", type=float, default=3.0)
    parser.add_argument("--per-latent-sigma-ratio", type=float, default=10.0)
    parser.add_argument("--per-latent-amp-ratio", type=float, default=3.0)
    parser.add_argument("--region-heterogeneity", type=float, default=1.0)
    parser.add_argument("--snr", type=float, default=20.0)
    parser.add_argument("--data-seed", type=int, default=0)
    # Fit
    parser.add_argument("--num-iters", type=int, default=300)
    parser.add_argument("--init-gamma-across", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-min", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gp-steps-per-em", type=int, default=1)
    parser.add_argument("--ard-prior-shape", type=float, default=1e-3)
    parser.add_argument("--ard-prior-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demo_mdlag_ssm] device={args.device}  out_dir={args.out_dir}")
    scenario = build_scenario(args)
    print(
        f"[demo_mdlag_ssm] scenario: T={scenario.T} B={scenario.n_trials} "
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
    engine = VEMKalmanARDEngine(
        lr=args.lr,
        lr_min=args.lr_min,
        weight_decay=args.weight_decay,
        gp_steps_per_em=args.gp_steps_per_em,
        cosine_anneal=True,
        log_every=args.log_every,
    )
    _gamma_a = float(args.init_gamma_across)
    _n_regions = len(scenario.y_dims)
    model = MDLAG(
        latent_spec=spec,
        y_dims=scenario.y_dims,
        T=scenario.T,
        kernel_factory_across=lambda: MOSEKernel(num_regions=_n_regions, init_sigma=_gamma_a),
        engine="kalman",
        engine_override=engine,
        lag_across=args.lag_across,
        device=args.device,
        dtype=torch.float64,
    ).to(args.device)

    print("[demo_mdlag_ssm] initializing from data (pCCA) ...")
    model.initialize_from_data(data)

    print(f"[demo_mdlag_ssm] fitting ({args.num_iters} iters) ...")
    t0 = time.perf_counter()
    result = model.fit(data, max_iter=args.num_iters, tol=1e-8)
    wall_s = time.perf_counter() - t0
    print(f"[demo_mdlag_ssm] fit done in {wall_s:.1f}s  ({result.n_iter} iters)")

    # The fitted model has ``K_init`` across-latents but ground truth has
    # ``K_true``. The delay-recovery metric uses the model's K_init; the
    # ARD α figure shows which K_init columns are actually used.
    fitted_delay = demo.extract_delay(model, scenario.T)
    fitted_obs = demo.extract_observable(model, data)
    fitted_y = demo.extract_y_recon(model, data)
    alpha_mean = demo.extract_ard_alpha(model)

    # Two delay metrics:
    #
    # - The default ``delay_rmse`` produced by :func:`write_method_outputs`
    #   averages over every K_init column of the fitted model, including
    #   spurious columns that ARD has pruned. Pruned columns' δ drifts on
    #   numerical noise, so this number is **misleadingly large** even
    #   when the active columns recover δ perfectly.
    # - The ARD-aware metric below restricts to ARD-active columns
    #   (and best-permutation matches them against truth), producing
    #   the recovery quality on columns the model is actually using.
    #
    # Both numbers are saved; ``extra_summary`` highlights the ARD-aware
    # one because that's the meaningful recovery score for mDLAG.
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

    # Pad ground-truth delay to K_init columns (zeros for the spurious
    # columns) so the pairwise plot helper has matching shapes.
    truth_delay_padded = truth["delay"]
    if truth_delay_padded.shape[-1] < args.k_init:
        pad_n = args.k_init - truth_delay_padded.shape[-1]
        import numpy as np

        truth_delay_padded = np.concatenate(
            [truth_delay_padded, np.zeros((truth_delay_padded.shape[0], truth_delay_padded.shape[1], pad_n))],
            axis=-1,
        )
    # Also pad truth_observable so latent-trace plot has K_init columns.
    truth_obs_padded = truth["observable"]
    if args.k_init > args.k_true:
        import numpy as np

        # Each region's observable block has K_true columns; insert
        # K_init - K_true zero columns per region after them. Since
        # n_within = 0 here, the truth observable is shaped (B, T, R*K_true).
        B = truth_obs_padded.shape[0]
        T = truth_obs_padded.shape[1]
        R = len(scenario.y_dims)
        truth_obs_grouped = truth_obs_padded.reshape(B, T, R, args.k_true)
        pad = np.zeros((B, T, R, args.k_init - args.k_true))
        truth_obs_padded = np.concatenate([truth_obs_grouped, pad], axis=-1).reshape(B, T, R * args.k_init)

    truth_padded = dict(truth)
    truth_padded["delay"] = truth_delay_padded
    truth_padded["observable"] = truth_obs_padded
    truth_padded["n_across"] = args.k_init

    record = demo.write_method_outputs(
        method_name="mdlag_ssm",
        model_label="mDLAG-SSM",
        truth=truth_padded,
        fitted_delay=fitted_delay,
        fitted_obs=fitted_obs,
        fitted_y=fitted_y,
        score_trace=result.score_trace,
        score_ylabel="proxy ELBO",
        wall_s=wall_s,
        out_dir=args.out_dir,
        alpha_mean=alpha_mean,
        extra_summary=extra,
    )
    demo.print_summary(record, "mdlag_ssm")


if __name__ == "__main__":
    main()
