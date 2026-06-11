"""Tutorial: plug a user-defined kernel into DLAG-SSM.

This script writes a :class:`mbrila.kernels.base.BaseKernel` subclass
and hands it to :class:`mbrila.DLAG` via ``kernel_factory_across`` /
``kernel_factory_within``. The library does not need to be modified
to support a new stationary scalar kernel.

The custom kernel is a Rational Quadratic (RQ) kernel:

    k(τ) = (1 + τ² / (2 α ℓ²))^(-α)

α controls the spectral "scale mixture": ``α → ∞`` recovers RBF
(light tail), ``α = 1`` is the Cauchy kernel (heavy tail). Leaving
α learnable lets the optimiser tune the tail shape per latent block.

RQ has no exact finite-dimensional SDE form, so the Kalman engine
consumes it through the same AR(P) approximation bridge that MOSE
uses. For a kernel that does have an exact SDE form, see
:mod:`examples.demo_matern`.

Steps:

1. Define :class:`RationalQuadraticKernel` (one ``cov`` method, two
   learnable parameters ℓ and α).
2. Call :func:`check_kernel` to verify symmetry / PSD / Fourier
   sanity before building a model that depends on it.
3. Build DLAG-SSM with
   ``kernel_factory_across=lambda: RationalQuadraticKernel(...)``.
4. Fit on a constant-delay multi-region synthetic scenario.

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
from torch import Tensor, nn

from mbrila import DLAG, KalmanEMEngine, LatentSpec
from mbrila.kernels.base import BaseKernel
from mbrila.kernels.validate import check_kernel
from mbrila.synthetic.multiregion import MultiRegionScenario

# ---------------------------------------------------------------------------
# Step 1 — the custom kernel itself. This is the only mbrila-specific
# code a downstream user has to write.
# ---------------------------------------------------------------------------


class RationalQuadraticKernel(BaseKernel):
    """Rational Quadratic (RQ) stationary kernel.

    ``k(τ) = (1 + τ² / (2 α ℓ²))^(-α)``

    RQ is a scale mixture of RBF kernels with inverse-Gamma weights.
    ``α`` controls the spread:

    - ``α → ∞`` → degenerate mixture, recovers RBF (light-tailed PSD)
    - ``α = 1`` → Cauchy kernel (heavy-tailed)
    - ``α`` in between → smooth interpolation

    ``K(0) = 1`` is fixed (only ℓ and α are learnable), so the GP
    prior variance does not compete with the emission matrix C during
    optimisation.
    """

    is_markovian = False  # No exact finite-state SDE → use AR(P) bridge.
    is_complex = False

    def __init__(self, *, init_lengthscale: float = 2.0, init_alpha: float = 2.0) -> None:
        super().__init__()
        if init_lengthscale <= 0 or init_alpha <= 0:
            raise ValueError(
                f"init_lengthscale and init_alpha must be positive; got {init_lengthscale}, {init_alpha}"
            )
        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(float(init_lengthscale))))
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(float(init_alpha))))

    def cov(self, tau: Tensor) -> Tensor:
        ell = torch.exp(self.log_lengthscale)
        alpha = torch.exp(self.log_alpha)
        return (1.0 + (tau / ell).square() / (2.0 * alpha)) ** (-alpha)


# ---------------------------------------------------------------------------
# Step 2 onwards — standard DLAG-SSM fit; the only difference from
# demo_dlag_ssm.py is the kernel factory.
# ---------------------------------------------------------------------------


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
    parser = argparse.ArgumentParser(description="DLAG-SSM with a custom Rational Quadratic kernel.")
    parser.add_argument("--out-dir", type=Path, default=Path("examples/synthetic/demo_outputs/custom_kernel"))
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
    parser.add_argument("--init-lengthscale", type=float, default=2.0)
    parser.add_argument("--init-alpha", type=float, default=2.0)
    parser.add_argument("--num-iters", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lr-min", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    # Rank-1 deflation init (per across-latent). Data-driven per-latent
    # kernel + δ + C seed; on by default with ``--no-deflation`` for
    # A/B comparison.
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

    # Step 2: validate the custom kernel before building a model that
    # depends on it. ``check_kernel`` raises if the kernel is not a
    # valid PSD stationary kernel.
    proto = RationalQuadraticKernel(init_lengthscale=args.init_lengthscale, init_alpha=args.init_alpha).to(
        dtype=torch.float64
    )
    check_kernel(proto)
    print(
        f"[demo_custom_kernel] RationalQuadraticKernel passed check_kernel  "
        f"(n_params={proto.n_params}, capabilities={sorted(proto.capabilities())})"
    )

    print(f"[demo_custom_kernel] device={args.device}  out_dir={args.out_dir}")
    scenario = build_scenario(args)
    print(
        f"[demo_custom_kernel] scenario: T={scenario.T} B={scenario.n_trials} "
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
    # Step 3: hand the kernel factory to DLAG. Across-latent blocks and
    # within-latent blocks get independent RQ kernels (separate parameter
    # copies, since each block has its own ``log_lengthscale`` /
    # ``log_alpha``).
    init_ell = args.init_lengthscale
    init_alpha = args.init_alpha

    def factory_across() -> BaseKernel:
        return RationalQuadraticKernel(init_lengthscale=init_ell, init_alpha=init_alpha)

    def factory_within() -> BaseKernel:
        return RationalQuadraticKernel(init_lengthscale=init_ell, init_alpha=init_alpha)

    model = DLAG(
        latent_spec=spec,
        y_dims=scenario.y_dims,
        T=scenario.T,
        kernel_factory_across=factory_across,
        kernel_factory_within=factory_within,
        engine="kalman",
        engine_override=engine,
        device=args.device,
        dtype=torch.float64,
    ).to(args.device)

    print("[demo_custom_kernel] initializing from data (pCCA) ...")
    model.initialize_from_data(data, mode="pcca")

    if args.deflation and args.deflation_iters_per_round > 0:
        print(
            f"[demo_custom_kernel] rank-1 deflation init "
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
        print(f"[demo_custom_kernel]   per-round losses = {[f'{x:.0f}' for x in info['rank1_losses']]}")
    else:
        print("[demo_custom_kernel] deflation skipped (--no-deflation)")

    print(f"[demo_custom_kernel] fitting ({args.num_iters} iters) ...")
    t_fit = time.perf_counter()
    result = model.fit(data, max_iter=args.num_iters, tol=1e-8)
    wall_s = time.perf_counter() - t_fit
    print(f"[demo_custom_kernel] fit done in {wall_s:.1f}s  ({result.n_iter} iters)")

    fitted_delay = demo.extract_delay(model, scenario.T)
    fitted_obs = demo.extract_observable(model, data)
    fitted_y = demo.extract_y_recon(model, data)

    record = demo.write_method_outputs(
        method_name="custom_kernel",
        model_label="DLAG-SSM + RationalQuadratic kernel",
        truth=truth,
        fitted_delay=fitted_delay,
        fitted_obs=fitted_obs,
        fitted_y=fitted_y,
        score_trace=np.asarray(result.score_trace, dtype=float),
        score_ylabel="joint log-likelihood",
        wall_s=wall_s,
        out_dir=args.out_dir,
    )
    demo.print_summary(record, "custom_kernel")


if __name__ == "__main__":
    main()
