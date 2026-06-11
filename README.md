# mbrila

**M**ultiple **B**rain **R**egion **I**nteraction using **L**atent **A**nalysis

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

A unified, GPU-native PyTorch library for **multi-region neural latent
variable models** — methods that infer shared low-dimensional latent
processes across simultaneously recorded brain regions, optionally with
**inter-region communication delays** (constant or time-varying).

## Why GP-SSM

Multi-region neural population dynamics are typically slow and smooth —
they have a **Gaussian process structure** on the latent level (this is
the modelling assumption behind GPFA, DLAG, mDLAG, …). The price of a
dense GP prior is **O(T³) inference cost** in the trial length T:
fitting 200-bin trials with 5 latents already needs a 1000-dim Cholesky
per EM iter. Free state-space models (linear LDS, RNN) scale to long T
but throw away GP smoothness and timescale priors.

`mbrila` builds on the recent observation
([Li et al., ICML 2025](https://proceedings.mlr.press/v267/li25ck.html))
that **any stationary GP can be approximately lifted into a Markov
state-space model** via an AR(P) realisation. The resulting GP-SSM
keeps the GP modelling structure (timescale priors, ARD, inter-region
delays as kernel arguments) but inherits the **O(log T) GPU parallel
Kalman scan** for inference.

## What you can do with this library

- **Drop in any of the standard methods** — ADM, DLAG (exact GP),
  DLAG-SSM, mDLAG (time-domain / frequency-domain circulant /
  Kalman-SSM), GPFA-SSM, free LDS — with a single `model = Preset(...)`
  call and a unified `model.fit(data)` interface.
- **Swap GP kernels** without touching the inference engine. Any
  stationary scalar kernel — MOSE/RBF, Matérn-1/2/3/2/5/2, or your own
  `BaseKernel` subclass — slots in via `kernel_factory_*` callables.
  Matérn-5/2 even has an **exact** finite-state SDE form (no AR(P)
  approximation needed); see [`notebooks/synthetic/demo_matern.ipynb`](notebooks/synthetic/demo_matern.ipynb).
- **Write your own kernel in ~10 lines** — implement `cov(τ)` on a
  `BaseKernel` subclass and plug it into every GP-prior model. See
  [`notebooks/synthetic/demo_custom_kernel.ipynb`](notebooks/synthetic/demo_custom_kernel.ipynb)
  for a Rational-Quadratic-kernel tutorial.
- **Compare methods fairly** — every method ships with the same
  evaluation pipeline (delay-recovery RMSE on synthetic, NLB-style
  held-out-neuron co-smoothing on real data, both with multi-seed
  averaging).
- **Pick speed vs accuracy per use case** — exact GP (`O(T³)`,
  reference) ↔ SSM-GP via Markov AR(P) lift (`O(log T)` on GPU) ↔
  frequency-domain circulant approximation.
- **Browse results in Jupyter** — every method has a self-contained
  notebook (see [`notebooks/`](notebooks/)) with results baked in.

All operations are **fully batched over trials** and **pure PyTorch**
— no per-trial Python loops, GPU by
default.

---

## The GP-SSM framework

The headline modelling choice in `mbrila` is the **latent prior class**
— what assumption you make about how latent state evolves in time. Four
families:

| Latent prior | What it assumes | Inference cost | Engine class |
|---|---|---|---|
| **Dense exact GP** | latent ~ GP with a stationary kernel; covariance is a full `T × T` matrix | `O(T³)` per EM iter | `ExactEMEngine` · `VEMARDEngine` (ARD) · `VEMARDFreqEngine` (circulant ≈ dense in freq domain) |
| **SSM-GP via AR(P) lift** | latent ~ same GP, **approximated** by a `P`-step Markov state-space model | `O(log T)` (parallel scan) | `KalmanEMEngine` · `VEMKalmanARDEngine` (ARD) |
| **SSM-GP via exact SDE** | latent ~ Matérn-`p/2` GP, **exactly** representable as a finite-state SDE | `O(log T)` | `KalmanEMEngine` |
| **Free SSM** | latent ~ generic linear-Gaussian Markov chain, learnable `(A, Q)`, no GP / no kernel | `O(log T)` | `KalmanEMEngine` |

Within a chosen latent prior, the model is further specified by:

- **GP kernel** (only for the three GP families): MOSE/RBF, Matérn,
  or any user-defined `BaseKernel`. **The kernel also encodes the
  inter-region communication delay** through its lagged covariance
  `cov(τ + δ_j − δ_i)`. The delay parameterisation — `NoDelay`,
  `FixedDelay` (constant `δ`), `TimeVaryingDelay` (`δ(t)`) — is
  implemented at the dynamics layer (`mbrila.delays`) and is what
  separates GPFA-SSM (no delay) from DLAG (fixed) from ADM (time-varying).
  Practical note: `FixedDelay` drops in cleanly with any kernel, but
  `TimeVaryingDelay` is significantly more expressive and its
  high-dimensional `δ(t)` parameter space needs a careful
  initialisation — specifically the rank-1 deflation init that
  `ADM` ships, which breaks the symmetry between latent components
  before joint training. If you want a time-varying-delay variant on a
  custom kernel, follow `ADM` end-to-end as the reference — not just
  the delay class, but the initialisation recipe.
- **Observation model**: linear-Gaussian
  (`MultiRegionLinearObservation`) for ADM / DLAG / GPFA / LDS, or
  variational ARD (`ARDObservation`) for the mDLAG family.
- **Model structure**: `LatentSpec(n_regions, n_across, n_within)` —
  the per-region neuron-count tuple plus the across-region (shared)
  and within-region (private) latent counts.

Engine compatibility is enforced by capability matching, e.g. `LDS`
cannot accept `ExactEMEngine` because it has no kernel; `MDLAG` with
ARD cannot accept the non-ARD `KalmanEMEngine`.

### Model presets

A "method" is just a name for a configured combination. The headline
presets are all the field-known multi-region methods, plus their
SSM-approximate cousins:

| Preset | Latent prior class | Delay | Engine class | Notes |
|---|---|---|---|---|
| **`ADM`** | **SSM-GP** (AR(P) lift of MOSE/RBF) | time-varying `δ(t)` | `KalmanEMEngine` | `O(log T)` parallel scan |
| **`DLAG`** | dense exact GP (MOSE/RBF) | constant `δ` | `ExactEMEngine` (default) **or** `KalmanEMEngine` | the second route gives a DLAG-SSM AR(P) approximation |
| **`MDLAG`** | dense exact GP + **ARD** | constant `δ` | `VEMARDEngine` (time) / `VEMARDFreqEngine` (freq, ~22× faster) / `VEMKalmanARDEngine` (SSM) | ARD prunes redundant latents automatically |
| **`GPFA-SSM`** | SSM-GP, **no delay** | — | `KalmanEMEngine` | SSM (AR(P) lift) approximation of GPFA; shared-only baseline (`n_within = 0`). |
| **`LDS`** | **free SSM** (no GP prior) | — | `KalmanEMEngine` | no-kernel baseline |

The framework is **user-extensible** along the kernel dimension: writing a new stationary
kernel by subclassing `BaseKernel` and supplying `cov(τ)` lets you plug
that kernel into any GP-prior model. See
[`notebooks/synthetic/demo_custom_kernel.ipynb`](notebooks/synthetic/demo_custom_kernel.ipynb)
for an end-to-end Rational-Quadratic-kernel tutorial.

The presets shipped here all model inter-region interaction as a
**communication delay** in the kernel's lagged covariance. This is one
particular hypothesis about how brain regions interact — and the GP
kernel is the natural place to encode others. We encourage users to
design new kernels that capture different forms of inter-region
interaction and contribute them back.

---

## Installation

`mbrila` targets Python 3.12 (PyTorch does not yet support 3.13) and is
managed with [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/BRAINML-GT/MBRILA.git mbrila
cd mbrila
uv sync              # runtime dependencies
uv sync --extra dev  # + dev tools (pytest, ruff, mypy)
```

Or with plain pip:

```bash
pip install -e .
```

Default device is CUDA when available, CPU otherwise; nothing is
hard-coded — pass `--device cpu` or `device="cpu"` to force CPU.

### Quickstart

The fastest way to see the library in action is to open one of the
Jupyter notebooks — they already have results baked in:

```bash
jupyter lab notebooks/synthetic/demo_adm.ipynb       # ADM on synthetic delay-recovery
jupyter lab notebooks/v1v2/demo_dlag_ssm.ipynb       # DLAG-SSM on real V1/V2 data
jupyter lab notebooks/synthetic/demo_custom_kernel.ipynb  # plug in your own GP kernel
```

Each notebook is self-contained and produces every diagnostic figure
inline. See [Examples](#examples) below for the full list.

---

## Examples

> 📓 **Start with the Jupyter notebooks** in [`notebooks/`](notebooks/) —
> they are the easiest entry point. Every method has its own
> self-contained notebook that loads data, builds the model, fits it,
> and produces every diagnostic figure inline (convergence,
> inter-region delay, smoother latents, PSTH heatmap, co-smoothing
> reconstruction, ARD α bar, headline metric). Just open one and read
> top-to-bottom — no CLI needed, no shell scripts to read, results are
> baked into the file so you can browse them even before running.

For automation / sweeps / SLURM jobs, the CLI demos in
[`examples/`](examples/) cover the same methods with the same configs
(every notebook has a one-to-one CLI counterpart with identical
defaults).

### Notebooks (recommended)

```
notebooks/
├── synthetic/      # ground-truth-delay recovery on synthetic GP data
│   ├── demo_adm.ipynb
│   ├── demo_dlag.ipynb              (exact-GP engine)
│   ├── demo_dlag_ssm.ipynb          (SSM-GP engine)
│   ├── demo_mdlag_time.ipynb / demo_mdlag_freq.ipynb / demo_mdlag_ssm.ipynb
│   ├── demo_gpfa_ssm.ipynb
│   ├── demo_lds.ipynb
│   ├── demo_matern.ipynb            (Matérn-5/2 with exact SDE form)
│   └── demo_custom_kernel.ipynb     (user-defined Rational Quadratic — kernel-as-axis tutorial)
└── v1v2/           # real-data co-smoothing on V1/V2 visual-cortex recordings
    ├── demo_adm.ipynb / demo_dlag.ipynb / demo_dlag_ssm.ipynb
    ├── demo_mdlag_time.ipynb / demo_mdlag_freq.ipynb / demo_mdlag_ssm.ipynb
    └── demo_custom_kernel.ipynb
```

Every notebook begins with a markdown banner stating the engine class
(dense exact GP / SSM-GP / SSM-GP exact-SDE / free SSM), a config
table, then runs the full fit-evaluate-plot pipeline. Diagnostic
figures are produced inline using shared helpers in
[`notebooks/nb_helpers.py`](notebooks/nb_helpers.py).

### CLI scripts (for sweeps / SLURM)

```
examples/
├── synthetic/      # same as notebooks/synthetic/ but CLI
│   ├── demo_adm.py / demo_dlag.py / demo_dlag_ssm.py
│   ├── demo_mdlag_time.py / demo_mdlag_freq.py / demo_mdlag_ssm.py
│   ├── demo_gpfa_ssm.py / demo_lds.py
│   ├── demo_matern.py / demo_custom_kernel.py
│   └── demo_common.py
└── v1v2/           # same as notebooks/v1v2/ but CLI
    ├── demo_adm.py / demo_dlag.py / demo_dlag_ssm.py
    ├── demo_mdlag_time.py / demo_mdlag_freq.py / demo_mdlag_ssm.py
    ├── demo_custom_kernel.py
    └── v1v2_common.py
```

Each CLI demo accepts `--help` for the full flag list.

### Synthetic data — delay recovery

Synthetic multi-region data is sampled from exact Gaussian processes
with a **known ground-truth delay**, so the headline metric is
delay-recovery RMSE against truth.

```bash
# CLI (one method per command)
uv run python examples/synthetic/demo_adm.py
uv run python examples/synthetic/demo_dlag.py            # exact-GP DLAG
uv run python examples/synthetic/demo_dlag_ssm.py        # SSM-GP DLAG
uv run python examples/synthetic/demo_mdlag_time.py      # dense time-domain mDLAG
uv run python examples/synthetic/demo_mdlag_freq.py      # frequency-domain mDLAG
uv run python examples/synthetic/demo_mdlag_ssm.py       # mDLAG-SSM (Kalman + ARD)
uv run python examples/synthetic/demo_gpfa_ssm.py        # shared-only SSM-GP, no delay
uv run python examples/synthetic/demo_lds.py             # free-SSM baseline
uv run python examples/synthetic/demo_matern.py          # Matérn-5/2 kernel
uv run python examples/synthetic/demo_custom_kernel.py   # custom RQ kernel
```

Each run writes per-pair delay overlays, per-region latent traces, y
reconstruction, convergence trace, and `summary.json` into the
preset's output directory.

### Real data — V1/V2 visual cortex

The V1/V2 dataset shipped with the demos is from **Semedo et al.,
*Cortical Areas Interact through a Communication Subspace*, Neuron
2019** — see [Citation](#citation) below.

The shipped pickle ([`data/demo_v1v2_data.pkl`](data/demo_v1v2_data.pkl))
is one recording session, 400 trials, with spike counts
Gaussian-smoothed in time and z-scored so the linear-Gaussian
emission models in this library see well-behaved inputs. The layout is
a dict with `V1` / `V2` arrays of shape `(n_trials, T, n_neurons)` =
`(400, 64, 72)` / `(400, 64, 22)`.

Real recordings have **no ground-truth delay**, so the headline metric
is **held-out-neuron co-smoothing RMSE**: a fraction of neurons per
region is hidden from inference and predicted from the posterior latent
inferred on the remaining context neurons. Reported per region:
`holdout_psth_rmse_{V1, V2}` (PSTH-level prediction).

V1V2 demos vary the **`--split-seed`** (train/val/test partition) and
**average over `--n-holdout-seeds`** different held-out-neuron masks per
split. The 3-split-seed std is the reported method-stability error bar:

```bash
DATA=data/demo_v1v2_data.pkl      # swap in your own pickle here
SEEDS=(0 1 2)
N_HOLDOUT_SEEDS=3

for SPLIT_SEED in "${SEEDS[@]}"; do
    uv run python examples/v1v2/demo_adm.py \
        --data-path "$DATA" \
        --seed 0 --split-seed "${SPLIT_SEED}" \
        --holdout-seed 0 --n-holdout-seeds "${N_HOLDOUT_SEEDS}" \
        --out-dir "examples/v1v2/demo_outputs/adm/split_${SPLIT_SEED}"
done

# Other methods: swap `demo_adm.py` for any of
#   demo_dlag.py | demo_dlag_ssm.py
#   demo_mdlag_time.py | demo_mdlag_freq.py | demo_mdlag_ssm.py
#   demo_custom_kernel.py
```

Then aggregate the methods into one comparison:

```bash
uv run python examples/v1v2/compare_v1v2_runs.py \
    --label adm           --runs "examples/v1v2/demo_outputs/adm/split_*" \
    --label dlag          --runs "examples/v1v2/demo_outputs/dlag/split_*" \
    --label dlag_ssm      --runs "examples/v1v2/demo_outputs/dlag_ssm/split_*" \
    --label mdlag_time    --runs "examples/v1v2/demo_outputs/mdlag_time/split_*" \
    --label mdlag_freq    --runs "examples/v1v2/demo_outputs/mdlag_freq/split_*" \
    --label mdlag_ssm     --runs "examples/v1v2/demo_outputs/mdlag_ssm/split_*" \
    --label custom_kernel --runs "examples/v1v2/demo_outputs/custom_kernel/split_*" \
    --out-dir examples/v1v2/demo_outputs/_compare
```

---

## Directory layout

```
src/mbrila/
├── core/         abstract base classes + MultiRegionData container + LatentSpec
├── kernels/      MOSE (RBF) · Matérn-1/2, 3/2, 5/2 · BaseKernel ABC (user extension point)
├── delays/       NoDelay · FixedDelay · TimeVaryingDelay
├── dynamics/     MarkovianGPLatent (kernel → AR(P) lift) · ExactGPLatent · FreeLDSLatent
├── observations/ MultiRegionLinearObservation · ARDObservation
├── inference/    ExactEMEngine · KalmanEMEngine · VEMARDEngine (time / freq) · VEMKalmanARDEngine
│                 (parallel-scan Kalman filter/smoother, Särkkä & García-Fernández 2021)
├── init/         pCCA emission init · rank-1 deflation init · scale anchor
├── frequency/    FFT utilities for the frequency-domain mDLAG engine
├── models/       ADM · DLAG · MDLAG · GPFA · LDS — assembled presets
├── synthetic/    multi-region scenario generator (exact-GP sampling, configurable
│                 delay shapes / per-latent heterogeneity / SNR)
├── metrics/      evaluation metrics
└── utils/        device handling + shared helpers

examples/synthetic/      end-to-end CLI demos on synthetic data (ground-truth delay)
examples/v1v2/           end-to-end CLI demos on V1/V2 data (co-smoothing metric)
notebooks/synthetic/     Jupyter version of every synthetic demo
notebooks/v1v2/          Jupyter version of every V1V2 demo
```

---

## Evaluation metrics

- **Synthetic data** — delay-recovery RMSE: how closely the fitted
  delay matches the known ground-truth delay (in time bins).
- **Real data** — co-smoothing RMSE on held-out neurons (NLB-style),
  per region, both trial-mean (PSTH) and trial-by-trial. This is the metric that fairly compares model classes on real data, because full-set reconstruction RMSE saturates at the spike-noise floor.

Log-likelihood / ELBO traces are kept as **convergence diagnostics
only**, never as a cross-model performance metric — different engines
optimise different surrogates (joint LL, marginal LL, true ELBO,
proxy ELBO), and absolute values are not comparable across model
classes.

---

## Citation

If you use `mbrila`, please cite the ADM paper that introduces this
GP-SSM framework:

```bibtex
@inproceedings{li2025learning,
  title={Learning Time-Varying Multi-Region Brain Communications via Scalable Markovian Gaussian Processes},
  author={Li, Weihan and Wang, Yule and Li, Chengrui and Wu, Anqi},
  booktitle={International Conference on Machine Learning},
  pages={36021--36041},
  year={2025},
  organization={PMLR}
}
```

If you additionally use models reimplemented here, please also cite
their original publications:

**DLAG** — Gokcen et al., Nature Computational Science 2022.
[doi:10.1038/s43588-022-00282-5](https://doi.org/10.1038/s43588-022-00282-5)

**mDLAG** — Gokcen et al., NeurIPS 2023.
[nips.cc/virtual/2023/poster/70171](https://nips.cc/virtual/2023/poster/70171)

**fast-mDLAG** (the `--mdlag-engine freq` path) — Gokcen et al., Neural
Computation 2025. [doi:10.1162/neco.a.22](https://doi.org/10.1162/neco.a.22)

### Datasets

The **V1/V2 visual cortex** data used in `examples/v1v2/` and
`notebooks/v1v2/` is from:

> Semedo, J. D., Zandvakili, A., Machens, C. K., Yu, B. M., & Kohn, A.
> (2019). *Cortical Areas Interact through a Communication Subspace*.
> **Neuron**, 102(1), 249–259.
> [doi:10.1016/j.neuron.2019.01.026](https://doi.org/10.1016/j.neuron.2019.01.026)

If you use that data in published work, please cite Semedo et al. 2019
in addition to `mbrila`.

---

## Contributing

Contributions are warmly welcomed — new kernels, new presets, bug
fixes, documentation improvements, or simply opening an issue with
your use case. Open a PR or issue on
[GitHub](https://github.com/BRAINML-GT/MBRILA).

---

## License & Acknowledgements

`mbrila` is released under the **MIT License** — see [`LICENSE`](LICENSE).

`mbrila` is an **independent PyTorch reimplementation**: it does not
import or copy any upstream source code. Its models reimplement
algorithms from separate research codebases:

| Model | Reimplemented from | Original author | Original license |
|---|---|---|---|
| ADM | Adaptive Delay Model (Python) | Li et al. 2025 | MIT |
| DLAG | DLAG (MATLAB) | Evren Gokcen et al. 2022 | MIT |
| mDLAG / fast-mDLAG | fast-mDLAG (MATLAB) | Evren Gokcen et al. 2023, 2025 | MIT |

All upstream projects are MIT-licensed; their copyright notices are
reproduced in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) as
an acknowledgement.
