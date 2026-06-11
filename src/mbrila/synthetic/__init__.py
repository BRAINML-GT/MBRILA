"""Synthetic data generators for benchmarking multi-region SSM methods.

This module is the public, model-agnostic API: it builds a known
ground-truth multi-region SSM via mbrila's own ADM components, then
samples observations under controlled (delay shape, SNR, trial count,
within-region structure) regimes. The same generator drives recovery
tests, ablations across methods (ADM / DLAG / mDLAG / MRM-GP), and
paper figures.
"""

from mbrila.synthetic.multiregion import (
    DelayShape,
    MultiRegionScenario,
    SyntheticDataset,
    generate_multiregion_synthetic,
)
from mbrila.synthetic.scenarios import (
    COMPLEX_WITHIN,
    EASY,
    FEW_TRIALS,
    HARD_MULTI_LATENT,
    LOW_SNR,
    MULTI_LATENT,
    PRESETS,
    SMOOTH_DELAY,
)

__all__ = [
    "COMPLEX_WITHIN",
    "EASY",
    "FEW_TRIALS",
    "HARD_MULTI_LATENT",
    "LOW_SNR",
    "MULTI_LATENT",
    "PRESETS",
    "SMOOTH_DELAY",
    "DelayShape",
    "MultiRegionScenario",
    "SyntheticDataset",
    "generate_multiregion_synthetic",
]
