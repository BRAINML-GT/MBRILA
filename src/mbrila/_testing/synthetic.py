"""Backward-compat shim for ``mbrila._testing.synthetic``.

The actual generator lives in :mod:`mbrila.synthetic`. This module keeps
the legacy ``generate_adm_synthetic`` signature so existing recovery
tests do not need to be rewritten. New code should use
:func:`mbrila.synthetic.generate_multiregion_synthetic` together with
:class:`mbrila.synthetic.MultiRegionScenario`.
"""

from __future__ import annotations

import torch

from mbrila.synthetic.multiregion import (
    MultiRegionScenario,
    SyntheticDataset,
    generate_multiregion_synthetic,
)

__all__ = ["SyntheticDataset", "generate_adm_synthetic"]


def generate_adm_synthetic(
    *,
    n_trials: int,
    T: int,
    y_dims: tuple[int, ...],
    n_across: int = 1,
    n_within: int = 1,
    lag_across: int = 5,
    lag_within: int = 2,
    sigma_across: float = 0.05,
    sigma_within: float = 0.05,
    delay_amplitude: float = 1.0,
    snr: float = 5.0,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> SyntheticDataset:
    """Legacy entry point preserved for the recovery test.

    Wraps :func:`generate_multiregion_synthetic` with a sin-shaped delay
    (the original behaviour). Prefer the public API in
    :mod:`mbrila.synthetic` for new code.
    """
    scenario = MultiRegionScenario(
        n_trials=n_trials,
        T=T,
        y_dims=y_dims,
        n_across=n_across,
        n_within=n_within,
        lag_across=lag_across,
        lag_within=lag_within,
        sigma_across=sigma_across,
        sigma_within=sigma_within,
        delay_shape="sin",
        delay_amplitude=delay_amplitude,
        snr=snr,
        seed=seed,
        dtype=dtype,
        device=device,
    )
    return generate_multiregion_synthetic(scenario)
