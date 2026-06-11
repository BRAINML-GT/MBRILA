"""Optimiser helpers shared across inference engines.

Parameters that act as **linear shifts** (delay offsets, emission
bias) are excluded from weight decay, because pulling them toward zero
is both inconsistent with their semantic (they are not "scale"
weights) and empirically harmful: decayed-toward-zero δ causes
catastrophic ``delay_rmse`` blow-up on SSM-GP models.

The decay exclusion list is **name-based**: any parameter whose qualified
name (via ``named_parameters``) contains ``"raw_delay"``,
``"beta"``, or ``"d_param"`` skips weight decay. These cover:

- :class:`mbrila.delays.time_varying.TimeVaryingDelay.raw_delay` — ADM's δ(t)
- :class:`mbrila.delays.fixed.FixedDelay.beta` — DLAG/MDLAG/GPFA's δ
- :class:`mbrila.observations.multi_region.MultiRegionLinearObservation.d_param` — emission bias
"""

from __future__ import annotations

import torch


def build_grouped_adamw(
    model: torch.nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Build an AdamW with linear-shift parameters excluded from weight decay.

    Parameters whose qualified name (via :func:`named_parameters`) contains
    any of ``"raw_delay"``, ``"beta"``, or ``"d_param"`` are placed in a
    decay-free parameter group. All other parameters get the requested
    decay. With ``weight_decay=0`` the result is behaviourally identical
    to plain :class:`torch.optim.Adam`.

    See the module docstring for why this exclusion exists.
    """
    wd_params: list[torch.nn.Parameter] = []
    no_wd_params: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "raw_delay" in name or "beta" in name or "d_param" in name:
            no_wd_params.append(p)
        else:
            wd_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": wd_params, "weight_decay": weight_decay},
            {"params": no_wd_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )
