"""Identically-zero delay (GPFA / multi-region GPFA / naive SSM).

A degenerate :class:`Delay` whose :meth:`as_tensor` is constant zero. It
exists so callers can wire a uniform ``Delay`` slot regardless of whether
the model has region-to-region delays: GPFA-style models use it to assert
"no delay", and the dynamics layer can dispatch on the delay type rather
than on ``delay is None``. Carries no learnable parameters; the inherited
:meth:`phase_shift` reduces to all-ones as expected.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mbrila.core.delay_spec import Delay


class NoDelay(Delay):
    """Per-region per-latent delay model that is identically zero.

    Parameters
    ----------
    n_regions:
        Number of regions. The ``r=0`` reference convention is preserved
        (every row is zero so the reference invariant is trivially met).
    n_latent:
        Number of latents this delay applies to.
    dtype:
        Floating point dtype for the materialised tensor. ``torch.float64``
        by default to match the rest of the library.
    """

    dtype: torch.dtype

    def __init__(
        self,
        n_regions: int,
        n_latent: int,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(n_regions=n_regions, n_latent=n_latent)
        self.dtype = dtype
        # Register a zero-element buffer so ``.to(device=...)`` on the module
        # still has somewhere to land its device/dtype routing decisions.
        self.register_buffer("_device_anchor", torch.zeros(0, dtype=dtype), persistent=False)

    @property
    def is_time_varying(self) -> bool:
        return False

    def as_tensor(self, T: int | None = None) -> Tensor:
        """Return an ``(n_regions, n_latent)`` zero tensor.

        ``T`` is accepted for interface compatibility but ignored — the
        delay is time-invariant (and zero).
        """
        del T  # not used; the zero delay does not depend on T.
        anchor = self._device_anchor
        assert isinstance(anchor, Tensor)
        return torch.zeros(self.n_regions, self.n_latent, dtype=anchor.dtype, device=anchor.device)
