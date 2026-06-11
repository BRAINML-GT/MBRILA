"""Fixed scalar per-region per-latent delay (DLAG / mDLAG).

One real-valued delay per ``(region, latent)`` pair, optimised by EM. The
delay is unbounded in raw parameter space and squashed through a tanh so
the effective delay stays in ``(-D_max, +D_max)`` bins:

    δ_{r, k} = D_max · tanh(β_{r, k} / 2)

Region 0 is the reference and is fixed at exactly zero — only
``n_regions - 1`` rows of ``β`` are learnable parameters; the zero row is
prepended at materialisation time.

The ``/2`` inside ``tanh`` matches the DLAG MATLAB parameterisation
``δ = 2·D_max / (1 + exp(-β)) - D_max`` and gives ``dδ/dβ = (D_max/2) · (1 - tanh²(β/2))``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from mbrila.core.delay_spec import Delay


class FixedDelay(Delay):
    """Tanh-bounded scalar delay per ``(region, latent)`` pair.

    Parameters
    ----------
    n_regions:
        Number of regions. Region 0 is the (fixed-zero) reference.
    n_latent:
        Number of across-region latents this delay applies to.
    max_delay:
        Bound ``D_max`` on the effective delay magnitude, in bin units.
        Effective delay stays in ``(-D_max, +D_max)``.
    init_scale:
        Width of the uniform initialisation for the raw parameter ``β``
        on the non-reference rows. ``0`` (default) starts every delay at
        exactly zero.
    dtype:
        Floating point dtype for the parameter. ``torch.float32`` by
        default; pass ``torch.float64`` for DLAG / mDLAG fits on real data
        where exact-GP Cholesky is sensitive to precision.
    """

    n_latent: int
    max_delay: float
    beta: torch.nn.Parameter

    def __init__(
        self,
        n_regions: int,
        n_latent: int,
        *,
        max_delay: float,
        init_scale: float = 0.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(n_regions=n_regions, n_latent=n_latent)
        if max_delay <= 0:
            raise ValueError(f"max_delay must be positive; got {max_delay}")
        if init_scale < 0:
            raise ValueError(f"init_scale must be >= 0; got {init_scale}")
        self.max_delay = float(max_delay)
        if n_regions == 1:
            init = torch.zeros(0, n_latent, dtype=dtype)
        elif init_scale == 0.0:
            init = torch.zeros(n_regions - 1, n_latent, dtype=dtype)
        else:
            init = init_scale * (2.0 * torch.rand(n_regions - 1, n_latent, dtype=dtype) - 1.0)
        self.beta = torch.nn.Parameter(init)

    @property
    def is_time_varying(self) -> bool:
        return False

    def as_tensor(self, T: int | None = None) -> Tensor:
        """Return the ``(n_regions, n_latent)`` delay tensor in bin units.

        ``T`` is accepted for interface compatibility but ignored — the
        delay is time-invariant.
        """
        del T  # not used; fixed delays do not depend on T.
        if self.n_regions == 1:
            return self.beta.new_zeros(1, self.n_latent)
        delta_nz = self.max_delay * torch.tanh(0.5 * self.beta)
        zero_ref = self.beta.new_zeros(1, self.n_latent)
        return torch.cat([zero_ref, delta_nz], dim=0)

    def grad_delta_wrt_beta(self) -> Tensor:
        """Return ``∂δ / ∂β`` of shape ``(n_regions - 1, n_latent)``.

        Excludes the reference row (region 0) because that row is not a
        free parameter. Used by the DLAG M-step to chain analytical
        ``dK/dδ`` gradients into ``β``-space.
        """
        if self.n_regions == 1:
            return self.beta.new_zeros(0, self.n_latent)
        return 0.5 * self.max_delay * (1.0 - torch.tanh(0.5 * self.beta).square())

    def phase_at_freq(self, freqs: Tensor) -> Tensor:
        """Per-frequency phase shifts ``Q(f) = exp(-i · 2π · f · δ)``.

        The frequency-domain emission for DLAG/mDLAG factors the per-
        region time delays into a complex multiplicative phase shift:
        if the time-domain emission for region ``r`` reads
        ``y_r(t) = C_r · x(t − δ_r) + ...``, the unitary-FFT'd form is
        ``y_r[f] = C_r · diag(Q_r(f)) · x[f] + ...`` with
        ``Q_r(f) = exp(-i · 2π · f · δ_r)``. This matches fast-mDLAG's
        ``Q = exp(-1i*2*pi*freqs(f).*params.D)`` in ``inferX_freq.m``.

        Parameters
        ----------
        freqs:
            Real frequency tensor of shape ``(T,)`` in cycles-per-bin
            (e.g. the output of :func:`mbrila.frequency.centered_freqs`).

        Returns
        -------
        Complex tensor of shape ``(T, n_regions, n_latent)``. Region 0
        is the reference and always carries ``Q_0(f) = 1`` for all
        ``f`` because the reference delay is zero by construction.
        """
        if freqs.ndim != 1:
            raise ValueError(f"freqs must be 1-D; got shape {tuple(freqs.shape)}")
        if not freqs.is_floating_point():
            raise TypeError(f"freqs must be a real floating tensor; got dtype {freqs.dtype}")
        delta = self.as_tensor()  # (R, K_latent), real
        # Promote to the complex dtype matching the underlying parameter dtype.
        complex_dtype = torch.complex128 if delta.dtype == torch.float64 else torch.complex64
        # Use a tensor for 2π to keep mypy happy with dtype/device propagation.
        two_pi = torch.tensor(2.0 * torch.pi, dtype=freqs.dtype, device=freqs.device)
        # Exponent shape: (T, 1, 1) · (1, R, K) → (T, R, K)
        phase = -1j * two_pi * freqs.view(-1, 1, 1) * delta.to(device=freqs.device).unsqueeze(0)
        return torch.exp(phase.to(complex_dtype))
