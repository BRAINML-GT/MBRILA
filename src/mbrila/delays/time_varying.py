"""Time-varying per-region delay (ADM's adaptive delay).

A learnable trajectory ``δ_r(t)`` of shape ``(T, num_regions)`` smoothed
along the time axis by a small Gaussian kernel. The first region is the
reference and is fixed at zero — only ``num_regions - 1`` columns are
parameters.

The smoothing kernel is the same as ADM's: a 6-tap Gaussian with
``σ = 0.05``, applied via ``F.conv1d`` with reflect padding. Smoothing
is essential because raw per-time delay parameters tend to develop
high-frequency noise during training (each ``δ(t)`` only weakly couples
to the likelihood through one bin).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from mbrila.core.delay_spec import Delay


def smoothing_params_for_kernel_sigma(kernel_sigma: float) -> tuple[int, float]:
    """Derive δ(t) smoothing parameters from the latent GP kernel σ.

    The latent's MOSE kernel ``K(τ) = exp(-σ/2 · τ²)`` has correlation
    length ``τ_x = 1/√σ``. The time-varying delay ``δ(t)`` should vary no
    faster than the latent ``x(t)`` it modulates — any wiggle on a
    shorter timescale gets absorbed by ``x``'s own autocorrelation and
    just fits noise. So the smoothing kernel is matched to the latent
    timescale:

    * ``smoothing_sigma = kernel_sigma``  → smoothing timescale = ``τ_x``.
    * ``smoothing_size  = round(4 / √σ)`` → window covers ±2σ of the
      resulting Gaussian (≈ 95% of its mass), floored at 8 taps.
    """
    if kernel_sigma <= 0:
        raise ValueError(f"kernel_sigma must be positive; got {kernel_sigma}")
    smoothing_sigma = float(kernel_sigma)
    smoothing_size = max(8, round(4.0 / math.sqrt(smoothing_sigma)))
    return smoothing_size, smoothing_sigma


def _gaussian_kernel_1d(size: int, sigma: float, *, dtype: torch.dtype) -> Tensor:
    """Return a normalised 1-D Gaussian kernel of exactly ``size`` taps.

    For ``size = 2k`` the taps are ``[-(k-1), …, k]`` (asymmetric by one
    bin); for ``size = 2k+1`` they are ``[-k, …, k]`` (symmetric).
    """
    if size < 1:
        raise ValueError(f"size must be >= 1; got {size}")
    half = (size - 1) // 2
    if size % 2 == 1:
        x = torch.arange(-half, half + 1, dtype=dtype)
    else:
        x = torch.arange(-(half), half + 2, dtype=dtype)
    assert x.shape[0] == size
    k = torch.exp(-0.5 * sigma * x.square())
    return (k / k.sum()).view(1, 1, -1)


class TimeVaryingDelay(Delay):
    """Per-region per-latent learnable delay trajectory with Gaussian smoothing.

    Parameters
    ----------
    n_regions:
        Number of regions. Region 0 is the (fixed-zero) reference.
    n_latent:
        Number of cross-region latents the delay applies to. The
        parameter tensor stores one trajectory per ``(latent, region)``
        pair; the first region's trajectory is identically zero (not
        learned).
    T:
        Trial length in bins.
    smoothing_size:
        Width of the Gaussian smoothing kernel. Must be positive; ADM
        uses 6.
    smoothing_sigma:
        ``σ`` of the Gaussian smoothing kernel, in inverse squared bins
        (``exp(-½ σ x²)``).
    init_scale:
        Initialisation scale for the unsmoothed delay parameters
        (uniform in ``(-init_scale, +init_scale)``). ``0`` (the ADM
        default) starts every delay at exactly zero.
    """

    n_latent: int

    def __init__(
        self,
        n_regions: int,
        n_latent: int,
        *,
        T: int,
        smoothing_size: int = 6,
        smoothing_sigma: float = 0.05,
        init_scale: float = 0.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__(n_regions=n_regions, n_latent=n_latent)
        if T < 1:
            raise ValueError(f"T must be >= 1; got {T}")
        if smoothing_size < 1:
            raise ValueError(f"smoothing_size must be >= 1; got {smoothing_size}")
        if smoothing_sigma <= 0:
            raise ValueError(f"smoothing_sigma must be positive; got {smoothing_sigma}")
        self.T = T
        self._smoothing_size = smoothing_size
        # Raw, unsmoothed trajectory for the non-reference regions.
        # Shape: (T, n_regions - 1, n_latent). Reference region is fixed at 0.
        if n_regions == 1:
            # Degenerate: no delay to learn (a single-region model has nothing
            # to align). We still register an empty parameter so state_dict is
            # consistent across configurations.
            init = torch.zeros(T, 0, n_latent, dtype=dtype)
        else:
            init = (
                torch.zeros(T, n_regions - 1, n_latent, dtype=dtype)
                if init_scale == 0.0
                else (init_scale * (2 * torch.rand(T, n_regions - 1, n_latent, dtype=dtype) - 1))
            )
        self.raw_delay = torch.nn.Parameter(init)
        self.register_buffer(
            "_smooth_kernel", _gaussian_kernel_1d(smoothing_size, smoothing_sigma, dtype=dtype)
        )

    @property
    def is_time_varying(self) -> bool:
        return True

    def _smooth(self, raw: Tensor) -> Tensor:
        """Apply 1-D Gaussian smoothing along the time axis with reflect padding.

        ``raw`` has shape ``(T, num_regions - 1, n_latent)``. The output has
        the same shape; padding chosen so that the temporal length is
        preserved.
        """
        if raw.shape[1] == 0:
            return raw  # nothing to smooth in the single-region case
        T_dim, R_minus_1, L = raw.shape
        # The buffer registration types ``self._smooth_kernel`` as
        # ``Tensor | Module`` to mypy's eyes; assert it back to a Tensor.
        kernel = self._smooth_kernel
        assert isinstance(kernel, Tensor)
        pad_right = int(kernel.shape[-1]) - 1

        # F.conv1d expects (batch, channels, length). We collapse (R-1, L) into
        # the channel dim and treat them as independent sequences.
        x = raw.permute(1, 2, 0).reshape(R_minus_1 * L, 1, T_dim)  # (R*L, 1, T)
        x_pad = F.pad(x, (0, pad_right), mode="reflect")
        y = F.conv1d(x_pad, kernel)  # (R*L, 1, T)
        out = y.view(R_minus_1, L, T_dim).permute(2, 0, 1).contiguous()  # (T, R-1, L)
        return out[:T_dim]

    def as_tensor(self, T: int | None = None) -> Tensor:
        """Return the full ``(T, n_regions, n_latent)`` delay tensor.

        ``T`` is honoured only if it equals the configured trial length;
        otherwise a ``ValueError`` is raised. (This is a deliberate
        limitation — the smoothing kernel is fixed-width and reflect
        padding is anchored to the model's ``T``.)
        """
        if T is not None and T != self.T:
            raise ValueError(f"TimeVaryingDelay was built for T={self.T}; got T={T}")
        smoothed = self._smooth(self.raw_delay)  # (T, n_regions - 1, n_latent)
        zero_ref = torch.zeros(self.T, 1, self.n_latent, dtype=smoothed.dtype, device=smoothed.device)
        full = torch.cat([zero_ref, smoothed], dim=1)  # (T, n_regions, n_latent)
        return full
