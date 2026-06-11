"""FFT utilities for the frequency-domain DLAG / mDLAG engines.

Two responsibilities:

1. **Frequency vector convention.** fast-mDLAG (and the wider DLAG
   literature) parameterises spectral densities ``S(f)`` against the
   *centered* frequency grid

   ::

       f[k] = (k - floor(T/2)) / T,    k = 0, …, T-1

   which corresponds to MATLAB's ``(-floor(T/2):floor((T-1)/2))/T``
   layout — what you'd get after applying ``fftshift`` to the natural
   ``torch.fft.fft`` output. We expose :func:`centered_freqs` so engines
   never have to roll their own.

2. **Unitary FFT.** mDLAG's ``inferX_freq`` operates on the *unitary*
   FFT ``y_f = fft(y) / √T``. The unitary normalisation makes the DFT
   matrix orthonormal so the per-trial Gaussian quadratic form
   ``y^T R⁻¹ y`` is preserved across the time / frequency boundary
   (Parseval). We expose :func:`unitary_fft` / :func:`unitary_ifft`
   wrappers that combine the ``1/√T`` factor with ``fftshift`` /
   ``ifftshift`` so callers receive / supply data in the centered
   convention directly.

Conventions
-----------
- "Unitary" here means the linear map ``x → fftshift(fft(x)) / √T``.
- The forward / inverse pair satisfies ``unitary_ifft(unitary_fft(x)) = x``.
- Real input → complex output (``torch.complex128`` by default per
  :mod:`mbrila`'s float64 policy); the inverse maps back to whatever
  dtype the caller's input had (real if the inverse round-trips a
  unitary-FFT'd real signal, complex otherwise).

Trial-batched
-------------
All functions are pure tensor ops with no Python-level batching
over trials.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def centered_freqs(
    T: int,
    *,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> Tensor:
    """Centered frequency grid for a length-``T`` signal.

    Returns a tensor of shape ``(T,)`` with entries
    ``f[k] = (k - floor(T/2)) / T``. Equivalent to MATLAB's
    ``((-floor(T/2):floor((T-1)/2))/T).'``.

    For even ``T`` the layout is ``[-T/2, …, T/2 - 1] / T``; for odd
    ``T`` it is ``[-(T-1)/2, …, (T-1)/2] / T``. The zero-frequency bin
    sits at index ``floor(T/2)`` (0-indexed).
    """
    if T < 1:
        raise ValueError(f"T must be >= 1; got {T}")
    return (torch.arange(T, dtype=dtype, device=device) - (T // 2)) / float(T)


def zero_freq_index(T: int) -> int:
    """0-based index of ``f = 0`` in :func:`centered_freqs`."""
    return int(T // 2)


def unitary_fft(x: Tensor, *, dim: int = -1) -> Tensor:
    """Unitary forward DFT in the centered convention.

    ``X = fftshift(fft(x, dim=dim), dim=dim) / √T``.

    Parameters
    ----------
    x:
        Real or complex tensor; the transform is taken along ``dim``.
    dim:
        Axis along which to FFT. Defaults to the last axis.

    Returns
    -------
    Complex tensor with the same shape as ``x``. The k-th entry
    corresponds to frequency ``centered_freqs(T)[k]``.
    """
    T = int(x.shape[dim])
    X: Tensor = torch.fft.fft(x, dim=dim)
    X = torch.fft.fftshift(X, dim=dim)
    return X / math.sqrt(T)


def unitary_ifft(X: Tensor, *, dim: int = -1) -> Tensor:
    """Unitary inverse DFT, the inverse of :func:`unitary_fft`.

    ``x = √T · ifft(ifftshift(X, dim=dim), dim=dim)``.

    Round-trips :func:`unitary_fft` exactly; for an input that came
    from a real signal, the imaginary part of the output is
    floating-point noise and may be discarded by the caller via
    ``.real``.
    """
    T = int(X.shape[dim])
    X = torch.fft.ifftshift(X, dim=dim)
    x: Tensor = torch.fft.ifft(X, dim=dim)
    return math.sqrt(T) * x
