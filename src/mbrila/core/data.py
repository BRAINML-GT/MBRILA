"""Multi-region neural data container.

Throughout mbrila the convention is that the leading tensor dimension is the
trial dimension and is *always* preserved by inference / dynamics / observation
code paths. ``MultiRegionData`` enforces and documents this convention.

Layout
------
``y`` has shape ``(n_trials, T, sum(y_dims))``. Per-region columns are
contiguous, in the order given by ``y_dims``::

    y[:, :, region_slices[r]]   # neurons of region r

Variable-length trials are tracked by ``trial_lengths`` (a ``(n_trials,)``
int64 tensor). Time bins beyond a trial's length should be masked by inference
engines that consume them; v1 model implementations may assume equal-length
trials and ignore this field, but the container always carries it so future
backends can opt in without changing the data layout.

The ``fft_cache`` field is a stash for frequency-domain inference engines
(Phase 4). Its concrete type lives in :mod:`mbrila.frequency`; here we keep it
as ``object | None`` so the data container does not depend on Phase-4 modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Self

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from torch.utils.data import DataLoader, Dataset


def _slices_from_dims(y_dims: tuple[int, ...]) -> tuple[slice, ...]:
    """Build per-region column slices from neuron counts."""
    out: list[slice] = []
    start = 0
    for n in y_dims:
        out.append(slice(start, start + n))
        start += n
    return tuple(out)


@dataclass(slots=True)
class MultiRegionData:
    """Trial-batched multi-region neural recordings.

    Parameters
    ----------
    y:
        Float tensor of shape ``(n_trials, T, sum(y_dims))``. Continuous
        values (e.g. firing-rate estimates) or pre-binned spike counts cast
        to float. Inference engines treat it as a Gaussian observation by
        default; a future Poisson observation will reinterpret it as integer
        counts.
    y_dims:
        Number of neurons in each region, in column order.
    bin_width:
        Bin width in milliseconds. Used to convert dimensionless model
        parameters (timescale gamma, delay) to physical units when reporting.
    trial_lengths:
        Optional ``(n_trials,)`` int64 tensor of true trial lengths
        (``<= T``). When ``None`` all trials are assumed to be of length
        ``T`` (the second axis of ``y``).
    fft_cache:
        Opaque slot for frequency-domain engines to cache ``torch.fft.fft(y)``
        and friends.
    """

    y: Tensor
    y_dims: tuple[int, ...]
    bin_width: float = 1.0
    trial_lengths: Tensor | None = None
    fft_cache: object | None = None
    region_slices: tuple[slice, ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.y.ndim != 3:
            raise ValueError(f"y must have shape (n_trials, T, n_neurons); got {tuple(self.y.shape)}")
        total = int(sum(self.y_dims))
        if total != int(self.y.shape[-1]):
            raise ValueError(f"sum(y_dims)={total} does not match y.shape[-1]={int(self.y.shape[-1])}")
        if any(d <= 0 for d in self.y_dims):
            raise ValueError(f"y_dims must all be positive; got {self.y_dims}")
        if self.bin_width <= 0:
            raise ValueError(f"bin_width must be positive; got {self.bin_width}")
        if self.trial_lengths is not None:
            tl = self.trial_lengths
            if tl.ndim != 1 or int(tl.shape[0]) != int(self.y.shape[0]):
                raise ValueError(
                    f"trial_lengths must have shape (n_trials,)={int(self.y.shape[0])}; got {tuple(tl.shape)}"
                )
            if not tl.dtype.is_signed or tl.dtype.is_floating_point:
                raise ValueError(f"trial_lengths must be an integer tensor; got dtype {tl.dtype}")
            T = int(self.y.shape[1])
            if int(tl.max().item()) > T or int(tl.min().item()) <= 0:
                raise ValueError(
                    f"trial_lengths must be in [1, T={T}]; got "
                    f"min={int(tl.min().item())}, max={int(tl.max().item())}"
                )
        # Region slices are derived; keep them in the dataclass for cheap
        # repeated lookup without recomputing.
        object.__setattr__(self, "region_slices", _slices_from_dims(self.y_dims))

    # --- shape helpers --------------------------------------------------

    @property
    def n_trials(self) -> int:
        return int(self.y.shape[0])

    @property
    def T(self) -> int:
        return int(self.y.shape[1])

    @property
    def n_regions(self) -> int:
        return len(self.y_dims)

    @property
    def n_neurons(self) -> int:
        return int(self.y.shape[-1])

    @property
    def device(self) -> torch.device:
        return self.y.device

    @property
    def dtype(self) -> torch.dtype:
        return self.y.dtype

    # --- moves & projections -------------------------------------------

    def to(self, *, device: str | torch.device | None = None, dtype: torch.dtype | None = None) -> Self:
        """Return a copy moved to the given device / dtype.

        The underlying tensors are not copied if they already match.
        """
        new_y = self.y
        if device is not None or dtype is not None:
            new_y = new_y.to(device=device, dtype=dtype)
        new_lengths = self.trial_lengths
        if new_lengths is not None and device is not None:
            new_lengths = new_lengths.to(device=device)
        # FFT cache is per-device/dtype; invalidate on a move.
        new_cache = self.fft_cache if (device is None and dtype is None) else None
        return replace(self, y=new_y, trial_lengths=new_lengths, fft_cache=new_cache)

    def split_by_region(self) -> tuple[Tensor, ...]:
        """Return per-region views of ``y``, each retaining the trial batch dim.

        The returned tensors are *views* and share storage with ``self.y``.
        """
        return tuple(self.y[..., s] for s in self.region_slices)

    def region(self, r: int) -> Tensor:
        """Return a view of region ``r``'s neurons."""
        return self.y[..., self.region_slices[r]]

    # --- iteration & loaders -------------------------------------------

    def __len__(self) -> int:
        return self.n_trials

    def iter_minibatches(self, batch_size: int) -> Iterator[Self]:
        """Yield ``MultiRegionData`` slices of size ``batch_size``.

        This iterates over *trials* but does so in fixed-size chunks; each
        yielded chunk preserves the batched layout so downstream code never
        sees a Python loop over individual trials.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}")
        n = self.n_trials
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield self._take_trials(slice(start, end))

    def _take_trials(self, idx: slice | Tensor) -> Self:
        new_y = self.y[idx]
        new_lengths = self.trial_lengths[idx] if self.trial_lengths is not None else None
        return replace(self, y=new_y, trial_lengths=new_lengths, fft_cache=None)

    def as_torch_dataset(self) -> Dataset[Any]:
        """Adapter for ``torch.utils.data.DataLoader`` users.

        Each item is ``(y_trial, length)`` where ``length`` is the trial
        length (or ``T`` if no per-trial lengths are tracked). Engines that
        want their own batched layout should iterate ``iter_minibatches``
        instead and skip the ``DataLoader`` machinery entirely.
        """
        from torch.utils.data import TensorDataset

        lengths = (
            self.trial_lengths
            if self.trial_lengths is not None
            else torch.full((self.n_trials,), self.T, dtype=torch.int64, device=self.device)
        )
        return TensorDataset(self.y, lengths)

    def as_loader(self, batch_size: int, *, shuffle: bool = False) -> DataLoader[Any]:
        """Wrap ``self`` in a ``DataLoader`` for ad-hoc usage."""
        from torch.utils.data import DataLoader

        return DataLoader(self.as_torch_dataset(), batch_size=batch_size, shuffle=shuffle)
