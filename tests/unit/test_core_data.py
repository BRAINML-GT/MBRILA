"""Sanity checks for the MultiRegionData container."""

from __future__ import annotations

import pytest
import torch

from mbrila import MultiRegionData


def make_data(n_trials: int = 4, T: int = 10, y_dims: tuple[int, ...] = (3, 5)) -> MultiRegionData:
    y = torch.randn(n_trials, T, sum(y_dims))
    return MultiRegionData(y=y, y_dims=y_dims, bin_width=20.0)


class TestShape:
    def test_basic_properties(self) -> None:
        d = make_data()
        assert d.n_trials == 4
        assert d.T == 10
        assert d.n_regions == 2
        assert d.n_neurons == 8
        assert d.region_slices == (slice(0, 3), slice(3, 8))

    def test_split_by_region_preserves_batch(self) -> None:
        d = make_data(n_trials=7, T=11, y_dims=(2, 3, 4))
        parts = d.split_by_region()
        assert len(parts) == 3
        for part, n in zip(parts, (2, 3, 4), strict=True):
            assert part.shape == (7, 11, n)
        # Views, not copies — modifying part 0 mutates d.y.
        parts[0][0, 0, 0] = 999.0
        assert d.y[0, 0, 0].item() == 999.0

    def test_region_indexer(self) -> None:
        d = make_data(y_dims=(3, 5))
        assert d.region(0).shape == (4, 10, 3)
        assert d.region(1).shape == (4, 10, 5)


class TestValidation:
    def test_rejects_2d_input(self) -> None:
        with pytest.raises(ValueError, match="must have shape"):
            MultiRegionData(y=torch.randn(10, 8), y_dims=(3, 5))

    def test_rejects_dim_mismatch(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            MultiRegionData(y=torch.randn(2, 4, 7), y_dims=(3, 5))

    def test_rejects_zero_neurons_in_region(self) -> None:
        with pytest.raises(ValueError, match="must all be positive"):
            MultiRegionData(y=torch.randn(2, 4, 8), y_dims=(0, 8))

    def test_rejects_nonpositive_bin_width(self) -> None:
        with pytest.raises(ValueError, match="bin_width"):
            MultiRegionData(y=torch.randn(2, 4, 8), y_dims=(3, 5), bin_width=0)

    def test_rejects_float_trial_lengths(self) -> None:
        with pytest.raises(ValueError, match="integer tensor"):
            MultiRegionData(
                y=torch.randn(2, 4, 8),
                y_dims=(3, 5),
                trial_lengths=torch.tensor([3.0, 4.0]),
            )

    def test_rejects_oversized_trial_length(self) -> None:
        with pytest.raises(ValueError, match=r"\[1, T="):
            MultiRegionData(
                y=torch.randn(2, 4, 8),
                y_dims=(3, 5),
                trial_lengths=torch.tensor([5, 3], dtype=torch.int64),
            )


class TestMoves:
    def test_to_dtype_is_lazy(self) -> None:
        d = make_data().to(dtype=torch.float64)
        assert d.dtype == torch.float64
        assert d.y.dtype == torch.float64

    def test_to_invalidates_fft_cache(self) -> None:
        d = make_data()
        # Stash a sentinel — any move should drop it.
        d.fft_cache = "sentinel"
        moved = d.to(dtype=torch.float64)
        assert moved.fft_cache is None
        # But a no-op move keeps it.
        same = d.to()
        assert same.fft_cache == "sentinel"


class TestMinibatching:
    def test_iter_minibatches_partitions_trials(self) -> None:
        d = make_data(n_trials=10, T=5, y_dims=(2, 3))
        chunks = list(d.iter_minibatches(batch_size=4))
        assert [c.n_trials for c in chunks] == [4, 4, 2]
        # Re-stacking should reproduce the original tensor.
        reassembled = torch.cat([c.y for c in chunks], dim=0)
        assert torch.equal(reassembled, d.y)

    def test_iter_minibatches_carries_lengths(self) -> None:
        lengths = torch.tensor([5, 4, 3, 2, 5, 4, 3, 2, 5, 4], dtype=torch.int64)
        d = MultiRegionData(
            y=torch.randn(10, 5, 8),
            y_dims=(3, 5),
            trial_lengths=lengths,
        )
        chunks = list(d.iter_minibatches(batch_size=3))
        assert torch.equal(chunks[0].trial_lengths, lengths[:3])
        assert torch.equal(chunks[-1].trial_lengths, lengths[9:10])

    def test_iter_minibatches_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match="batch_size"):
            list(make_data().iter_minibatches(batch_size=0))
