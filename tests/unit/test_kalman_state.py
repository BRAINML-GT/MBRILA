"""Unit tests for GaussianState."""

from __future__ import annotations

import math

import pytest
import torch

from mbrila.inference.kalman import GaussianState


class TestShape:
    def test_basic_shape(self) -> None:
        s = GaussianState(mean=torch.zeros(3, 5), covariance=torch.eye(5).expand(3, 5, 5))
        assert s.state_dim == 5
        assert s.dtype == torch.float32

    def test_rejects_dim_mismatch(self) -> None:
        with pytest.raises(ValueError, match="trailing dim"):
            GaussianState(mean=torch.zeros(3), covariance=torch.zeros(5, 5))

    def test_rejects_non_square_cov(self) -> None:
        with pytest.raises(ValueError, match="square"):
            GaussianState(mean=torch.zeros(5), covariance=torch.zeros(5, 4))

    def test_rejects_precision_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="precision shape"):
            GaussianState(
                mean=torch.zeros(5),
                covariance=torch.eye(5),
                precision=torch.eye(4),
            )


class TestClone:
    def test_clone_independent(self) -> None:
        s = GaussianState(mean=torch.ones(3), covariance=torch.eye(3))
        c = s.clone()
        c.mean[0] = 99.0
        assert s.mean[0].item() == 1.0
        assert c.mean[0].item() == 99.0

    def test_clone_carries_precision(self) -> None:
        prec = torch.eye(3) * 2
        s = GaussianState(mean=torch.zeros(3), covariance=torch.eye(3), precision=prec)
        c = s.clone()
        assert c.precision is not None
        assert torch.equal(c.precision, prec)


class TestLogDensity:
    def test_unit_gaussian_at_origin(self) -> None:
        # log N(0 | 0, I_d) = -d/2 log(2π)
        d = 4
        s = GaussianState(mean=torch.zeros(d), covariance=torch.eye(d))
        x = torch.zeros(d)
        expected = -0.5 * d * math.log(2.0 * math.pi)
        assert s.log_density(x).item() == pytest.approx(expected, abs=1e-6)

    def test_batched_log_density(self) -> None:
        d = 3
        B = 7
        s = GaussianState(
            mean=torch.zeros(B, d),
            covariance=torch.eye(d).expand(B, d, d).contiguous(),
        )
        xs = torch.randn(B, d)
        out = s.log_density(xs)
        # Compare to per-batch identity Gaussian log-pdf.
        expected = -0.5 * (xs.pow(2).sum(dim=-1) + d * math.log(2.0 * math.pi))
        torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
