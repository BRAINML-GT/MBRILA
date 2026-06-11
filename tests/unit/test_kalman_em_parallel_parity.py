"""Parity test for ``KalmanEMEngine.use_parallel``.

Auto-dispatch picks parallel-scan on CUDA and sequential on CPU, but
both code paths must produce the same ``(filter, smoother)`` output up
to float64 round-off. This regression guard catches divergence if
either implementation drifts.
"""

from __future__ import annotations

import torch

from mbrila import ADM, KalmanEMEngine, LatentSpec, MOSEKernel


def _build_model_and_data() -> tuple[ADM, object]:
    spec = LatentSpec(n_across=1, n_within=(1, 1))
    model = ADM(
        latent_spec=spec,
        y_dims=(3, 3),
        T=6,
        kernel_factory_across=lambda: MOSEKernel(num_regions=2, init_sigma=0.1),
        kernel_factory_within=lambda: MOSEKernel(num_regions=1, init_sigma=0.1),
        device="cpu",
        dtype=torch.float64,
    )
    data = model.sample(n_trials=3, T=6, seed=0)
    return model, data


class TestKalmanEMParallelSequentialParity:
    def test_filter_smoother_outputs_match(self) -> None:
        model, data = _build_model_and_data()

        def run(use_parallel: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            engine = KalmanEMEngine(use_parallel=use_parallel)
            A, Q, H_eff, diag_R, d = engine._assemble(model)
            R = torch.diag(diag_R)
            m0, P0 = engine._initial_prior(A)
            y_centred = data.y - d
            _, _, s_means, s_covs, pair = engine._filter_then_smooth(y_centred, A, Q, H_eff, R, m0, P0)
            return s_means, s_covs, pair

        s_p, c_p, pair_p = run(True)
        s_s, c_s, pair_s = run(False)
        torch.testing.assert_close(s_p, s_s, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(c_p, c_s, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(pair_p, pair_s, atol=1e-9, rtol=1e-9)
