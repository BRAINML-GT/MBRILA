"""Non-Markovian exact-GP latent dynamics for DLAG / mDLAG.

DLAG-style models drop the Markovian lifted-LDS construction used by ADM
and instead retain the full ``(M·T, M·T)`` time-domain GP covariance
``K_big``. The state at each time bin is laid out region-by-region:

    s_t = [reg0_across (K_a), reg0_within (K_w[0]),
           reg1_across (K_a), reg1_within (K_w[1]),
           ...]

so the slot index of "region ``r``, across latent ``k``" is
``region_base[r] + k`` and "region ``r``, within latent ``l``" is
``region_base[r] + K_a + l`` with
``region_base[r] = Σ_{r'<r} (K_a + K_w[r'])``. The flat row/column index
into ``K_big`` is ``t · M + slot``.

For across latent ``k`` the kernel is region-coupled through delays:

    cov(x_k(r1, t1), x_k(r2, t2)) = (1 - ε_a[k]) · k(Δt)
                                      + ε_a[k] · 𝟙[r1=r2 ∧ t1=t2]
    Δt = (t1 - t2) - (δ_{r1, k} - δ_{r2, k})

where ``k(τ)`` is the per-block kernel's stationary covariance (any
:class:`~mbrila.kernels.base.BaseKernel`; not necessarily RBF). Different
across latents are independent (block-diagonal in ``k``). Within latents
(region ``r``, latent ``l``) have no delay and are independent across
regions and latents.

The exact-GP path is what makes DLAG distinct from ADM at small ``T``:
no Markov-order truncation, delays appear directly in the kernel
without lag-block jitter, and the inference engine pays a single
``(M·T)³`` Cholesky in exchange for tight likelihood estimates.

Kernel decoupling
-----------------
Each across / within block owns an independent kernel instance built by
the caller-supplied ``kernel_factory_*`` zero-arg callables. The kernel
contract is the standard :class:`~mbrila.kernels.base.BaseKernel`:

- :meth:`BaseKernel.cov(τ)` for the time-domain construction.
- :meth:`BaseKernel.spectral_density(ω)` for the frequency-domain
  construction (required for ``cov_freq``; raises if the kernel does not
  implement it).

The noise floor ``ε`` is a dynamics-level decision (not a kernel
property) and is applied via the standard ``(1 - ε) · k(τ) + ε · I``
blend inside this module.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from mbrila.delays.fixed import FixedDelay
from mbrila.kernels.base import BaseKernel


class ExactGPLatent(nn.Module):
    """Exact GP latent dynamics over the full DLAG latent space.

    Owns per-block kernel instances (one :class:`BaseKernel` per across
    latent + one per (region, within-latent) pair) and the
    :class:`FixedDelay` controlling across-latent region phase. Within
    latents have no cross-region coupling.

    Parameters
    ----------
    n_regions:
        Number of regions ``R``.
    n_across:
        Number of across-region latents ``K_a``. ``0`` is allowed (a
        purely region-local model — degenerate but supported for unit
        tests).
    n_within:
        Per-region within-latent counts ``(K_w[0], …, K_w[R-1])``. May
        contain zeros.
    kernel_factory_across, kernel_factory_within:
        Zero-arg callables that return a freshly-constructed
        :class:`~mbrila.kernels.base.BaseKernel`. Called once per block
        so each latent owns an independent parameter set. Any subclass
        of ``BaseKernel`` is accepted; the frequency engine additionally
        requires :meth:`BaseKernel.spectral_density` to be implemented.
    delay:
        :class:`FixedDelay` instance with ``n_regions == R`` and
        ``n_latent == K_a``. Required when ``K_a > 0`` and ignored
        otherwise (a small empty instance is constructed internally if
        the caller passes ``None`` and ``K_a == 0``).
    eps_across, eps_within:
        White-noise floors ``ε ∈ [0, 1)``. Fixed (non-learnable) by
        default to match the DLAG MATLAB convention
        (``learnGPNoise = false``); kept as buffers so they move with
        :meth:`Module.to`.
    dtype:
        Dtype for the registered buffers and kernel parameters. The
        eventual device is set by the parent :class:`BaseModel.to`.
    """

    CAPABILITIES = frozenset({"cov_full", "cov_freq"})

    n_regions: int
    n_across: int
    n_within: tuple[int, ...]
    eps_across: Tensor
    eps_within_flat: Tensor
    region_base: tuple[int, ...]
    within_offset: tuple[int, ...]
    delay: FixedDelay
    kernel_across: nn.ModuleList
    kernel_within_flat: nn.ModuleList

    def __init__(
        self,
        *,
        n_regions: int,
        n_across: int,
        n_within: tuple[int, ...],
        kernel_factory_across: Callable[[], BaseKernel],
        kernel_factory_within: Callable[[], BaseKernel],
        delay: FixedDelay | None = None,
        eps_across: float = 1e-3,
        eps_within: float = 1e-3,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        self._validate_layout(n_regions, n_across, n_within)
        if not (0.0 <= eps_across < 1.0) or not (0.0 <= eps_within < 1.0):
            raise ValueError(f"eps_* must lie in [0, 1); got across={eps_across}, within={eps_within}")
        self.n_regions = n_regions
        self.n_across = n_across
        self.n_within = tuple(n_within)

        # Pre-compute the per-time slot layout.
        region_sizes = [n_across + n_within[r] for r in range(n_regions)]
        self.region_base = tuple(int(sum(region_sizes[:r])) for r in range(n_regions))
        within_cum = [0]
        for r in range(n_regions):
            within_cum.append(within_cum[-1] + n_within[r])
        self.within_offset = tuple(within_cum)
        self._M = int(sum(region_sizes))

        # Per-block kernel instances. K_a across kernels + Σ K_w[r] within
        # kernels (flattened in region order, matching ``within_offset``).
        self.kernel_across = nn.ModuleList([kernel_factory_across() for _ in range(n_across)])
        self.kernel_within_flat = nn.ModuleList([kernel_factory_within() for _ in range(within_cum[-1])])
        # Coerce all kernel parameters to ``dtype``. Device follows the
        # parent ``BaseModel.to``.
        for k in self.kernel_across:
            k.to(dtype=dtype)
        for k in self.kernel_within_flat:
            k.to(dtype=dtype)

        # Noise floors as buffers (fixed in v1).
        self.register_buffer("eps_across", torch.full((n_across,), float(eps_across), dtype=dtype))
        self.register_buffer("eps_within_flat", torch.full((within_cum[-1],), float(eps_within), dtype=dtype))

        # Delay submodule.
        if n_across == 0:
            if delay is not None and delay.n_latent != 0:
                raise ValueError(f"delay.n_latent must equal n_across=0; got {delay.n_latent}")
            self.delay = FixedDelay(
                n_regions=n_regions, n_latent=1, max_delay=1.0, dtype=dtype
            )  # placeholder; never consulted because the across branch is skipped
        else:
            if delay is None:
                raise ValueError("delay must be provided when n_across > 0")
            if delay.n_regions != n_regions:
                raise ValueError(f"delay.n_regions ({delay.n_regions}) must equal n_regions ({n_regions})")
            if delay.n_latent != n_across:
                raise ValueError(f"delay.n_latent ({delay.n_latent}) must equal n_across ({n_across})")
            self.delay = delay

    # --- Static layout validation ----------------------------------------

    @staticmethod
    def _validate_layout(R: int, K_a: int, n_within: tuple[int, ...]) -> None:
        if R < 1:
            raise ValueError(f"n_regions must be >= 1; got {R}")
        if K_a < 0:
            raise ValueError(f"n_across must be >= 0; got {K_a}")
        if len(n_within) != R:
            raise ValueError(f"n_within has length {len(n_within)} but n_regions={R}")
        if any(d < 0 for d in n_within):
            raise ValueError(f"n_within counts must be >= 0; got {n_within}")
        if K_a == 0 and not any(n_within):
            raise ValueError("at least one latent (across or within) must be present")

    # --- Layout accessors ------------------------------------------------

    @property
    def state_dim_per_time(self) -> int:
        """``M = R · K_a + Σ K_w[r]`` — slots per time bin in ``K_big``."""
        return self._M

    def slot_across(self, r: int, k: int) -> int:
        """Flat slot index within a single time bin for region ``r``, across latent ``k``."""
        if not 0 <= r < self.n_regions:
            raise IndexError(f"region index {r} out of range [0, {self.n_regions})")
        if not 0 <= k < self.n_across:
            raise IndexError(f"across latent index {k} out of range [0, {self.n_across})")
        return self.region_base[r] + k

    def slot_within(self, r: int, w: int) -> int:
        """Flat slot index within a single time bin for region ``r``, within latent ``w``."""
        if not 0 <= r < self.n_regions:
            raise IndexError(f"region index {r} out of range [0, {self.n_regions})")
        if not 0 <= w < self.n_within[r]:
            raise IndexError(f"within latent index {w} out of range [0, {self.n_within[r]}) for region {r}")
        return self.region_base[r] + self.n_across + w

    def eps_within(self, r: int) -> Tensor:
        """Slice ``ε_w[r]`` of shape ``(K_w[r],)`` (buffer)."""
        return self.eps_within_flat[self.within_offset[r] : self.within_offset[r + 1]]

    def kernel_within(self, r: int, w: int) -> BaseKernel:
        """Kernel instance for region ``r``, within latent ``w``."""
        if not 0 <= r < self.n_regions:
            raise IndexError(f"region index {r} out of range [0, {self.n_regions})")
        if not 0 <= w < self.n_within[r]:
            raise IndexError(f"within latent index {w} out of range [0, {self.n_within[r]}) for region {r}")
        kernel = self.kernel_within_flat[self.within_offset[r] + w]
        assert isinstance(kernel, BaseKernel)
        return kernel

    # --- Covariance constructors ----------------------------------------

    def cov_full(self, T: int) -> Tensor:
        """Construct the full ``(M·T, M·T)`` GP covariance ``K_big``.

        Differentiable end-to-end w.r.t. all kernel parameters and the
        delay parameter. The construction loops over the K_a across
        kernels and over per-(region, within-latent) blocks; only the
        trial axis is never looped (and there is no trial axis here).
        """
        if T < 1:
            raise ValueError(f"T must be >= 1; got {T}")
        R = self.n_regions
        K_a = self.n_across
        M = self._M
        MT = M * T

        dtype = self.eps_across.dtype
        device = self.eps_across.device

        t_grid = torch.arange(T, dtype=dtype, device=device)
        tdiff = t_grid[:, None] - t_grid[None, :]  # (T, T)
        eye_T = torch.eye(T, dtype=dtype, device=device)
        eye_R = torch.eye(R, dtype=dtype, device=device)

        K_big = torch.zeros(MT, MT, dtype=dtype, device=device)

        # --- Across block: one kernel per across latent. ---
        if K_a > 0:
            delta = self.delay.as_tensor()  # (R, K_a)
            d_kr = delta.transpose(0, 1)  # (K_a, R)
            ddiff_k = d_kr.unsqueeze(2) - d_kr.unsqueeze(1)  # (K_a, R, R)
            diag_mask_rt = eye_R.view(R, R, 1, 1) * eye_T.view(1, 1, T, T)  # (R, R, T, T)

            rb_t = torch.as_tensor(self.region_base, dtype=torch.long, device=device)
            t_idx_l = torch.arange(T, dtype=torch.long, device=device)
            for k in range(K_a):
                # δt[r1, r2, t1, t2] = tdiff[t1, t2] - ddiff_k[k, r1, r2]
                delta_t_k = tdiff.view(1, 1, T, T) - ddiff_k[k].view(R, R, 1, 1)
                kernel_k = self.kernel_across[k]
                assert isinstance(kernel_k, BaseKernel)
                temp = kernel_k.cov(delta_t_k)
                eps_k = self.eps_across[k]
                K_block = (1.0 - eps_k) * temp + eps_k * diag_mask_rt  # (R, R, T, T)

                rows = (t_idx_l.view(1, 1, T, 1) * M + rb_t.view(R, 1, 1, 1) + k).expand(R, R, T, T)
                cols = (t_idx_l.view(1, 1, 1, T) * M + rb_t.view(1, R, 1, 1) + k).expand(R, R, T, T)
                K_big = K_big.index_put((rows.reshape(-1), cols.reshape(-1)), K_block.reshape(-1))

        # --- Within block: one kernel per (region, within latent). ---
        for r in range(R):
            K_w_r = self.n_within[r]
            if K_w_r == 0:
                continue
            base_r = self.region_base[r]
            offset_r = self.within_offset[r]
            t_idx_l = torch.arange(T, dtype=torch.long, device=device)
            for w in range(K_w_r):
                kernel_w = self.kernel_within_flat[offset_r + w]
                assert isinstance(kernel_w, BaseKernel)
                temp_w = kernel_w.cov(tdiff)  # (T, T)
                eps_w = self.eps_within_flat[offset_r + w]
                K_w = (1.0 - eps_w) * temp_w + eps_w * eye_T  # (T, T)

                slot = base_r + K_a + w
                rows_w = (t_idx_l.view(T, 1) * M + slot).expand(T, T)
                cols_w = (t_idx_l.view(1, T) * M + slot).expand(T, T)
                K_big = K_big.index_put((rows_w.reshape(-1), cols_w.reshape(-1)), K_w.reshape(-1))

        return K_big

    def cov_freq(self, T: int) -> Tensor:
        """Per-latent power spectral densities at the centered freqs.

        Returns the diagonal eigenvalues of the kernel prior in the
        unitary FFT basis. Across latents come first, then per-region
        within latents flattened in the same order as
        :attr:`region_base`. The result has shape
        ``(T, K_a + Σ K_w[r])``. **Delays are not applied here** — the
        per-region phase shifts ``Q_r(f) = exp(-i · 2π · f · δ_{r,k})``
        are a separate component (see :meth:`FixedDelay.phase_at_freq`).

        Each block calls its own :meth:`BaseKernel.spectral_density`. If
        a kernel does not implement it, a :class:`RuntimeError` is raised
        — frequency engines require this capability per :class:`BaseKernel`'s
        capability set.

        The dynamics-level noise floor blend ``(1 - ε) · S_kernel(ω) + ε``
        is applied here, consistent with the time-domain construction.
        """
        if T < 1:
            raise ValueError(f"T must be >= 1; got {T}")
        from mbrila.frequency.fft import centered_freqs

        dtype = self.eps_across.dtype
        device = self.eps_across.device
        freqs = centered_freqs(T, dtype=dtype, device=device)  # (T,)
        omega = 2.0 * torch.pi * freqs  # (T,)

        psd_blocks: list[Tensor] = []

        if self.n_across > 0:
            cols_a: list[Tensor] = []
            for k in range(self.n_across):
                kernel_k = self.kernel_across[k]
                assert isinstance(kernel_k, BaseKernel)
                s_k = kernel_k.spectral_density(omega)
                if s_k is None:
                    raise RuntimeError(
                        f"kernel {type(kernel_k).__name__} does not implement spectral_density; "
                        "frequency engines require it"
                    )
                eps_k = self.eps_across[k]
                cols_a.append((1.0 - eps_k) * s_k + eps_k)
            psd_blocks.append(torch.stack(cols_a, dim=-1))  # (T, K_a)
        else:
            psd_blocks.append(torch.empty(T, 0, dtype=dtype, device=device))

        n_within_total = self.within_offset[-1]
        if n_within_total > 0:
            cols_w: list[Tensor] = []
            for j in range(n_within_total):
                kernel_w = self.kernel_within_flat[j]
                assert isinstance(kernel_w, BaseKernel)
                s_w = kernel_w.spectral_density(omega)
                if s_w is None:
                    raise RuntimeError(
                        f"kernel {type(kernel_w).__name__} does not implement spectral_density; "
                        "frequency engines require it"
                    )
                eps_w = self.eps_within_flat[j]
                cols_w.append((1.0 - eps_w) * s_w + eps_w)
            psd_blocks.append(torch.stack(cols_w, dim=-1))  # (T, Σ K_w[r])
        else:
            psd_blocks.append(torch.empty(T, 0, dtype=dtype, device=device))

        return torch.cat(psd_blocks, dim=-1)

    def to_lds(self, T: int) -> tuple[Tensor, Tensor]:
        """Markovian LDS lifting is not provided by the exact-GP path.

        Exact GPs do not have a finite-state Markovian representation,
        so callers using a Kalman engine should pick :class:`MarkovianGPLatent`
        instead.
        """
        del T  # not used.
        raise NotImplementedError(
            "ExactGPLatent does not support to_lds(); use MarkovianGPLatent "
            "for Kalman-based inference engines."
        )
