"""ARD emission model for mDLAG (variational mean-field q(C, α, φ, d)).

mDLAG replaces DLAG's point-estimate emission ``(C, d, diag(R))`` with a
mean-field variational posterior over four blocks:

* ``q(C_r)`` row-wise Gaussian: ``C_r[i, :] ~ N(μ_{r,i}, Σ_{r,i})``.
* ``q(α_r)`` column-wise Gamma over per-region per-latent precision
  ``α_{r, k}`` (ARD prior — large ``α_{r,k}`` shrinks the matching
  column of ``C_r`` to zero, automatically pruning latent ``k`` for
  region ``r``).
* ``q(φ_i)`` per-neuron Gamma noise precision (replacing ``diag(R)``).
* ``q(d_i)`` per-neuron Gaussian offset.

All four blocks have conjugate closed-form variational updates given
posterior expectations of the latent ``x``. They are implemented here
without any reference to the X posterior or the GP prior, so this
module can be unit-tested against hand-computed sufficient statistics.
The full VEM engine that wires this together with the X E-step lives
in :mod:`mbrila.inference.vem_ard`.

Hyperpriors
-----------
::

    p(d_i)   = N(0, β_d^{-1})
    p(α_{r,k}) = Gamma(a₀, b₀)        # shape, rate
    p(φ_i)   = Gamma(a_φ⁰, b_φ⁰)

Defaults are ``a = b = 1e-3`` (matching fast-mDLAG's MATLAB code, which
relies on ``minVarFrac`` to clamp ``φ_mean`` from above).

State shapes
------------
``R`` = number of regions, ``k`` = ``n_obs_per_region`` (uniform per
region, matching DLAG's ``_uniform_within`` constraint), ``y_r`` =
``y_dims[r]``, ``Y = sum(y_dims)``.

==================  =============  ===================================
Field               Shape          Meaning
==================  =============  ===================================
``d_mean``          ``(Y,)``       posterior mean of offset
``d_cov``           ``(Y,)``       posterior diagonal cov of offset
``alpha_a``         ``(R,)``       fixed Gamma shape (``a₀ + y_r/2``)
``alpha_b``         ``(R, k)``     Gamma rate (updated each iter)
``alpha_mean``      ``(R, k)``     ``α_a / α_b`` cache
``phi_a``           ``()``         fixed Gamma shape (``a_φ⁰ + NT/2``)
``phi_b``           ``(Y,)``       Gamma rate (updated each iter)
``phi_mean``        ``(Y,)``       ``φ_a / φ_b`` cache
``C_mean_r``        ``(y_r, k)``   posterior mean of region-r loadings
``C_cov_r``         ``(y_r, k, k)`` per-row posterior cov of ``C_r``
``C_moment_r``      ``(y_r, k, k)`` per-row second moment (cov + mμᵀ)
==================  =============  ===================================

The ``alpha_a``/``phi_a`` parameters are set once (at fit start, when
the engine knows ``NT``) and constant thereafter — only the ``b``
fields are updated each EM iteration.

Trial-batched
-------------
This module has no Python loops over trials. Updates take sufficient
statistics ``(XX, XY, sum_x, sum_y, sum_y2)`` already aggregated over
``(B, T)`` by the upstream engine, so the dimensions inside
``ARDObservation`` are only ``(R, k, y_r)``. Loops over regions are
permitted by ``check_no_trial_loops.py`` (region-axis loops are
explicitly allowed).

Float64
-------
Per mbrila policy ``float64`` is the default. Some of the per-row
Cholesky solves on ``(k, k)`` precision matrices accumulate small
errors at ``float32`` once ``k > 8``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor

from mbrila.core.observation_spec import Observation


class ARDObservation(Observation):
    """Variational emission with ARD prior on C for mDLAG.

    Parameters
    ----------
    y_dims:
        Per-region neuron counts.
    n_obs_per_region:
        Per-region latent column count of ``C_r`` (= ``n_across +
        n_within`` in DLAG layout; typically ``n_across`` for mDLAG
        with ARD doing within/across discovery).
    prior_d_beta, prior_phi_a, prior_phi_b, prior_alpha_a, prior_alpha_b:
        Hyperprior parameters. Defaults are ``1e-3`` for all (the
        fast-mDLAG MATLAB reference uses ``1e-12``).
    min_var_frac:
        Fraction of per-neuron sample variance used as the upper bound
        on ``diag(R) = 1/φ_mean``. Matches fast-mDLAG's ``minVarFrac``
        floor on the private variance. Set via ``set_variance_floor``
        once data is available; until then the floor is zero (inactive).
    init_alpha_mean, init_phi_mean:
        Initial values for ``α_mean`` and ``φ_mean`` before
        ``initialize_from_pcca`` is called.
    """

    # Buffers declared at class level so mypy --strict can narrow types
    # after ``nn.Module.__getattr__`` returns ``Tensor | Module``.
    d_mean: Tensor
    d_cov: Tensor
    alpha_a: Tensor
    alpha_b: Tensor
    alpha_mean: Tensor
    phi_a: Tensor
    phi_b: Tensor
    phi_mean: Tensor
    prior_d_beta: Tensor
    prior_phi_a: Tensor
    prior_phi_b: Tensor
    prior_alpha_a: Tensor
    prior_alpha_b: Tensor
    var_floor: Tensor

    n_obs_per_region: int

    def __init__(
        self,
        y_dims: tuple[int, ...],
        n_obs_per_region: int,
        *,
        prior_d_beta: float = 1e-3,
        prior_phi_a: float = 1e-3,
        prior_phi_b: float = 1e-3,
        prior_alpha_a: float = 1e-3,
        prior_alpha_b: float = 1e-3,
        min_var_frac: float = 1e-3,
        init_alpha_mean: float = 1.0,
        init_phi_mean: float = 1.0,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if n_obs_per_region < 1:
            raise ValueError(f"n_obs_per_region must be >= 1; got {n_obs_per_region}")
        n_total_obs = len(y_dims) * n_obs_per_region
        super().__init__(y_dims=y_dims, n_latent_total=n_total_obs)
        self.n_obs_per_region = int(n_obs_per_region)
        self._min_var_frac = float(min_var_frac)
        k = int(n_obs_per_region)
        R = self.n_regions
        sum_y = self.n_neurons

        # Hyperpriors registered as buffers so they move with .to(device).
        self.register_buffer("prior_d_beta", torch.tensor(prior_d_beta, dtype=dtype))
        self.register_buffer("prior_phi_a", torch.tensor(prior_phi_a, dtype=dtype))
        self.register_buffer("prior_phi_b", torch.tensor(prior_phi_b, dtype=dtype))
        self.register_buffer("prior_alpha_a", torch.tensor(prior_alpha_a, dtype=dtype))
        self.register_buffer("prior_alpha_b", torch.tensor(prior_alpha_b, dtype=dtype))
        # Variance floor stays at zero (inactive) until set_variance_floor is called.
        self.register_buffer("var_floor", torch.zeros(sum_y, dtype=dtype))

        # q(d) — diagonal Gaussian
        self.register_buffer("d_mean", torch.zeros(sum_y, dtype=dtype))
        self.register_buffer("d_cov", torch.full((sum_y,), 1.0 / prior_d_beta, dtype=dtype))

        # q(α) — per-region Gamma. ``a`` constant once set; only ``b`` updates.
        alpha_a_init = torch.tensor([prior_alpha_a + 0.5 * y_r for y_r in y_dims], dtype=dtype)  # (R,)
        self.register_buffer("alpha_a", alpha_a_init)
        alpha_b_init = (alpha_a_init.unsqueeze(-1) / init_alpha_mean).expand(R, k).contiguous()
        self.register_buffer("alpha_b", alpha_b_init.clone())
        self.register_buffer("alpha_mean", torch.full((R, k), float(init_alpha_mean), dtype=dtype))

        # q(φ) — per-neuron Gamma. ``a`` set by ``set_phi_shape_from_NT`` once
        # the engine knows the (B·T) total bin count; until then it stays at
        # the prior, and ``phi_mean`` is still queryable for sampling priors.
        self.register_buffer("phi_a", torch.tensor(prior_phi_a, dtype=dtype))
        phi_mean_init = torch.full((sum_y,), float(init_phi_mean), dtype=dtype)
        self.register_buffer("phi_mean", phi_mean_init.clone())
        self.register_buffer("phi_b", self.phi_a / self.phi_mean)

        # q(C_r) — per-region row-Gaussian. Each region gets three buffers
        # (mean, cov, second moment) so state_dict picks them up automatically.
        # Note: ``.expand().contiguous()`` would *not* allocate fresh storage
        # when y_r == k == 1 (expand on a same-shape tensor is a no-op view, and
        # contiguous on a contiguous tensor returns ``self``), so we clone
        # explicitly to keep each buffer independent.
        self._n_regions_int = R
        eye_k = torch.eye(k, dtype=dtype)
        for r, y_r in enumerate(y_dims):
            self.register_buffer(f"C_mean_{r}", torch.zeros(y_r, k, dtype=dtype))
            self.register_buffer(f"C_cov_{r}", eye_k.unsqueeze(0).expand(y_r, k, k).clone())
            self.register_buffer(f"C_moment_{r}", eye_k.unsqueeze(0).expand(y_r, k, k).clone())

    # ------------------------------------------------------------------
    # Convenience accessors (mirror MultiRegionLinearObservation.Cs)
    # ------------------------------------------------------------------

    @property
    def C_means(self) -> list[Tensor]:
        return [getattr(self, f"C_mean_{r}") for r in range(self._n_regions_int)]

    @property
    def C_covs(self) -> list[Tensor]:
        return [getattr(self, f"C_cov_{r}") for r in range(self._n_regions_int)]

    @property
    def C_moments(self) -> list[Tensor]:
        return [getattr(self, f"C_moment_{r}") for r in range(self._n_regions_int)]

    # ------------------------------------------------------------------
    # Observation contract
    # ------------------------------------------------------------------

    def block_diag_C(self) -> Tensor:
        out: Tensor = torch.block_diag(*self.C_means)  # type: ignore[no-untyped-call]
        return out

    def diag_R(self) -> Tensor:
        """Posterior mean of the diagonal noise variance, ``1 / φ_mean``."""
        return 1.0 / self.phi_mean

    def offset(self) -> Tensor:
        return self.d_mean

    def forward(self, x: Tensor) -> Tensor:
        """Predict noiseless ``E[y | x] = blkdiag(C_mean)·x + d_mean``."""
        C = self.block_diag_C()
        return torch.einsum("ij,btj->bti", C, x) + self.d_mean

    # ------------------------------------------------------------------
    # One-shot configuration (called by engine before fit)
    # ------------------------------------------------------------------

    def set_phi_shape_from_NT(self, NT: int) -> None:
        """Set the Gamma shape ``a_φ = a_φ⁰ + NT/2``.

        Must be called once after the engine knows the total bin count
        ``NT = B · T`` (and again only if the data shape changes). The
        rate ``b_φ`` is rescaled so the current ``φ_mean`` is preserved.
        """
        if NT < 1:
            raise ValueError(f"NT must be >= 1; got {NT}")
        with torch.no_grad():
            self.phi_a.copy_(self.prior_phi_a + 0.5 * NT)
            # Keep the same ``φ_mean = a / b`` after rescaling ``a``.
            self.phi_b.copy_(self.phi_a / torch.clamp(self.phi_mean, min=1e-30))

    def set_variance_floor(self, y_var: Tensor) -> None:
        """Set per-neuron variance floor ``min_var_frac · Var[y_i]``.

        Used to upper-bound ``φ_mean`` so the private variance ``1/φ_mean``
        stays above ``min_var_frac · Var[y_i]`` (fast-mDLAG default).
        """
        if y_var.shape != (self.n_neurons,):
            raise ValueError(f"y_var must have shape ({self.n_neurons},); got {tuple(y_var.shape)}")
        with torch.no_grad():
            self.var_floor.copy_(
                self._min_var_frac * y_var.to(dtype=self.var_floor.dtype, device=self.var_floor.device)
            )

    # ------------------------------------------------------------------
    # Initialisation from data
    # ------------------------------------------------------------------

    def initialize_from_pcca(
        self,
        data: Tensor,
        *,
        init_C_cov: float = 1e-6,
        zero_offset: bool = False,
        fa_max_iter: int = 50,
    ) -> None:
        """Seed ``q(C, α, φ, d)`` from a multi-view pCCA fit on ``data``.

        Mirrors DLAG's ``initialize_from_data(mode='pcca')``:

        1. Run :func:`mbrila.init.pcca.pcca_init_C` with ``n_across =
           n_obs_per_region`` and ``n_within = 0``. (mDLAG treats every
           latent column uniformly; ARD discovers within/across
           structure rather than declaring it up front.)
        2. Copy ``Cs[r]`` into ``C_mean_r`` and seed ``C_cov_r =
           init_C_cov · I`` (a tiny numerical floor — see below).
        3. Set ``φ_mean = 1 / ψ_global``, ``d_mean = mu_global`` (or
           zero), and refresh the dependent ``φ_b``, ``d_cov``.
        4. Set the per-neuron variance floor from sample variance.

        ``α_mean`` is left at the constructor default (1.0) — the first
        ``update_alpha`` call after one ``update_C`` will move it to a
        data-appropriate value automatically.

        Why ``init_C_cov`` must be tiny
        -------------------------------
        ``C_cov`` is the **variational posterior** covariance of each
        row of ``C``, not a prior. The engine's E-step builds the
        emission precision from the second moment
        ``⟨C_r[i] C_r[i]ᵀ⟩ = C_cov[r,i] + C_mean[r,i] C_mean[r,i]ᵀ``,
        so an inflated ``C_cov`` directly inflates
        ``CPhiC = Σ_i φ_i ⟨C_r[i] C_r[i]ᵀ⟩``. Seeding ``C_cov = I``
        (the old behaviour) makes the ``Σ_i φ_i · I`` term — typically
        ``O(Σφ)`` ≈ hundreds — swamp the genuine ``Σ_i φ_i C_mean C_meanᵀ``
        term (``O(10)``), so the iter-0 latent posterior is crushed
        ``~100×``, the Whittle γ-MLE collapses the timescale, and the
        ARD / C / φ updates spiral. A tiny ``init_C_cov`` makes the
        iter-0 E-step use an effectively point-estimate ``CPhiC`` (the
        same quantity DLAG uses, known good); the first ``update_C``
        then overwrites ``C_cov`` with the proper variational value
        ``(diag(α) + φ·XX)⁻¹``. The init value never enters a reported
        ELBO (the engine measures ELBO only after the M-step).
        """
        if data.ndim != 3:
            raise ValueError(f"data must have shape (B, T, sum_y); got {tuple(data.shape)}")
        if int(data.shape[-1]) != self.n_neurons:
            raise ValueError(f"data last dim {int(data.shape[-1])} must equal sum(y_dims)={self.n_neurons}")

        from mbrila.init.pcca import pcca_init_C

        device = self.d_mean.device
        dtype = self.d_mean.dtype
        y = data.to(device=device, dtype=dtype)

        Cs, psi, mu = pcca_init_C(
            y,
            y_dims=self.y_dims,
            n_across=self.n_obs_per_region,
            n_within=0,
            max_iter=fa_max_iter,
        )

        if init_C_cov <= 0:
            raise ValueError(f"init_C_cov must be positive; got {init_C_cov}")
        k = self.n_obs_per_region
        eye_k = torch.eye(k, dtype=dtype, device=device)
        with torch.no_grad():
            for r, C_r in enumerate(Cs):
                C_mean_r = C_r.to(dtype=dtype, device=device).contiguous()
                C_cov_r = (float(init_C_cov) * eye_k).expand(C_mean_r.shape[0], k, k).contiguous()
                moment_r = C_cov_r + C_mean_r.unsqueeze(-1) * C_mean_r.unsqueeze(-2)
                getattr(self, f"C_mean_{r}").copy_(C_mean_r)
                getattr(self, f"C_cov_{r}").copy_(C_cov_r)
                getattr(self, f"C_moment_{r}").copy_(moment_r)

            phi_mean_new = 1.0 / torch.clamp(psi.to(dtype=dtype, device=device), min=1e-12)
            self.phi_mean.copy_(phi_mean_new)
            self.phi_b.copy_(self.phi_a / phi_mean_new)

            if zero_offset:
                self.d_mean.zero_()
            else:
                self.d_mean.copy_(mu.to(dtype=dtype, device=device))
            self.d_cov.copy_(
                1.0 / (self.prior_d_beta + 0.0)  # NT unknown here — d_cov refreshed on first update_d
                + torch.zeros_like(self.d_cov)
            )

            # Variance floor from sample variance over (B, T).
            y_flat = y.reshape(-1, self.n_neurons)
            y_var = y_flat.var(dim=0, unbiased=False)
            self.set_variance_floor(y_var)

    # ------------------------------------------------------------------
    # Closed-form variational updates (one per block)
    # ------------------------------------------------------------------
    #
    # All updates are transcribed from fast-mDLAG/mDLAG/core_mdlag/em_mdlag.m
    # (specifically the M-step blocks at lines 432-510). The arithmetic
    # matches the MATLAB reference exactly; only the loops over
    # ``trialIdx`` are pushed into the engine that builds the
    # sufficient statistics, so this method body is pure linear algebra.

    def update_d(self, sum_y: Tensor, sum_x_per_region: Tensor, NT: int) -> None:
        """Update ``q(d)`` — diagonal Gaussian. em_mdlag.m:432-440.

        Parameters
        ----------
        sum_y:
            ``Σ_{b,t} y_{b,t}`` — shape ``(Y,)``.
        sum_x_per_region:
            ``Σ_{b,t} ⟨x_{b,t,r}⟩`` — shape ``(R, k)``.
        NT:
            Total bin count ``B·T``.
        """
        if sum_y.shape != (self.n_neurons,):
            raise ValueError(f"sum_y must have shape ({self.n_neurons},); got {tuple(sum_y.shape)}")
        if sum_x_per_region.shape != (self._n_regions_int, self.n_obs_per_region):
            raise ValueError(
                "sum_x_per_region must have shape "
                f"({self._n_regions_int}, {self.n_obs_per_region}); "
                f"got {tuple(sum_x_per_region.shape)}"
            )
        if NT < 1:
            raise ValueError(f"NT must be >= 1; got {NT}")

        with torch.no_grad():
            d_cov_new = 1.0 / (self.prior_d_beta + NT * self.phi_mean)  # (Y,)

            # sum_Cx[i] = Σ_{b,t} ⟨ blkdiag(C_mean) · x_{b,t} ⟩_i
            # Linearity of expectation lets us push the sum inside the matmul:
            #   sum_Cx_per_region[r] = C_mean_r · sum_x_per_region[r]  → (y_r,)
            sum_Cx_parts: list[Tensor] = []
            for r in range(self._n_regions_int):
                C_mean_r = getattr(self, f"C_mean_{r}")  # (y_r, k)
                sum_Cx_r = C_mean_r @ sum_x_per_region[r]  # (y_r,)
                sum_Cx_parts.append(sum_Cx_r)
            sum_Cx = torch.cat(sum_Cx_parts)  # (Y,)

            d_mean_new = d_cov_new * self.phi_mean * (sum_y - sum_Cx)
            self.d_cov.copy_(d_cov_new)
            self.d_mean.copy_(d_mean_new)

    def update_C(self, XX: Tensor, XY: Sequence[Tensor], sum_x_per_region: Tensor) -> None:
        """Update ``q(C_r)`` row-by-row. em_mdlag.m:444-466.

        Centres ``XY`` against the current ``d_mean`` internally (so the
        engine can keep ``XY`` defined against the raw data ``Y``):
        ``XY0[r] = XY[r] − sum_x[r] · d_mean_r``.

        Parameters
        ----------
        XX:
            Per-region second moments ``Σ_{b,t} ⟨x_{b,t,r} x_{b,t,r}ᵀ⟩``,
            shape ``(R, k, k)``. Must already include the per-time
            posterior covariance (i.e. cov_sum + mean·meanᵀ), not just
            the outer product of the means.
        XY:
            List of length ``R``; element ``r`` is ``Σ_{b,t} ⟨x_{b,t,r}⟩
            · y_{b,t,r}ᵀ`` of shape ``(k, y_r)``. Note the data is the
            **raw** ``y`` (un-centred).
        sum_x_per_region:
            ``Σ_{b,t} ⟨x_{b,t,r}⟩`` — shape ``(R, k)``. Used to centre
            ``XY`` against ``d_mean``.
        """
        R = self._n_regions_int
        k = self.n_obs_per_region
        if XX.shape != (R, k, k):
            raise ValueError(f"XX must have shape ({R}, {k}, {k}); got {tuple(XX.shape)}")
        if len(XY) != R:
            raise ValueError(f"XY must have {R} entries; got {len(XY)}")
        if sum_x_per_region.shape != (R, k):
            raise ValueError(
                f"sum_x_per_region must have shape ({R}, {k}); got {tuple(sum_x_per_region.shape)}"
            )

        with torch.no_grad():
            cum = 0
            for r, y_r in enumerate(self.y_dims):
                if XY[r].shape != (k, y_r):
                    raise ValueError(f"XY[{r}] must have shape ({k}, {y_r}); got {tuple(XY[r].shape)}")
                d_mean_r = self.d_mean[cum : cum + y_r]  # (y_r,)
                phi_m_r = self.phi_mean[cum : cum + y_r]  # (y_r,)
                alpha_m_diag = torch.diag(self.alpha_mean[r])  # (k, k)

                # XY0[r] = XY[r] - sum_x[r] · d_mean_rᵀ → shape (k, y_r)
                XY0_r = XY[r] - sum_x_per_region[r].unsqueeze(-1) * d_mean_r.unsqueeze(0)

                # Per-row precision: (y_r, k, k) — broadcast scalar phi over rows.
                Sigma_inv = alpha_m_diag.unsqueeze(0) + phi_m_r.view(-1, 1, 1) * XX[r].unsqueeze(0)
                # Symmetrise to absorb any floating-point asymmetry before Cholesky.
                Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv.transpose(-2, -1))

                L = torch.linalg.cholesky(Sigma_inv)
                eye_yk = torch.eye(k, dtype=Sigma_inv.dtype, device=Sigma_inv.device).expand(y_r, k, k)
                Sigma = torch.cholesky_solve(eye_yk, L)
                Sigma = 0.5 * (Sigma + Sigma.transpose(-2, -1))

                # Mean: (y_r, k) = phi_i · Σ_i · XY0[:, i]
                XY0_per_row = XY0_r.transpose(0, 1).unsqueeze(-1)  # (y_r, k, 1)
                mean_unscaled = (Sigma @ XY0_per_row).squeeze(-1)  # (y_r, k)
                mean_r = phi_m_r.unsqueeze(-1) * mean_unscaled

                moment_r = Sigma + mean_r.unsqueeze(-1) * mean_r.unsqueeze(-2)

                getattr(self, f"C_mean_{r}").copy_(mean_r)
                getattr(self, f"C_cov_{r}").copy_(Sigma)
                getattr(self, f"C_moment_{r}").copy_(moment_r)
                cum += y_r

    def update_alpha(self) -> None:
        """Update ``q(α_r)`` — Gamma rate from current ``C`` second moments.

        em_mdlag.m:472-480. ``alpha_a`` is fixed at construction;
        ``alpha_b[r, k] = b₀ + 0.5 · Σ_i ⟨C_r[i, k]²⟩``.
        """
        with torch.no_grad():
            for r in range(self._n_regions_int):
                moment_r = getattr(self, f"C_moment_{r}")  # (y_r, k, k)
                diag_sum = torch.diagonal(moment_r, dim1=-2, dim2=-1).sum(dim=0)  # (k,)
                self.alpha_b[r].copy_(self.prior_alpha_b + 0.5 * diag_sum)
                self.alpha_mean[r].copy_(self.alpha_a[r] / self.alpha_b[r])

    def update_phi(
        self,
        NT: int,
        sum_y: Tensor,
        sum_y2: Tensor,
        XX: Tensor,
        XY: Sequence[Tensor],
        sum_x_per_region: Tensor,
    ) -> None:
        """Update ``q(φ_i)`` — per-neuron Gamma rate. em_mdlag.m:489-510.

        Parameters
        ----------
        NT:
            Total bin count ``B·T``.
        sum_y:
            ``Σ_{b,t} y_{b,t}`` — shape ``(Y,)``.
        sum_y2:
            ``Σ_{b,t} y²_{b,t}`` — shape ``(Y,)``.
        XX, XY, sum_x_per_region:
            Same conventions as in :meth:`update_C`. ``XY`` is the
            raw (un-centred) cross-moment; ``d_mean`` from the most
            recent ``update_d`` is used to subtract the offset
            contribution.
        """
        if sum_y.shape != (self.n_neurons,) or sum_y2.shape != (self.n_neurons,):
            raise ValueError(
                f"sum_y / sum_y2 must have shape ({self.n_neurons},); "
                f"got {tuple(sum_y.shape)} / {tuple(sum_y2.shape)}"
            )

        with torch.no_grad():
            cum = 0
            new_phi_b_parts: list[Tensor] = []
            for r, y_r in enumerate(self.y_dims):
                d_mean_r = self.d_mean[cum : cum + y_r]  # (y_r,)
                d_cov_r = self.d_cov[cum : cum + y_r]  # (y_r,)
                dd_r = d_cov_r + d_mean_r * d_mean_r  # ⟨d_i²⟩
                sum_y_r = sum_y[cum : cum + y_r]
                sum_y2_r = sum_y2[cum : cum + y_r]
                C_mean_r = getattr(self, f"C_mean_{r}")  # (y_r, k)
                C_moment_r = getattr(self, f"C_moment_{r}")  # (y_r, k, k)
                XY0_r = XY[r] - sum_x_per_region[r].unsqueeze(-1) * d_mean_r.unsqueeze(0)

                # NT · ⟨d_i²⟩ + Σ y² - 2·d·Σ y
                term1 = NT * dd_r + sum_y2_r - 2.0 * d_mean_r * sum_y_r
                # -2 · ⟨C_r[i, :]⟩ · XY0_r[:, i]  (einsum over k)
                term2 = -2.0 * torch.einsum("ik,ki->i", C_mean_r, XY0_r)
                # tr(⟨C_r[i, :, :]⟩ · XX[r]) — note transpose convention:
                # tr(A B) = Σ_{p, q} A_{p, q} B_{q, p}, so we contract
                # (i, p, q) with (q, p).
                term3 = torch.einsum("ipq,qp->i", C_moment_r, XX[r])

                phi_b_r = self.prior_phi_b + 0.5 * (term1 + term2 + term3)
                new_phi_b_parts.append(phi_b_r)
                cum += y_r

            phi_b_new = torch.cat(new_phi_b_parts)  # (Y,)
            phi_mean_new = self.phi_a / phi_b_new
            # Apply variance floor: 1/φ_mean ≥ min_var_frac · Var[y_i]
            # equivalent to φ_mean ≤ 1 / var_floor (when var_floor > 0).
            if torch.any(self.var_floor > 0):
                cap = 1.0 / torch.clamp(self.var_floor, min=1e-30)
                phi_mean_new = torch.minimum(phi_mean_new, cap)
                phi_b_new = self.phi_a / phi_mean_new
            self.phi_mean.copy_(phi_mean_new)
            self.phi_b.copy_(phi_b_new)

    # ------------------------------------------------------------------
    # ELBO contributions (emission side only)
    # ------------------------------------------------------------------
    #
    # The total ELBO splits into (a) an emission-side contribution
    # implemented here, plus (b) X / GP terms supplied by the upstream
    # engine. Each helper below mirrors a labelled block of fast-mDLAG's
    # ``em_mdlag.m`` lines 521-567.

    def elbo_data_likelihood(self, NT: int) -> Tensor:
        """``E_q[log p(y|x,C,d,φ)]`` rewritten via ``b_φ - b_φ⁰``.

        em_mdlag.m:521-524. Equals
        ``-(NT·Y/2)·log 2π + (NT/2) Σ_i ⟨log φ_i⟩ - Σ_i φ_mean_i · (b_φ_i - b_φ⁰)``,
        which folds in the half of the ``q(φ)`` KL that contains the
        residual second-moment trick.
        """
        log_phi = torch.digamma(self.phi_a) - torch.log(self.phi_b)  # (Y,)
        const = -0.5 * NT * self.n_neurons * math.log(2.0 * math.pi)
        return const + 0.5 * NT * log_phi.sum() - (self.phi_mean * (self.phi_b - self.prior_phi_b)).sum()

    def elbo_C(self) -> Tensor:
        """``q(C)`` ELBO contribution. em_mdlag.m:536-543.

        ``0.5 Σ_r,i log|Σ_C[r,i]| + Σ_r (y_r/2)·Σ_k ⟨log α_{r,k}⟩
         + Σ_r,i 0.5·tr(I_k - ⟨C_r[i,:] C_r[i,:]ᵀ⟩ · diag(α_mean_r))``.
        """
        log_alpha = torch.digamma(self.alpha_a).unsqueeze(-1) - torch.log(self.alpha_b)  # (R, k)
        total = torch.zeros((), dtype=self.alpha_b.dtype, device=self.alpha_b.device)
        for r, y_r in enumerate(self.y_dims):
            cov_r = getattr(self, f"C_cov_{r}")  # (y_r, k, k)
            moment_r = getattr(self, f"C_moment_{r}")  # (y_r, k, k)
            sign, logabsdet = torch.linalg.slogdet(cov_r)
            if torch.any(sign <= 0):
                raise RuntimeError(f"region {r}: C_cov has non-positive determinant")
            total = total + 0.5 * logabsdet.sum()
            total = total + 0.5 * y_r * log_alpha[r].sum()
            # tr(I - moment · diag(α_mean)) summed over rows i:
            #   = y_r · k - Σ_i Σ_k moment[i, k, k] · α_mean[r, k]
            diag_moment = torch.diagonal(moment_r, dim1=-2, dim2=-1)  # (y_r, k)
            tr_term = y_r * self.n_obs_per_region - (diag_moment * self.alpha_mean[r]).sum()
            total = total + 0.5 * tr_term
        return total

    def elbo_alpha(self) -> Tensor:
        """``KL[q(α) || p(α)]`` contribution. em_mdlag.m:546-555.

        Returns the ELBO term ``-KL`` (i.e. positive contribution).
        """
        R = self._n_regions_int
        k = self.n_obs_per_region
        alogb = self.prior_alpha_a * torch.log(self.prior_alpha_b)
        log_g_prior = torch.lgamma(self.prior_alpha_a)
        log_alpha = torch.digamma(self.alpha_a).unsqueeze(-1) - torch.log(self.alpha_b)  # (R, k)

        total = (R * k) * (alogb - log_g_prior)
        for r in range(R):
            term = (
                -self.alpha_a[r] * torch.log(self.alpha_b[r])
                - self.prior_alpha_b * self.alpha_mean[r]
                + (self.prior_alpha_a - self.alpha_a[r]) * log_alpha[r]
            ).sum()
            total = total + term + k * torch.lgamma(self.alpha_a[r]) + k * self.alpha_a[r]
        return total

    def elbo_phi(self) -> Tensor:
        """``KL[q(φ) || p(φ)]`` (rest of the φ contribution; complements
        :meth:`elbo_data_likelihood`). em_mdlag.m:558-563.
        """
        alogb = self.prior_phi_a * torch.log(self.prior_phi_b)
        log_g_prior = torch.lgamma(self.prior_phi_a)
        log_g_post = torch.lgamma(self.phi_a)
        digamma_a = torch.digamma(self.phi_a)
        Y = self.n_neurons

        head = Y * (alogb + log_g_post - log_g_prior + self.phi_a)
        tail = (
            -self.phi_a * torch.log(self.phi_b)
            - self.prior_phi_b * self.phi_mean
            + (self.prior_phi_a - self.phi_a) * (digamma_a - torch.log(self.phi_b))
        ).sum()
        return head + tail

    def elbo_d(self) -> Tensor:
        """``KL[q(d) || p(d)]`` contribution. em_mdlag.m:566-567.

        ``Y/2 + (Y/2)·log β_d + 0.5 Σ_i log d_cov_i - 0.5·β_d·Σ_i ⟨d_i²⟩``.
        """
        Y = self.n_neurons
        dd = self.d_cov + self.d_mean * self.d_mean  # ⟨d_i²⟩
        return (
            0.5 * Y
            + 0.5 * Y * torch.log(self.prior_d_beta)
            + 0.5 * torch.log(self.d_cov).sum()
            - 0.5 * self.prior_d_beta * dd.sum()
        )

    def elbo_emission(self, NT: int) -> Tensor:
        """Sum of all emission-side ELBO pieces.

        The full mDLAG ELBO equals ``elbo_emission(NT) + X_KL`` where the
        ``X_KL = (R·k·NT)/2 + lb_gp + 0.5·logdet(Σ_X)`` piece is computed
        by the upstream VEM engine and depends on the GP prior.
        """
        return (
            self.elbo_data_likelihood(NT)
            + self.elbo_C()
            + self.elbo_alpha()
            + self.elbo_phi()
            + self.elbo_d()
        )
