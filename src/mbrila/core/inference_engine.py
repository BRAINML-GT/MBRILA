"""Inference engine contract.

An :class:`InferenceEngine` knows *how* to fit a model: exact EM, frequency-EM,
ARD variational EM, Laplace-EM, Kalman-EM, or pure SGD. It is intentionally
decoupled from :class:`~mbrila.core.base_model.BaseModel` so the same model
class can be paired with different engines (e.g. DLAG with either time-domain
or frequency-domain EM).

Engines declare their capability requirements via
:attr:`required_capabilities`; :meth:`check_compatible` raises a clear error
at fit time if the model wires up the wrong dynamics / kernel / observation
combo. The capability strings themselves are the names of methods the engine
will call on the model's components, e.g. ``"cov_full"`` for ``ExactGPLatent``
or ``"to_lds"`` for ``MarkovianGPLatent``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from torch import Tensor

if TYPE_CHECKING:
    from mbrila.core.base_model import BaseModel
    from mbrila.core.data import MultiRegionData


@dataclass(slots=True)
class Posterior:
    """Approximate posterior over latents.

    ``mean`` is always populated; ``cov`` may be one of three forms depending
    on the engine, distinguished by :attr:`cov_form`:

    - ``"diagonal"``: ``cov`` has shape ``(n_trials, T, n_latent_total)`` and
      stores marginal variances.
    - ``"per_time_block"``: ``cov`` has shape
      ``(n_trials, T, n_latent_total, n_latent_total)`` and stores the full
      latent covariance at each time bin (no cross-time terms).
    - ``"full_chol"``: ``cov`` is a Cholesky factor of the joint posterior
      over all time bins, shape
      ``(n_trials, T * n_latent_total, T * n_latent_total)``. Reserved for
      diagnostic use; v1 engines all return ``"per_time_block"`` or
      ``"diagonal"``.

    ``discrete_marginals`` is populated by SLDS-style models (MRM-GP) and
    holds ``E[z_t]`` of shape ``(n_trials, T, n_states)``.
    """

    mean: Tensor
    cov: Tensor
    cov_form: str = "per_time_block"
    discrete_marginals: Tensor | None = None
    extras: dict[str, Tensor] = field(default_factory=dict)


@dataclass(slots=True)
class FitResult:
    """Summary of a fit run.

    ``score_trace`` holds the per-iteration objective (LL for exact-EM
    engines, ELBO for variational ones). It is only meaningful *within* a
    single fit and must not be compared across methods (see plan section 7).
    """

    score_trace: list[float]
    converged: bool
    n_iter: int
    wall_time_s: float
    reason: str = ""


class InferenceEngine(ABC):
    """Inference algorithm abstract base class.

    Subclasses are stateless w.r.t. the model: they receive the model as an
    argument and read/write its parameters. This keeps a single ``Engine``
    instance reusable across many models.
    """

    name: str
    required_capabilities: frozenset[str]

    def check_compatible(self, model: BaseModel) -> None:
        """Validate that ``model`` exposes everything this engine needs.

        Raises :class:`ValueError` with a message naming the missing
        capability so the user can fix the wiring before a long run.
        """
        missing = self.required_capabilities - model.capabilities()
        if missing:
            raise ValueError(
                f"{type(self).__name__} requires capabilities {sorted(self.required_capabilities)} "
                f"but {type(model).__name__} only provides {sorted(model.capabilities())}; "
                f"missing: {sorted(missing)}"
            )

    @abstractmethod
    def fit(
        self,
        model: BaseModel,
        data: MultiRegionData,
        *,
        max_iter: int,
        tol: float,
        **kwargs: object,
    ) -> FitResult:
        """Run inference until convergence or ``max_iter`` is reached."""

    @abstractmethod
    def infer(self, model: BaseModel, data: MultiRegionData) -> Posterior:
        """Compute the posterior over latents at the current parameters."""

    @abstractmethod
    def score(self, model: BaseModel, data: MultiRegionData) -> float:
        """Objective value (LL or ELBO) at current parameters.

        For monitoring convergence within a single fit only — never compare
        across engines (different normalising constants).
        """
