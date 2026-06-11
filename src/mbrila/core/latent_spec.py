"""Description of the latent space shared by every model.

The latent layout is the one common ground between the very different methods
mbrila wraps:

- DLAG / mDLAG split latents into ``n_across`` cross-region latents and a
  per-region tuple ``n_within``. Across-region latents carry a per-region
  delay; within-region ones do not.
- ADM uses the same across/within split but with a single ``n_within`` count
  shared across regions in its default configuration.
- MRM-GP carries a discrete state ``z_t`` on top of the across/within split;
  see :class:`DiscreteStateSpec`.

The :class:`LatentSpec` here is a *structural* description — it declares the
dimensions and the regularisation regime; concrete distributional parameters
live on the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class ARDPriorConfig:
    """Hyperparameters for the ARD precision prior used by mDLAG.

    The model places a Gamma prior on per-region per-latent precisions
    ``alpha[r, k]``. Larger ``shape``/``rate`` collapse more aggressively;
    the defaults match the values from the fast-mDLAG MATLAB implementation.
    """

    shape: float = 1e-3
    rate: float = 1e-3

    def __post_init__(self) -> None:
        if self.shape <= 0 or self.rate <= 0:
            raise ValueError(
                f"ARD prior shape/rate must be positive; got shape={self.shape}, rate={self.rate}"
            )


SelectionMode = Literal["fixed", "ard", "inducing"]


@dataclass(frozen=True, slots=True)
class DiscreteStateSpec:
    """Discrete state configuration for SLDS-style models (MRM-GP).

    ``n_states`` is the number of discrete regimes. ``sticky`` is a
    diagonal-bias hyperparameter for the stationary transition matrix
    (a value of 0 leaves the transitions uninformed).
    """

    n_states: int
    sticky: float = 0.0

    def __post_init__(self) -> None:
        if self.n_states < 1:
            raise ValueError(f"n_states must be >= 1; got {self.n_states}")
        if self.sticky < 0:
            raise ValueError(f"sticky must be >= 0; got {self.sticky}")


@dataclass(frozen=True, slots=True)
class LatentSpec:
    """Structural description of the latent space.

    Parameters
    ----------
    n_across:
        Number of cross-region latents. Under ``selection="ard"`` this is an
        upper bound; the ARD prior may collapse some columns to zero.
    n_within:
        Per-region within-region latent counts. ``len(n_within)`` must equal
        the number of regions in the data.
    selection:
        ``"fixed"``: no regularisation, all dimensions kept.
        ``"ard"``: ARD column-pruning prior on the emission matrix
        (mDLAG-style). Requires ``ard_prior``.
        ``"inducing"``: sparse variational with inducing points (smDLAG;
        deferred to v1.1). Requires ``n_inducing``.
    n_inducing:
        Number of inducing points per latent (``selection="inducing"`` only).
    ard_prior:
        Gamma prior hyperparameters (``selection="ard"`` only).
    discrete:
        Discrete state configuration for SLDS-style models (e.g. MRM-GP).
        ``None`` for purely continuous latents.
    """

    n_across: int
    n_within: tuple[int, ...]
    selection: SelectionMode = "fixed"
    n_inducing: int | None = None
    ard_prior: ARDPriorConfig | None = field(default=None)
    discrete: DiscreteStateSpec | None = None

    def __post_init__(self) -> None:
        self._validate_dims()
        self._validate_selection()

    def _validate_dims(self) -> None:
        if self.n_across < 0:
            raise ValueError(f"n_across must be >= 0; got {self.n_across}")
        if not self.n_within:
            raise ValueError("n_within must list at least one region")
        if any(d < 0 for d in self.n_within):
            raise ValueError(f"n_within counts must be >= 0; got {self.n_within}")
        if self.n_across == 0 and not any(self.n_within):
            raise ValueError("at least one latent (across or within) must be present")

    def _validate_selection(self) -> None:
        if self.selection == "ard":
            if self.ard_prior is None:
                # default to mDLAG values rather than failing — they are uninformative.
                object.__setattr__(self, "ard_prior", ARDPriorConfig())
            if self.n_inducing is not None:
                raise ValueError("n_inducing is only valid with selection='inducing'")
        elif self.selection == "inducing":
            if self.n_inducing is None or self.n_inducing < 1:
                raise ValueError(f"selection='inducing' requires n_inducing >= 1; got {self.n_inducing}")
            if self.ard_prior is not None:
                raise ValueError("ard_prior is only valid with selection='ard'")
        else:  # fixed
            if self.n_inducing is not None:
                raise ValueError("n_inducing is only valid with selection='inducing'")
            if self.ard_prior is not None:
                raise ValueError("ard_prior is only valid with selection='ard'")

    @property
    def n_regions(self) -> int:
        return len(self.n_within)

    @property
    def n_latent_total(self) -> int:
        """Sum of all latent dimensions (across + within for every region)."""
        return self.n_across + int(sum(self.n_within))
