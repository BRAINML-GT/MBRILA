"""Cross-method abstractions used by every mbrila model.

These types describe *what* a multi-region latent model looks like; the
concrete machinery — kernels, dynamics, inference engines — lives in the
sibling subpackages.
"""

from mbrila.core.base_model import SAVE_FORMAT_VERSION, BaseModel
from mbrila.core.data import MultiRegionData
from mbrila.core.delay_spec import Delay
from mbrila.core.inference_engine import FitResult, InferenceEngine, Posterior
from mbrila.core.kernel_spec import Kernel, SDECoefficients
from mbrila.core.latent_spec import (
    ARDPriorConfig,
    DiscreteStateSpec,
    LatentSpec,
    SelectionMode,
)
from mbrila.core.observation_spec import Observation
from mbrila.core.registry import Registry

__all__ = [
    "SAVE_FORMAT_VERSION",
    "ARDPriorConfig",
    "BaseModel",
    "Delay",
    "DiscreteStateSpec",
    "FitResult",
    "InferenceEngine",
    "Kernel",
    "LatentSpec",
    "MultiRegionData",
    "Observation",
    "Posterior",
    "Registry",
    "SDECoefficients",
    "SelectionMode",
]
