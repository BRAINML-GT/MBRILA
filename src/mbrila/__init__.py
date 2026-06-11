"""mbrila — Multiple Brain Region Interaction using Latent Analysis.

Top-level re-exports are added incrementally as concrete model classes land.
For now only the cross-method abstractions in :mod:`mbrila.core`, the ADM
model, and the shared kernel / delay / observation primitives are public.
"""

from mbrila.api import build_model, model_registry
from mbrila.core import (
    SAVE_FORMAT_VERSION,
    ARDPriorConfig,
    BaseModel,
    Delay,
    DiscreteStateSpec,
    FitResult,
    InferenceEngine,
    Kernel,
    LatentSpec,
    MultiRegionData,
    Observation,
    Posterior,
    Registry,
    SDECoefficients,
)
from mbrila.delays import FixedDelay, NoDelay, TimeVaryingDelay
from mbrila.dynamics import ExactGPLatent, FreeLDSLatent
from mbrila.inference import (
    ExactEMEngine,
    KalmanEMEngine,
    VEMARDEngine,
    VEMARDFreqEngine,
    VEMKalmanARDEngine,
)
from mbrila.init import (
    fa_em,
    fa_init_per_region,
    normalize_latent_scales,
    pcca_init_C,
)
from mbrila.kernels import (
    BaseKernel,
    Matern12Kernel,
    Matern32Kernel,
    Matern52Kernel,
    MOSEKernel,
    check_kernel,
    kernel_registry,
)
from mbrila.models import ADM, DLAG, GPFA, LDS, MDLAG
from mbrila.observations import (
    ARDObservation,
    LinearRegressionResult,
    MultiRegionLinearObservation,
    bayesian_linear_regression,
)

__version__ = "0.1.0"

__all__ = [
    "ADM",
    "DLAG",
    "GPFA",
    "LDS",
    "MDLAG",
    "SAVE_FORMAT_VERSION",
    "ARDObservation",
    "ARDPriorConfig",
    "BaseKernel",
    "BaseModel",
    "Delay",
    "DiscreteStateSpec",
    "ExactEMEngine",
    "ExactGPLatent",
    "FitResult",
    "FixedDelay",
    "FreeLDSLatent",
    "InferenceEngine",
    "KalmanEMEngine",
    "Kernel",
    "LatentSpec",
    "LinearRegressionResult",
    "MOSEKernel",
    "Matern12Kernel",
    "Matern32Kernel",
    "Matern52Kernel",
    "MultiRegionData",
    "MultiRegionLinearObservation",
    "NoDelay",
    "Observation",
    "Posterior",
    "Registry",
    "SDECoefficients",
    "TimeVaryingDelay",
    "VEMARDEngine",
    "VEMARDFreqEngine",
    "VEMKalmanARDEngine",
    "__version__",
    "bayesian_linear_regression",
    "build_model",
    "check_kernel",
    "fa_em",
    "fa_init_per_region",
    "kernel_registry",
    "model_registry",
    "normalize_latent_scales",
    "pcca_init_C",
]
