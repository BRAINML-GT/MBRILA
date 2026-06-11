"""Initialisation strategies for mbrila models."""

from mbrila.init.factor_analysis import fa_em, fa_init_per_region
from mbrila.init.pcca import pcca_init_C
from mbrila.init.scale_anchor import normalize_latent_scales

__all__ = [
    "fa_em",
    "fa_init_per_region",
    "normalize_latent_scales",
    "pcca_init_C",
]
