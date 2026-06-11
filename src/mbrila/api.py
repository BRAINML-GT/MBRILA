"""Top-level facade for assembling mbrila models from the 4-axis space.

A single entry point that lets a user dispatch to any preset model by
name. Preset names map to concrete ``BaseModel`` subclasses registered
in :data:`model_registry`. Third-party packages can contribute
additional presets via the same registry (e.g. through entry points),
so a ``build_model("my_method", ...)`` call works uniformly.

The function is a thin shim: each preset's constructor defines its own
valid kwargs. ``build_model`` does no argument massaging — it
dispatches to the registered class. This keeps individual model
classes' signatures authoritative and avoids a brittle "central
schema" that has to be kept in sync with every preset.

Presets registered out of the box:

- ``"adm"``   → :class:`mbrila.ADM` (time-varying delay, Kalman engine)
- ``"dlag"``  → :class:`mbrila.DLAG` (fixed delay; ``engine="exact"``
  for the dense-GP path, ``engine="kalman"`` for DLAG-SSM)
- ``"mdlag"`` → :class:`mbrila.MDLAG` (fixed delay + ARD;
  ``engine="time"`` / ``"freq"`` for dense-GP paths,
  ``engine="kalman"`` for mDLAG-SSM)
- ``"gpfa"``  → :class:`mbrila.GPFA` (no delay, Kalman engine,
  kernel-pluggable)
- ``"lds"``   → :class:`mbrila.LDS` (naive multi-region LDS — no
  kernel, free ``(A, Q)``, shared latents, Kalman engine)
"""

from __future__ import annotations

from typing import Any

from mbrila.core.base_model import BaseModel
from mbrila.core.registry import Registry
from mbrila.models.adm import ADM
from mbrila.models.dlag import DLAG
from mbrila.models.gpfa import GPFA
from mbrila.models.lds import LDS
from mbrila.models.mdlag import MDLAG

model_registry: Registry[BaseModel] = Registry("model")
model_registry.register("adm", ADM)
model_registry.register("dlag", DLAG)
model_registry.register("mdlag", MDLAG)
model_registry.register("gpfa", GPFA)
model_registry.register("lds", LDS)


def build_model(preset: str, /, **kwargs: Any) -> BaseModel:
    """Construct a mbrila model by preset name.

    Parameters
    ----------
    preset:
        Registered preset key (case-insensitive). See module docstring for
        the built-in set; use ``model_registry.names()`` to list available
        presets at runtime.
    **kwargs:
        Forwarded verbatim to the preset's constructor. Inspect the chosen
        preset's ``__init__`` signature for the valid arguments.

    Returns
    -------
    A fully constructed :class:`BaseModel` instance, ready for
    ``.fit(data)``.

    Examples
    --------
    Build a default-MOSE GPFA model::

        from mbrila import LatentSpec, build_model
        model = build_model(
            "gpfa",
            latent_spec=LatentSpec(n_across=2, n_within=(1, 1)),
            y_dims=(10, 12),
            T=50,
        )

    Build a Matérn-3/2 GPFA model — same preset, different kernel via the
    factory hook::

        from mbrila import Matern32Kernel
        model = build_model(
            "gpfa",
            latent_spec=LatentSpec(n_across=2, n_within=(1, 1)),
            y_dims=(10, 12),
            T=50,
            kernel_factory_across=lambda: Matern32Kernel(lengthscale=2.0),
        )

    Build a DLAG-SSM model — DLAG's fixed-delay structure with the
    AR(``P``) Kalman engine instead of the dense-GP engine::

        model = build_model(
            "dlag",
            latent_spec=LatentSpec(n_across=2, n_within=(1, 1)),
            y_dims=(10, 12),
            T=50,
            engine="kalman",
            lag_across=5,
            lag_within=2,
        )
    """
    cls = model_registry.get(preset)
    return cls(**kwargs)
