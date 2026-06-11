"""Kernel functions used by mbrila models.

The :data:`kernel_registry` exposes named kernel classes for the
ergonomic string-based API (``mbrila.api.fit(method=..., kernel="matern_32")``)
and lets third-party packages contribute new kernels without touching
the core library. Built-in kernels (MOSE, Matérn-½/3/2/5/2) self-register
on import.
"""

from mbrila.core.registry import Registry
from mbrila.kernels.base import BaseKernel
from mbrila.kernels.matern import Matern12Kernel, Matern32Kernel, Matern52Kernel
from mbrila.kernels.mose import (
    MOSEKernel,
    rbf_grad_delta_t,
    rbf_grad_log_gamma,
    rbf_kernel_with_eps,
    rbf_psd,
    rbf_psd_grad_log_gamma,
)
from mbrila.kernels.validate import check_kernel

kernel_registry: Registry[BaseKernel] = Registry("kernel")
kernel_registry.register("mose", MOSEKernel)
kernel_registry.register("rbf", MOSEKernel)  # alias — MOSE *is* RBF
kernel_registry.register("matern_12", Matern12Kernel)
kernel_registry.register("matern_32", Matern32Kernel)
kernel_registry.register("matern_52", Matern52Kernel)

__all__ = [
    "BaseKernel",
    "MOSEKernel",
    "Matern12Kernel",
    "Matern32Kernel",
    "Matern52Kernel",
    "check_kernel",
    "kernel_registry",
    "rbf_grad_delta_t",
    "rbf_grad_log_gamma",
    "rbf_kernel_with_eps",
    "rbf_psd",
    "rbf_psd_grad_log_gamma",
]
