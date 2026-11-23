"""Shared kernel: shared-lengthscale Matérn-5/2."""
from __future__ import annotations

import math

from gpytorch.constraints import GreaterThan
from gpytorch.kernels import Kernel, MaternKernel
from gpytorch.priors import LogNormalPrior


KERNEL_KIND = "matern52"
ARD = False


def build_kernel(dimension: int) -> Kernel:
    """Build the exact classic kernel selected for the OCR prompt task."""
    if type(dimension) is not int or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    prior = LogNormalPrior(math.sqrt(2) + 0.5 * math.log(dimension), math.sqrt(3))
    return MaternKernel(
        nu=2.5,
        ard_num_dims=None,
        lengthscale_prior=prior,
        lengthscale_constraint=GreaterThan(
            0.025, transform=None, initial_value=prior.mode
        ),
    )
