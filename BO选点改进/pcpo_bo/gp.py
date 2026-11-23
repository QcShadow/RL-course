"""Fit the frozen kernel without changing the RL process's random state."""
from contextlib import contextmanager
import random

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch import settings
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import LogNormalPrior

from .kernel import build_kernel


@contextmanager
def bo_cpu_context(seed):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    threads = torch.get_num_threads()
    # fork_rng saves only CPU state: BO never touches the diffusion CUDA RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        np.random.seed(seed % (2**32))
        random.seed(seed)
        torch.set_num_threads(1)
        try:
            yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            torch.set_num_threads(threads)


def fit_model(train_x, train_y):
    prior = LogNormalPrior(-4.0, 1.0)
    likelihood = GaussianLikelihood(
        noise_prior=prior,
        noise_constraint=GreaterThan(1e-4, transform=None, initial_value=prior.mode),
    )
    # The frozen features are already scaled; unit-cube rescaling would change the design.
    with settings.validate_input_scaling(False):
        model = SingleTaskGP(
            train_x, train_y, covar_module=build_kernel(train_x.shape[1]),
            likelihood=likelihood, outcome_transform=Standardize(m=1),
        )
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model), max_attempts=1)
    model.eval()
    return model
