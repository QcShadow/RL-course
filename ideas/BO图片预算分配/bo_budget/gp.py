import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from gpytorch.kernels import MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood


def fit_model(x, y):
    model = SingleTaskGP(x.double(), y.double(), covar_module=MaternKernel(nu=2.5),
                         outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model), max_attempts=1)
    model.eval()
    return model


def ucb(model, x, beta=2.0):
    with torch.no_grad():
        posterior = model.posterior(x.double(), observation_noise=False)
        return posterior.mean.reshape(-1) + beta * posterior.variance.clamp_min(0).sqrt().reshape(-1)
