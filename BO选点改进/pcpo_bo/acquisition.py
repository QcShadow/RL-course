"""Marginal scores for a finite pool; choose a batch by the highest scores."""
import torch
from botorch.acquisition.analytic import _log_ei_helper


def score_candidates(model, candidates, *, method, beta=2.0, best_f=None, chunk_size=128):
    scores = []
    with torch.no_grad():
        for chunk in candidates.split(chunk_size):
            # A small matrix query reuses the training covariance without broadcasting
            # 400 x 1530 training coordinates once for every candidate in the chunk.
            posterior = model.posterior(chunk, observation_noise=False)
            mean = posterior.mean.reshape(-1)
            variance = posterior.variance.reshape(-1)
            if method == "ucb":
                value = mean + beta * variance.clamp_min(0).sqrt()
            elif method == "logei":
                # Same stable formula and variance floor as BoTorch 0.18 LogEI.
                sigma = variance.clamp_min(1e-12).sqrt()
                value = _log_ei_helper((mean - best_f) / sigma) + sigma.log()
            else:
                raise ValueError("method must be ucb or logei")
            scores.append(value.reshape(-1))
    result = torch.cat(scores)
    if not torch.isfinite(result).all():
        raise ValueError("Acquisition produced a nonfinite score; no random fallback was used")
    return result
