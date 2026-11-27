import numpy as np
import torch


def flatten_parameter_grads(parameters):
    """Return a detached float64 vector for the currently retained gradients."""
    chunks = [p.grad.detach().reshape(-1).cpu().double().numpy()
              for p in parameters if p.grad is not None]
    if not chunks:
        raise ValueError("No parameter gradients are available")
    return np.concatenate(chunks)


def random_project(vector, projection):
    vector = np.asarray(vector, dtype=float).reshape(-1)
    projection = np.asarray(projection, dtype=float)
    if projection.ndim != 2 or projection.shape[1] != vector.size:
        raise ValueError("projection must have shape (output_dim, gradient_dim)")
    return projection @ vector


def group_gradient_matrix(group_gradients, projection=None):
    values = np.asarray(group_gradients, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("group_gradients must be a finite 2-D array")
    if projection is None:
        return values
    return np.stack([random_project(row, projection) for row in values])
