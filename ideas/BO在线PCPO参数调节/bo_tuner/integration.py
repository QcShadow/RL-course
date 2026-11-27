import numpy as np


CONTEXT_KEYS = ("validation_reward", "train_reward", "kl", "clipfrac", "advantage_std")


def context_from_metrics(metrics):
    """Read the fixed context order used by the tuner from training metrics."""
    return np.asarray([float(metrics[key]) for key in CONTEXT_KEYS], dtype=float)


def apply_learning_rate(optimizer, learning_rate):
    """Apply a suggested learning rate to every existing optimizer group."""
    value = float(learning_rate)
    for group in optimizer.param_groups:
        group["lr"] = value


def selected_kl_coefficient(choice):
    """Return the scalar to use in the PCPO KL term."""
    return float(choice["kl_coefficient"])
