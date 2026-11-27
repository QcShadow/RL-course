import numpy as np


def expand_prompt_indices(prompt_indices, image_counts):
    if len(prompt_indices) != len(image_counts) or any(int(n) < 1 for n in image_counts):
        raise ValueError("prompt_indices and image_counts must have matching positive lengths")
    return np.asarray([int(i) for i, n in zip(prompt_indices, image_counts) for _ in range(int(n))], dtype=int)


def group_flat_rewards(dataset_indices, rewards, prompt_indices, image_counts):
    indices = np.asarray(dataset_indices)
    values = np.asarray(rewards, dtype=float)
    if indices.ndim != 1 or values.shape != indices.shape:
        raise ValueError("dataset_indices and rewards must be matching 1-D arrays")
    expected = expand_prompt_indices(prompt_indices, image_counts)
    if len(expected) != len(indices) or set(indices.tolist()) != set(prompt_indices):
        raise ValueError("flat rewards do not match the selected prompt groups")
    groups = [values[indices == i] for i in prompt_indices]
    if [len(g) for g in groups] != [int(n) for n in image_counts]:
        raise ValueError("flat rewards do not match image_counts")
    return groups
