import numpy as np
import torch


def prompt_rows(train_prompts):
    """Build rows in the same order as the PCPO Dataset."""
    return [{"prompt": text} for text in train_prompts]


def validate_features(features, prompt_count):
    values = torch.as_tensor(features)
    if values.ndim != 2 or values.shape[0] != prompt_count or values.shape[1] != 1530:
        raise ValueError("features must have shape (candidate_count, 1530)")
    return values.double()


def observe_flat(selector, dataset_indices, ocr_rewards):
    """Restore shuffled 64-image rewards to the selector's pending 8 groups."""
    indices = np.asarray(dataset_indices)
    rewards = np.asarray(ocr_rewards, dtype=float)
    expected = selector.pending["prompt_indices"] if selector.pending else None
    if expected is None:
        raise RuntimeError("Call selector.next_batch before observe_flat")
    count = selector.config["batch_size"] * selector.config["images_per_prompt"]
    if indices.shape != (count,) or rewards.shape != (count,):
        raise ValueError(f"Expected {count} dataset indices and OCR rewards")
    if set(indices.tolist()) != set(expected):
        raise ValueError("dataset_indices differ from the pending prompt selection")
    groups = [rewards[indices == i] for i in expected]
    if any(len(group) != selector.config["images_per_prompt"] for group in groups):
        raise ValueError("Each selected prompt must have exactly images_per_prompt rewards")
    return selector.observe(expected, np.stack(groups))
