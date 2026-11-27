from pathlib import Path
import json
import numpy as np
import torch

from .gp import fit_model, ucb


def effective_signal(image_rewards, adv_clip_max=5.0):
    """Return the per-prompt mean absolute clipped PCPO advantage.

    The denominator follows the supplied OCR PCPO configuration: one standard
    deviation over the whole 64-image rollout, not one standard deviation per prompt.
    """
    rewards = np.asarray(image_rewards, dtype=float)
    if rewards.ndim != 2 or not np.isfinite(rewards).all():
        raise ValueError("image_rewards must be a finite 2-D array")
    global_std = rewards.std() + 1e-4
    centered = rewards - rewards.mean(axis=1, keepdims=True)
    advantages = np.clip(centered / global_std, -adv_clip_max, adv_clip_max)
    return np.mean(np.abs(advantages), axis=1)


class EffectiveSignalSelector:
    """Select unseen prompts whose predicted PCPO signal is largest."""
    state_version = "effective-pcpo-signal-v1"

    def __init__(self, features, *, seed=42, warmup_rounds=20, batch_size=8,
                 images_per_prompt=8, window_rounds=50, beta=2.0):
        self.features = torch.as_tensor(features, dtype=torch.double)
        self.config = dict(seed=seed, warmup_rounds=warmup_rounds, batch_size=batch_size,
                           images_per_prompt=images_per_prompt, window_rounds=window_rounds,
                           beta=beta)
        self.history, self.pending = [], None

    @property
    def completed_rounds(self):
        return len(self.history)

    def _unseen(self):
        used = {int(i) for record in self.history for i in record["prompt_indices"]}
        return [i for i in range(len(self.features)) if i not in used]

    def next_batch(self):
        if self.pending is not None:
            return np.asarray(self.pending["prompt_indices"], dtype=int)
        unseen = self._unseen()
        if len(unseen) < self.config["batch_size"]:
            raise ValueError("Not enough untrained prompts for a full batch")
        rng = np.random.default_rng(self.config["seed"] + self.completed_rounds)
        if self.completed_rounds < self.config["warmup_rounds"] or not self.history:
            order, method = rng.permutation(unseen).tolist(), "random"
            scores = None
        else:
            recent = self.history[-self.config["window_rounds"]:]
            ids = [i for r in recent for i in r["prompt_indices"]]
            y = torch.tensor([v for r in recent for v in r["signal"]], dtype=torch.double).reshape(-1, 1)
            model = fit_model(self.features[ids], y)
            all_scores = ucb(model, self.features, self.config["beta"])
            order = sorted(unseen, key=lambda i: (-float(all_scores[i]), i))
            method, scores = "ucb", all_scores
        selected = order[:self.config["batch_size"]]
        self.pending = {"prompt_indices": selected, "selection_method": method,
                        "selection_objective": "maximize_effective_pcpo_signal",
                        "acquisition_scores": None if scores is None else [float(scores[i]) for i in selected]}
        return np.asarray(selected, dtype=int)

    def observe(self, prompt_indices, image_rewards):
        if self.pending is None or not np.array_equal(prompt_indices, self.pending["prompt_indices"]):
            raise ValueError("prompt_indices must match the pending batch")
        rewards = np.asarray(image_rewards, dtype=float)
        shape = (self.config["batch_size"], self.config["images_per_prompt"])
        if rewards.shape != shape or np.any((rewards < 0) | (rewards > 1)):
            raise ValueError(f"Expected OCR rewards in [0, 1] with shape {shape}")
        record = {**self.pending, "image_rewards": rewards.tolist(),
                  "signal": effective_signal(rewards).tolist(),
                  "round_mean_reward": float(rewards.mean())}
        self.history.append(record)
        self.pending = None
        return record

    def save_state(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": self.state_version, "config": self.config,
                                    "history": self.history, "pending": self.pending}, indent=2) + "\n")

    @classmethod
    def load_state(cls, path, features):
        state = json.loads(Path(path).read_text())
        if state["version"] != cls.state_version:
            raise ValueError("Signal selector state version differs")
        selector = cls(features, **state["config"])
        selector.history, selector.pending = state["history"], state["pending"]
        return selector
