from pathlib import Path
import json
import numpy as np
import torch

from .gp import fit_model, ucb


def group_reward_signal(rewards):
    values = np.asarray(rewards, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("each prompt needs at least two finite image rewards")
    return float(values.std(ddof=1))


class BudgetAllocator:
    """Select unseen prompts and allocate a fixed image budget across them."""
    state_version = "pcpo-image-budget-bo-v1"

    def __init__(self, features, *, seed=42, warmup_rounds=20, prompt_count=8,
                 total_images=64, min_images=4, max_images=12, window_rounds=50, beta=2.0):
        if prompt_count * min_images > total_images or prompt_count * max_images < total_images:
            raise ValueError("image budget cannot satisfy min_images and max_images")
        self.features = torch.as_tensor(features, dtype=torch.double)
        self.config = dict(seed=seed, warmup_rounds=warmup_rounds, prompt_count=prompt_count,
                           total_images=total_images, min_images=min_images, max_images=max_images,
                           window_rounds=window_rounds, beta=beta)
        self.history, self.pending = [], None

    @property
    def completed_rounds(self):
        return len(self.history)

    def _unseen(self):
        used = {int(i) for r in self.history for i in r["prompt_indices"]}
        return [i for i in range(len(self.features)) if i not in used]

    def next_batch(self):
        if self.pending is not None:
            return dict(self.pending)
        unseen = self._unseen()
        if len(unseen) < self.config["prompt_count"]:
            raise ValueError("Not enough untrained prompts for a full batch")
        rng = np.random.default_rng(self.config["seed"] + self.completed_rounds)
        scores = None
        if self.completed_rounds < self.config["warmup_rounds"] or not self.history:
            selected = rng.permutation(unseen)[:self.config["prompt_count"]].tolist()
            method = "random"
        else:
            recent = self.history[-self.config["window_rounds"]:]
            ids = [i for r in recent for i in r["prompt_indices"]]
            y = torch.tensor([v for r in recent for v in r["signal"]], dtype=torch.double).reshape(-1, 1)
            model = fit_model(self.features[ids], y)
            scores_all = ucb(model, self.features, self.config["beta"])
            selected = sorted(unseen, key=lambda i: (-float(scores_all[i]), i))[:self.config["prompt_count"]]
            scores = [float(scores_all[i]) for i in selected]
            method = "ucb"
        counts = [self.config["min_images"]] * len(selected)
        for i in np.argsort(np.asarray(scores if scores is not None else rng.random(len(selected))))[::-1]:
            if sum(counts) == self.config["total_images"]:
                break
            counts[int(i)] += 1
            if counts[int(i)] == self.config["max_images"]:
                continue
            # The next loop pass can keep assigning to the same high-score prompt.
            while sum(counts) < self.config["total_images"] and counts[int(i)] < self.config["max_images"]:
                counts[int(i)] += 1
        self.pending = {"prompt_indices": selected, "image_counts": counts,
                        "selection_method": method,
                        "selection_objective": "maximize_group_signal_per_prompt_budget",
                        "acquisition_scores": scores}
        return dict(self.pending)

    def observe(self, prompt_indices, image_rewards):
        if self.pending is None or list(prompt_indices) != self.pending["prompt_indices"]:
            raise ValueError("prompt_indices must match the pending batch")
        rewards = [np.asarray(v, dtype=float) for v in image_rewards]
        if [len(v) for v in rewards] != self.pending["image_counts"]:
            raise ValueError("image_rewards lengths must match image_counts")
        signal = [group_reward_signal(v) for v in rewards]
        record = {**self.pending, "signal": signal,
                  "round_mean_reward": float(np.concatenate(rewards).mean())}
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
            raise ValueError("Budget allocator state version differs")
        allocator = cls(features, **state["config"])
        allocator.history, allocator.pending = state["history"], state["pending"]
        return allocator
