from pathlib import Path
import json
import numpy as np
import torch

from .gp import fit_model, ucb


def cosine_alignment(gradients, reference_gradient):
    gradients = np.asarray(gradients, dtype=float)
    reference = np.asarray(reference_gradient, dtype=float).reshape(-1)
    if gradients.ndim != 2 or gradients.shape[1] != reference.size:
        raise ValueError("gradients must be [batch, dimension] and match reference_gradient")
    denom = np.linalg.norm(gradients, axis=1) * np.linalg.norm(reference)
    return np.divide(gradients @ reference, denom, out=np.zeros(len(gradients)), where=denom > 0)


class GradientAlignmentSelector:
    """Use PCPO gradient alignment as the BO target for unseen prompts."""
    state_version = "gradient-alignment-pcpo-v1"

    def __init__(self, features, *, seed=42, warmup_rounds=20, batch_size=8,
                 window_rounds=50, beta=2.0):
        self.features = torch.as_tensor(features, dtype=torch.double)
        self.config = dict(seed=seed, warmup_rounds=warmup_rounds, batch_size=batch_size,
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
            return np.asarray(self.pending["prompt_indices"], dtype=int)
        unseen = self._unseen()
        if len(unseen) < self.config["batch_size"]:
            raise ValueError("Not enough untrained prompts for a full batch")
        rng = np.random.default_rng(self.config["seed"] + self.completed_rounds)
        if self.completed_rounds < self.config["warmup_rounds"] or not self.history:
            order, method, scores = rng.permutation(unseen).tolist(), "random", None
        else:
            recent = self.history[-self.config["window_rounds"]:]
            ids = [i for r in recent for i in r["prompt_indices"]]
            y = torch.tensor([v for r in recent for v in r["alignment"]], dtype=torch.double).reshape(-1, 1)
            model = fit_model(self.features[ids], y)
            all_scores = ucb(model, self.features, self.config["beta"])
            order = sorted(unseen, key=lambda i: (-float(all_scores[i]), i))
            method, scores = "ucb", all_scores
        selected = order[:self.config["batch_size"]]
        self.pending = {"prompt_indices": selected, "selection_method": method,
                        "selection_objective": "maximize_pcpo_gradient_alignment",
                        "acquisition_scores": None if scores is None else [float(scores[i]) for i in selected]}
        return np.asarray(selected, dtype=int)

    def observe(self, prompt_indices, gradients, reference_gradient):
        if self.pending is None or not np.array_equal(prompt_indices, self.pending["prompt_indices"]):
            raise ValueError("prompt_indices must match the pending batch")
        alignment = cosine_alignment(gradients, reference_gradient)
        if len(alignment) != self.config["batch_size"] or not np.isfinite(alignment).all():
            raise ValueError("Expected one finite gradient alignment per selected prompt")
        record = {**self.pending, "alignment": alignment.tolist(),
                  "reference_gradient_norm": float(np.linalg.norm(reference_gradient))}
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
            raise ValueError("Gradient alignment state version differs")
        selector = cls(features, **state["config"])
        selector.history, selector.pending = state["history"], state["pending"]
        return selector
