from pathlib import Path
import json
import numpy as np
import torch

from .gp import fit_model, ucb


class PCPOParameterBO:
    """Contextual BO over a small safe grid of KL coefficient and learning rate."""
    state_version = "pcpo-parameter-bo-v1"

    def __init__(self, *, beta_values=(.02, .04, .08), learning_rate_values=(1.5e-4, 3e-4, 6e-4),
                 seed=42, warmup_trials=4, beta=2.0, kl_budget=None):
        self.beta_values = tuple(float(v) for v in beta_values)
        self.learning_rate_values = tuple(float(v) for v in learning_rate_values)
        self.config = dict(beta_values=self.beta_values, learning_rate_values=self.learning_rate_values,
                           seed=seed, warmup_trials=warmup_trials, beta=beta, kl_budget=kl_budget)
        self.history, self.pending = [], None

    def _candidates(self, context):
        context = np.asarray(context, dtype=float).reshape(-1)
        return [(context, b, lr) for b in self.beta_values for lr in self.learning_rate_values]

    @staticmethod
    def _vector(context, beta, learning_rate):
        context = np.asarray(context, dtype=float).reshape(-1)
        return np.r_[context, np.log(beta), np.log(learning_rate)]

    def suggest(self, context):
        if self.pending is not None:
            return dict(self.pending)
        candidates = self._candidates(context)
        rng = np.random.default_rng(self.config["seed"] + len(self.history))
        if len(self.history) < self.config["warmup_trials"]:
            choice = int(rng.integers(len(candidates)))
            method, scores = "random", None
        else:
            x = torch.tensor([r["x"] for r in self.history], dtype=torch.double)
            y = torch.tensor([r["validation_delta"] for r in self.history], dtype=torch.double).reshape(-1, 1)
            model = fit_model(x, y)
            cx = torch.tensor([self._vector(*candidate) for candidate in candidates], dtype=torch.double)
            scores = ucb(model, cx, self.config["beta"])
            if self.config["kl_budget"] is not None:
                kl = torch.tensor([r["kl"] for r in self.history], dtype=torch.double).reshape(-1, 1)
                kl_model = fit_model(x, kl)
                kl_ucb = ucb(kl_model, cx, self.config["beta"])
                scores = scores - 2.0 * torch.relu(kl_ucb - self.config["kl_budget"])
            choice = int(torch.argmax(scores).item())
            method = "ucb_constrained" if self.config["kl_budget"] is not None else "ucb"
        context_value, beta_value, lr_value = candidates[choice]
        self.pending = {"context": context_value.tolist(), "kl_coefficient": beta_value,
                        "learning_rate": lr_value, "selection_method": method,
                        "acquisition_score": None if scores is None else float(scores[choice])}
        return dict(self.pending)

    def observe(self, validation_delta, kl):
        if self.pending is None:
            raise RuntimeError("Call suggest before observe")
        if not np.isfinite(validation_delta) or not np.isfinite(kl):
            raise ValueError("validation_delta and kl must be finite")
        record = {**self.pending, "validation_delta": float(validation_delta), "kl": float(kl)}
        self.history.append({**record, "x": self._vector(record["context"], record["kl_coefficient"], record["learning_rate"]).tolist()})
        self.pending = None
        return record

    def save_state(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"version": self.state_version, "config": self.config,
                                    "history": self.history, "pending": self.pending}, indent=2) + "\n")

    @classmethod
    def load_state(cls, path):
        state = json.loads(Path(path).read_text())
        if state["version"] != cls.state_version:
            raise ValueError("Parameter BO state version differs")
        tuner = cls(**state["config"])
        tuner.history, tuner.pending = state["history"], state["pending"]
        return tuner
