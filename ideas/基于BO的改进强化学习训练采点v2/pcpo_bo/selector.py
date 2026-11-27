"""Select 8 prompts, observe their 8-image means, and repeat."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from .acquisition import score_candidates
from .features import encode_rows
from .gp import bo_cpu_context, fit_model

STATE_VERSION = "pcpo-prompt-bo-v4-objectives-no-repeat"
OBJECTIVE_NAMES = {
    "min_reward": "minimize_prompt_mean_ocr_reward",
    "max_group_variance": "maximize_prompt_ocr_reward_variance",
}


class PromptBOSelector:
    def __init__(self, candidate_rows, *, seed=42, acquisition="ucb", objective="max_group_variance", beta=2.0,
                 warmup_rounds=20, batch_size=8, images_per_prompt=8, window_rounds=50):
        if acquisition not in {"ucb", "logei", "random"}:
            raise ValueError("acquisition must be ucb, logei or random")
        if objective not in {"min_reward", "max_group_variance"}:
            raise ValueError("objective must be min_reward or max_group_variance")
        self.candidate_rows = list(candidate_rows)
        self.features = encode_rows(self.candidate_rows).cpu()
        self.config = dict(seed=seed, acquisition=acquisition, objective=objective, beta=beta,
                           warmup_rounds=warmup_rounds, batch_size=batch_size,
                           images_per_prompt=images_per_prompt, window_rounds=window_rounds)
        if len(set(row['prompt'] for row in self.candidate_rows)) < batch_size:
            raise ValueError("The pool must contain at least batch_size distinct prompt texts")
        if warmup_rounds < 1 or window_rounds < 1 or batch_size < 2 or images_per_prompt < 1:
            raise ValueError("Use at least one warmup/window round, two prompts and one image")
        self.pool_sha256 = hashlib.sha256(json.dumps(
            self.candidate_rows, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        self.history = []
        self.pending = None

    @property
    def completed_rounds(self):
        return len(self.history)

    def next_batch(self):
        """Idempotent until observe_round/observe_flat commits this pending batch."""
        if self.pending is not None:
            return np.array(self.pending['prompt_indices'], dtype=int)
        cfg = self.config
        seed = cfg['seed'] + self.completed_rounds
        if self.completed_rounds < cfg['warmup_rounds'] or cfg['acquisition'] == 'random':
            generator = torch.Generator(device='cpu').manual_seed(seed)
            order = torch.randperm(len(self.candidate_rows), generator=generator).tolist()
            mode, scores = 'random', None
        else:
            recent = self.history[-cfg['window_rounds']:]
            indices = [i for record in recent for i in record['prompt_indices']]
            target = 'prompt_mean_rewards' if cfg['objective'] == 'min_reward' else 'prompt_reward_variances'
            values = [y for record in recent for y in record[target]]
            best_f = min(values) if cfg['objective'] == 'min_reward' else max(values)
            train_y = torch.tensor(values, dtype=torch.double).reshape(-1, 1)
            if cfg['objective'] == 'min_reward':
                # Acquisition functions maximize, so fit the negative raw reward.
                train_y = -train_y
                best_f = -best_f
            with bo_cpu_context(seed):
                model = fit_model(self.features[indices], train_y)
                scores = score_candidates(model, self.features, method=cfg['acquisition'],
                                          beta=cfg['beta'], best_f=best_f)
            order = torch.argsort(scores, descending=True, stable=True).tolist()
            mode = cfg['acquisition']
        # Use all history, not the GP window, so old prompts stay excluded after resume.
        texts = {self.candidate_rows[i]['prompt'] for record in self.history
                 for i in record['prompt_indices']}
        selected = []
        for index in order:
            prompt = self.candidate_rows[index]['prompt']
            if prompt in texts:
                continue
            selected.append(index)
            texts.add(prompt)
            if len(selected) == cfg['batch_size']:
                break
        if len(selected) != cfg['batch_size']:
            raise ValueError("Not enough untrained distinct prompts for a full batch")
        self.pending = {
            'epoch': self.completed_rounds, 'selection_method': mode,
            'selection_objective': (
                'random_without_replacement' if cfg['acquisition'] == 'random'
                else OBJECTIVE_NAMES[cfg['objective']]
            ),
            'prompt_indices': selected,
            'acquisition_scores': None if scores is None else scores[selected].tolist(),
        }
        return np.array(selected, dtype=int)

    def observe_round(self, prompt_indices, image_rewards, *, policy_step=None):
        """Rows must follow the pending indices; columns are repeated images."""
        if self.pending is None:
            raise RuntimeError("Call next_batch before observing a round")
        indices = np.asarray(prompt_indices)
        expected = np.asarray(self.pending['prompt_indices'])
        if indices.dtype.kind not in 'iu' or not np.array_equal(indices, expected):
            raise ValueError("prompt_indices must equal the pending indices in the same order")
        rewards = np.asarray(image_rewards, dtype=float)
        shape = (self.config['batch_size'], self.config['images_per_prompt'])
        if rewards.shape != shape or not np.isfinite(rewards).all():
            raise ValueError(f"Expected finite image_rewards shaped {shape}; not a scalar round mean")
        if np.any((rewards < 0) | (rewards > 1)):
            raise ValueError("OCR rewards must lie in [0, 1]")
        record = {
            **self.pending, 'policy_step': policy_step,
            'image_rewards': rewards.tolist(),
            'prompt_mean_rewards': rewards.mean(axis=1).tolist(),
            'prompt_reward_variances': rewards.var(axis=1).tolist(),
            'round_mean_reward': float(rewards.mean()),
        }
        self.history.append(record)
        self.pending = None
        return record

    def observe_flat(self, dataset_indices, rewards, *, policy_step=None):
        """Group shuffled per-image rewards by original dataset row index."""
        if self.pending is None:
            raise RuntimeError("Call next_batch before observing a round")
        indices, values = np.asarray(dataset_indices), np.asarray(rewards, dtype=float)
        count = self.config['batch_size'] * self.config['images_per_prompt']
        if indices.shape != (count,) or values.shape != (count,) or indices.dtype.kind not in 'iu':
            raise ValueError(f"Expected {count} integer dataset indices and {count} scalar rewards")
        expected = self.pending['prompt_indices']
        if set(indices.tolist()) != set(expected):
            raise ValueError("Returned dataset indices differ from the pending selection")
        groups = [values[indices == i] for i in expected]
        if any(len(group) != self.config['images_per_prompt'] for group in groups):
            raise ValueError("Each selected dataset index must have exactly images_per_prompt rewards")
        return self.observe_round(expected, np.stack(groups), policy_step=policy_step)

    def save_state(self, path):
        """Write only BO state; the caller must pair it with the same RL epoch."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = dict(version=STATE_VERSION, pool_sha256=self.pool_sha256,
                     config=self.config, history=self.history, pending=self.pending)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)

    @classmethod
    def load_state(cls, path, candidate_rows):
        state = json.loads(Path(path).read_text())
        if state['version'] != STATE_VERSION:
            raise ValueError("BO state version differs: start a new run for the selected no-repeat BO objective")
        selector = cls(candidate_rows, **state['config'])
        if selector.pool_sha256 != state['pool_sha256']:
            raise ValueError("Prompt text or pool order differs from saved state")
        selector.history, selector.pending = state['history'], state['pending']
        return selector
