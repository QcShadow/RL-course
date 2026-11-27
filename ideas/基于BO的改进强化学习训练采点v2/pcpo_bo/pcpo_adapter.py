"""Replacement for KRepeatSampler in the supplied single-GPU PCPO driver."""
from pathlib import Path
import torch

from .selector import PromptBOSelector


class SelectedPromptBatchSampler:
    def __init__(self, dataset_size, microbatch_size, *, images_per_prompt=8,
                 prompts_per_round=8, seed=42):
        self.dataset_size = dataset_size
        self.batch_size = microbatch_size
        self.drop_last = False
        self.images_per_prompt = images_per_prompt
        self.prompts_per_round = prompts_per_round
        self.seed, self.epoch, self.indices = seed, 0, None
        if prompts_per_round * images_per_prompt % microbatch_size:
            raise ValueError("Microbatch size must divide the round image budget")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_indices(self, indices):
        indices = [int(i) for i in indices]
        if len(indices) != self.prompts_per_round or len(set(indices)) != len(indices):
            raise ValueError("Expected one distinct dataset index per selected prompt")
        if any(i < 0 or i >= self.dataset_size for i in indices):
            raise IndexError("Selected dataset index is outside the training pool")
        self.indices = indices

    def __iter__(self):
        if self.indices is None:
            raise RuntimeError("Call set_indices before iterating the training DataLoader")
        g = torch.Generator(device='cpu').manual_seed(self.seed + self.epoch)
        # Match the original KRepeatSampler's shuffle RNG after its prompt permutation.
        torch.randperm(self.dataset_size, generator=g)
        repeated = [i for i in self.indices for _ in range(self.images_per_prompt)]
        shuffled = [repeated[j] for j in torch.randperm(len(repeated), generator=g).tolist()]
        for start in range(0, len(shuffled), self.batch_size):
            yield shuffled[start:start + self.batch_size]

    def __len__(self):
        return self.prompts_per_round * self.images_per_prompt // self.batch_size


def load_for_epoch(rows, directory, epoch, **options):
    """Recover the sidecar matching the restored PCPO checkpoint, never 'latest'."""
    path = Path(directory) / f'epoch-{epoch:06d}.json'
    if path.exists():
        selector = PromptBOSelector.load_state(path, rows)
        if selector.completed_rounds != epoch or selector.config != options:
            raise ValueError("BO state epoch/settings do not match this PCPO run")
        return selector
    if epoch != 0:
        raise FileNotFoundError(f"No BO state for resumed PCPO epoch {epoch}: {path}")
    return PromptBOSelector(rows, **options)
