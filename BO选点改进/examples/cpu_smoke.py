"""Exercise actual BO with synthetic scores; no images or RL model."""
import argparse
from pathlib import Path
import tempfile
import time
import numpy as np
from pcpo_bo import PromptBOSelector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--acquisition', choices=['ucb', 'logei', 'random'], default='ucb')
    parser.add_argument('--pool', type=Path)
    parser.add_argument('--warmup-rounds', type=int, default=1)
    args = parser.parse_args()
    if args.pool:
        rows = [{'prompt': line.strip()} for line in args.pool.read_text().splitlines() if line.strip()]
    else:
        rows = [{'prompt': f'A printed sign reading "word {i}" in a bright room.'} for i in range(64)]
    selector = PromptBOSelector(rows, acquisition=args.acquisition, warmup_rounds=args.warmup_rounds)
    rng = np.random.default_rng(123)
    start = time.monotonic()
    for epoch in range(args.warmup_rounds + 1):
        indices = selector.next_batch()
        # These artificial rewards test interfaces only, never reuse them in real training.
        rewards = rng.uniform(0, 1, size=(8, 8))
        summary = selector.observe_round(indices, rewards, policy_step=epoch)
        if epoch in (0, args.warmup_rounds):
            print(f"synthetic epoch={epoch}, selection={summary['selection_method']}, indices={indices.tolist()}")
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'state.json'
        selector.save_state(path)
        restored = PromptBOSelector.load_state(path, rows)
        assert restored.history == selector.history
    print(f"CPU smoke passed, pool={len(rows)}, acquisition={args.acquisition}, elapsed={time.monotonic()-start:.2f}s")


if __name__ == '__main__':
    main()
