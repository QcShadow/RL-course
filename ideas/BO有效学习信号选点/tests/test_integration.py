import unittest
import numpy as np
import torch
from bo_signal import EffectiveSignalSelector
from bo_signal.integration import observe_flat, prompt_rows, validate_features


class IntegrationTests(unittest.TestCase):
    def test_flat_reward_grouping(self):
        selector = EffectiveSignalSelector(np.eye(32, 1530), images_per_prompt=2)
        ids = selector.next_batch()
        flat_ids = np.repeat(ids, 2)
        rewards = np.arange(16, dtype=float) / 16
        order = np.random.default_rng(1).permutation(16)
        record = observe_flat(selector, flat_ids[order], rewards[order])
        self.assertEqual(len(record["signal"]), 8)

    def test_feature_contract(self):
        self.assertEqual(prompt_rows(["a"])[0]["prompt"], "a")
        self.assertEqual(validate_features(np.zeros((3, 1530)), 3).dtype, torch.float64)


if __name__ == "__main__":
    unittest.main()
