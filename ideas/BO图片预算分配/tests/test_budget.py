import unittest
import numpy as np
from bo_budget import BudgetAllocator, group_reward_signal


class BudgetTests(unittest.TestCase):
    def test_signal_and_fixed_budget(self):
        self.assertAlmostEqual(group_reward_signal([0, .5, 1]), .5)
        allocator = BudgetAllocator(np.eye(40), warmup_rounds=1, min_images=4, max_images=12)
        batch = allocator.next_batch()
        self.assertEqual(sum(batch["image_counts"]), 64)
        self.assertTrue(all(4 <= n <= 12 for n in batch["image_counts"]))
        allocator.observe(batch["prompt_indices"], [np.linspace(0, 1, n) for n in batch["image_counts"]])
        self.assertTrue(set(batch["prompt_indices"]).isdisjoint(allocator.next_batch()["prompt_indices"]))


if __name__ == "__main__":
    unittest.main()
