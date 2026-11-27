import unittest
import torch
from bo_tuner.integration import apply_learning_rate, context_from_metrics, selected_kl_coefficient


class IntegrationTests(unittest.TestCase):
    def test_context_and_optimizer_hook(self):
        context = context_from_metrics({key: i for i, key in enumerate(("validation_reward", "train_reward", "kl", "clipfrac", "advantage_std"))})
        self.assertEqual(context.tolist(), [0., 1., 2., 3., 4.])
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=.1)
        apply_learning_rate(optimizer, .02)
        self.assertEqual(optimizer.param_groups[0]["lr"], .02)
        self.assertEqual(selected_kl_coefficient({"kl_coefficient": .04}), .04)


if __name__ == "__main__":
    unittest.main()
