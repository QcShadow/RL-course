import unittest
import numpy as np
from bo_budget.integration import expand_prompt_indices, group_flat_rewards


class IntegrationTests(unittest.TestCase):
    def test_variable_group_expansion(self):
        ids = expand_prompt_indices([3, 5], [2, 3])
        np.testing.assert_array_equal(ids, [3, 3, 5, 5, 5])
        groups = group_flat_rewards([5, 3, 5, 3, 5], [.5, .1, .7, .2, .9], [3, 5], [2, 3])
        self.assertEqual([len(g) for g in groups], [2, 3])


if __name__ == "__main__":
    unittest.main()
