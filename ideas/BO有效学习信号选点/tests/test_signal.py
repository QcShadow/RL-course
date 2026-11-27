import unittest
import numpy as np
from bo_signal import EffectiveSignalSelector, effective_signal


class SignalTests(unittest.TestCase):
    def test_constant_group_has_no_signal(self):
        rewards = np.array([[0, 0, 0, 0], [0, .2, .4, .6]], dtype=float)
        np.testing.assert_allclose(effective_signal(rewards)[0], 0)
        self.assertGreater(effective_signal(rewards)[1], 0)

    def test_unseen_selection_and_round_observation(self):
        selector = EffectiveSignalSelector(np.eye(24), warmup_rounds=1, images_per_prompt=2)
        ids = selector.next_batch()
        selector.observe(ids, np.tile([[0, 0], [.2, .8], [.3, .3], [.1, .9], [.4, .4], [.6, .2], [.1, .7], [.5, .5]], (1, 1)))
        next_ids = selector.next_batch()
        self.assertTrue(set(ids).isdisjoint(next_ids))


if __name__ == "__main__":
    unittest.main()
