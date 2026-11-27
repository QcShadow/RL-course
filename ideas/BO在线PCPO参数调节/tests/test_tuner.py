import unittest
from bo_tuner import PCPOParameterBO


class TunerTests(unittest.TestCase):
    def test_candidates_and_pending_are_idempotent(self):
        tuner = PCPOParameterBO(warmup_trials=1)
        first = tuner.suggest([.2, .01, .3])
        self.assertEqual(first, tuner.suggest([999]))
        tuner.observe(.1, .02)
        self.assertEqual(len(tuner.history), 1)

    def test_invalid_observation(self):
        tuner = PCPOParameterBO()
        tuner.suggest([0, 0])
        with self.assertRaises(ValueError):
            tuner.observe(float("nan"), .1)


if __name__ == "__main__":
    unittest.main()
