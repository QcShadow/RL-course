import unittest
import numpy as np
from bo_align import GradientAlignmentSelector, cosine_alignment


class AlignmentTests(unittest.TestCase):
    def test_cosine_alignment(self):
        values = cosine_alignment([[1, 0], [0, 1], [0, 0]], [1, 0])
        np.testing.assert_allclose(values, [1, 0, 0])

    def test_selector_records_alignment(self):
        selector = GradientAlignmentSelector(np.eye(24), warmup_rounds=1)
        ids = selector.next_batch()
        record = selector.observe(ids, np.eye(8), np.ones(8))
        self.assertEqual(len(record["alignment"]), 8)
        self.assertTrue(set(ids).isdisjoint(selector.next_batch()))


if __name__ == "__main__":
    unittest.main()
