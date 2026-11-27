import numpy as np
from bo_align import GradientAlignmentSelector

selector = GradientAlignmentSelector(np.random.default_rng(0).normal(size=(40, 12)), warmup_rounds=1)
for _ in range(3):
    ids = selector.next_batch()
    gradients = np.random.default_rng(int(ids[0])).normal(size=(8, 16))
    reference = np.ones(16)
    selector.observe(ids, gradients, reference)
print("gradient-alignment smoke passed", selector.completed_rounds)
