import numpy as np
from bo_signal import EffectiveSignalSelector

selector = EffectiveSignalSelector(np.random.default_rng(0).normal(size=(40, 12)), warmup_rounds=1)
for _ in range(3):
    ids = selector.next_batch()
    selector.observe(ids, np.random.default_rng(int(ids[0])).random((8, 8)))
print("effective-signal smoke passed", selector.completed_rounds)
