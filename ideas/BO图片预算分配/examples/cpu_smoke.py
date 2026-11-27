import numpy as np
from bo_budget import BudgetAllocator

allocator = BudgetAllocator(np.random.default_rng(0).normal(size=(40, 12)), warmup_rounds=1)
for _ in range(3):
    batch = allocator.next_batch()
    allocator.observe(batch["prompt_indices"], [np.random.default_rng(i).random(n) for i, n in zip(batch["prompt_indices"], batch["image_counts"])])
print("image-budget smoke passed", allocator.completed_rounds)
