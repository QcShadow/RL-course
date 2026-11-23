"""Initialize one explicitly assigned GPU before importing the training stack."""
import json
import os
from pathlib import Path
import runpy

import torch

fields = {key: os.environ.get(key) for key in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'CUDA_VISIBLE_DEVICES']}
fields.update(cuda_available=torch.cuda.is_available(), visible_gpu_count=torch.cuda.device_count())
print(json.dumps({'event': 'rank_gpu_probe', **fields}), flush=True)
if not fields['cuda_available'] or fields['visible_gpu_count'] != 1:
    raise RuntimeError('Each paper worker must receive exactly one usable assigned GPU')
torch.cuda.init()
torch.cuda.set_device(0)
print(json.dumps({'event': 'rank_gpu_initialized', **fields,
                  'gpu': torch.cuda.get_device_name(0),
                  'gpu_bytes': torch.cuda.get_device_properties(0).total_memory}), flush=True)
# Check the same FP16 BLAS path before loading models. Zero inputs consume no
# RNG, and all allocations are released before restoring the saved RNG/state.
with torch.no_grad():
    left = torch.zeros((1024, 1536), device='cuda', dtype=torch.float16)
    right = torch.zeros((1536, 32), device='cuda', dtype=torch.float16)
    for _ in range(16):
        product = left @ right
    torch.cuda.synchronize()
    if not bool(torch.isfinite(product).all()) or bool(product.any()):
        raise RuntimeError('Assigned GPU failed FP16 zero-GEMM sanity check')
    del left, right, product
torch.cuda.empty_cache()
print(json.dumps({'event': 'rank_fp16_blas_probe_passed', **fields}), flush=True)
runpy.run_path(str(Path(__file__).resolve().parent / 'train_sd3_PCPO_bo.py'), run_name='__main__')
