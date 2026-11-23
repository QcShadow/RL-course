#!/usr/bin/env python3
"""Train and independently reload the final EMA within one assigned task slot."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

signal.signal(signal.SIGUSR1, signal.SIG_IGN)
project = Path(os.environ['PCPO_PROJECT'])
run = Path(os.environ['PCPO_RUN_DIR'])
world = int(os.environ['PCPO_WORLD_SIZE'])
job = os.environ['SLURM_JOB_ID']
stop_file = run / ('STOP-' + job)


def launch(eval_only=False):
    env = dict(os.environ)
    env['PCPO_EVAL_ONLY'] = '1' if eval_only else '0'
    assigned = env['CUDA_VISIBLE_DEVICES'].split(',')
    if len(assigned) != world:
        raise RuntimeError('Assigned UUID count does not match task world size')
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    command = [sys.executable, str(project / 'source/bo_rank_start.py'),
               '--config', str(project / 'source/paper_config.py')]
    print(json.dumps({'event': 'launch', 'task': os.environ['PCPO_TASK'],
                      'world_size': world, 'eval_only': eval_only,
                      'cuda_visible_devices': env.get('CUDA_VISIBLE_DEVICES'),
                      'accelerate_use_cpu': env.get('ACCELERATE_USE_CPU')}), flush=True)
    workers = []
    for rank, device in enumerate(assigned):
        rank_env = dict(env, CUDA_VISIBLE_DEVICES=device, ACCELERATE_USE_CPU='false')
        if world > 1:
            # Every process owns one visible GPU (local CUDA index zero). Global
            # RANK still identifies its distinct NCCL/DataLoaderShard participant.
            rank_env.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK='0',
                            LOCAL_WORLD_SIZE=str(world), MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
        else:
            for name in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE']:
                rank_env.pop(name, None)
        workers.append(subprocess.Popen(command, env=rank_env))
    first_failure = None
    while True:
        codes = [worker.poll() for worker in workers]
        bad = [code for code in codes if code not in (None, 0, 75)]
        if bad and first_failure is None:
            first_failure = bad[0]
            # A failed collective cannot safely checkpoint a partially updated
            # peer. Preserve the last complete boundary and release the GPUs.
            for worker in workers:
                if worker.poll() is None:
                    worker.terminate()
        if all(code is not None for code in codes):
            break
        time.sleep(1)
    code = first_failure or next((value for value in codes if value), 0)
    if code and stop_file.exists() and all(
        (run / f'STOPPED-{job}-rank-{rank}.json').exists() for rank in range(world)
    ):
        return 75
    return code


if (run / 'summary.json').exists():
    existing = json.loads((run / 'summary.json').read_text())
    if existing.get('status') == 'completed':
        raise SystemExit(0)
if not (run / 'TRAINING_DONE.json').exists():
    code = launch()
    if code:
        raise SystemExit(code)
if stop_file.exists():
    raise SystemExit(75)
raise SystemExit(launch(eval_only=True))
