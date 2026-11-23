#!/usr/bin/env bash
set -euo pipefail
trap ':' USR1
pcpo_task="$1"
pcpo_devices="$2"
pcpo_world="$3"
if [[ "$pcpo_task" != ocr || "$pcpo_world" != 1 ]]; then
    echo 'This handoff supports OCR with one GPU per run.' >&2
    exit 2
fi
pcpo_group="$(id -gn)"
export PCPO_PROJECT="$HOME/projects/pcpo-baseline"
export PCPO_DATA2="/data2/$pcpo_group/$USER/pcpo-baseline"
export PCPO_ENV_PREFIX="/data1/$pcpo_group/$USER/envs/pcpo-baseline"
source /opt/Software/Anaconda3/etc/profile.d/conda.sh
conda activate "$PCPO_ENV_PREFIX"
export PCPO_CODE_ROOT="$PCPO_PROJECT/repo/flowgrpo"
export PCPO_TASK="$pcpo_task"
export PCPO_WORLD_SIZE="$pcpo_world"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$pcpo_devices"
export ACCELERATE_USE_CPU=false
unset ACCELERATE_TORCH_DEVICE
export HF_HOME="$PCPO_DATA2/models/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export PCPO_MODEL_DIR="$HF_HUB_CACHE/models--stabilityai--stable-diffusion-3.5-medium/snapshots/b940f670f0eda2d07fbb75229e779da1ad11eb80"
export PCPO_OCR_MODEL_ROOT="$PCPO_DATA2/models/paddleocr"
export PCPO_EMBEDDINGS_ROOT="$PCPO_DATA2/embeddings"
export PCPO_RUN_DIR="$PCPO_RUN_ROOT/$pcpo_task-seed42"
export PCPO_TARGET_EPOCHS="${PCPO_TARGET_EPOCHS:-250}"
export PCPO_EVAL_SEED=42
export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export WANDB_MODE=offline
export WANDB_DIR="$PCPO_RUN_DIR/wandb"
export WANDB_SILENT=true
export ALBUMENTATIONS_OFFLINE=1
export NO_ALBUMENTATIONS_UPDATE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PCPO_PROJECT/source:$PCPO_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$PCPO_RUN_DIR" "$WANDB_DIR"
cd "$PCPO_CODE_ROOT"
if [[ "$pcpo_task" == pickscore ]]; then pcpo_dataset=pickscore_sfw; else pcpo_dataset=ocr; fi
if [[ -s "$PCPO_EMBEDDINGS_ROOT/$pcpo_dataset/train_embeddings.h5" && -s "$PCPO_EMBEDDINGS_ROOT/$pcpo_dataset/test_embeddings.h5" ]]; then
    printf 'Reusing previously validated full embeddings: %s\n' "$pcpo_dataset"
else
    "$PCPO_ENV_PREFIX/bin/python" -u "$PCPO_PROJECT/source/precompute_embeddings.py" \
        --prompt-dir "$PCPO_CODE_ROOT/dataset/$pcpo_dataset" \
        --output-dir "$PCPO_EMBEDDINGS_ROOT/$pcpo_dataset" --batch-size 64
fi
if [[ "$pcpo_task" == ocr ]]; then
    CUDA_VISIBLE_DEVICES="" "$PCPO_ENV_PREFIX/bin/python" -u "$PCPO_PROJECT/source/check_ocr_execution.py"
fi
exec "$PCPO_ENV_PREFIX/bin/python" -u "$PCPO_PROJECT/source/bo_launch.py"
