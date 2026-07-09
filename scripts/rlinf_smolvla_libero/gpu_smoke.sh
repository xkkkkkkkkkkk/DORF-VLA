#!/usr/bin/env bash
set -euo pipefail

cd "${LEROBOT_SRC:-/root/autodl-tmp/lerobot/src}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-lerobot}"

export LEROBOT_LIBERO_ROOT="${LEROBOT_LIBERO_ROOT:-/root/autodl-fs/hf_libero_full}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/autodl-fs/hf_datasets_cache}"
export TMPDIR="${TMPDIR:-/root/autodl-fs/tmp}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE=disabled

python lerobot/scripts/rlinf_smolvla_libero_sac_flow_train.py \
  --gpu-smoke \
  --confirm-gpu-smoke \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.root=/root/autodl-fs/hf_libero_full \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids=[0] \
  --sac-flow.device="${SAC_FLOW_DEVICE:-cuda:0}" \
  --sac-flow.actor-train-scope=action_path \
  --sac-flow.max-train-steps=2 \
  --sac-flow.max-chunk-steps=1 \
  --sac-flow.num-updates-per-step=1 \
  --sac-flow.batch-size=1 \
  --sac-flow.min-buffer-size=1 \
  --sac-flow.wandb-enable=false
