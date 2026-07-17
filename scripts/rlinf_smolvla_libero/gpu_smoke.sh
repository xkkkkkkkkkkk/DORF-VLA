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

# Keep the historical offline-safe default. Set WANDB_ENABLE=true to publish
# the two-step smoke metrics to WandB without changing the smoke budget.
WANDB_ENABLE="${WANDB_ENABLE:-false}"
WANDB_ARGS=(--sac-flow.wandb-enable=false)
if [[ "${WANDB_ENABLE}" == "true" ]]; then
  RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
  WANDB_PROJECT="${WANDB_PROJECT:-smolvla-sac-flow-smoke-${RUN_STAMP}}"
  WANDB_RUN_NAME="${WANDB_RUN_NAME:-libero_object_task0_action_path_smoke-${RUN_STAMP}}"
  export WANDB_MODE="${WANDB_MODE:-online}"
  export WANDB_PROJECT
  export WANDB_RUN_NAME
  WANDB_ARGS=(
    --sac-flow.wandb-enable=true
    --sac-flow.wandb-project="${WANDB_PROJECT}"
    --sac-flow.wandb-run-name="${WANDB_RUN_NAME}"
    --sac-flow.wandb-tags=smoke,action_path
  )
else
  export WANDB_MODE=disabled
fi

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
  "${WANDB_ARGS[@]}"
