#!/usr/bin/env bash
set -euo pipefail

cd "${LEROBOT_SRC:-/root/autodl-tmp/lerobot/src}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-lerobot}"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
WANDB_PROJECT="${WANDB_PROJECT:-smolvla-sac-flow-short-${RUN_STAMP}}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-libero_object_task0_action_path_short-${RUN_STAMP}}"
RESUME_ARGS=()
if [[ -n "${SAC_FLOW_RESUME_CHECKPOINT:-}" ]]; then
  RESUME_ARGS+=("--sac-flow.resume-checkpoint=${SAC_FLOW_RESUME_CHECKPOINT}")
fi

export LEROBOT_LIBERO_ROOT="${LEROBOT_LIBERO_ROOT:-/root/autodl-fs/hf_libero_full}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/autodl-fs/hf_datasets_cache}"
export TMPDIR="${TMPDIR:-/root/autodl-fs/tmp}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE=online
export WANDB_PROJECT
export WANDB_RUN_NAME

python lerobot/scripts/rlinf_smolvla_libero_sac_flow_train.py \
  --train-run \
  "${RESUME_ARGS[@]}" \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.root=/root/autodl-fs/hf_libero_full \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids=[0] \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --sac-flow.device="${SAC_FLOW_DEVICE:-cuda:0}" \
  --sac-flow.actor-train-scope=action_path \
  --sac-flow.max-train-steps=280 \
  --sac-flow.max-chunk-steps=1 \
  --sac-flow.num-envs=4 \
  --sac-flow.num-updates-per-step=4 \
  --sac-flow.batch-size=8 \
  --sac-flow.min-buffer-size=8 \
  --sac-flow.replay-capacity=256 \
  --sac-flow.actor-warmup-updates=2000 \
  --sac-flow.noise-std-train=0.02 \
  --sac-flow.noise-std-rollout=0.02 \
  --sac-flow.actor-lr=1e-5 \
  --sac-flow.critic-lr=3e-4 \
  --sac-flow.alpha-lr=3e-4 \
  --sac-flow.wandb-enable=true \
  --sac-flow.wandb-project="${WANDB_PROJECT}" \
  --sac-flow.wandb-run-name="${WANDB_RUN_NAME}" \
  --sac-flow.wandb-tags=short,action_path
