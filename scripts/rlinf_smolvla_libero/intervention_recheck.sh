#!/usr/bin/env bash
set -euo pipefail

cd "${LEROBOT_SRC:-/root/autodl-tmp/lerobot/src}"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-lerobot}"

RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
OUTPUT_DIR="${SAC_FLOW_OUTPUT_DIR:-/root/autodl-fs/sac_flow_intervention_recheck-${RUN_STAMP}}"
TASK_IDS="${SAC_FLOW_TASK_IDS:-[9]}"
NUM_ENVS="${SAC_FLOW_NUM_ENVS:-10}"

export LEROBOT_LIBERO_ROOT="${LEROBOT_LIBERO_ROOT:-/root/autodl-fs/hf_libero_full}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/autodl-fs/hf_datasets_cache}"
export TMPDIR="${TMPDIR:-/root/autodl-fs/tmp}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE=disabled

python lerobot/scripts/rlinf_smolvla_libero_sac_flow_train.py \
  --train-run \
  --policy.path=HuggingFaceVLA/smolvla_libero \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --dataset.root=/root/autodl-fs/hf_libero_full \
  --output_dir="${OUTPUT_DIR}" \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids="${TASK_IDS}" \
  --env.paired_init_states=true \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --sac-flow.device="${SAC_FLOW_DEVICE:-cuda:0}" \
  --sac-flow.seed="${SAC_FLOW_SEED:-0}" \
  --sac-flow.actor-train-scope=action_path \
  --sac-flow.max-train-steps=280 \
  --sac-flow.max-chunk-steps=1 \
  --sac-flow.num-envs="${NUM_ENVS}" \
  --sac-flow.num-updates-per-step=4 \
  --sac-flow.batch-size=8 \
  --sac-flow.min-buffer-size=256 \
  --sac-flow.replay-capacity=8192 \
  --sac-flow.actor-warmup-updates=2000 \
  --sac-flow.actor-updates-enabled=false \
  --sac-flow.save-checkpoint=true \
  --sac-flow.heldout-num-steps=256 \
  --sac-flow.heldout-seed=2000 \
  --sac-flow.root-cause-diagnostics=true \
  --sac-flow.root-cause-max-transitions-per-task=32 \
  --sac-flow.root-cause-gradient-repeats=1 \
  --sac-flow.noise-std-train=0.02 \
  --sac-flow.noise-std-rollout=0.02 \
  --sac-flow.actor-lr=3e-6 \
  --sac-flow.actor-agg-q=min \
  --sac-flow.entropy-regularization=false \
  --sac-flow.backup-entropy=false \
  --sac-flow.critic-lr=3e-4 \
  --sac-flow.alpha-lr=3e-4 \
  --sac-flow.critic-positive-sample-fraction=0.5 \
  --sac-flow.critic-task-balanced-sampling=true \
  --sac-flow.critic-intervention-fraction=0.5 \
  --sac-flow.critic-intervention-noise-std=0.3 \
  --sac-flow.critic-intervention-balanced-sampling=true \
  --sac-flow.critic-intervention-pairing=true \
  --sac-flow.critic-pairwise-coef=1.0 \
  --sac-flow.critic-pairwise-margin=0.05 \
  --sac-flow.critic-pairwise-min-length-gap=5 \
  --sac-flow.critic-step-penalty=0.001 \
  --sac-flow.critic-conservative-coef=0 \
  --sac-flow.critic-monte-carlo-coef=0.1 \
  --sac-flow.critic-random-action-strategy=replay_local_gaussian \
  --sac-flow.critic-random-action-std=0.05 \
  --sac-flow.wandb-enable=false
