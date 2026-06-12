# SmolVLA FM-RL post-training smoke run

This is the clean RL post-training baseline for the current SmolVLA flow-matching policy.
It is intentionally separate from `lerobot/scripts/tlofVLA_train.py`: no DORF reward model,
no critic, no staged readiness gate, and no offline expert reward loss are trained here.

## Standard flow aligned to RLinf

```text
SFT SmolVLA checkpoint
  -> collect one rollout group on one LIBERO task
  -> compute true trajectory returns / success
  -> group-normalize returns as GRPO-style advantages
  -> map advantages to clipped FM weights
  -> sample fixed-observation policy-space action chunks
  -> update SmolVLA with weighted flow-matching loss
```

The implemented training entry is:

```text
lerobot/scripts/smolvla_fm_rl_train.py
```

Core helpers live in:

```text
lerobot/scripts/smolvla_fm_rl_utils.py
```

## Why `synthetic` labels are the default

`lerobot_eval.rollout()` returns actions after policy and environment postprocessors.
Those actions are suitable for `env.step(...)`, but they are not guaranteed to be in the same
normalized action space consumed by `SmolVLAPolicy.forward(...)` during FM training.

Therefore the default is:

```text
SMOLVLA_FM_RL_LABEL_MODE=synthetic
```

In this mode, rollout supplies only the RL signal, while the FM label is sampled via
`policy.predict_action_chunk(...)` at the fixed initial rollout observation.

Use `rollout_prefix` only after confirming rollout actions are policy-space labels.

## Smoke command template

Start from the same config/overrides you used for the current SmolVLA or `tlofVLA_train.py` run,
but swap the script path and keep the run very short:

```bash
export LEROBOT_LIBERO_ROOT=/root/autodl-fs/hf_libero_full
export SMOLVLA_FM_RL_GROUP_SIZE=8
export SMOLVLA_FM_RL_WEIGHT_BETA=0.5
export SMOLVLA_FM_RL_MIN_WEIGHT=0.05
export SMOLVLA_FM_RL_MAX_WEIGHT=5.0
export SMOLVLA_FM_RL_ACCUMULATION_STEPS=4
export SMOLVLA_FM_RL_LABEL_MODE=synthetic

python lerobot/scripts/smolvla_fm_rl_train.py \
  <your existing train config / CLI overrides> \
  steps=3 \
  log_freq=1 \
  save_freq=1 \
  eval_freq=-1 \
  batch_size=4
```

If the parser uses `--config_path`, keep your original `--config_path ...` exactly as before.
Do not add custom `rl_*` CLI fields; use the environment variables above instead.

## Metrics to inspect first

A valid smoke run should reach all of these at least once:

```text
train/rollout_return_mean
train/rollout_success_rate
train/policy/fm_loss
train/policy_weight_mean
train/policy_advantage_std
train/policy/final_grad_norm
train/policy_micro_batches
```

Minimum smoke pass criteria:

1. The script creates the dataset metadata and rollout env.
2. One rollout group is collected.
3. Weighted FM update runs without shape/device/action-space errors.
4. `train/policy/fm_loss` is finite.
5. `train/policy/final_grad_norm` is finite.
6. A checkpoint can be written when `save_checkpoint=True` and `save_freq=1`.

## First failure triage

- Missing local data: set `LEROBOT_LIBERO_ROOT` or `SMOLVLA_FM_RL_DATA_ROOT`.
- Group size too large for memory: reduce `SMOLVLA_FM_RL_GROUP_SIZE` to 4 or 2.
- NaN weights: check `train/policy_advantage_std`; singleton or identical returns should produce zero advantages and weight 1.
- Shape mismatch in `policy.forward`: keep `SMOLVLA_FM_RL_LABEL_MODE=synthetic`; do not switch to `rollout_prefix` yet.
- Immediate reward collapse: run a 20-50 step diagnostic and compare `label_mode=synthetic` vs frozen policy/eval-only behavior before adding DORF back.
