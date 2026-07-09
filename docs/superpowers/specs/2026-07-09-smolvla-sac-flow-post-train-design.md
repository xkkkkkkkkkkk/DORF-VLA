# SmolVLA SAC-Flow Post-Training Design

Date: 2026-07-09
Status: Draft for user review
Workspace: `C:\Users\w1985\.codex\worktrees\aecd\DORF-VLA`

## 1. Purpose

Task14 has proven that the SmolVLA SAC-Flow path can run a real GPU smoke test on LIBERO:

- SmolVLA can act as the SAC actor.
- LIBERO rollout can collect transitions.
- Replay, critic, actor update, entropy temperature update, and checkpoint writing all execute.

This design moves the work from "smoke test passes" to "controlled short-run post-training". The goal is to make the training run:

- Explicit about which SmolVLA parameters are updated.
- Observable in Weights & Biases.
- Reproducible through scripts.
- Conservative enough to avoid silently damaging the VLM backbone.

## 2. Decisions

### 2.1 Baseline Trainable Scope

The first baseline will use `action_path` training.

Default trainable modules:

- `vlm_with_expert.lm_expert`
- `action_in_proj`
- `action_out_proj`
- `action_time_mlp_in`
- `action_time_mlp_out`

Default frozen modules:

- VLM backbone
- vision encoder
- language backbone
- `state_proj`

Rationale:

- This follows the VLA post-training principle used in RLinf-style models: do not update the full VLM by default.
- It isolates the update to the action generation path.
- It avoids adding LoRA as a second variable before the full action-path baseline has shown whether SAC-Flow produces a useful signal.
- Freezing `state_proj` is intentionally conservative. It prevents the first baseline from changing state/proprioception encoding while we evaluate SAC-Flow's effect on the action generator itself.

### 2.2 LoRA Is Not First Baseline

LoRA will not be used in the first implementation.

Rationale:

- LoRA saves memory, but it restricts the learnable update to a low-rank subspace.
- If action-path LoRA does not improve performance, the result would be ambiguous: SAC-Flow might be ineffective, or the LoRA rank/target modules might be too restrictive.
- A full update of the action path is the clearer baseline for testing whether SAC-Flow can improve SmolVLA behavior.

Future optional scopes:

- `action_path_lora`
- `action_path_state_proj`
- `all`

These are not part of the first implementation unless explicitly requested later.

### 2.3 WandB Default

For actual training scripts, WandB is enabled by default.

Smoke tests and automated tests may explicitly disable WandB.

Rationale:

- The user needs real-time visibility into SAC-Flow losses and environment signals.
- Silent training runs are not acceptable for this stage.
- Scripts should allow the user to set `WANDB_PROJECT` manually. If unset, the script should generate a timestamped project/run name.

## 3. Current Code Facts

Relevant current code:

- `lerobot/rlinf_smolvla_libero/actor_adapter.py`
  - `SmolVLASACFlowActor.parameters()` delegates to `policy.parameters()`.
- `lerobot/rlinf_smolvla_libero/trainer.py`
  - `actor_optimizer` is built from `self.actor.parameters()`.
- `lerobot/policies/smolvla/configuration_smolvla.py`
  - Existing defaults include `train_expert_only=True`, `freeze_vision_encoder=True`, and `train_state_proj=True`.
- `lerobot/policies/smolvla/smolvlm_with_expert.py`
  - `train_expert_only=True` freezes `self.vlm.parameters()`.
- `lerobot/policies/smolvla/modeling_smolvla.py`
  - `state_proj` is controlled separately by `train_state_proj`.
  - action path projections are `action_in_proj`, `action_out_proj`, `action_time_mlp_in`, and `action_time_mlp_out`.
- `lerobot/rlinf_smolvla_libero/training_loop.py`
  - Training metrics are returned from `update_sac()` but are not yet logged to WandB.

Implication:

The current implementation may avoid full VLM updates through `requires_grad=False`, but the training scope is not explicit or audited. The next implementation should make the trainable scope visible, testable, and logged.

## 4. Proposed Architecture

### 4.1 Trainable Scope Control

Add an explicit SAC-Flow actor trainable scope.

Initial supported value:

```text
action_path
```

Meaning:

```text
Train:
- vlm_with_expert.lm_expert
- action_in_proj
- action_out_proj
- action_time_mlp_in
- action_time_mlp_out

Freeze:
- VLM backbone
- vision encoder
- language backbone
- state_proj
```

The scope should be applied before optimizer construction.

The optimizer should receive only trainable parameters, not all `policy.parameters()`.

Why:

- Passing all parameters relies on `requires_grad` indirectly.
- Passing only trainable parameters makes optimizer state smaller and the training intent easier to audit.

### 4.2 Trainable Parameter Audit

At startup, report:

- total parameter count
- trainable parameter count
- trainable ratio
- trainable module groups
- first N trainable parameter names
- frozen high-level modules

The same summary should be logged into WandB config when WandB is enabled.

The audit is not cosmetic. It is the guardrail that prevents accidental full-model RL updates.

### 4.3 WandB Logger

Add a small WandB logging layer around the existing training loop.

Required config:

```text
wandb_enable: bool
wandb_project: str | None
wandb_run_name: str | None
wandb_mode: online | offline | disabled
wandb_tags: list[str]
```

Training scripts default to:

```text
wandb_enable=true
wandb_mode=online
```

Smoke and tests explicitly set:

```text
wandb_enable=false
```

Metrics to log:

```text
train/sac/critic_loss
train/sac/actor_loss
train/sac/alpha_loss
train/sac/alpha
train/actor/entropy
train/actor/log_pi
train/replay_buffer/size
env/reward
env/discounted_return
env/success
env/chunk_steps
train/global_step
```

The logger should tolerate missing actor metrics on steps where the actor is not updated because of `critic_actor_ratio`.

### 4.4 Training Scripts

Add scripts under a dedicated path such as:

```text
scripts/rlinf_smolvla_libero/
```

Scripts:

```text
gpu_smoke.sh
short_run_wandb.sh
baseline_run_wandb.sh
```

`gpu_smoke.sh`:

- reproduces task14 GPU smoke
- disables WandB
- uses safe small values

`short_run_wandb.sh`:

- defaults WandB to online
- supports manual `WANDB_PROJECT`
- auto-generates project/run name if not supplied
- starts with conservative short-run hyperparameters

`baseline_run_wandb.sh`:

- used only after short-run stability is confirmed
- increases steps and update count gradually

## 5. Initial Hyperparameters

### 5.1 GPU Smoke

```text
max_train_steps = 2
max_chunk_steps = 1
num_updates_per_step = 1
batch_size = 1
min_buffer_size = 1
replay_capacity = 16
wandb_enable = false
actor_train_scope = action_path
```

### 5.2 Short Run

```text
max_train_steps = 100
max_chunk_steps = 1
num_updates_per_step = 4
batch_size = 2
min_buffer_size = 2
replay_capacity = 64
critic_actor_ratio = 4
actor_lr = 1e-5
critic_lr = 3e-4
alpha_lr = 3e-4
gamma = 0.96
tau = 0.005
noise_std_train = 0.3
noise_std_rollout = 0.02
wandb_enable = true
actor_train_scope = action_path
```

### 5.3 Baseline Run

Start only after short-run metrics look stable.

Candidate progression:

```text
num_updates_per_step: 4 -> 8 -> 16 -> 32 -> 64
batch_size: 2 -> 4 -> 8
replay_capacity: 64 -> 200
max_train_steps: 100 -> 500+
```

Do not jump directly to 64 updates per step before confirming stability.

## 6. Acceptance Criteria

The implementation is acceptable when:

- startup audit shows VLM backbone frozen
- startup audit shows `state_proj` frozen by default
- startup audit shows action expert and action projections trainable
- optimizer is built only from trainable actor parameters
- WandB is enabled by default in training scripts
- WandB can be disabled for smoke/test runs
- short-run logs SAC losses, alpha, entropy, replay size, and environment reward/success
- GPU smoke still passes
- AutoDL short-run starts and logs to WandB without requiring code edits

## 7. Test Plan

Unit tests:

- trainable scope freezes `state_proj` for `action_path`
- trainable scope keeps action projections trainable
- trainable scope keeps action expert trainable
- trainable scope freezes VLM backbone
- optimizer receives only parameters with `requires_grad=True`
- WandB disabled mode does not import or initialize WandB
- WandB enabled mode logs expected metric keys through a fake logger

Server validation:

1. Run existing Task14 test suite.
2. Run `gpu_smoke.sh`.
3. Run `short_run_wandb.sh` with a manual `WANDB_PROJECT`.
4. Confirm WandB dashboard receives expected metrics.

## 8. Non-Goals

The first implementation will not:

- add LoRA training
- train a reward model
- update VLM backbone
- run full long training
- add multi-task LIBERO scheduling
- tune hyperparameters beyond the conservative short-run defaults

## 9. Open Follow-Up Experiments

After the action-path baseline is verified:

- compare `action_path` vs `action_path_state_proj`
- compare full action-path update vs action-path LoRA
- increase `num_updates_per_step`
- increase `max_chunk_steps`
- evaluate multi-task LIBERO training
- consider VLM LoRA only if action-path-only learning is insufficient
