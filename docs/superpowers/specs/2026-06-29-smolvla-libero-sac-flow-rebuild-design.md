# SmolVLA LIBERO SAC-Flow Rebuild Design

## Goal

Build a clean SAC-Flow post-training path that mirrors RLinf's embodied SAC implementation while replacing only the actor with SmolVLA and the task suite with LIBERO.

## Non-goals

- Do not reuse existing modified SmolVLA RL/DORF training scripts in this repository.
- Do not add DORF reward models, GRPO, weighted flow-matching, IQL/CQL, dense reward shaping, or new algorithmic tricks.
- Do not optimize for final performance before the minimal SAC-Flow loop passes smoke tests.

## Primary references checked

- RLinf `examples/embodiment/config/maniskill_sac_flow_state.yaml`: SAC hyperparameters, replay settings, entropy tuning, rollout/update ratio.
- RLinf `examples/embodiment/config/realworld_sac_flow_image.yaml`: image SAC-Flow settings, fixed train/rollout noise, real-world replay sizing.
- RLinf `rlinf/workers/actor/fsdp_sac_policy_worker.py`: critic target, actor loss, alpha loss, target network update.
- RLinf `rlinf/models/embodiment/modules/flow_actor.py`: JaxFlowTActor log-prob accumulation across denoising steps.
- RLinf `examples/embodiment/config/env/libero_10.yaml`: LIBERO environment settings and sparse/success reward convention.

## Core design decisions

### D1: Rebuild a new isolated path

Create a new implementation under `lerobot/rlinf_smolvla_libero/` and one new script `lerobot/scripts/rlinf_smolvla_libero_sac_flow_train.py`. Existing modified RL scripts are not imported by this path.

### D2: Use SmolVLA as the SAC-Flow actor

The adapter must expose the same conceptual calls used by RLinf SAC:

```text
sample_action_with_logprob(obs, train) -> action_variable, log_pi, obs_feature
q_values(obs, action_variable) -> [batch, num_q_heads]
target_q_values(obs, action_variable) -> [batch, num_q_heads]
```

### D3: Treat a SmolVLA action chunk as the SAC action variable

RLinf's example `flow_policy` currently uses `num_action_chunks: 1`, but SmolVLA is natively an action-chunk flow policy. Forcing only the first action would make the log-prob/action pair inconsistent with the sampled flow output. Therefore the first runnable SmolVLA version stores and trains on the flattened chunk as `A`, executes the chunk in LIBERO, and bootstraps from the next observation after the executed chunk.

The SAC target is:

```text
target_q = chunk_reward + (1 - done) * gamma^h * (min target_q(next_obs, next_A) - alpha * log_pi(next_A | next_obs))
```

where `h` is the number of actually executed environment steps in the chunk.

### D4: Mirror RLinf SAC losses

Critic loss:

```text
MSE(Q_i(obs, A), target_q) over all Q heads
```

Actor loss:

```text
(alpha * log_pi(A | obs) - aggregate_q(Q(obs, A))).mean()
```

Alpha loss:

```text
-alpha * (log_pi(A | obs).detach().mean() + target_entropy)
```

Use RLinf defaults unless the SmolVLA chunk dimension forces a shape change:

```text
gamma = 0.96 for LIBERO image tasks
tau = 0.005
initial_alpha = 0.01
critic_actor_ratio = 4
num_updates_per_step = 64
noise_std_train = 0.3
noise_std_rollout = 0.02
num_q_heads = 10 for image/LIBERO-style runs
```

### D5: LIBERO adapter follows RLinf LIBERO config semantics

Start with one LIBERO suite, preferably `libero_spatial` or `libero_10` depending on local assets. Use success/sparse reward first:

```text
reward = reward_coef * env_reward_or_success
max_episode_steps = 512
use_fixed_reset_state_ids = true for evaluation
```

### D6: First success criterion is smoke correctness

The first milestone is not performance improvement. It is:

```text
reset -> SmolVLA chunk sample -> LIBERO step loop -> replay add -> critic update -> actor update -> alpha update -> eval success logging
```

Every scalar loss/log-prob/Q value must be finite.

## Data flow

```text
LIBERO obs + task text
  -> SmolVLA processor
  -> SmolVLA flow decoder samples action chunk and log_pi
  -> execute chunk in LIBERO
  -> collect chunk reward, done, next obs
  -> replay buffer stores chunk transition
  -> SAC critic/actor/alpha updates sample from replay
  -> target critic soft update
```

## Files to create

```text
lerobot/rlinf_smolvla_libero/config.py
lerobot/rlinf_smolvla_libero/replay.py
lerobot/rlinf_smolvla_libero/critic.py
lerobot/rlinf_smolvla_libero/actor_adapter.py
lerobot/rlinf_smolvla_libero/libero_adapter.py
lerobot/rlinf_smolvla_libero/trainer.py
lerobot/scripts/rlinf_smolvla_libero_sac_flow_train.py
tests/test_rlinf_smolvla_sac_flow_math.py
tests/test_rlinf_smolvla_sac_flow_replay.py
tests/test_rlinf_smolvla_sac_flow_critic.py
```

## Review notes

- The design intentionally avoids importing old failed training scripts.
- The only necessary SmolVLA-specific adaptation is chunk-level action handling.
- If chunk-level log-prob cannot be made finite and differentiable, the project should stop and inspect SmolVLA's flow sampler rather than silently switching to first-action training.
