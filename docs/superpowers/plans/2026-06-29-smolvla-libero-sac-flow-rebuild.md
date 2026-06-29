# SmolVLA LIBERO SAC-Flow Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean RLinf-style SAC-Flow training path for SmolVLA on LIBERO without importing the repository's previous failed RL training scripts.

**Architecture:** The new path is isolated under `lerobot/rlinf_smolvla_libero/`. It mirrors RLinf SAC: rollout produces replay transitions, critic learns bootstrapped Q targets, actor maximizes Q minus entropy, alpha is tuned automatically, and target critic is softly updated. SmolVLA is wrapped by an adapter that samples flow action chunks with log-prob and exposes Q-compatible observation features.

**Tech Stack:** Python, PyTorch, LeRobot SmolVLA policy/processor/env factories, LIBERO envs, unittest/pytest.

---

## File structure

- Create `lerobot/rlinf_smolvla_libero/__init__.py`: package marker.
- Create `lerobot/rlinf_smolvla_libero/config.py`: dataclass config matching RLinf SAC fields.
- Create `lerobot/rlinf_smolvla_libero/replay.py`: chunk transition object and replay buffer.
- Create `lerobot/rlinf_smolvla_libero/critic.py`: RLinf-style multi-Q head, alpha temperature, soft update.
- Create `lerobot/rlinf_smolvla_libero/actor_adapter.py`: SmolVLA SAC-Flow adapter.
- Create `lerobot/rlinf_smolvla_libero/libero_adapter.py`: LIBERO rollout adapter.
- Create `lerobot/rlinf_smolvla_libero/trainer.py`: SAC update loop and smoke metrics.
- Create `lerobot/scripts/rlinf_smolvla_libero_sac_flow_train.py`: parser entry point.
- Create `tests/test_rlinf_smolvla_sac_flow_math.py`: reward/discount/log-prob shape tests.
- Create `tests/test_rlinf_smolvla_sac_flow_replay.py`: replay storage and sampling tests.
- Create `tests/test_rlinf_smolvla_sac_flow_critic.py`: Q head/target/actor loss tests with dummy tensors.

---

### Task 1: SAC math helpers

**Files:**
- Create: `C:\Users\w1985\.codex\worktrees\aecd\DORF-VLA\lerobot\rlinf_smolvla_libero\__init__.py`
- Create: `C:\Users\w1985\.codex\worktrees\aecd\DORF-VLA\lerobot\rlinf_smolvla_libero\replay.py`
- Test: `C:\Users\w1985\.codex\worktrees\aecd\DORF-VLA\tests\test_rlinf_smolvla_sac_flow_math.py`

- [ ] **Step 1: Write the failing math tests**

```python
import unittest

from lerobot.rlinf_smolvla_libero.replay import chunk_discount, discounted_chunk_reward, flatten_chunk


class SACFlowMathTest(unittest.TestCase):
    def test_discounted_chunk_reward_uses_stepwise_gamma(self):
        self.assertAlmostEqual(discounted_chunk_reward([1.0, 2.0, 3.0], gamma=0.5), 2.75)

    def test_chunk_discount_uses_actual_executed_horizon(self):
        self.assertAlmostEqual(chunk_discount(horizon=3, gamma=0.5), 0.125)

    def test_flatten_chunk_preserves_batch_dimension(self):
        flat = flatten_chunk([[[1.0, 2.0], [3.0, 4.0]]])
        self.assertEqual(flat, [[1.0, 2.0, 3.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rlinf_smolvla_sac_flow_math.py -q`

Expected: FAIL because `lerobot.rlinf_smolvla_libero.replay` does not exist.

- [ ] **Step 3: Implement the helpers**

Create `__init__.py` as an empty UTF-8 file.

Create `replay.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def discounted_chunk_reward(rewards: list[float], *, gamma: float) -> float:
    total = 0.0
    discount = 1.0
    for reward in rewards:
        total += discount * float(reward)
        discount *= gamma
    return float(total)


def chunk_discount(*, horizon: int, gamma: float) -> float:
    if horizon < 0:
        raise ValueError(f"horizon must be non-negative, got {horizon}.")
    return float(gamma**horizon)


def flatten_chunk(action_chunk: Any) -> Any:
    if action_chunk.__class__.__module__.startswith("torch") and action_chunk.__class__.__name__ == "Tensor":
        if action_chunk.ndim != 3:
            raise ValueError(f"Expected action chunk shape [batch, chunk, action_dim], got {tuple(action_chunk.shape)}.")
        return action_chunk.reshape(action_chunk.shape[0], -1)
    flattened = []
    for batch_item in action_chunk:
        one = []
        for action in batch_item:
            one.extend(float(value) for value in action)
        flattened.append(one)
    return flattened


@dataclass
class ChunkTransition:
    curr_obs: dict[str, Any]
    actions: Any
    next_obs: dict[str, Any]
    rewards: list[float]
    done: bool
    horizon: int
    discount: float
    chunk_reward: float
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rlinf_smolvla_sac_flow_math.py -q`

Expected: PASS.

---

### Task 2: Replay buffer

**Files:**
- Modify: `C:\Users\w1985\.codex\worktrees\aecd\DORF-VLA\lerobot\rlinf_smolvla_libero\replay.py`
- Test: `C:\Users\w1985\.codex\worktrees\aecd\DORF-VLA\tests\test_rlinf_smolvla_sac_flow_replay.py`

- [ ] **Step 1: Write the failing replay tests**

```python
import unittest

import torch

from lerobot.rlinf_smolvla_libero.replay import ChunkReplayBuffer, ChunkTransition


class ChunkReplayBufferTest(unittest.TestCase):
    def make_transition(self, value: float) -> ChunkTransition:
        obs = {"states": torch.tensor([[value, value + 1]], dtype=torch.float32)}
        next_obs = {"states": torch.tensor([[value + 2, value + 3]], dtype=torch.float32)}
        return ChunkTransition(
            curr_obs=obs,
            actions=torch.tensor([[value, value + 0.5]], dtype=torch.float32),
            next_obs=next_obs,
            rewards=[1.0],
            done=False,
            horizon=1,
            discount=0.96,
            chunk_reward=1.0,
        )

    def test_sample_returns_batched_transition_tensors(self):
        buffer = ChunkReplayBuffer(capacity=4, seed=123)
        buffer.add(self.make_transition(1.0))
        buffer.add(self.make_transition(2.0))
        batch = buffer.sample(batch_size=2, device=torch.device("cpu"))
        self.assertEqual(batch["actions"].shape, (2, 2))
        self.assertEqual(batch["curr_obs"]["states"].shape, (2, 2))
        self.assertEqual(batch["next_obs"]["states"].shape, (2, 2))
        self.assertEqual(batch["rewards"].shape, (2, 1))
        self.assertEqual(batch["terminations"].shape, (2, 1))
        self.assertEqual(batch["discounts"].shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rlinf_smolvla_sac_flow_replay.py -q`

Expected: FAIL because `ChunkReplayBuffer` is missing.

- [ ] **Step 3: Implement minimal replay buffer**

Append to `replay.py`:

```python
import random
from collections import deque


class ChunkReplayBuffer:
    def __init__(self, *, capacity: int, seed: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        self._storage = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, transition: ChunkTransition) -> None:
        if transition.horizon <= 0:
            raise ValueError(f"transition horizon must be positive, got {transition.horizon}.")
        self._storage.append(transition)

    def sample(self, *, batch_size: int, device):
        if not self._storage:
            raise RuntimeError("Cannot sample from an empty ChunkReplayBuffer.")
        import torch

        sampled = self._rng.sample(list(self._storage), k=min(batch_size, len(self._storage)))
        obs_keys = sampled[0].curr_obs.keys()
        curr_obs = {key: torch.cat([item.curr_obs[key] for item in sampled], dim=0).to(device) for key in obs_keys}
        next_obs = {key: torch.cat([item.next_obs[key] for item in sampled], dim=0).to(device) for key in obs_keys}
        return {
            "curr_obs": curr_obs,
            "next_obs": next_obs,
            "actions": torch.cat([item.actions for item in sampled], dim=0).to(device),
            "rewards": torch.tensor([[item.chunk_reward] for item in sampled], dtype=torch.float32, device=device),
            "terminations": torch.tensor([[item.done] for item in sampled], dtype=torch.bool, device=device),
            "discounts": torch.tensor([[item.discount] for item in sampled], dtype=torch.float32, device=device),
            "horizons": torch.tensor([[item.horizon] for item in sampled], dtype=torch.long, device=device),
        }
```

- [ ] **Step 4: Run replay tests**

Run: `python -m pytest tests/test_rlinf_smolvla_sac_flow_replay.py -q`

Expected: PASS.

---

### Task 3: Critic, alpha, and SAC target math

**Files:**
- Create: `C:\Users\w1985\.codex\worktrees\aecd\DORF-VLA\lerobot\rlinf_smolvla_libero\critic.py`
- Test: `C:\Users\w1985\.codex\worktrees\aecd\DORF-VLA\tests\test_rlinf_smolvla_sac_flow_critic.py`

- [ ] **Step 1: Write the failing critic tests**

```python
import unittest

import torch

from lerobot.rlinf_smolvla_libero.critic import EntropyTemperature, MultiQHead, actor_loss, critic_target


class SACFlowCriticTest(unittest.TestCase):
    def test_q_head_outputs_one_value_per_head(self):
        q = MultiQHead(obs_dim=5, action_dim=3, hidden_dim=16, num_q_heads=4)
        values = q(torch.zeros(2, 5), torch.zeros(2, 3))
        self.assertEqual(values.shape, (2, 4))

    def test_critic_target_uses_discount_and_entropy_backup(self):
        reward = torch.tensor([[1.0]])
        done = torch.tensor([[False]])
        discount = torch.tensor([[0.5]])
        next_q = torch.tensor([[2.0, 3.0]])
        next_log_pi = torch.tensor([[-4.0]])
        alpha = torch.tensor(0.25)
        target = critic_target(reward, done, discount, next_q, next_log_pi, alpha, agg="min", backup_entropy=True)
        self.assertAlmostEqual(float(target.item()), 1.0 + 0.5 * (2.0 - 0.25 * -4.0))

    def test_actor_loss_matches_sac_formula(self):
        q_pi = torch.tensor([[2.0, 3.0]])
        log_pi = torch.tensor([[-1.5]])
        loss = actor_loss(q_pi, log_pi, torch.tensor(0.2), agg="min")
        self.assertAlmostEqual(float(loss.item()), 0.2 * -1.5 - 2.0)

    def test_alpha_is_positive(self):
        temp = EntropyTemperature(initial_alpha=0.01)
        self.assertGreater(float(temp.alpha.item()), 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rlinf_smolvla_sac_flow_critic.py -q`

Expected: FAIL because `critic.py` does not exist.

- [ ] **Step 3: Implement critic.py**

Create `critic.py` with:

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQHead(nn.Module):
    def __init__(self, *, obs_dim: int, action_dim: int, hidden_dim: int, num_q_heads: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(obs_dim + action_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_q_heads)
            ]
        )

    def forward(self, obs_features: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs_features, actions], dim=-1)
        return torch.cat([head(x) for head in self.heads], dim=-1)


class EntropyTemperature(nn.Module):
    def __init__(self, *, initial_alpha: float):
        super().__init__()
        if initial_alpha <= 0:
            raise ValueError(f"initial_alpha must be positive, got {initial_alpha}.")
        self.log_alpha = nn.Parameter(torch.log(torch.tensor(float(initial_alpha), dtype=torch.float32)))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()


def aggregate_q(q_values: torch.Tensor, *, agg: str) -> torch.Tensor:
    if agg == "min":
        return q_values.min(dim=-1, keepdim=True).values
    if agg == "mean":
        return q_values.mean(dim=-1, keepdim=True)
    raise ValueError(f"Unsupported Q aggregation: {agg}")


def critic_target(reward, done, discount, next_q, next_log_pi, alpha, *, agg: str, backup_entropy: bool):
    next_value = aggregate_q(next_q, agg=agg)
    if backup_entropy:
        next_value = next_value - alpha * next_log_pi
    not_done = (~done).to(dtype=reward.dtype)
    return reward + not_done * discount * next_value


def critic_loss(q_data, target):
    return F.mse_loss(q_data, target.expand_as(q_data))


def actor_loss(q_pi, log_pi, alpha, *, agg: str):
    return (alpha * log_pi - aggregate_q(q_pi, agg=agg)).mean()


def alpha_loss(alpha: torch.Tensor, log_pi: torch.Tensor, *, target_entropy: float):
    return -alpha * (log_pi.detach().mean() + target_entropy)


def soft_update(source: nn.Module, target: nn.Module, *, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters(), strict=True):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)
```

- [ ] **Step 4: Run critic tests**

Run: `python -m pytest tests/test_rlinf_smolvla_sac_flow_critic.py -q`

Expected: PASS.

---

### Task 4: SmolVLA actor adapter interface

**Files:**
- Create: `C:\Users\w1985\.codex\worktrees\aecd\DORF-VLA\lerobot\rlinf_smolvla_libero\actor_adapter.py`
- Modify: `C:\Users\w1985\.codex\worktrees\aecd\DORF-VLA\lerobot\policies\smolvla\modeling_smolvla.py` only if the upstream SmolVLA sampler lacks a differentiable `sample_action_chunk_with_logprob` method.

- [ ] **Step 1: Add adapter shell with explicit runtime errors**

Create `actor_adapter.py` with a `SmolVLASACFlowActor` class that accepts a SmolVLA policy, preprocessor, postprocessor, device, train noise, and rollout noise. The constructor verifies that the policy can provide a chunk sampler with log-prob; if not, it raises `AttributeError` naming the missing method.

- [ ] **Step 2: Implement `sample_chunk`**

`sample_chunk(obs, train)` must return:

```text
actions: torch.Tensor [batch, chunk_size * action_dim]
log_pi: torch.Tensor [batch, 1]
obs_features: torch.Tensor [batch, hidden_dim]
raw_chunk: torch.Tensor [batch, chunk_size, action_dim]
```

- [ ] **Step 3: Implement `encode_obs`**

`encode_obs(obs)` must return the same hidden feature dimensionality as `sample_chunk(...).obs_features`.

- [ ] **Step 4: Add a dummy-policy unit test before using real SmolVLA**

The test defines a fake policy with deterministic tensors and verifies adapter output shapes. This test must not instantiate SmolVLA or LIBERO.

Run: `python -m pytest tests/test_rlinf_smolvla_sac_flow_actor_adapter.py -q`

Expected: PASS.

---

### Task 5: LIBERO rollout adapter

**Files:**
- Create: `C:\Users\w1985\.codex\worktrees\aecd\DORF-VLA\lerobot\rlinf_smolvla_libero\libero_adapter.py`

- [ ] **Step 1: Implement `execute_action_chunk`**

The function accepts one vector env, one policy-space chunk, postprocessors, `gamma`, and `max_chunk_steps`. It executes chunk actions sequentially, stops on termination/truncation, and returns a `ChunkTransition` with actual `horizon`.

- [ ] **Step 2: Preserve LIBERO task text**

The adapter must pass the task/language field through the same observation path used by SmolVLA baseline inference.

- [ ] **Step 3: Add smoke-only logging**

Log per rollout:

```text
rollout/return
rollout/success
rollout/horizon
rollout/chunks
```

---

### Task 6: SAC trainer

**Files:**
- Create: `C:\Users\w1985\.codex\worktrees\aecd\DORF-VLA\lerobot\rlinf_smolvla_libero\config.py`
- Create: `C:\Users\w1985\.codex\worktrees\aecd\DORF-VLA\lerobot\rlinf_smolvla_libero\trainer.py`

- [ ] **Step 1: Implement config dataclass**

Use these first-run defaults:

```python
SACFlowConfig(
    gamma=0.96,
    tau=0.005,
    initial_alpha=0.01,
    target_entropy=None,
    critic_actor_ratio=4,
    num_updates_per_step=64,
    replay_capacity=200,
    min_buffer_size=2,
    batch_size=8,
    num_q_heads=10,
    hidden_dim=256,
    noise_std_train=0.3,
    noise_std_rollout=0.02,
    backup_entropy=True,
    agg_q="min",
    actor_agg_q="mean",
)
```

If `target_entropy` is unset, set it to `-flattened_action_dim` after the actor adapter reports its action dimension.

- [ ] **Step 2: Implement one SAC update function**

`update_sac(batch)` performs critic update every call, actor/alpha update every `critic_actor_ratio`, and target soft update every call.

- [ ] **Step 3: Implement trainer loop**

Loop order:

```text
collect rollout chunk transitions
add to replay
if replay size >= min_buffer_size: run num_updates_per_step updates
if eval interval reached: run LIBERO eval
if save interval reached: save policy, critic, target critic, alpha, config
```

---

### Task 7: Entry script and smoke command

**Files:**
- Create: `C:\Users\w1985\.codex\worktrees\aecd\DORF-VLA\lerobot\scripts\rlinf_smolvla_libero_sac_flow_train.py`

- [ ] **Step 1: Implement script entry**

The script should parse the normal LeRobot `TrainPipelineConfig`, create dataset metadata, instantiate SmolVLA, create LIBERO envs, then instantiate `SACFlowTrainer`.

- [ ] **Step 2: Add entry-script usage guard to docstring**

The docstring must state that a real training run requires `LEROBOT_LIBERO_ROOT` and the same SmolVLA checkpoint/config overrides used for baseline SmolVLA evaluation. The script must fail before environment creation with this exact message when the variable is absent:

```text
LEROBOT_LIBERO_ROOT is required for LIBERO SAC-Flow smoke runs.
```

- [ ] **Step 3: Run import smoke**

Run: `python -m py_compile lerobot/scripts/rlinf_smolvla_libero_sac_flow_train.py`

Expected: command exits with code 0.

---

## Self-review

- Spec coverage: Tasks 1-3 cover SAC math/replay/critic; Task 4 covers SmolVLA actor replacement; Task 5 covers LIBERO replacement; Task 6 covers RLinf SAC update loop; Task 7 covers runnable entry.
- Empty-field scan: no待填内容 remains; local runtime paths are handled by script guards such as `LEROBOT_LIBERO_ROOT` instead of hard-coded examples.
- Type consistency: replay batch names match RLinf's `curr_obs`, `next_obs`, `actions`, `rewards`, and `terminations`; critic helpers expect `[batch, num_q_heads]`; actor adapter emits flattened chunk actions for the critic and raw chunks for environment execution.


