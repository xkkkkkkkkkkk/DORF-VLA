#!/usr/bin/env python

"""Utilities for SmolVLA SAC-flow post-training.

This module keeps scalar/list helpers importable without torch so that basic
behavior can still be unit-tested on lightweight hosts. Tensor-heavy helpers
import torch lazily inside the relevant call sites.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and value.__class__.__name__ == "Tensor"


def flatten_action_chunk(action_chunk: Any) -> Any:
    """Flatten [B, T, A] action chunks into [B, T*A].

    Supports nested Python lists for tests and torch tensors at runtime.
    """

    if _is_torch_tensor(action_chunk):
        if action_chunk.ndim != 3:
            raise ValueError(
                f"Expected action chunk tensor with shape [batch, chunk, action_dim], got {tuple(action_chunk.shape)}."
            )
        return action_chunk.reshape(action_chunk.shape[0], -1)

    flat_chunks: list[list[float]] = []
    for batch_item in action_chunk:
        flat_item: list[float] = []
        for step in batch_item:
            flat_item.extend(step)
        flat_chunks.append(flat_item)
    return flat_chunks


def unflatten_action_chunk(flat_action: Any, *, chunk_size: int, action_dim: int) -> Any:
    """Inverse of :func:`flatten_action_chunk`."""

    if _is_torch_tensor(flat_action):
        return flat_action.reshape(flat_action.shape[0], chunk_size, action_dim)

    restored: list[list[list[float]]] = []
    for batch_item in flat_action:
        if len(batch_item) != chunk_size * action_dim:
            raise ValueError(
                f"Expected flattened action length {chunk_size * action_dim}, got {len(batch_item)}."
            )
        restored.append(
            [batch_item[start : start + action_dim] for start in range(0, len(batch_item), action_dim)]
        )
    return restored


def chunk_reward_sum(rewards: list[float], *, gamma: float) -> float:
    """Compute discounted SMDP reward over one action chunk decision."""

    discounted = 0.0
    running_discount = 1.0
    for reward in rewards:
        discounted += running_discount * float(reward)
        running_discount *= gamma
    return discounted


def chunk_discount_factor(*, num_steps: int, gamma: float) -> float:
    """Return the bootstrap discount for a chunk covering ``num_steps`` env steps."""

    if num_steps < 0:
        raise ValueError(f"num_steps must be non-negative, got {num_steps}.")
    return float(gamma**num_steps)


@dataclass
class ChunkDecisionStats:
    """Track one chunk-level decision while actions from the queue are executed."""

    horizon: int = 0
    raw_reward_sum: float = 0.0
    terminal: bool = False

    def add_step(self, *, reward: float, terminal: bool = False) -> None:
        self.horizon += 1
        self.raw_reward_sum += float(reward)
        self.terminal = self.terminal or terminal


def _lazy_torch():
    import torch
    import torch.nn as nn

    return torch, nn


class MLPQEnsemble:  # pragma: no cover - tensor runtime only
    """Small Q ensemble used by the SmolVLA SAC-flow rewrite.

    Constructed lazily so importing this module does not require torch.
    """

    def __new__(cls, *args, **kwargs):
        torch, nn = _lazy_torch()

        class _MLPQEnsemble(nn.Module):
            def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, num_q_heads: int):
                super().__init__()
                self.obs_dim = obs_dim
                self.action_dim = action_dim
                self.hidden_dim = hidden_dim
                self.num_q_heads = num_q_heads
                self.q_heads = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(obs_dim + action_dim, hidden_dim),
                            nn.LayerNorm(hidden_dim),
                            nn.SiLU(),
                            nn.Linear(hidden_dim, hidden_dim),
                            nn.SiLU(),
                            nn.Linear(hidden_dim, 1),
                        )
                        for _ in range(num_q_heads)
                    ]
                )

            def forward(self, obs_features, flat_actions):
                if obs_features.ndim != 2:
                    raise ValueError(
                        f"Expected obs_features shape [batch, obs_dim], got {tuple(obs_features.shape)}."
                    )
                if flat_actions.ndim != 2:
                    raise ValueError(
                        f"Expected flat_actions shape [batch, action_dim], got {tuple(flat_actions.shape)}."
                    )
                if obs_features.shape[0] != flat_actions.shape[0]:
                    raise ValueError(
                        "obs_features and flat_actions batch sizes must match, "
                        f"got {obs_features.shape[0]} and {flat_actions.shape[0]}."
                    )
                if obs_features.shape[1] != self.obs_dim:
                    raise ValueError(
                        f"Expected obs_features hidden dim {self.obs_dim}, got {obs_features.shape[1]}."
                    )
                if flat_actions.shape[1] != self.action_dim:
                    raise ValueError(
                        f"Expected flat action dim {self.action_dim}, got {flat_actions.shape[1]}."
                    )
                critic_input = torch.cat([obs_features, flat_actions], dim=-1)
                q_values = [head(critic_input).squeeze(-1) for head in self.q_heads]
                return torch.stack(q_values, dim=0)

        return _MLPQEnsemble(*args, **kwargs)


class EntropyTemperature:  # pragma: no cover - tensor runtime only
    """Minimal learnable entropy temperature."""

    def __new__(cls, *args, **kwargs):
        torch, nn = _lazy_torch()

        class _EntropyTemperature(nn.Module):
            def __init__(self, initial_alpha: float):
                super().__init__()
                if initial_alpha <= 0:
                    raise ValueError(f"initial_alpha must be positive, got {initial_alpha}.")
                self.log_alpha = nn.Parameter(torch.log(torch.tensor(float(initial_alpha), dtype=torch.float32)))

            @property
            def alpha(self):
                return self.log_alpha.exp()

        return _EntropyTemperature(*args, **kwargs)


def soft_update_module(source_module, target_module, *, tau: float) -> None:
    """Polyak-average ``source_module`` into ``target_module``."""

    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau}.")

    with __import__("contextlib").nullcontext():
        pass

    torch, _ = _lazy_torch()
    with torch.no_grad():
        for source_param, target_param in zip(source_module.parameters(), target_module.parameters(), strict=True):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(source_param.data, alpha=tau)


def min_q_value(q_values):
    """Reduce [num_q_heads, batch] Q-values with the SAC min operator."""

    if _is_torch_tensor(q_values):
        return q_values.min(dim=0).values
    return [min(values) for values in zip(*q_values, strict=True)]


def freeze_module(module) -> None:  # pragma: no cover - tensor runtime only
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def unfreeze_module(module) -> None:  # pragma: no cover - tensor runtime only
    for parameter in module.parameters():
        parameter.requires_grad_(True)


class TypedReplayBuffer:  # pragma: no cover - tensor runtime only
    """Small replay buffer that preserves tensor dtypes for VLM observations."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        self.capacity = capacity
        self._storage = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, transition: dict[str, Any]) -> None:
        required_keys = {
            "state",
            "next_state",
            "action",
            "reward",
            "done",
            "truncated",
            "discount",
            "horizon",
            "raw_reward_sum",
        }
        missing = sorted(required_keys.difference(transition))
        if missing:
            raise KeyError(f"Replay transition is missing required keys: {missing}")
        if not _is_torch_tensor(transition["action"]):
            raise TypeError("Replay transition 'action' must be a torch tensor.")
        if transition["action"].ndim != 2:
            raise ValueError(
                f"Replay transition 'action' must have shape [1, flat_action_dim], got {tuple(transition['action'].shape)}."
            )
        self._storage.append(transition)

    def sample(self, batch_size: int, device) -> dict[str, Any]:
        if not self._storage:
            raise RuntimeError("Cannot sample from an empty replay buffer.")

        torch, _ = _lazy_torch()
        sampled = random.sample(list(self._storage), k=min(batch_size, len(self._storage)))
        state_keys = sampled[0]["state"].keys()
        batch_state = {
            key: torch.cat([transition["state"][key] for transition in sampled], dim=0).to(device)
            for key in state_keys
        }
        batch_next_state = {
            key: torch.cat([transition["next_state"][key] for transition in sampled], dim=0).to(device)
            for key in state_keys
        }
        batch_action = torch.cat([transition["action"] for transition in sampled], dim=0).to(device)
        if batch_action.ndim != 2:
            raise RuntimeError(
                f"Expected sampled replay actions with shape [batch, flat_action_dim], got {tuple(batch_action.shape)}."
            )
        batch_reward = torch.tensor([transition["reward"] for transition in sampled], dtype=torch.float32, device=device)
        batch_done = torch.tensor([transition["done"] for transition in sampled], dtype=torch.float32, device=device)
        batch_truncated = torch.tensor(
            [transition["truncated"] for transition in sampled], dtype=torch.float32, device=device
        )
        batch_discount = torch.tensor(
            [transition["discount"] for transition in sampled], dtype=torch.float32, device=device
        )
        batch_horizon = torch.tensor([transition["horizon"] for transition in sampled], dtype=torch.long, device=device)
        batch_raw_reward = torch.tensor(
            [transition["raw_reward_sum"] for transition in sampled], dtype=torch.float32, device=device
        )

        return {
            "state": batch_state,
            "next_state": batch_next_state,
            "action": batch_action,
            "reward": batch_reward,
            "done": batch_done,
            "truncated": batch_truncated,
            "discount": batch_discount,
            "horizon": batch_horizon,
            "raw_reward_sum": batch_raw_reward,
        }
