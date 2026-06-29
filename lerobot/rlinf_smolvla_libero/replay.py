from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Any


def discounted_chunk_reward(rewards: list[float], *, gamma: float) -> float:
    if rewards is None:
        raise ValueError("rewards must be a sequence.")
    try:
        reward_items = list(rewards)
    except TypeError as exc:
        raise ValueError("rewards must be a sequence.") from exc

    total = 0.0
    discount = 1.0
    for reward in reward_items:
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

    shape_error = "Expected action chunk shape [batch, chunk, action_dim]."
    try:
        batch_items = list(action_chunk)
    except TypeError as exc:
        raise ValueError(shape_error) from exc
    if not batch_items:
        raise ValueError(shape_error)

    expected_chunk_len = None
    expected_action_dim = None
    flattened = []
    for batch_item in batch_items:
        try:
            actions = list(batch_item)
        except TypeError as exc:
            raise ValueError(shape_error) from exc
        if not actions:
            raise ValueError(shape_error)
        if expected_chunk_len is None:
            expected_chunk_len = len(actions)
        elif len(actions) != expected_chunk_len:
            raise ValueError(shape_error)

        one = []
        for action in actions:
            try:
                values = list(action)
            except TypeError as exc:
                raise ValueError(shape_error) from exc
            if expected_action_dim is None:
                expected_action_dim = len(values)
            elif len(values) != expected_action_dim:
                raise ValueError(shape_error)
            one.extend(float(value) for value in values)
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


class ChunkReplayBuffer:
    def __init__(self, *, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items: deque[ChunkTransition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: ChunkTransition) -> None:
        if transition.horizon <= 0:
            raise ValueError("transition horizon must be positive")
        self._items.append(transition)

    def sample(self, *, batch_size: int, device: Any) -> dict[str, Any]:
        if not self._items:
            raise RuntimeError("empty ChunkReplayBuffer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        import torch

        sample_size = min(batch_size, len(self._items))
        transitions = self._rng.sample(list(self._items), sample_size)

        for transition in transitions:
            if getattr(transition.actions, "ndim", None) != 2:
                raise ValueError(
                    "transition.actions must have shape [1, action_dim] or [batch, action_dim]"
                )

        return {
            "curr_obs": self._cat_obs([transition.curr_obs for transition in transitions], device=device),
            "next_obs": self._cat_obs([transition.next_obs for transition in transitions], device=device),
            "actions": torch.cat([transition.actions.to(device) for transition in transitions], dim=0),
            "rewards": torch.cat(
                [torch.tensor([[transition.chunk_reward]], dtype=torch.float32, device=device) for transition in transitions],
                dim=0,
            ),
            "terminations": torch.cat(
                [torch.tensor([[transition.done]], dtype=torch.bool, device=device) for transition in transitions],
                dim=0,
            ),
            "discounts": torch.cat(
                [torch.tensor([[transition.discount]], dtype=torch.float32, device=device) for transition in transitions],
                dim=0,
            ),
            "horizons": torch.cat(
                [torch.tensor([[transition.horizon]], dtype=torch.long, device=device) for transition in transitions],
                dim=0,
            ),
        }

    @staticmethod
    def _cat_obs(observations: list[dict[str, Any]], *, device: Any) -> dict[str, Any]:
        import torch

        expected_keys = set(observations[0].keys())
        for observation in observations[1:]:
            if set(observation.keys()) != expected_keys:
                raise ValueError("observation keys must match across sampled transitions")

        return {
            key: torch.cat([observation[key].to(device) for observation in observations], dim=0)
            for key in observations[0].keys()
        }
