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
    # These fields are deliberately separate from ``chunk_reward``.  In
    # LIBERO the sparse positive reward is usually observed only on the
    # terminal step, while ``episode_success`` comes from the environment
    # info and is the ground-truth episode outcome.
    episode_success: bool | None = None
    episode_completed: bool | None = None
    truncated: bool | None = None
    episode_return_to_go: float | None = None
    intervention_applied: bool | None = None
    intervention_noise_l2: float | None = None
    intervention_noise_std: float | None = None
    intervention_task_index: int | None = None
    intervention_slot_index: int | None = None
    pair_id: str | None = None
    pair_branch: str | None = None
    pair_anchor: bool = False
    episode_step_index: int | None = None
    episode_length: int | None = None
    raw_rewards: list[float] | None = None
    raw_chunk_reward: float | None = None


class ChunkReplayBuffer:
    def __init__(self, *, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._items: deque[ChunkTransition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def items(self) -> tuple[ChunkTransition, ...]:
        """Return an immutable snapshot for diagnostics, never for training sampling."""
        return tuple(self._items)

    def positive_reward_fraction(self) -> float:
        if not self._items:
            return 0.0
        return sum(transition.chunk_reward > 0 for transition in self._items) / len(self._items)

    def positive_outcome_fraction(self) -> float:
        if not self._items:
            return 0.0
        return sum(_is_positive_outcome(transition) for transition in self._items) / len(self._items)

    def intervention_fraction(self) -> float:
        labeled = [
            transition
            for transition in self._items
            if getattr(transition, "intervention_applied", None) is not None
        ]
        if not labeled:
            return 0.0
        return sum(bool(transition.intervention_applied) for transition in labeled) / len(labeled)

    def intervention_labeled_fraction(self) -> float:
        if not self._items:
            return 0.0
        return sum(
            getattr(transition, "intervention_applied", None) is not None
            for transition in self._items
        ) / len(self._items)

    def add(self, transition: ChunkTransition) -> None:
        if transition.horizon <= 0:
            raise ValueError("transition horizon must be positive")
        self._items.append(transition)

    def state_dict(self) -> dict[str, Any]:
        """Return all replay state needed for deterministic continuation."""
        return {
            "capacity": self._items.maxlen,
            "items": list(self._items),
            "rng_state": self._rng.getstate(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore replay contents and sampling RNG without silently dropping data."""
        for key in ("capacity", "items", "rng_state"):
            if key not in state:
                raise RuntimeError(f"Replay checkpoint is missing {key!r}.")

        saved_capacity = int(state["capacity"])
        configured_capacity = int(self._items.maxlen)
        if saved_capacity > configured_capacity:
            raise ValueError(
                "Replay capacity cannot be reduced when resuming "
                f"(checkpoint={saved_capacity}, configured={configured_capacity})."
            )

        items = list(state["items"])
        if len(items) > saved_capacity:
            raise RuntimeError("Replay checkpoint contains more items than its declared capacity.")
        for transition in items:
            if not isinstance(transition, ChunkTransition):
                raise TypeError("Replay checkpoint contains an invalid transition.")

        # A continuation may intentionally increase capacity to retain
        # cross-task positive transitions during a critic-only coverage phase.
        self._items = deque(items, maxlen=configured_capacity)
        self._rng.setstate(state["rng_state"])

    def sample(
        self,
        *,
        batch_size: int,
        device: Any,
        positive_fraction: float = 0.0,
        task_balanced: bool = False,
        intervention_balanced: bool = False,
    ) -> dict[str, Any]:
        if not self._items:
            raise RuntimeError("empty ChunkReplayBuffer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= float(positive_fraction) <= 1.0:
            raise ValueError(f"positive_fraction must be in [0, 1], got {positive_fraction}.")
        if not isinstance(task_balanced, bool):
            raise ValueError(f"task_balanced must be a bool, got {task_balanced!r}.")
        if not isinstance(intervention_balanced, bool):
            raise ValueError(
                f"intervention_balanced must be a bool, got {intervention_balanced!r}."
            )

        import torch

        sample_size = min(batch_size, len(self._items))
        transitions = self._sample_transitions(
            sample_size,
            positive_fraction=float(positive_fraction),
            task_balanced=task_balanced,
            intervention_balanced=intervention_balanced,
        )

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
            "replay_positive_reward_fraction": self.positive_reward_fraction(),
            "batch_positive_reward_fraction": float(
                sum(transition.chunk_reward > 0 for transition in transitions) / len(transitions)
            ),
            "replay_positive_outcome_fraction": self.positive_outcome_fraction(),
            "batch_positive_outcome_fraction": float(
                sum(_is_positive_outcome(transition) for transition in transitions) / len(transitions)
            ),
            "replay_intervention_fraction": self.intervention_fraction(),
            "batch_intervention_fraction": _intervention_fraction(transitions),
            "replay_intervention_labeled_fraction": self.intervention_labeled_fraction(),
            "batch_intervention_labeled_fraction": float(
                sum(
                    getattr(transition, "intervention_applied", None) is not None
                    for transition in transitions
                )
                / len(transitions)
            ),
            "intervention_applied": torch.tensor(
                [
                    [bool(getattr(transition, "intervention_applied", False))]
                    for transition in transitions
                ],
                dtype=torch.bool,
                device=device,
            ),
            "intervention_noise_l2": torch.tensor(
                [
                    [
                        0.0
                        if getattr(transition, "intervention_noise_l2", None) is None
                        else float(transition.intervention_noise_l2)
                    ]
                    for transition in transitions
                ],
                dtype=torch.float32,
                device=device,
            ),
            **_episode_return_batch(transitions, device=device),
            "batch_task_count": float(
                len({_task_key(transition.curr_obs) for transition in transitions})
            ),
            "horizons": torch.cat(
                [torch.tensor([[transition.horizon]], dtype=torch.long, device=device) for transition in transitions],
                dim=0,
            ),
        }

    def sample_pairwise(
        self,
        *,
        batch_size: int,
        device: Any,
        min_length_gap: int = 5,
    ) -> dict[str, Any] | None:
        """Sample completed same-state action pairs with a real outcome order."""
        import torch

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if isinstance(min_length_gap, bool) or not isinstance(min_length_gap, int) or min_length_gap < 0:
            raise ValueError("min_length_gap must be a non-negative integer")

        by_pair: dict[str, dict[str, ChunkTransition]] = {}
        for transition in self._items:
            pair_id = getattr(transition, "pair_id", None)
            branch = getattr(transition, "pair_branch", None)
            if (
                pair_id is None
                or branch not in {"clean", "intervention"}
                or not bool(getattr(transition, "pair_anchor", False))
                or getattr(transition, "episode_length", None) is None
            ):
                continue
            by_pair.setdefault(str(pair_id), {})[branch] = transition

        ordered_pairs: list[tuple[ChunkTransition, ChunkTransition]] = []
        for pair in by_pair.values():
            clean = pair.get("clean")
            intervention = pair.get("intervention")
            if clean is None or intervention is None:
                continue
            clean_success = bool(getattr(clean, "episode_success", False))
            intervention_success = bool(getattr(intervention, "episode_success", False))
            if clean_success != intervention_success:
                preferred, rejected = (
                    (clean, intervention)
                    if clean_success
                    else (intervention, clean)
                )
                ordered_pairs.append((preferred, rejected))
                continue
            if clean_success and intervention_success:
                clean_length = getattr(clean, "episode_length", None)
                intervention_length = getattr(intervention, "episode_length", None)
                if clean_length is None or intervention_length is None:
                    continue
                if abs(int(clean_length) - int(intervention_length)) < min_length_gap:
                    continue
                preferred, rejected = (
                    (clean, intervention)
                    if int(clean_length) < int(intervention_length)
                    else (intervention, clean)
                )
                ordered_pairs.append((preferred, rejected))

        if not ordered_pairs:
            return None
        if len(ordered_pairs) >= batch_size:
            selected = self._rng.sample(ordered_pairs, batch_size)
        else:
            selected = [
                ordered_pairs[index % len(ordered_pairs)]
                for index in range(batch_size)
            ]
        preferred = [pair[0] for pair in selected]
        rejected = [pair[1] for pair in selected]
        return {
            "pair_preferred_curr_obs": _cat_observations(
                [item.curr_obs for item in preferred],
                device=device,
            ),
            "pair_preferred_actions": torch.cat(
                [item.actions.to(device) for item in preferred],
                dim=0,
            ),
            "pair_rejected_curr_obs": _cat_observations(
                [item.curr_obs for item in rejected],
                device=device,
            ),
            "pair_rejected_actions": torch.cat(
                [item.actions.to(device) for item in rejected],
                dim=0,
            ),
            "pair_count": float(len(selected)),
            "pair_available_count": float(len(ordered_pairs)),
        }

    def verified_pair_count(self, *, min_length_gap: int = 5) -> int:
        """Count complete, outcome-ordered clean/intervention pairs without sampling."""
        if isinstance(min_length_gap, bool) or not isinstance(min_length_gap, int) or min_length_gap < 0:
            raise ValueError("min_length_gap must be a non-negative integer")
        by_pair: dict[str, dict[str, ChunkTransition]] = {}
        for transition in self._items:
            pair_id = getattr(transition, "pair_id", None)
            branch = getattr(transition, "pair_branch", None)
            if (
                pair_id is None
                or branch not in {"clean", "intervention"}
                or not bool(getattr(transition, "pair_anchor", False))
                or getattr(transition, "episode_length", None) is None
            ):
                continue
            by_pair.setdefault(str(pair_id), {})[branch] = transition

        count = 0
        for pair in by_pair.values():
            clean = pair.get("clean")
            intervention = pair.get("intervention")
            if clean is None or intervention is None:
                continue
            clean_success = bool(getattr(clean, "episode_success", False))
            intervention_success = bool(getattr(intervention, "episode_success", False))
            if clean_success != intervention_success:
                count += 1
            elif clean_success and abs(
                int(clean.episode_length) - int(intervention.episode_length)
            ) >= min_length_gap:
                count += 1
        return count

    def _sample_transitions(
        self,
        sample_size: int,
        *,
        positive_fraction: float,
        task_balanced: bool,
        intervention_balanced: bool,
    ) -> list[ChunkTransition]:
        items = list(self._items)
        if positive_fraction <= 0.0:
            return self._sample_pool(
                items,
                sample_size,
                task_balanced=task_balanced,
                intervention_balanced=intervention_balanced,
            )

        positives = [item for item in items if _is_positive_outcome(item)]
        if not positives:
            return self._sample_pool(
                items,
                sample_size,
                task_balanced=task_balanced,
                intervention_balanced=intervention_balanced,
            )
        ordinary = [item for item in items if not _is_positive_outcome(item)]
        if not ordinary:
            return self._sample_pool(
                positives,
                sample_size,
                task_balanced=task_balanced,
                intervention_balanced=intervention_balanced,
            )
        positive_count = min(sample_size, max(1, round(sample_size * positive_fraction)))
        ordinary_count = sample_size - positive_count
        transitions = self._sample_pool(
            positives,
            positive_count,
            task_balanced=task_balanced,
            intervention_balanced=intervention_balanced,
        )
        if ordinary_count:
            transitions.extend(
                self._sample_pool(
                    ordinary,
                    ordinary_count,
                    task_balanced=task_balanced,
                    intervention_balanced=intervention_balanced,
                )
            )
        self._rng.shuffle(transitions)
        return transitions

    def _sample_pool(
        self,
        pool: list[ChunkTransition],
        sample_size: int,
        *,
        task_balanced: bool,
        intervention_balanced: bool,
    ) -> list[ChunkTransition]:
        if sample_size <= 0:
            return []
        if not pool:
            raise RuntimeError("cannot sample from an empty replay sub-pool")
        if not task_balanced and not intervention_balanced:
            if sample_size <= len(pool):
                return self._rng.sample(pool, sample_size)
            return [self._rng.choice(pool) for _ in range(sample_size)]

        by_group: dict[tuple[Any, ...], list[ChunkTransition]] = {}
        for transition in pool:
            group = []
            if task_balanced:
                group.append(_task_key(transition.curr_obs))
            if intervention_balanced:
                group.append(_intervention_group_key(transition))
            by_group.setdefault(tuple(group), []).append(transition)
        groups = sorted(by_group, key=repr)
        return [
            self._rng.choice(by_group[groups[index % len(groups)]])
            for index in range(sample_size)
        ]

    @staticmethod
    def _cat_obs(observations: list[dict[str, Any]], *, device: Any) -> dict[str, Any]:
        return _cat_observations(observations, device=device)


def collate_transitions(transitions: list[ChunkTransition] | tuple[ChunkTransition, ...], *, device: Any) -> dict[str, Any]:
    """Collate explicit transitions without sampling from a training replay.

    This is used for independent held-out Bellman checks.  Keeping collation
    separate from ``ChunkReplayBuffer.sample`` prevents held-out transitions
    from accidentally entering the optimizer's replay distribution.
    """
    import torch

    transitions = list(transitions)
    if not transitions:
        raise ValueError("cannot collate an empty transition collection")
    for transition in transitions:
        if getattr(transition.actions, "ndim", None) != 2:
            raise ValueError(
                "transition.actions must have shape [1, action_dim] or [batch, action_dim]"
            )

    return {
        "curr_obs": _cat_observations([transition.curr_obs for transition in transitions], device=device),
        "next_obs": _cat_observations([transition.next_obs for transition in transitions], device=device),
        "actions": torch.cat([transition.actions.to(device) for transition in transitions], dim=0),
        "rewards": torch.cat(
            [
                torch.tensor([[transition.chunk_reward]], dtype=torch.float32, device=device)
                for transition in transitions
            ],
            dim=0,
        ),
        "terminations": torch.cat(
            [
                torch.tensor(
                    [[bool(transition.done)]],
                    dtype=torch.bool,
                    device=device,
                )
                for transition in transitions
            ],
            dim=0,
        ),
        "discounts": torch.cat(
            [
                torch.tensor([[transition.discount]], dtype=torch.float32, device=device)
                for transition in transitions
            ],
            dim=0,
        ),
        "intervention_applied": torch.tensor(
            [
                [bool(getattr(transition, "intervention_applied", False))]
                for transition in transitions
            ],
            dtype=torch.bool,
            device=device,
        ),
        "intervention_noise_l2": torch.tensor(
            [
                [
                    0.0
                    if getattr(transition, "intervention_noise_l2", None) is None
                    else float(transition.intervention_noise_l2)
                ]
                for transition in transitions
            ],
            dtype=torch.float32,
            device=device,
        ),
        **_episode_return_batch(transitions, device=device),
        "horizons": torch.cat(
            [
                torch.tensor([[transition.horizon]], dtype=torch.long, device=device)
                for transition in transitions
            ],
            dim=0,
        ),
    }


def _tensor_observation_fields(observation: dict[str, Any]) -> dict[str, Any]:
    """只保留 SAC actor/critic 可消费的 tensor observation 字段，跳过 transition 元数据。"""
    return {
        key: value
        for key, value in observation.items()
        if value is not None and callable(getattr(value, "to", None)) and hasattr(value, "shape")
    }


def _cat_observations(observations: list[dict[str, Any]], *, device: Any) -> dict[str, Any]:
    tensor_observations = [_tensor_observation_fields(observation) for observation in observations]
    expected_keys = set(tensor_observations[0].keys())
    for observation in tensor_observations[1:]:
        if set(observation.keys()) != expected_keys:
            raise ValueError("observation keys must match across sampled transitions")

    return {
        key: _cat_observation_tensors(
            key,
            [observation[key].to(device) for observation in tensor_observations],
        )
        for key in expected_keys
    }


def _task_key(observation: dict[str, Any]) -> str:
    return repr(observation.get("task"))


def _is_positive_outcome(transition: ChunkTransition) -> bool:
    return bool(getattr(transition, "episode_success", False)) or float(
        getattr(transition, "chunk_reward", 0.0)
    ) > 0.0


def _intervention_group_key(transition: ChunkTransition) -> str:
    applied = getattr(transition, "intervention_applied", None)
    if applied is None:
        return "unlabeled"
    return "intervention" if bool(applied) else "clean"


def _intervention_fraction(transitions: list[ChunkTransition]) -> float:
    labeled = [
        transition
        for transition in transitions
        if getattr(transition, "intervention_applied", None) is not None
    ]
    if not labeled:
        return 0.0
    return sum(bool(transition.intervention_applied) for transition in labeled) / len(labeled)


def _episode_return_batch(
    transitions: list[ChunkTransition],
    *,
    device: Any,
) -> dict[str, Any]:
    import torch

    values = [getattr(transition, "episode_return_to_go", None) for transition in transitions]
    return {
        "episode_returns": torch.tensor(
            [[0.0 if value is None else float(value)] for value in values],
            dtype=torch.float32,
            device=device,
        ),
        "episode_return_mask": torch.tensor(
            [[value is not None] for value in values],
            dtype=torch.bool,
            device=device,
        ),
    }


def _cat_observation_tensors(key: str, values: list[Any]) -> Any:
    import torch
    import torch.nn.functional as F

    reference_shape = tuple(values[0].shape)
    if all(tuple(value.shape) == reference_shape for value in values[1:]):
        return torch.cat(values, dim=0)

    # Different LIBERO task descriptions can tokenize to different lengths.
    # Right-pad only the language sequence fields so mixed-task replay batches
    # remain valid without changing image/state/action semantics.
    language_keys = {
        "observation.language.tokens",
        "observation.language.attention_mask",
    }
    if key not in language_keys or any(value.ndim != 2 for value in values):
        raise ValueError("observation tensor shapes must match across sampled transitions")

    max_length = max(int(value.shape[1]) for value in values)
    padded = [
        F.pad(value, (0, max_length - int(value.shape[1])))
        for value in values
    ]
    return torch.cat(padded, dim=0)
