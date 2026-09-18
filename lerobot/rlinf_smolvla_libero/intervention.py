from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any


@dataclass(frozen=True)
class InterventionBatch:
    raw_chunk: Any
    applied: tuple[bool, ...]
    noise_l2: tuple[float, ...]
    pair_noise_stds: tuple[float, ...]
    task_indices: tuple[int, ...]
    slot_indices: tuple[int, ...]


class FixedSlotInterventionCollector:
    """Apply external action noise to a fixed subset of vector slots per task."""

    def __init__(
        self,
        *,
        num_tasks: int,
        slots_per_task: int,
        intervention_fraction: float,
        noise_std: float,
        seed: int,
        task_indices: tuple[int, ...] | None = None,
        pair_actions: bool = False,
        noise_stds: tuple[float, ...] | None = None,
    ) -> None:
        if isinstance(num_tasks, bool) or not isinstance(num_tasks, int) or num_tasks <= 0:
            raise ValueError(f"num_tasks must be a positive integer, got {num_tasks!r}.")
        if (
            isinstance(slots_per_task, bool)
            or not isinstance(slots_per_task, int)
            or slots_per_task <= 0
        ):
            raise ValueError(
                f"slots_per_task must be a positive integer, got {slots_per_task!r}."
            )
        if (
            isinstance(intervention_fraction, bool)
            or not isinstance(intervention_fraction, (int, float))
            or not 0.0 <= float(intervention_fraction) <= 1.0
        ):
            raise ValueError(
                "intervention_fraction must be a number in [0, 1], "
                f"got {intervention_fraction!r}."
            )
        if (
            isinstance(noise_std, bool)
            or not isinstance(noise_std, (int, float))
            or float(noise_std) < 0.0
        ):
            raise ValueError(f"noise_std must be non-negative, got {noise_std!r}.")
        if not isinstance(pair_actions, bool):
            raise ValueError(f"pair_actions must be a bool, got {pair_actions!r}.")
        if noise_stds is not None:
            if not pair_actions:
                raise ValueError("noise_stds requires pair_actions=True.")
            if len(noise_stds) != slots_per_task // 2:
                raise ValueError(
                    "noise_stds must contain one value per paired intervention slot, "
                    f"got {len(noise_stds)} for slots_per_task={slots_per_task}."
                )
            if any(isinstance(value, bool) or float(value) < 0.0 for value in noise_stds):
                raise ValueError(f"noise_stds must be non-negative, got {noise_stds!r}.")
            noise_stds = tuple(float(value) for value in noise_stds)

        fraction = float(intervention_fraction)
        if pair_actions and (
            slots_per_task < 2
            or slots_per_task % 2 != 0
            or abs(fraction - 0.5) > 1e-12
        ):
            raise ValueError(
                "pair_actions requires an even number of slots and "
                "intervention_fraction=0.5."
            )
        if 0.0 < fraction < 1.0 and slots_per_task < 2:
            raise ValueError(
                "Partial slot intervention requires at least two vector slots per task."
            )
        intervention_count = int(math.floor(slots_per_task * fraction + 0.5))
        if 0.0 < fraction < 1.0:
            intervention_count = min(slots_per_task - 1, max(1, intervention_count))
        if pair_actions and intervention_count >= slots_per_task:
            raise ValueError("pair_actions requires at least one clean slot per task.")

        if task_indices is None:
            task_indices = tuple(range(num_tasks))
        if len(task_indices) != num_tasks or any(
            isinstance(task_index, bool) or not isinstance(task_index, int)
            for task_index in task_indices
        ):
            raise ValueError(
                f"task_indices must contain {num_tasks} integers, got {task_indices!r}."
            )

        slot_rng = random.Random(int(seed))
        applied: list[bool] = []
        batch_task_indices: list[int] = []
        slot_indices: list[int] = []
        pair_noise_stds: list[float] = []
        for task_index in task_indices:
            if pair_actions:
                intervention_slots = set(range(1, slots_per_task, 2))
            else:
                intervention_pool = range(slots_per_task)
                intervention_slots = set(slot_rng.sample(intervention_pool, intervention_count))
            for slot_index in range(slots_per_task):
                batch_task_indices.append(task_index)
                slot_indices.append(slot_index)
                applied.append(slot_index in intervention_slots)
                pair_noise_stds.append(
                    float(noise_stds[slot_index // 2])
                    if noise_stds is not None
                    else float(noise_std)
                )

        self.num_tasks = num_tasks
        self.slots_per_task = slots_per_task
        self.intervention_fraction = fraction
        self.noise_std = float(noise_std)
        self.pair_actions = pair_actions
        self.noise_stds = noise_stds
        self.seed = int(seed)
        self.applied = tuple(applied)
        self.task_indices = tuple(batch_task_indices)
        self.slot_indices = tuple(slot_indices)
        self.pair_noise_stds = tuple(pair_noise_stds)
        self._generator = None
        self._generator_device = None

    @property
    def batch_size(self) -> int:
        return self.num_tasks * self.slots_per_task

    def reset_noise_rng(self, seed: int | None = None) -> None:
        self.seed = self.seed if seed is None else int(seed)
        self._generator = None
        self._generator_device = None

    def apply(self, raw_chunk: Any) -> InterventionBatch:
        import torch

        if not isinstance(raw_chunk, torch.Tensor) or raw_chunk.ndim != 3:
            raise ValueError("raw_chunk must be a torch tensor with shape [batch, chunk, action_dim].")
        if int(raw_chunk.shape[0]) != self.batch_size:
            raise ValueError(
                f"raw_chunk batch size must be {self.batch_size}, got {int(raw_chunk.shape[0])}."
            )
        if int(raw_chunk.shape[1]) <= 0 or int(raw_chunk.shape[2]) <= 0:
            raise ValueError("raw_chunk must have non-empty shape [batch, chunk, action_dim].")

        output = raw_chunk.clone()
        if self.pair_actions:
            for task_offset in range(self.num_tasks):
                start = task_offset * self.slots_per_task
                for pair_offset in range(0, self.slots_per_task, 2):
                    output[start + pair_offset + 1] = output[start + pair_offset]
        noise_l2 = torch.zeros(self.batch_size, device=output.device, dtype=output.dtype)
        intervention_indices = [
            index for index, applied in enumerate(self.applied) if applied
        ]
        intervention_noise_stds = [
            self.pair_noise_stds[index] for index in intervention_indices
        ]
        if intervention_indices and any(value > 0.0 for value in intervention_noise_stds):
            generator = self._noise_generator(output.device)
            index_tensor = torch.tensor(
                intervention_indices,
                dtype=torch.long,
                device=output.device,
            )
            noise = torch.randn(
                (len(intervention_indices), int(output.shape[2])),
                device=output.device,
                dtype=output.dtype,
                generator=generator,
            )
            noise = noise * torch.tensor(
                intervention_noise_stds,
                device=output.device,
                dtype=output.dtype,
            ).unsqueeze(-1)
            output[index_tensor, 0, :] = output[index_tensor, 0, :] + noise
            noise_l2[index_tensor] = noise.norm(dim=-1)

        return InterventionBatch(
            raw_chunk=output,
            applied=self.applied,
            noise_l2=tuple(float(value) for value in noise_l2.detach().cpu()),
            pair_noise_stds=self.pair_noise_stds,
            task_indices=self.task_indices,
            slot_indices=self.slot_indices,
        )

    def _noise_generator(self, device: Any) -> Any:
        import torch

        device_key = str(device)
        if self._generator is None or self._generator_device != device_key:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.seed)
            self._generator_device = device_key
        return self._generator
