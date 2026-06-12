#!/usr/bin/env python

"""Utilities for SmolVLA flow-matching RL post-training.

This module is intentionally light-weight: scalar helpers work without importing
torch, so their RL math can be tested on machines that do not have the training
runtime installed. Tensor/batch helpers import torch lazily and are used by the
actual training script.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

OBS_LANGUAGE_TOKENS = "observation.language.tokens"
OBS_LANGUAGE_ATTENTION_MASK = "observation.language.attention_mask"


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and value.__class__.__name__ == "Tensor"


def _sample_std(values: Sequence[float]) -> float:
    """Return torch.std-compatible sample std for a one-dimensional group."""

    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def compute_group_advantages(
    returns: Any,
    *,
    group_size: int,
    eps: float = 1e-6,
) -> Any:
    """Compute RLinf-style GRPO advantages over contiguous trajectory groups.

    RLinf's embodied GRPO path normalizes each group as
    ``(reward - group_mean) / (group_std + eps)``. This function mirrors that
    contract for the SmolVLA weighted-FM approximation.
    """

    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}.")

    if _is_torch_tensor(returns):
        if returns.ndim != 1:
            returns = returns.reshape(-1)
        if returns.numel() % group_size != 0:
            raise ValueError(
                f"returns length {returns.numel()} must be divisible by group_size {group_size}."
            )
        if group_size == 1:
            return returns.new_zeros(returns.shape)
        grouped = returns.reshape(-1, group_size)
        group_mean = grouped.mean(dim=-1, keepdim=True)
        group_std = grouped.std(dim=-1, keepdim=True)
        advantages = (grouped - group_mean) / (group_std + eps)
        advantages = advantages.masked_fill(group_std <= eps, 0.0)
        return advantages.reshape(-1)

    values = [float(value) for value in returns]
    if len(values) % group_size != 0:
        raise ValueError(f"returns length {len(values)} must be divisible by group_size {group_size}.")

    advantages: list[float] = []
    for start in range(0, len(values), group_size):
        group = values[start : start + group_size]
        mean = sum(group) / len(group)
        std = _sample_std(group)
        if std <= eps:
            advantages.extend([0.0] * len(group))
        else:
            advantages.extend([(value - mean) / (std + eps) for value in group])
    return advantages


def compute_rl_fm_weights(
    advantages: Any,
    *,
    beta: float,
    min_weight: float,
    max_weight: float,
) -> Any:
    """Map group-normalized advantages to clipped flow-matching sample weights."""

    if min_weight <= 0:
        raise ValueError(f"min_weight must be positive, got {min_weight}.")
    if max_weight < min_weight:
        raise ValueError(f"max_weight must be >= min_weight, got {max_weight} < {min_weight}.")

    if _is_torch_tensor(advantages):
        return (beta * advantages).exp().clamp(min=min_weight, max=max_weight)

    weights: list[float] = []
    for advantage in advantages:
        weight = math.exp(beta * float(advantage))
        weights.append(min(max(weight, min_weight), max_weight))
    return weights


def _lazy_torch():
    import torch
    from torch.nn.utils.rnn import pad_sequence

    return torch, pad_sequence


def build_weighted_fm_batch(
    entries: list[dict[str, Any]],
    *,
    beta: float,
    min_weight: float,
    max_weight: float,
) -> tuple[dict[str, Any], Any, dict[str, float]]:
    """Stack trajectory entries into a policy batch and compute RL-FM weights."""

    if not entries:
        raise ValueError("Cannot build a weighted FM batch from an empty entry list.")

    torch, pad_sequence = _lazy_torch()
    returns = torch.tensor([entry["trajectory_return"] for entry in entries], dtype=torch.float32)
    group_size = _infer_single_group_size(entries)
    advantages = compute_group_advantages(returns, group_size=group_size)
    weights = compute_rl_fm_weights(advantages, beta=beta, min_weight=min_weight, max_weight=max_weight)

    language_tokens = [entry[OBS_LANGUAGE_TOKENS] for entry in entries]
    language_attention_masks = [entry[OBS_LANGUAGE_ATTENTION_MASK] for entry in entries]
    if any(tokens.shape != language_tokens[0].shape for tokens in language_tokens):
        language_tokens = pad_sequence(language_tokens, batch_first=True, padding_value=0)
        language_attention_masks = pad_sequence(language_attention_masks, batch_first=True, padding_value=0)
    else:
        language_tokens = torch.stack(language_tokens)
        language_attention_masks = torch.stack(language_attention_masks)

    batch = {
        "action": torch.stack([entry["action"] for entry in entries]),
        "actions_id_pad": torch.stack([entry["actions_id_pad"] for entry in entries]),
        "observation.state": torch.stack([entry["observation.state"] for entry in entries]),
        "observation.images.image": torch.stack([entry["observation.images.image"] for entry in entries]),
        "observation.images.image2": torch.stack([entry["observation.images.image2"] for entry in entries]),
        OBS_LANGUAGE_TOKENS: language_tokens,
        OBS_LANGUAGE_ATTENTION_MASK: language_attention_masks,
    }

    stats = {
        "trajectory_return_mean": returns.mean().item(),
        "trajectory_return_std": returns.std().item() if returns.numel() > 1 else 0.0,
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item() if advantages.numel() > 1 else 0.0,
        "weight_mean": weights.mean().item(),
        "weight_max": weights.max().item(),
        "weight_min": weights.min().item(),
    }
    return batch, weights, stats


def _infer_single_group_size(entries: list[dict[str, Any]]) -> int:
    """Infer one contiguous group size from entries.

    The first clean implementation deliberately supports one on-policy group per
    update. This keeps the update semantics transparent and avoids accidentally
    mixing tasks or reset groups while computing advantages.
    """

    group_ids = {entry.get("group_id", 0) for entry in entries}
    if len(group_ids) != 1:
        raise ValueError(
            "build_weighted_fm_batch expects one contiguous rollout group; "
            f"got group_ids={sorted(group_ids)}."
        )
    return len(entries)
