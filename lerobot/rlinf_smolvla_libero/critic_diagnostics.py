from __future__ import annotations

import hashlib
from contextlib import contextmanager
from collections import defaultdict
from typing import Any

import torch

from .critic import aggregate_q
from .replay import ChunkTransition, collate_transitions


def evaluate_intervention_coverage(
    transitions: list[ChunkTransition] | tuple[ChunkTransition, ...],
) -> dict[str, float]:
    """Summarize observed clean/intervention outcomes without assigning causality."""
    transitions = list(transitions)
    if not transitions:
        raise ValueError("cannot evaluate an empty transition collection")

    labeled = [
        transition
        for transition in transitions
        if getattr(transition, "intervention_applied", None) is not None
    ]
    metrics: dict[str, float] = {
        "transition_count": float(len(transitions)),
        "labeled_transition_count": float(len(labeled)),
        "labeled_transition_fraction": len(labeled) / len(transitions),
    }
    for group_name, applied in (("clean", False), ("intervention", True)):
        group = [
            transition
            for transition in labeled
            if bool(transition.intervention_applied) is applied
        ]
        completed = [
            transition
            for transition in group
            if bool(getattr(transition, "episode_completed", False))
        ]
        successful = [
            transition for transition in completed if _transition_success(transition)
        ]
        metrics[f"{group_name}_transition_count"] = float(len(group))
        metrics[f"{group_name}_completed_episode_count"] = float(len(completed))
        metrics[f"{group_name}_successful_episode_count"] = float(len(successful))
        metrics[f"{group_name}_episode_success_rate"] = (
            len(successful) / len(completed) if completed else 0.0
        )
    metrics["intervention_transition_fraction"] = (
        metrics["intervention_transition_count"] / len(labeled) if labeled else 0.0
    )
    intervention_noise = [
        float(transition.intervention_noise_l2)
        for transition in labeled
        if bool(transition.intervention_applied)
        and getattr(transition, "intervention_noise_l2", None) is not None
    ]
    metrics["intervention_noise_l2_mean"] = (
        sum(intervention_noise) / len(intervention_noise) if intervention_noise else 0.0
    )
    if (
        metrics["clean_completed_episode_count"] > 0.0
        and metrics["intervention_completed_episode_count"] > 0.0
    ):
        metrics["intervention_minus_clean_success_rate"] = (
            metrics["intervention_episode_success_rate"]
            - metrics["clean_episode_success_rate"]
        )
    else:
        metrics["intervention_minus_clean_success_rate"] = 0.0

    task_indices = sorted(
        {
            int(transition.intervention_task_index)
            for transition in labeled
            if getattr(transition, "intervention_task_index", None) is not None
        }
    )
    for task_index in task_indices:
        task_transitions = [
            transition
            for transition in labeled
            if getattr(transition, "intervention_task_index", None) == task_index
        ]
        for group_name, applied in (("clean", False), ("intervention", True)):
            group = [
                transition
                for transition in task_transitions
                if bool(transition.intervention_applied) is applied
            ]
            completed = [
                transition
                for transition in group
                if bool(getattr(transition, "episode_completed", False))
            ]
            successful = [
                transition for transition in completed if _transition_success(transition)
            ]
            prefix = f"task_{task_index}_{group_name}"
            metrics[f"{prefix}_transition_count"] = float(len(group))
            metrics[f"{prefix}_completed_episode_count"] = float(len(completed))
            metrics[f"{prefix}_successful_episode_count"] = float(len(successful))
            metrics[f"{prefix}_episode_success_rate"] = (
                len(successful) / len(completed) if completed else 0.0
            )
    return metrics


def evaluate_pairwise_action_ranking(
    *,
    actor: Any,
    q_network: Any,
    transitions: list[ChunkTransition] | tuple[ChunkTransition, ...],
    device: Any,
    agg: str = "min",
    margin: float = 0.05,
    min_length_gap: int = 5,
    batch_size: int = 16,
) -> dict[str, float]:
    """Measure whether the critic ranks verified paired actions correctly."""
    if margin < 0.0:
        raise ValueError(f"margin must be non-negative, got {margin}.")
    if min_length_gap < 0:
        raise ValueError(f"min_length_gap must be non-negative, got {min_length_gap}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    by_pair: dict[str, dict[str, ChunkTransition]] = defaultdict(dict)
    for transition in transitions:
        pair_id = getattr(transition, "pair_id", None)
        branch = getattr(transition, "pair_branch", None)
        if (
            pair_id is None
            or branch not in {"clean", "intervention"}
            or not bool(getattr(transition, "pair_anchor", False))
            or getattr(transition, "episode_length", None) is None
        ):
            continue
        by_pair[str(pair_id)][branch] = transition

    pair_missing_branch_count = 0
    pair_same_outcome_count = 0
    pair_length_gap_rejected_count = 0
    pairs: list[tuple[ChunkTransition, ChunkTransition]] = []
    for branches in by_pair.values():
        clean = branches.get("clean")
        intervention = branches.get("intervention")
        if clean is None or intervention is None:
            pair_missing_branch_count += 1
            continue
        clean_success = bool(getattr(clean, "episode_success", False))
        intervention_success = bool(getattr(intervention, "episode_success", False))
        if clean_success != intervention_success:
            pairs.append(
                (clean, intervention)
                if clean_success
                else (intervention, clean)
            )
            continue
        if clean_success and intervention_success:
            clean_length = int(clean.episode_length)
            intervention_length = int(intervention.episode_length)
            if abs(clean_length - intervention_length) >= min_length_gap:
                pairs.append(
                    (clean, intervention)
                    if clean_length < intervention_length
                    else (intervention, clean)
                )
            else:
                pair_length_gap_rejected_count += 1
        else:
            pair_same_outcome_count += 1

    if not pairs:
        return {
            "pair_id_group_count": float(len(by_pair)),
            "pair_missing_branch_count": float(pair_missing_branch_count),
            "pair_same_outcome_count": float(pair_same_outcome_count),
            "pair_length_gap_rejected_count": float(pair_length_gap_rejected_count),
            "pair_available_count": 0.0,
            "pair_evaluated_count": 0.0,
            "pairwise_ranking_accuracy": 0.0,
            "pairwise_q_gap_mean": 0.0,
            "pairwise_margin_violation_fraction": 0.0,
        }

    preferred = [pair[0] for pair in pairs]
    rejected = [pair[1] for pair in pairs]
    q_training = bool(getattr(q_network, "training", False))
    q_network.eval()
    policy = getattr(actor, "policy", None)
    policy_training = None
    if policy is not None and callable(getattr(policy, "eval", None)):
        policy_training = bool(getattr(policy, "training", False))
        policy.eval()
    preferred_values: list[torch.Tensor] = []
    rejected_values: list[torch.Tensor] = []
    try:
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                preferred_batch = collate_transitions(
                    preferred[start : start + batch_size],
                    device=device,
                )
                rejected_batch = collate_transitions(
                    rejected[start : start + batch_size],
                    device=device,
                )
                preferred_features = actor.encode_obs(preferred_batch["curr_obs"])
                rejected_features = actor.encode_obs(rejected_batch["curr_obs"])
                preferred_values.append(
                    aggregate_q(
                        q_network(preferred_features, preferred_batch["actions"]),
                        agg=agg,
                    ).squeeze(-1).cpu()
                )
                rejected_values.append(
                    aggregate_q(
                        q_network(rejected_features, rejected_batch["actions"]),
                        agg=agg,
                    ).squeeze(-1).cpu()
                )
    finally:
        if q_training:
            q_network.train()
        if policy is not None and policy_training:
            policy.train()

    gap = torch.cat(preferred_values) - torch.cat(rejected_values)
    return {
        "pair_id_group_count": float(len(by_pair)),
        "pair_missing_branch_count": float(pair_missing_branch_count),
        "pair_same_outcome_count": float(pair_same_outcome_count),
        "pair_length_gap_rejected_count": float(pair_length_gap_rejected_count),
        "pair_available_count": float(len(pairs)),
        "pair_evaluated_count": float(gap.numel()),
        "pairwise_ranking_accuracy": float((gap > 0.0).float().mean()),
        "pairwise_q_gap_mean": float(gap.mean()),
        "pairwise_margin_violation_fraction": float((gap < float(margin)).float().mean()),
    }


def evaluate_critic_action_gate(
    *,
    actor: Any,
    q_network: Any,
    transitions: list[ChunkTransition] | tuple[ChunkTransition, ...],
    device: Any,
    agg: str = "min",
    perturb_std: float = 0.05,
    clamp_perturbations: bool = False,
    batch_size: int = 16,
    seed: int = 0,
    max_transitions_per_task: int | None = None,
) -> dict[str, float]:
    """Evaluate action sensitivity and explicit success-state separation.

    ``replay_vs_perturbed`` compares the action actually executed in the
    replay with a small action perturbation at the same observation.  Clamping
    to ``[-1, 1]`` is opt-in because SmolVLA actions use mean/std
    normalization and are not guaranteed to be in that interval.
    The success/ordinary comparison uses the environment ``success`` metadata,
    not the sparse reward alone.
    """
    if perturb_std < 0.0:
        raise ValueError(f"perturb_std must be non-negative, got {perturb_std}.")
    if not isinstance(clamp_perturbations, bool):
        raise ValueError(f"clamp_perturbations must be a bool, got {clamp_perturbations!r}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if max_transitions_per_task is not None and (
        isinstance(max_transitions_per_task, bool) or max_transitions_per_task <= 0
    ):
        raise ValueError(
            "max_transitions_per_task must be positive or None, "
            f"got {max_transitions_per_task}."
        )

    transitions, _ = _select_diagnostic_transitions(
        transitions,
        max_transitions_per_task=max_transitions_per_task,
    )
    if not transitions:
        raise ValueError("cannot evaluate an empty transition collection")

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    q_replay_values: list[torch.Tensor] = []
    q_perturbed_values: list[torch.Tensor] = []
    q_head_spans: list[torch.Tensor] = []
    success_values: list[torch.Tensor] = []

    q_training = bool(getattr(q_network, "training", False))
    q_network.eval()
    policy = getattr(actor, "policy", None)
    policy_training = None
    if policy is not None and callable(getattr(policy, "eval", None)):
        policy_training = bool(getattr(policy, "training", False))
        policy.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(transitions), batch_size):
                chunk = transitions[start : start + batch_size]
                batch = collate_transitions(chunk, device=device)
                features = actor.encode_obs(batch["curr_obs"])
                replay_q_heads = q_network(features, batch["actions"])
                perturbation = torch.randn(
                    batch["actions"].shape,
                    device=batch["actions"].device,
                    dtype=batch["actions"].dtype,
                    generator=generator,
                ).mul(float(perturb_std))
                perturbed_actions = batch["actions"] + perturbation
                if clamp_perturbations:
                    perturbed_actions = perturbed_actions.clamp(-1.0, 1.0)
                perturbed_q_heads = q_network(features, perturbed_actions)

                q_replay_values.append(aggregate_q(replay_q_heads, agg=agg).squeeze(-1).cpu())
                q_perturbed_values.append(
                    aggregate_q(perturbed_q_heads, agg=agg).squeeze(-1).cpu()
                )
                q_head_spans.append(
                    (replay_q_heads.max(dim=-1).values - replay_q_heads.min(dim=-1).values).cpu()
                )
                success_values.append(
                    torch.tensor(
                        [_transition_success(transition) for transition in chunk],
                        dtype=torch.bool,
                    )
                )
    finally:
        if q_training:
            q_network.train()
        if policy is not None and policy_training:
            policy.train()

    q_replay = torch.cat(q_replay_values)
    q_perturbed = torch.cat(q_perturbed_values)
    q_span = torch.cat(q_head_spans)
    success_mask = torch.cat(success_values)
    ordinary_mask = ~success_mask
    replay_delta = q_replay - q_perturbed

    metrics = {
        "replay_transition_count": float(len(transitions)),
        "perturbation_clamp_enabled": float(clamp_perturbations),
        "replay_action_beats_perturbed_fraction": float((replay_delta > 0.0).float().mean()),
        "replay_minus_perturbed_q_mean": float(replay_delta.mean()),
        "success_transition_count": float(success_mask.sum()),
        "ordinary_transition_count": float(ordinary_mask.sum()),
        "q_head_span_mean": float(q_span.mean()),
        "q_head_span_max": float(q_span.max()),
    }
    if bool(success_mask.any()):
        metrics["success_q_mean"] = float(q_replay[success_mask].mean())
    else:
        metrics["success_q_mean"] = 0.0
    if bool(ordinary_mask.any()):
        metrics["ordinary_q_mean"] = float(q_replay[ordinary_mask].mean())
    else:
        metrics["ordinary_q_mean"] = 0.0
    metrics["success_minus_ordinary_q_gap"] = (
        metrics["success_q_mean"] - metrics["ordinary_q_mean"]
        if bool(success_mask.any()) and bool(ordinary_mask.any())
        else 0.0
    )
    return metrics


def evaluate_critic_root_causes(
    *,
    actor: Any,
    q_network: Any,
    transitions: list[ChunkTransition] | tuple[ChunkTransition, ...],
    device: Any,
    agg: str = "min",
    perturb_std: float = 0.05,
    perturb_repeats: int = 4,
    batch_size: int = 16,
    seed: int = 0,
    max_transitions_per_task: int | None = 32,
) -> dict[str, Any]:
    """Measure hypotheses that can explain a weak replay/action ranking.

    This is deliberately a diagnosis, not a training objective.  It checks:

    * whether replay actions are in the coordinate system assumed by the
      diagnostic/conservative loss (the old code assumed ``[-1, 1]`` even
      though SmolVLA actions are mean/std normalized);
    * whether the result is stable over repeated fixed-seed perturbations;
    * whether the critic changes more with action or observation features; and
    * whether changing actions within a task changes Q at all.

    No optimizer, parameter, replay buffer, or global RNG state is modified.
    ``max_transitions_per_task`` keeps the VLM forwards bounded and prevents a
    diagnostic from becoming an accidental second training run.
    """
    if perturb_std < 0.0:
        raise ValueError(f"perturb_std must be non-negative, got {perturb_std}.")
    if isinstance(perturb_repeats, bool) or perturb_repeats <= 0:
        raise ValueError(f"perturb_repeats must be a positive integer, got {perturb_repeats}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if max_transitions_per_task is not None and (
        isinstance(max_transitions_per_task, bool) or max_transitions_per_task <= 0
    ):
        raise ValueError(
            "max_transitions_per_task must be positive or None, "
            f"got {max_transitions_per_task}."
        )

    selected, task_labels = _select_diagnostic_transitions(
        transitions,
        max_transitions_per_task=max_transitions_per_task,
    )
    if not selected:
        raise ValueError("cannot evaluate an empty transition collection")

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    q_training, policy_training = _enter_diagnostic_eval_mode(actor, q_network)

    action_values: list[torch.Tensor] = []
    feature_values: list[torch.Tensor] = []
    replay_q_values: list[torch.Tensor] = []
    replay_q_spans: list[torch.Tensor] = []
    action_gradients: list[torch.Tensor] = []
    feature_gradients: list[torch.Tensor] = []
    unclamped_perturbed_q_values: list[torch.Tensor] = []
    clamped_perturbed_q_values: list[torch.Tensor] = []
    perturbation_clamp_fractions: list[torch.Tensor] = []

    try:
        for start in range(0, len(selected), batch_size):
            chunk = selected[start : start + batch_size]
            batch = collate_transitions(chunk, device=device)

            with torch.no_grad():
                features = actor.encode_obs(batch["curr_obs"]).detach()
                actions = batch["actions"].detach()
                replay_q_heads = q_network(features, actions)
                replay_q = aggregate_q(replay_q_heads, agg=agg).squeeze(-1)

            # Input Jacobians expose whether the Q network actually uses the
            # action dimensions.  The model parameters are not updated.
            action_input = actions.clone().detach().requires_grad_(True)
            action_q = aggregate_q(q_network(features, action_input), agg=agg).squeeze(-1)
            action_grad = torch.autograd.grad(action_q.sum(), action_input)[0].detach()

            feature_input = features.clone().detach().requires_grad_(True)
            feature_q = aggregate_q(q_network(feature_input, actions), agg=agg).squeeze(-1)
            feature_grad = torch.autograd.grad(feature_q.sum(), feature_input)[0].detach()

            unclamped_q_for_chunk: list[torch.Tensor] = []
            clamped_q_for_chunk: list[torch.Tensor] = []
            clamp_fraction_for_chunk: list[torch.Tensor] = []
            with torch.no_grad():
                for _ in range(perturb_repeats):
                    noise = torch.randn(
                        actions.shape,
                        device=actions.device,
                        dtype=actions.dtype,
                        generator=generator,
                    ).mul(float(perturb_std))
                    unclamped_actions = actions + noise
                    clamped_actions = unclamped_actions.clamp(-1.0, 1.0)
                    unclamped_q = aggregate_q(
                        q_network(features, unclamped_actions),
                        agg=agg,
                    ).squeeze(-1)
                    clamped_q = aggregate_q(
                        q_network(features, clamped_actions),
                        agg=agg,
                    ).squeeze(-1)
                    unclamped_q_for_chunk.append(unclamped_q.detach().cpu())
                    clamped_q_for_chunk.append(clamped_q.detach().cpu())
                    clamp_fraction_for_chunk.append(
                        (clamped_actions != unclamped_actions).float().mean(dim=-1).cpu()
                    )

            action_values.append(actions.cpu())
            feature_values.append(features.cpu())
            replay_q_values.append(replay_q.cpu())
            replay_q_spans.append(
                (replay_q_heads.max(dim=-1).values - replay_q_heads.min(dim=-1).values).cpu()
            )
            action_gradients.append(action_grad.cpu())
            feature_gradients.append(feature_grad.cpu())
            unclamped_perturbed_q_values.append(torch.stack(unclamped_q_for_chunk, dim=1))
            clamped_perturbed_q_values.append(torch.stack(clamped_q_for_chunk, dim=1))
            perturbation_clamp_fractions.append(torch.stack(clamp_fraction_for_chunk, dim=1))
    finally:
        _restore_diagnostic_eval_mode(actor, q_network, q_training, policy_training)

    actions_all = torch.cat(action_values, dim=0)
    features_all = torch.cat(feature_values, dim=0)
    replay_q_all = torch.cat(replay_q_values, dim=0)
    replay_q_span_all = torch.cat(replay_q_spans, dim=0)
    action_grad_all = torch.cat(action_gradients, dim=0)
    feature_grad_all = torch.cat(feature_gradients, dim=0)
    unclamped_q_all = torch.cat(unclamped_perturbed_q_values, dim=0)
    clamped_q_all = torch.cat(clamped_perturbed_q_values, dim=0)
    clamp_fraction_all = torch.cat(perturbation_clamp_fractions, dim=0)

    unclamped_delta = replay_q_all[:, None] - unclamped_q_all
    clamped_delta = replay_q_all[:, None] - clamped_q_all
    action_scale = actions_all.std(dim=0, unbiased=False).clamp_min(1e-6)
    feature_scale = features_all.std(dim=0, unbiased=False).clamp_min(1e-6)
    scaled_action_grad = action_grad_all * action_scale
    scaled_feature_grad = feature_grad_all * feature_scale

    metrics: dict[str, Any] = {
        "transition_count": float(len(selected)),
        "task_count": float(len(set(task_labels))),
        "missing_task_label_fraction": float(
            sum(label == "__missing_task__" for label in task_labels) / len(task_labels)
        ),
        "action_dim": float(actions_all.shape[1]),
        "action_abs_mean": float(actions_all.abs().mean()),
        "action_abs_max": float(actions_all.abs().max()),
        "action_rms": float(actions_all.square().mean().sqrt()),
        "action_outside_unit_fraction": float((actions_all.abs() > 1.0).float().mean()),
        "q_head_span_mean": float(replay_q_span_all.mean()),
        "q_head_span_max": float(replay_q_span_all.max()),
        "perturbation_std": float(perturb_std),
        "perturbation_repeats": float(perturb_repeats),
        "perturbation_clamp_fraction": float(clamp_fraction_all.mean()),
        "replay_vs_unclamped_perturbed_fraction_mean": float((unclamped_delta > 0.0).float().mean()),
        "replay_vs_unclamped_perturbed_fraction_std": float(
            (unclamped_delta > 0.0).float().mean(dim=0).std(unbiased=False)
        ),
        "replay_minus_unclamped_perturbed_q_mean": float(unclamped_delta.mean()),
        "replay_vs_clamped_perturbed_fraction_mean": float((clamped_delta > 0.0).float().mean()),
        "replay_vs_clamped_perturbed_fraction_std": float(
            (clamped_delta > 0.0).float().mean(dim=0).std(unbiased=False)
        ),
        "replay_minus_clamped_perturbed_q_mean": float(clamped_delta.mean()),
        "q_action_grad_norm_mean": float(action_grad_all.norm(dim=-1).mean()),
        "q_observation_grad_norm_mean": float(feature_grad_all.norm(dim=-1).mean()),
        "q_action_effect_mean": float(scaled_action_grad.abs().mean()),
        "q_observation_effect_mean": float(scaled_feature_grad.abs().mean()),
        "q_action_to_observation_effect_ratio": float(
            scaled_action_grad.abs().mean()
            / scaled_feature_grad.abs().mean().clamp_min(1e-12)
        ),
        "q_action_shuffle_abs_delta_mean": _action_shuffle_delta_mean(
            q_network=q_network,
            features=features_all,
            actions=actions_all,
            task_labels=task_labels,
            device=device,
            agg=agg,
        ),
    }

    metrics.update(
        _per_task_root_cause_metrics(
            q_network=q_network,
            features=features_all,
            actions=actions_all,
            replay_q=replay_q_all,
            unclamped_q=unclamped_q_all,
            clamped_q=clamped_q_all,
            task_labels=task_labels,
            device=device,
            agg=agg,
        )
    )
    return metrics


def evaluate_actor_gradient_compatibility(
    *,
    actor: Any,
    q_network: Any,
    transitions: list[ChunkTransition] | tuple[ChunkTransition, ...],
    device: Any,
    agg: str = "min",
    kl_penalty_coef: float = 0.0,
    include_kl: bool = True,
    repeats: int = 3,
    max_transitions_per_task: int | None = 16,
    seed: int = 0,
) -> dict[str, float]:
    """Measure task-wise actor-gradient compatibility without updating actor.

    The result separates the Q-driven gradient from the KL gradient.  This is
    necessary because a negative cosine can come from genuinely different task
    objectives, from an unreliable critic, or from the KL anchor rather than
    from a single undifferentiated "actor gradient conflict".
    """
    if kl_penalty_coef < 0.0:
        raise ValueError(f"kl_penalty_coef must be non-negative, got {kl_penalty_coef}.")
    if isinstance(repeats, bool) or repeats <= 0:
        raise ValueError(f"repeats must be a positive integer, got {repeats}.")

    selected, task_labels = _select_diagnostic_transitions(
        transitions,
        max_transitions_per_task=max_transitions_per_task,
    )
    if not selected:
        raise ValueError("cannot evaluate an empty transition collection")
    task_to_transitions: dict[str, list[ChunkTransition]] = defaultdict(list)
    for transition, task_label in zip(selected, task_labels, strict=True):
        task_to_transitions[task_label].append(transition)
    task_keys = sorted(task_to_transitions)

    parameters = [
        parameter
        for parameter in actor.parameters()
        if bool(getattr(parameter, "requires_grad", False))
    ]
    if not parameters:
        raise ValueError("actor has no trainable parameters for gradient diagnostics")

    q_training, policy_training = _enter_diagnostic_eval_mode(actor, q_network)
    q_gradients: list[list[torch.Tensor]] = []
    kl_gradients: list[list[torch.Tensor]] = []
    total_gradients: list[list[torch.Tensor]] = []
    try:
        for repeat in range(repeats):
            q_repeat: list[torch.Tensor] = []
            kl_repeat: list[torch.Tensor] = []
            total_repeat: list[torch.Tensor] = []
            for task_key in task_keys:
                batch = collate_transitions(task_to_transitions[task_key], device=device)
                with _fork_torch_rng(device, seed=int(seed) + repeat):
                    sampled = _sample_actor_for_gradient_diagnostic(
                        actor=actor,
                        obs=batch["curr_obs"],
                        include_kl=include_kl and kl_penalty_coef > 0.0,
                    )
                actions, actor_features, kl_estimate = sampled
                q_values = q_network(actor_features, actions)
                q_objective = -aggregate_q(q_values, agg=agg).mean()
                q_vector = _autograd_vector(q_objective, parameters, retain_graph=kl_estimate is not None)

                if kl_estimate is not None:
                    kl_objective = kl_estimate.mean()
                    kl_vector = _autograd_vector(kl_objective, parameters, retain_graph=False)
                else:
                    kl_vector = torch.zeros_like(q_vector)
                total_vector = q_vector + float(kl_penalty_coef) * kl_vector
                q_repeat.append(q_vector.detach())
                kl_repeat.append(kl_vector.detach())
                total_repeat.append(total_vector.detach())
            q_gradients.append(q_repeat)
            kl_gradients.append(kl_repeat)
            total_gradients.append(total_repeat)
    finally:
        _restore_diagnostic_eval_mode(actor, q_network, q_training, policy_training)

    q_stack = torch.stack([vector for repeat in q_gradients for vector in repeat])
    kl_stack = torch.stack([vector for repeat in kl_gradients for vector in repeat])
    total_stack = torch.stack([vector for repeat in total_gradients for vector in repeat])
    q_pairwise = _pairwise_cosines(q_gradients)
    kl_pairwise = _pairwise_cosines(kl_gradients)
    total_pairwise = _pairwise_cosines(total_gradients)
    q_kl_cosines = [
        _cosine_similarity(q_vector, kl_vector)
        for q_repeat, kl_repeat in zip(q_gradients, kl_gradients, strict=True)
        for q_vector, kl_vector in zip(q_repeat, kl_repeat, strict=True)
    ]
    mean_total = total_stack.reshape(repeats, len(task_keys), -1).mean(dim=1)
    mean_task_norm = total_stack.norm(dim=-1).reshape(repeats, len(task_keys)).mean(dim=1)

    metrics = {
        "task_count": float(len(task_keys)),
        "transition_count": float(len(selected)),
        "gradient_repeats": float(repeats),
        "q_gradient_norm_mean": float(q_stack.norm(dim=-1).mean()),
        "q_gradient_norm_min": float(q_stack.norm(dim=-1).min()),
        "q_gradient_norm_max": float(q_stack.norm(dim=-1).max()),
        "kl_gradient_norm_mean": float(kl_stack.norm(dim=-1).mean()),
        "total_gradient_norm_mean": float(total_stack.norm(dim=-1).mean()),
        "q_pairwise_cosine_mean": _mean_or_zero(q_pairwise),
        "q_pairwise_cosine_min": _min_or_zero(q_pairwise),
        "q_pairwise_negative_fraction": _negative_fraction(q_pairwise),
        "kl_pairwise_cosine_mean": _mean_or_zero(kl_pairwise),
        "kl_pairwise_cosine_min": _min_or_zero(kl_pairwise),
        "total_pairwise_cosine_mean": _mean_or_zero(total_pairwise),
        "total_pairwise_cosine_min": _min_or_zero(total_pairwise),
        "total_pairwise_negative_fraction": _negative_fraction(total_pairwise),
        "q_kl_cosine_mean": _mean_or_zero(q_kl_cosines),
        "q_kl_cosine_min": _min_or_zero(q_kl_cosines),
        "total_mean_gradient_to_task_norm_ratio": float(
            mean_total.norm(dim=-1).mean() / mean_task_norm.mean().clamp_min(1e-12)
        ),
    }
    return metrics


def _select_diagnostic_transitions(
    transitions: list[ChunkTransition] | tuple[ChunkTransition, ...],
    *,
    max_transitions_per_task: int | None,
) -> tuple[list[ChunkTransition], list[str]]:
    transitions = list(transitions)
    if not transitions:
        return [], []
    grouped: dict[str, list[ChunkTransition]] = defaultdict(list)
    for transition in transitions:
        grouped[_transition_task_key(transition)].append(transition)

    selected: list[ChunkTransition] = []
    labels: list[str] = []
    for task_key in sorted(grouped):
        pool = grouped[task_key]
        if max_transitions_per_task is not None:
            pool = _deterministic_stratified_subset(pool, max_transitions_per_task)
        selected.extend(pool)
        labels.extend([task_key] * len(pool))
    return selected, labels


def _deterministic_stratified_subset(
    transitions: list[ChunkTransition],
    limit: int,
) -> list[ChunkTransition]:
    if len(transitions) <= limit:
        return list(transitions)

    outcome_indices = [
        index
        for index, transition in enumerate(transitions)
        if bool(getattr(transition, "episode_success", False))
        or float(getattr(transition, "chunk_reward", 0.0)) > 0.0
        or bool(getattr(transition, "episode_completed", False))
        or bool(getattr(transition, "truncated", False))
    ]
    outcome_budget = min(len(outcome_indices), max(1, limit // 4))
    selected_indices = set(_evenly_spaced_indices(outcome_indices, outcome_budget))

    remaining_budget = limit - len(selected_indices)
    remaining_indices = [
        index for index in range(len(transitions)) if index not in selected_indices
    ]
    selected_indices.update(_evenly_spaced_indices(remaining_indices, remaining_budget))
    return [transitions[index] for index in sorted(selected_indices)]


def _evenly_spaced_indices(indices: list[int], count: int) -> list[int]:
    if count <= 0 or not indices:
        return []
    if count >= len(indices):
        return list(indices)
    if count == 1:
        return [indices[len(indices) // 2]]
    last = len(indices) - 1
    return [indices[(offset * last) // (count - 1)] for offset in range(count)]


def _transition_task_key(transition: ChunkTransition) -> str:
    task = transition.curr_obs.get("task")
    if task is None:
        return "__missing_task__"
    return repr(task)


def _transition_success(transition: ChunkTransition) -> bool:
    explicit_success = getattr(transition, "episode_success", None)
    if explicit_success is not None:
        return bool(explicit_success)
    return bool(getattr(transition, "done", False)) and (
        float(getattr(transition, "chunk_reward", 0.0)) > 0.0
    )


def _enter_diagnostic_eval_mode(actor: Any, q_network: Any) -> tuple[bool, bool | None]:
    q_training = bool(getattr(q_network, "training", False))
    q_network.eval()
    policy = getattr(actor, "policy", None)
    if policy is None or not callable(getattr(policy, "eval", None)):
        return q_training, None
    policy_training = bool(getattr(policy, "training", False))
    policy.eval()
    return q_training, policy_training


def _restore_diagnostic_eval_mode(
    actor: Any,
    q_network: Any,
    q_training: bool,
    policy_training: bool | None,
) -> None:
    if q_training:
        q_network.train()
    policy = getattr(actor, "policy", None)
    if policy is not None and policy_training:
        policy.train()


def _action_shuffle_delta_mean(
    *,
    q_network: Any,
    features: torch.Tensor,
    actions: torch.Tensor,
    task_labels: list[str],
    device: Any,
    agg: str,
) -> float:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, task_label in enumerate(task_labels):
        groups[task_label].append(index)
    deltas: list[torch.Tensor] = []
    q_training = bool(getattr(q_network, "training", False))
    q_network.eval()
    try:
        with torch.no_grad():
            for indices in groups.values():
                if len(indices) < 2:
                    continue
                index_tensor = torch.tensor(indices, dtype=torch.long)
                group_features = features[index_tensor].to(device)
                group_actions = actions[index_tensor].to(device)
                shuffled_actions = torch.roll(group_actions, shifts=1, dims=0)
                original_q = aggregate_q(q_network(group_features, group_actions), agg=agg).squeeze(-1)
                shuffled_q = aggregate_q(q_network(group_features, shuffled_actions), agg=agg).squeeze(-1)
                deltas.append((original_q - shuffled_q).abs().cpu())
    finally:
        if q_training:
            q_network.train()
    if not deltas:
        return 0.0
    return float(torch.cat(deltas).mean())


def _per_task_root_cause_metrics(
    *,
    q_network: Any,
    features: torch.Tensor,
    actions: torch.Tensor,
    replay_q: torch.Tensor,
    unclamped_q: torch.Tensor,
    clamped_q: torch.Tensor,
    task_labels: list[str],
    device: Any,
    agg: str,
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, task_label in enumerate(task_labels):
        groups[task_label].append(index)
    metrics: dict[str, Any] = {}
    for task_index, task_key in enumerate(sorted(groups)):
        indices = groups[task_key]
        index_tensor = torch.tensor(indices, dtype=torch.long)
        task_features = features[index_tensor]
        task_actions = actions[index_tensor]
        task_replay_q = replay_q[index_tensor]
        task_unclamped_q = unclamped_q[index_tensor]
        task_clamped_q = clamped_q[index_tensor]
        task_unclamped_delta = task_replay_q[:, None] - task_unclamped_q
        task_clamped_delta = task_replay_q[:, None] - task_clamped_q
        prefix = f"task_{task_index}"
        metrics[f"{prefix}_label_hash"] = float(
            int(hashlib.sha1(task_key.encode("utf-8")).hexdigest()[:8], 16)
        )
        metrics[f"{prefix}_transition_count"] = float(len(indices))
        metrics[f"{prefix}_action_outside_unit_fraction"] = float(
            (task_actions.abs() > 1.0).float().mean()
        )
        metrics[f"{prefix}_replay_vs_unclamped_perturbed_fraction_mean"] = float(
            (task_unclamped_delta > 0.0).float().mean()
        )
        metrics[f"{prefix}_replay_minus_unclamped_perturbed_q_mean"] = float(
            task_unclamped_delta.mean()
        )
        metrics[f"{prefix}_replay_vs_clamped_perturbed_fraction_mean"] = float(
            (task_clamped_delta > 0.0).float().mean()
        )
        metrics[f"{prefix}_replay_minus_clamped_perturbed_q_mean"] = float(
            task_clamped_delta.mean()
        )
        metrics[f"{prefix}_q_action_shuffle_abs_delta_mean"] = _action_shuffle_delta_mean(
            q_network=q_network,
            features=task_features,
            actions=task_actions,
            task_labels=[task_key] * len(indices),
            device=device,
            agg=agg,
        )
    return metrics


def _sample_actor_for_gradient_diagnostic(
    *,
    actor: Any,
    obs: dict[str, Any],
    include_kl: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if include_kl:
        sample_with_kl = getattr(actor, "sample_chunk_with_kl", None)
        if callable(sample_with_kl):
            actions, _, features, _, kl_estimate = sample_with_kl(obs)
            return actions, features, kl_estimate
    actions, _, features, _ = actor.sample_chunk(obs, train=True)
    return actions, features, None


def _autograd_vector(
    objective: torch.Tensor,
    parameters: list[Any],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    if not objective.requires_grad:
        return torch.cat([torch.zeros_like(parameter).reshape(-1) for parameter in parameters])
    gradients = torch.autograd.grad(
        objective,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    pieces = [
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients, strict=True)
    ]
    return torch.cat([piece.reshape(-1) for piece in pieces])


@contextmanager
def _fork_torch_rng(device: Any, seed: int) -> Any:
    device_obj = torch.device(device)
    devices: list[int] = []
    if device_obj.type == "cuda":
        devices = [
            torch.cuda.current_device()
            if device_obj.index is None
            else int(device_obj.index)
        ]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if devices:
            torch.cuda.manual_seed_all(int(seed))
        yield


def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = left.norm() * right.norm()
    if float(denominator) <= 1e-12:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def _pairwise_cosines(gradients: list[list[torch.Tensor]]) -> list[float]:
    values: list[float] = []
    for repeat in gradients:
        for left_index in range(len(repeat)):
            for right_index in range(left_index + 1, len(repeat)):
                values.append(_cosine_similarity(repeat[left_index], repeat[right_index]))
    return values


def _mean_or_zero(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _min_or_zero(values: list[float]) -> float:
    return float(min(values)) if values else 0.0


def _negative_fraction(values: list[float]) -> float:
    return float(sum(value < 0.0 for value in values) / len(values)) if values else 0.0
