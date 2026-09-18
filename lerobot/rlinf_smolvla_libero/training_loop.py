from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .config import SACFlowConfig
from .libero_adapter import BatchedChunkRolloutResult, ChunkRolloutResult, execute_action_chunk, execute_batched_action_chunk


@dataclass(frozen=True)
class SACFlowTrainStepResult:
    """一次在线 SAC-Flow step 的轻量结果。"""

    rollout: ChunkRolloutResult
    update_metrics: list[dict[str, float]]
    next_obs: dict[str, Any]
    rollouts: tuple[ChunkRolloutResult, ...] = ()


@dataclass
class SACFlowCollectionStats:
    """Explicit episode/transition accounting for one collection split.

    ``chunk_reward > 0`` is a transition-level sparse-reward statistic.
    ``successful_episode_count`` is instead driven by the environment's
    success info and is counted only when the episode actually completes.
    Keeping the two counters separate prevents a terminal reward from being
    mistaken for an episode count.
    """

    transitions_collected: int = 0
    completed_episode_count: int = 0
    successful_episode_count: int = 0
    truncated_episode_count: int = 0
    terminal_transition_count: int = 0
    positive_reward_transition_count: int = 0
    raw_reward_sum: float = 0.0
    completed_episode_length_sum: int = 0
    _active_episode_steps: dict[int, int] = field(default_factory=dict, repr=False)
    group_transition_counts: dict[str, int] = field(default_factory=dict)
    group_completed_episode_counts: dict[str, int] = field(default_factory=dict)
    group_successful_episode_counts: dict[str, int] = field(default_factory=dict)
    task_group_transition_counts: dict[tuple[int, str], int] = field(default_factory=dict)
    task_group_completed_episode_counts: dict[tuple[int, str], int] = field(default_factory=dict)
    task_group_successful_episode_counts: dict[tuple[int, str], int] = field(default_factory=dict)
    intervention_noise_l2_sum: float = 0.0
    intervention_noise_l2_count: int = 0

    def update(self, rollouts: tuple[ChunkRolloutResult, ...] | list[ChunkRolloutResult]) -> None:
        for rollout in rollouts:
            transition = rollout.transition
            env_index = int(getattr(rollout, "env_index", 0))
            horizon = int(transition.horizon)
            if horizon <= 0:
                raise ValueError(f"rollout transition horizon must be positive, got {horizon}.")

            self.transitions_collected += 1
            group = _transition_intervention_group(transition)
            task_index = getattr(transition, "intervention_task_index", None)
            if group is not None:
                self.group_transition_counts[group] = self.group_transition_counts.get(group, 0) + 1
                if task_index is not None:
                    task_group = (int(task_index), group)
                    self.task_group_transition_counts[task_group] = (
                        self.task_group_transition_counts.get(task_group, 0) + 1
                    )
            noise_l2 = getattr(transition, "intervention_noise_l2", None)
            if group == "intervention" and noise_l2 is not None:
                self.intervention_noise_l2_sum += float(noise_l2)
                self.intervention_noise_l2_count += 1
            self._active_episode_steps[env_index] = self._active_episode_steps.get(env_index, 0) + horizon
            self.raw_reward_sum += float(sum(getattr(rollout, "raw_rewards", ()) or ()))
            if float(transition.chunk_reward) > 0.0:
                self.positive_reward_transition_count += 1
            transition_done = bool(getattr(transition, "done", False))
            if transition_done:
                self.terminal_transition_count += 1

            success_value = getattr(rollout, "success", None)
            if success_value is None:
                success_value = getattr(transition, "episode_success", False)
            success = bool(success_value)

            truncated_value = getattr(rollout, "truncated", None)
            if truncated_value is None:
                truncated_value = getattr(transition, "truncated", False)
            truncated = bool(truncated_value)

            completed_value = getattr(transition, "episode_completed", None)
            if completed_value is None:
                completed_value = transition_done or truncated
            completed = bool(completed_value)
            if not completed:
                continue

            self.completed_episode_count += 1
            if group is not None:
                self.group_completed_episode_counts[group] = (
                    self.group_completed_episode_counts.get(group, 0) + 1
                )
                if task_index is not None:
                    task_group = (int(task_index), group)
                    self.task_group_completed_episode_counts[task_group] = (
                        self.task_group_completed_episode_counts.get(task_group, 0) + 1
                    )
            self.completed_episode_length_sum += self._active_episode_steps.pop(env_index, horizon)
            if truncated:
                self.truncated_episode_count += 1
            if success:
                self.successful_episode_count += 1
                if group is not None:
                    self.group_successful_episode_counts[group] = (
                        self.group_successful_episode_counts.get(group, 0) + 1
                    )
                    if task_index is not None:
                        task_group = (int(task_index), group)
                        self.task_group_successful_episode_counts[task_group] = (
                            self.task_group_successful_episode_counts.get(task_group, 0) + 1
                        )

    @property
    def episode_success_rate(self) -> float:
        if self.completed_episode_count <= 0:
            return 0.0
        return self.successful_episode_count / self.completed_episode_count

    @property
    def positive_reward_transition_fraction(self) -> float:
        if self.transitions_collected <= 0:
            return 0.0
        return self.positive_reward_transition_count / self.transitions_collected

    @property
    def mean_completed_episode_length(self) -> float:
        if self.completed_episode_count <= 0:
            return 0.0
        return self.completed_episode_length_sum / self.completed_episode_count

    def metrics(self, *, prefix: str) -> dict[str, float]:
        prefix = prefix.rstrip("/")
        metrics = {
            f"{prefix}/transitions_collected": float(self.transitions_collected),
            f"{prefix}/episodes_completed": float(self.completed_episode_count),
            f"{prefix}/episodes_successful": float(self.successful_episode_count),
            f"{prefix}/episodes_truncated": float(self.truncated_episode_count),
            f"{prefix}/terminal_transitions": float(self.terminal_transition_count),
            f"{prefix}/positive_reward_transitions": float(self.positive_reward_transition_count),
            f"{prefix}/positive_reward_transition_fraction": float(self.positive_reward_transition_fraction),
            f"{prefix}/episode_success_rate": float(self.episode_success_rate),
            f"{prefix}/mean_completed_episode_length": float(self.mean_completed_episode_length),
            f"{prefix}/raw_reward_sum": float(self.raw_reward_sum),
        }
        labeled_count = sum(self.group_transition_counts.values())
        if not labeled_count:
            return metrics
        intervention_count = self.group_transition_counts.get("intervention", 0)
        metrics[f"{prefix}/intervention_labeled_transition_fraction"] = (
            labeled_count / self.transitions_collected if self.transitions_collected else 0.0
        )
        metrics[f"{prefix}/intervention_transition_fraction"] = (
            intervention_count / labeled_count if labeled_count else 0.0
        )
        metrics[f"{prefix}/intervention_noise_l2_mean"] = (
            self.intervention_noise_l2_sum / self.intervention_noise_l2_count
            if self.intervention_noise_l2_count
            else 0.0
        )
        for group in ("clean", "intervention"):
            completed = self.group_completed_episode_counts.get(group, 0)
            successful = self.group_successful_episode_counts.get(group, 0)
            metrics[f"{prefix}/{group}/transitions"] = float(
                self.group_transition_counts.get(group, 0)
            )
            metrics[f"{prefix}/{group}/episodes_completed"] = float(completed)
            metrics[f"{prefix}/{group}/episodes_successful"] = float(successful)
            metrics[f"{prefix}/{group}/episode_success_rate"] = (
                successful / completed if completed else 0.0
            )
        task_indices = sorted(
            {
                task_index
                for task_index, _ in (
                    set(self.task_group_transition_counts)
                    | set(self.task_group_completed_episode_counts)
                )
            }
        )
        for task_index in task_indices:
            for group in ("clean", "intervention"):
                key = (task_index, group)
                completed = self.task_group_completed_episode_counts.get(key, 0)
                successful = self.task_group_successful_episode_counts.get(key, 0)
                task_prefix = f"{prefix}/task_{task_index}/{group}"
                metrics[f"{task_prefix}/transitions"] = float(
                    self.task_group_transition_counts.get(key, 0)
                )
                metrics[f"{task_prefix}/episodes_completed"] = float(completed)
                metrics[f"{task_prefix}/episodes_successful"] = float(successful)
                metrics[f"{task_prefix}/episode_success_rate"] = (
                    successful / completed if completed else 0.0
                )
        return metrics

    def state_dict(self) -> dict[str, Any]:
        return {
            "transitions_collected": self.transitions_collected,
            "completed_episode_count": self.completed_episode_count,
            "successful_episode_count": self.successful_episode_count,
            "truncated_episode_count": self.truncated_episode_count,
            "terminal_transition_count": self.terminal_transition_count,
            "positive_reward_transition_count": self.positive_reward_transition_count,
            "raw_reward_sum": self.raw_reward_sum,
            "completed_episode_length_sum": self.completed_episode_length_sum,
            "active_episode_steps": dict(self._active_episode_steps),
            "group_transition_counts": dict(self.group_transition_counts),
            "group_completed_episode_counts": dict(self.group_completed_episode_counts),
            "group_successful_episode_counts": dict(self.group_successful_episode_counts),
            "task_group_transition_counts": dict(self.task_group_transition_counts),
            "task_group_completed_episode_counts": dict(
                self.task_group_completed_episode_counts
            ),
            "task_group_successful_episode_counts": dict(
                self.task_group_successful_episode_counts
            ),
            "intervention_noise_l2_sum": self.intervention_noise_l2_sum,
            "intervention_noise_l2_count": self.intervention_noise_l2_count,
        }


def summarize_rollouts(step_results: list[SACFlowTrainStepResult]) -> SACFlowCollectionStats:
    """Summarize one training or held-out collection split."""
    stats = SACFlowCollectionStats()
    for result in step_results:
        rollouts = getattr(result, "rollouts", ()) or (result.rollout,)
        stats.update(tuple(rollouts))
    return stats


class SACFlowOnlineLoop:
    """串联 rollout、replay 和 SAC update，不负责创建真实模型或环境。"""

    def __init__(
        self,
        *,
        actor: Any,
        env: Any,
        replay_buffer: Any,
        trainer: Any,
        config: SACFlowConfig,
        action_postprocessor: Callable[[Any], Any] | None = None,
        observation_preparer: Callable[..., Any] | None = None,
        max_chunk_steps: int | None = None,
        stop_on_success: bool = True,
        rollout_fn: Callable[..., ChunkRolloutResult] = execute_action_chunk,
        batched_rollout_fn: Callable[..., BatchedChunkRolloutResult] = execute_batched_action_chunk,
        update_callback: Callable[[dict[str, float], int], None] | None = None,
        step_callback: Callable[[SACFlowTrainStepResult, int], None] | None = None,
        intervention_collector: Any | None = None,
        collection_seed: int | None = None,
    ) -> None:
        self.actor = actor
        self.env = env
        self.replay_buffer = replay_buffer
        self.trainer = trainer
        self.config = config
        self.action_postprocessor = action_postprocessor
        self.observation_preparer = observation_preparer
        self.max_chunk_steps = max_chunk_steps
        self.stop_on_success = stop_on_success
        self.rollout_fn = rollout_fn
        self.batched_rollout_fn = batched_rollout_fn
        self.update_callback = update_callback
        self.step_callback = step_callback
        self.intervention_collector = intervention_collector
        self.collection_seed = 0 if collection_seed is None else int(collection_seed)
        self._episode_transitions: dict[int, list[Any]] = {}
        self._episode_step_counts: dict[int, int] = {}
        self._episode_numbers: dict[int, int] = {}

    def collect_transition(
        self,
        curr_obs: dict[str, Any],
        *,
        store_in_replay: bool = True,
    ) -> ChunkRolloutResult:
        """用 actor 采样 action chunk，执行环境前缀，并写入 replay。"""
        sample_result = self.actor.sample_chunk(curr_obs, train=False)
        if not isinstance(sample_result, (tuple, list)) or len(sample_result) != 4:
            raise ValueError("actor.sample_chunk must return (flat_actions, log_pi, obs_features, raw_chunk)")
        _, _, _, raw_chunk = sample_result
        intervention_batch = None
        if self.intervention_collector is not None:
            intervention_batch = self.intervention_collector.apply(raw_chunk)
            raw_chunk = intervention_batch.raw_chunk

        batch_size = int(getattr(raw_chunk, "shape", (1,))[0])
        if batch_size > 1:
            batched_rollout = self.batched_rollout_fn(
                env=self.env,
                curr_obs=curr_obs,
                raw_chunk=raw_chunk,
                gamma=self.config.gamma,
                max_chunk_steps=self.max_chunk_steps,
                action_postprocessor=self.action_postprocessor,
                stop_on_success=self.stop_on_success,
                observation_preparer=self.observation_preparer,
                step_penalty=self.config.critic_step_penalty,
            )
            for rollout in batched_rollout.rollouts:
                if intervention_batch is not None:
                    self._annotate_intervention_transition(
                        rollout.transition,
                        env_index=int(getattr(rollout, "env_index", 0)),
                        intervention_batch=intervention_batch,
                    )
                self._annotate_episode_position(
                    rollout.transition,
                    env_index=int(getattr(rollout, "env_index", 0)),
                )
                if store_in_replay:
                    self.replay_buffer.add(rollout.transition)
            self._record_episode_rollouts(batched_rollout.rollouts)
            if not batched_rollout.rollouts:
                raise ValueError("batched rollout must contain at least one transition")
            self._last_next_obs = batched_rollout.next_obs
            self._last_rollouts = tuple(batched_rollout.rollouts)
            return batched_rollout.rollouts[0]

        rollout = self.rollout_fn(
            env=self.env,
            curr_obs=curr_obs,
            raw_chunk=raw_chunk,
            gamma=self.config.gamma,
            max_chunk_steps=self.max_chunk_steps,
            action_postprocessor=self.action_postprocessor,
            stop_on_success=self.stop_on_success,
            observation_preparer=self.observation_preparer,
            step_penalty=self.config.critic_step_penalty,
        )
        if intervention_batch is not None:
            self._annotate_intervention_transition(
                rollout.transition,
                env_index=0,
                intervention_batch=intervention_batch,
            )
        self._annotate_episode_position(rollout.transition, env_index=0)
        if store_in_replay:
            self.replay_buffer.add(rollout.transition)
        self._record_episode_rollouts((rollout,))
        self._last_next_obs = rollout.transition.next_obs
        self._last_rollouts = (rollout,)
        return rollout

    def update_if_ready(
        self,
        *,
        collection_step: int = 0,
        enabled: bool = True,
    ) -> list[dict[str, float]]:
        """replay 达到 min_buffer_size 后执行固定次数 SAC update。"""
        if not enabled:
            return []
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return []

        update_metrics: list[dict[str, float]] = []
        for _ in range(self.config.num_updates_per_step):
            sample_kwargs = {
                "batch_size": self.config.batch_size,
                "device": self.config.device,
                "positive_fraction": self.config.critic_positive_sample_fraction,
                "task_balanced": self.config.critic_task_balanced_sampling,
            }
            if self.config.critic_intervention_fraction > 0.0:
                sample_kwargs["intervention_balanced"] = (
                    self.config.critic_intervention_balanced_sampling
                )
            batch = self.replay_buffer.sample(**sample_kwargs)
            if self.config.critic_pairwise_coef > 0.0:
                pairwise_sampler = getattr(self.replay_buffer, "sample_pairwise", None)
                if callable(pairwise_sampler):
                    pairwise_batch = pairwise_sampler(
                        batch_size=self.config.batch_size,
                        device=self.config.device,
                        min_length_gap=self.config.critic_pairwise_min_length_gap,
                    )
                    if pairwise_batch is not None:
                        batch.update(pairwise_batch)
            metrics = self.trainer.update_sac(batch)
            update_metrics.append(metrics)
            if self.update_callback is not None:
                self.update_callback(metrics, collection_step)
        return update_metrics

    def train_step(
        self,
        curr_obs: dict[str, Any],
        *,
        collection_step: int = 0,
        store_in_replay: bool = True,
        update: bool = True,
    ) -> SACFlowTrainStepResult:
        """执行一次 collect + optional updates，并返回下一步 observation。"""
        rollout = self.collect_transition(curr_obs, store_in_replay=store_in_replay)
        update_metrics = self.update_if_ready(collection_step=collection_step, enabled=update)
        return SACFlowTrainStepResult(
            rollout=rollout,
            update_metrics=update_metrics,
            next_obs=self._last_next_obs,
            rollouts=self._last_rollouts,
        )

    def run(
        self,
        initial_obs: dict[str, Any],
        *,
        num_steps: int,
        store_in_replay: bool = True,
        update: bool = True,
    ) -> list[SACFlowTrainStepResult]:
        """从 initial_obs 开始运行固定步数。

        ``update=False, store_in_replay=False`` is the collection-only path
        used for an independent held-out split.  It shares the actor and
        environment interface but cannot change the training replay or
        optimizer state.
        """
        if num_steps < 0:
            raise ValueError(f"num_steps must be non-negative, got {num_steps}.")
        obs = initial_obs
        results: list[SACFlowTrainStepResult] = []
        for collection_step in range(num_steps):
            result = self.train_step(
                obs,
                collection_step=collection_step,
                store_in_replay=store_in_replay,
                update=update,
            )
            results.append(result)
            if self.step_callback is not None:
                self.step_callback(result, collection_step)
            obs = result.next_obs
        return results

    def reset_episode_tracking(self) -> None:
        """Discard episode assembly state after an explicit environment reset."""
        self._episode_transitions.clear()
        self._episode_step_counts.clear()
        self._episode_numbers.clear()

    def reset_intervention_rng(self, seed: int | None = None) -> None:
        if self.intervention_collector is None:
            return
        reset = getattr(self.intervention_collector, "reset_noise_rng", None)
        if callable(reset):
            reset(seed)

    def _record_episode_rollouts(
        self,
        rollouts: tuple[ChunkRolloutResult, ...] | list[ChunkRolloutResult],
    ) -> None:
        for rollout in rollouts:
            env_index = int(getattr(rollout, "env_index", 0))
            episode = self._episode_transitions.setdefault(env_index, [])
            episode.append(rollout.transition)
            completed = bool(
                getattr(
                    rollout.transition,
                    "episode_completed",
                    False,
                )
            )
            if not completed:
                continue

            success = bool(
                getattr(
                    rollout,
                    "success",
                    getattr(rollout.transition, "episode_success", False),
                )
            )
            return_to_go = 0.0
            episode_length = sum(int(item.horizon) for item in episode)
            for transition in reversed(episode):
                return_to_go = float(transition.chunk_reward) + float(transition.discount) * return_to_go
                transition.episode_success = success
                transition.episode_return_to_go = return_to_go
                transition.episode_length = episode_length
            self._episode_transitions.pop(env_index, None)
            self._episode_step_counts[env_index] = 0
            self._episode_numbers[env_index] = self._episode_numbers.get(env_index, 0) + 1

    def _annotate_episode_position(self, transition: Any, *, env_index: int) -> None:
        step_index = self._episode_step_counts.get(env_index, 0)
        transition.episode_step_index = step_index
        transition.pair_anchor = step_index == 0 and getattr(transition, "pair_id", None) is not None
        self._episode_step_counts[env_index] = step_index + int(transition.horizon)

    def _annotate_intervention_transition(
        self,
        transition: Any,
        *,
        env_index: int,
        intervention_batch: Any,
    ) -> None:
        transition.intervention_applied = bool(intervention_batch.applied[env_index])
        transition.intervention_noise_l2 = float(intervention_batch.noise_l2[env_index])
        transition.intervention_task_index = int(intervention_batch.task_indices[env_index])
        transition.intervention_slot_index = int(intervention_batch.slot_indices[env_index])
        episode_number = self._episode_numbers.get(env_index, 0)
        transition.pair_id = (
            f"task={transition.intervention_task_index}:"
            f"seed={self.collection_seed}:"
            f"pair={transition.intervention_slot_index // 2}:"
            f"episode={episode_number}"
        )
        transition.pair_branch = (
            "intervention" if transition.intervention_applied else "clean"
        )


def _transition_intervention_group(transition: Any) -> str | None:
    applied = getattr(transition, "intervention_applied", None)
    if applied is None:
        return None
    return "intervention" if bool(applied) else "clean"
