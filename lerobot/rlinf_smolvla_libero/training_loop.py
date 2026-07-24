from __future__ import annotations

from dataclasses import dataclass
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

    def collect_transition(self, curr_obs: dict[str, Any]) -> ChunkRolloutResult:
        """用 actor 采样 action chunk，执行环境前缀，并写入 replay。"""
        sample_result = self.actor.sample_chunk(curr_obs, train=False)
        if not isinstance(sample_result, (tuple, list)) or len(sample_result) != 4:
            raise ValueError("actor.sample_chunk must return (flat_actions, log_pi, obs_features, raw_chunk)")
        _, _, _, raw_chunk = sample_result

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
            )
            for rollout in batched_rollout.rollouts:
                self.replay_buffer.add(rollout.transition)
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
        )
        self.replay_buffer.add(rollout.transition)
        self._last_next_obs = rollout.transition.next_obs
        self._last_rollouts = (rollout,)
        return rollout

    def update_if_ready(self, *, collection_step: int = 0) -> list[dict[str, float]]:
        """replay 达到 min_buffer_size 后执行固定次数 SAC update。"""
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return []

        update_metrics: list[dict[str, float]] = []
        for _ in range(self.config.num_updates_per_step):
            batch = self.replay_buffer.sample(batch_size=self.config.batch_size, device=self.config.device)
            metrics = self.trainer.update_sac(batch)
            update_metrics.append(metrics)
            if self.update_callback is not None:
                self.update_callback(metrics, collection_step)
        return update_metrics

    def train_step(self, curr_obs: dict[str, Any], *, collection_step: int = 0) -> SACFlowTrainStepResult:
        """执行一次 collect + optional updates，并返回下一步 observation。"""
        rollout = self.collect_transition(curr_obs)
        update_metrics = self.update_if_ready(collection_step=collection_step)
        return SACFlowTrainStepResult(
            rollout=rollout,
            update_metrics=update_metrics,
            next_obs=self._last_next_obs,
            rollouts=self._last_rollouts,
        )

    def run(self, initial_obs: dict[str, Any], *, num_steps: int) -> list[SACFlowTrainStepResult]:
        """从 initial_obs 开始运行固定步数；真实停止策略由上层训练脚本决定。"""
        if num_steps < 0:
            raise ValueError(f"num_steps must be non-negative, got {num_steps}.")
        obs = initial_obs
        results: list[SACFlowTrainStepResult] = []
        for collection_step in range(num_steps):
            result = self.train_step(obs, collection_step=collection_step)
            results.append(result)
            if self.step_callback is not None:
                self.step_callback(result, collection_step)
            obs = result.next_obs
        return results
