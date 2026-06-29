from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .config import SACFlowConfig
from .libero_adapter import ChunkRolloutResult, execute_action_chunk


@dataclass(frozen=True)
class SACFlowTrainStepResult:
    """一次在线 SAC-Flow step 的轻量结果。"""

    rollout: ChunkRolloutResult
    update_metrics: list[dict[str, float]]
    next_obs: dict[str, Any]


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
        stop_on_success: bool = True,
        rollout_fn: Callable[..., ChunkRolloutResult] = execute_action_chunk,
    ) -> None:
        self.actor = actor
        self.env = env
        self.replay_buffer = replay_buffer
        self.trainer = trainer
        self.config = config
        self.action_postprocessor = action_postprocessor
        self.observation_preparer = observation_preparer
        self.stop_on_success = stop_on_success
        self.rollout_fn = rollout_fn

    def collect_transition(self, curr_obs: dict[str, Any]) -> ChunkRolloutResult:
        """用 actor 采样 action chunk，执行环境前缀，并写入 replay。"""
        sample_result = self.actor.sample_chunk(curr_obs, train=False)
        if not isinstance(sample_result, (tuple, list)) or len(sample_result) != 4:
            raise ValueError("actor.sample_chunk must return (flat_actions, log_pi, obs_features, raw_chunk)")
        _, _, _, raw_chunk = sample_result

        rollout = self.rollout_fn(
            env=self.env,
            curr_obs=curr_obs,
            raw_chunk=raw_chunk,
            gamma=self.config.gamma,
            max_chunk_steps=None,
            action_postprocessor=self.action_postprocessor,
            stop_on_success=self.stop_on_success,
            observation_preparer=self.observation_preparer,
        )
        self.replay_buffer.add(rollout.transition)
        return rollout

    def update_if_ready(self) -> list[dict[str, float]]:
        """replay 达到 min_buffer_size 后执行固定次数 SAC update。"""
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return []

        update_metrics: list[dict[str, float]] = []
        for _ in range(self.config.num_updates_per_step):
            batch = self.replay_buffer.sample(batch_size=self.config.batch_size, device=self.config.device)
            update_metrics.append(self.trainer.update_sac(batch))
        return update_metrics

    def train_step(self, curr_obs: dict[str, Any]) -> SACFlowTrainStepResult:
        """执行一次 collect + optional updates，并返回下一步 observation。"""
        rollout = self.collect_transition(curr_obs)
        update_metrics = self.update_if_ready()
        return SACFlowTrainStepResult(
            rollout=rollout,
            update_metrics=update_metrics,
            next_obs=rollout.transition.next_obs,
        )

    def run(self, initial_obs: dict[str, Any], *, num_steps: int) -> list[SACFlowTrainStepResult]:
        """从 initial_obs 开始运行固定步数；真实停止策略由上层训练脚本决定。"""
        if num_steps < 0:
            raise ValueError(f"num_steps must be non-negative, got {num_steps}.")
        obs = initial_obs
        results: list[SACFlowTrainStepResult] = []
        for _ in range(num_steps):
            result = self.train_step(obs)
            results.append(result)
            obs = result.next_obs
        return results
