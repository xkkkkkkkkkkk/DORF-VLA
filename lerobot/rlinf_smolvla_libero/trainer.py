from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from .config import SACFlowConfig
from .critic import EntropyTemperature, actor_loss, alpha_loss, critic_loss, critic_target, soft_update
from .trainable_scope import iter_trainable_parameters


class SACFlowTrainer:
    """只包含 SAC-Flow 单步更新核心，不负责 rollout、环境连接或完整训练循环。"""

    _REQUIRED_BATCH_KEYS = (
        "curr_obs",
        "next_obs",
        "actions",
        "rewards",
        "terminations",
        "discounts",
    )

    def __init__(
        self,
        actor: nn.Module,
        q_network: nn.Module,
        target_q_network: nn.Module,
        config: SACFlowConfig,
        actor_optimizer: torch.optim.Optimizer | None = None,
        critic_optimizer: torch.optim.Optimizer | None = None,
        alpha_optimizer: torch.optim.Optimizer | None = None,
        temperature: EntropyTemperature | None = None,
    ) -> None:
        self.actor = actor
        self.q_network = q_network
        self.target_q_network = target_q_network
        self.config = config
        self.temperature = temperature if temperature is not None else EntropyTemperature(config.initial_alpha)

        self.actor_optimizer = actor_optimizer or torch.optim.Adam(
            self._default_actor_optimizer_parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = critic_optimizer or torch.optim.Adam(self.q_network.parameters(), lr=config.critic_lr)
        self.alpha_optimizer = alpha_optimizer or torch.optim.Adam(self.temperature.parameters(), lr=config.alpha_lr)

        action_dim = self._infer_action_dim(self.q_network)
        self.target_entropy = config.target_entropy if config.target_entropy is not None else -float(action_dim)
        self.update_step = 0

    def update_sac(self, batch: Mapping[str, Any]) -> dict[str, float]:
        missing_keys = [key for key in self._REQUIRED_BATCH_KEYS if key not in batch]
        if missing_keys:
            raise KeyError(f"SAC batch is missing required keys: {missing_keys}.")

        with torch.no_grad():
            next_actions, next_log_pi, next_features, _ = self.actor.sample_chunk(batch["next_obs"], train=True)
            target_q = self.target_q_network(next_features, next_actions)
            # batch["discounts"] 已是 chunk-level bootstrap discount；config.gamma 在 rollout/replay 构造 transition 时使用。
            target = critic_target(
                batch["rewards"],
                batch["terminations"],
                batch["discounts"],
                target_q,
                next_log_pi,
                self.temperature.alpha.detach(),
                self.config.agg_q,
                self.config.backup_entropy,
            )

        curr_features = self.actor.encode_obs(batch["curr_obs"]).detach()
        q_data = self.q_network(curr_features, batch["actions"])
        critic_objective = critic_loss(q_data, target)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_objective.backward()
        self._clip_grad_norm(self.q_network.parameters())
        self.critic_optimizer.step()

        metrics: dict[str, float] = {
            "critic_loss": float(critic_objective.detach().cpu()),
            "q_mean": float(q_data.detach().mean().cpu()),
            "alpha": float(self.temperature.alpha.detach().cpu()),
        }

        if (
            self.update_step >= self.config.actor_warmup_updates
            and self.update_step % self.config.critic_actor_ratio == 0
        ):
            self._set_requires_grad(self.q_network, requires_grad=False)
            try:
                curr_actions, log_pi, actor_features, _ = self.actor.sample_chunk(batch["curr_obs"], train=True)
                q_pi = self.q_network(actor_features, curr_actions)
                actor_objective = actor_loss(
                    q_pi,
                    log_pi,
                    self.temperature.alpha.detach(),
                    self.config.actor_agg_q,
                )

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_objective.backward()
                self._clip_grad_norm(self.actor.parameters())
                self.actor_optimizer.step()
            finally:
                self._set_requires_grad(self.q_network, requires_grad=True)

            alpha_objective = alpha_loss(self.temperature.log_alpha, log_pi.detach(), self.target_entropy)
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_objective.backward()
            self.alpha_optimizer.step()

            metrics.update(
                {
                    "actor_loss": float(actor_objective.detach().cpu()),
                    "alpha_loss": float(alpha_objective.detach().cpu()),
                    "entropy": float((-log_pi.detach()).mean().cpu()),
                    "log_pi": float(log_pi.detach().mean().cpu()),
                    "alpha": float(self.temperature.alpha.detach().cpu()),
                }
            )

        soft_update(self.q_network, self.target_q_network, self.config.tau)
        self.update_step += 1
        return metrics

    def _clip_grad_norm(self, parameters: Any) -> None:
        if self.config.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(parameters, self.config.grad_clip_norm)

    def _default_actor_optimizer_parameters(self) -> list[Any]:
        parameters = iter_trainable_parameters(self.actor)
        if not parameters:
            raise ValueError("SAC actor has no trainable parameters for optimizer construction.")
        return parameters

    @staticmethod
    def _infer_action_dim(q_network: nn.Module) -> int:
        if hasattr(q_network, "action_dim"):
            return int(q_network.action_dim)
        if hasattr(q_network, "heads") and hasattr(q_network, "obs_dim"):
            first_linear = q_network.heads[0][0]
            if hasattr(first_linear, "in_features"):
                return int(first_linear.in_features - q_network.obs_dim)
        raise ValueError("Unable to infer action_dim from q_network; expose action_dim or heads[0][0].in_features and obs_dim.")

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(requires_grad)
