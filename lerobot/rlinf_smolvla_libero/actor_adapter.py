from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

from lerobot.rlinf_smolvla_libero.replay import flatten_chunk


class SmolVLASACFlowActor:
    """把 SmolVLA policy 的 SAC-Flow 相关方法包装成稳定 actor 接口。"""

    def __init__(
        self,
        policy: Any,
        device: Any,
        train_noise_std: float,
        rollout_noise_std: float,
        reference_policy: Any | None = None,
        critic_action_steps: int = 1,
    ) -> None:
        self._require_callable(policy, "sac_sample_action_chunk")
        self._require_callable(policy, "sac_encode_observation")
        if reference_policy is not None:
            self._require_callable(reference_policy, "sac_log_prob_action_trajectory")
        self.policy = policy
        self.reference_policy = reference_policy
        self.device = device
        self.train_noise_std = train_noise_std
        self.rollout_noise_std = rollout_noise_std
        if (
            isinstance(critic_action_steps, bool)
            or not isinstance(critic_action_steps, int)
            or critic_action_steps != 1
        ):
            raise ValueError(
                "critic_action_steps must be 1: the online loop replans after each environment step."
            )
        self.critic_action_steps = critic_action_steps

    @staticmethod
    def _require_callable(policy: Any, method_name: str) -> None:
        if not callable(getattr(policy, method_name, None)):
            raise AttributeError(f"policy must provide callable {method_name}")

    @staticmethod
    def _require_tensor_like(value: Any, name: str) -> None:
        if not callable(getattr(value, "to", None)) or not hasattr(value, "shape") or not hasattr(value, "ndim"):
            raise ValueError(f"{name} must be a tensor")

    def _move_obs_to_device(self, obs: Any) -> Any:
        """递归移动 observation 中支持 .to(device) 的值，非 tensor-like 值保持不变。"""
        if isinstance(obs, Mapping):
            return {key: self._move_obs_to_device(value) for key, value in obs.items()}
        if isinstance(obs, tuple):
            return tuple(self._move_obs_to_device(value) for value in obs)
        if isinstance(obs, list):
            return [self._move_obs_to_device(value) for value in obs]
        to_device = getattr(obs, "to", None)
        if callable(to_device):
            return to_device(self.device)
        return obs

    def parameters(self) -> Any:
        """把优化器需要的参数迭代器委托给真实 SmolVLA policy。"""
        parameters_fn = getattr(self.policy, "parameters", None)
        if not callable(parameters_fn):
            raise AttributeError("wrapped policy must provide callable parameters")
        return parameters_fn()

    def named_parameters(self) -> Any:
        """把参数名和参数委托给真实 SmolVLA policy，便于训练范围审计。"""
        named_parameters_fn = getattr(self.policy, "named_parameters", None)
        if not callable(named_parameters_fn):
            raise AttributeError("wrapped policy must provide callable named_parameters")
        return named_parameters_fn()

    def sample_chunk(self, obs: Any, train: bool) -> tuple[Any, Any, Any, Any]:
        batch = self._move_obs_to_device(obs)
        context = nullcontext() if train else _no_grad_context()
        with context:
            sample_result = self.policy.sac_sample_action_chunk(
                batch,
                train=train,
                rollout_noise_std=self.rollout_noise_std,
                train_noise_std=self.train_noise_std,
            )
        return self._format_sample_result(sample_result)

    def sample_chunk_with_kl(self, obs: Any) -> tuple[Any, Any, Any, Any, Any]:
        """Sample from the live actor and estimate KL to the frozen phase-start policy.

        The estimate is the likelihood ratio over the complete stochastic flow
        trajectory, normalized per denoising state dimension.  The reference policy's
        parameters are frozen, but its score retains gradients through the supplied
        trajectory to the live actor.
        """

        if self.reference_policy is None:
            raise RuntimeError("KL sampling requires a frozen reference_policy.")
        batch = self._move_obs_to_device(obs)
        sample_result = self.policy.sac_sample_action_chunk(
            batch,
            train=True,
            rollout_noise_std=self.rollout_noise_std,
            train_noise_std=self.train_noise_std,
            return_trajectory=True,
        )
        if not isinstance(sample_result, (tuple, list)) or len(sample_result) != 4:
            raise ValueError(
                "policy.sac_sample_action_chunk(return_trajectory=True) must return "
                "(raw_chunk, log_pi, obs_features, trajectory)"
            )
        raw_chunk, log_pi, obs_features, trajectory = sample_result
        flat_actions, log_pi, obs_features, raw_chunk = self._format_sample_result(
            (raw_chunk, log_pi, obs_features)
        )
        self._require_tensor_like(trajectory, "trajectory")
        if trajectory.ndim != 4 or trajectory.shape[1] != flat_actions.shape[0]:
            raise ValueError(
                "trajectory must have shape [steps, batch, chunk, action_dim] with matching batch size, "
                f"got {getattr(trajectory, 'shape', None)}"
            )

        reference_log_pi = self.reference_policy.sac_log_prob_action_trajectory(
            batch,
            trajectory,
            noise_std=self.train_noise_std,
        )
        reference_log_pi = self._format_log_pi(reference_log_pi, batch_size=flat_actions.shape[0])
        normalization = trajectory.shape[0] * trajectory.shape[2] * trajectory.shape[3]
        if normalization <= 0:
            raise ValueError(f"trajectory normalization must be positive, got {normalization}.")
        kl_estimate = (log_pi - reference_log_pi) / float(normalization)
        self._require_tensor_like(kl_estimate, "kl_estimate")
        if not _is_finite(kl_estimate):
            raise RuntimeError("Encountered non-finite normalized SAC flow KL estimate.")
        return flat_actions, log_pi, obs_features, raw_chunk, kl_estimate

    def _format_sample_result(self, sample_result: Any) -> tuple[Any, Any, Any, Any]:
        if not isinstance(sample_result, (tuple, list)) or len(sample_result) != 3:
            raise ValueError("policy.sac_sample_action_chunk must return (raw_chunk, log_pi, obs_features)")
        raw_chunk, log_pi, obs_features = sample_result

        self._require_tensor_like(raw_chunk, "raw_chunk")
        if raw_chunk.ndim != 3:
            raise ValueError("raw_chunk must have shape [batch, chunk, action_dim]")
        if raw_chunk.shape[0] <= 0 or raw_chunk.shape[1] <= 0 or raw_chunk.shape[2] <= 0:
            raise ValueError(
                "raw_chunk must have non-empty shape [batch, chunk, action_dim], "
                f"got shape={tuple(raw_chunk.shape)}"
            )

        # The online MDP replans after one environment step, so Q(s, a) must only
        # receive the action that actually caused reward and next_obs.
        flat_actions = flatten_chunk(raw_chunk[:, : self.critic_action_steps])
        self._require_tensor_like(flat_actions, "flat_actions")
        if flat_actions.ndim != 2:
            raise ValueError("flat_actions must have shape [batch, action_dim]")

        batch_size = flat_actions.shape[0]
        self._require_tensor_like(log_pi, "log_pi")
        log_pi = self._format_log_pi(log_pi, batch_size=batch_size)
        self._require_tensor_like(obs_features, "obs_features")
        self._validate_2d_batch(obs_features, name="obs_features", batch_size=batch_size)
        return flat_actions, log_pi, obs_features, raw_chunk

    def encode_obs(self, obs: Any) -> Any:
        batch = self._move_obs_to_device(obs)
        obs_features = self.policy.sac_encode_observation(batch)
        self._require_tensor_like(obs_features, "obs_features")
        if obs_features.ndim != 2:
            raise ValueError("obs_features must have shape [batch, feature_dim]")
        return obs_features

    @staticmethod
    def _format_log_pi(log_pi: Any, *, batch_size: int) -> Any:
        if getattr(log_pi, "ndim", None) == 1 and log_pi.shape[0] == batch_size:
            return log_pi.reshape(batch_size, 1)
        if (
            getattr(log_pi, "ndim", None) == 2
            and log_pi.shape[0] == batch_size
            and log_pi.shape[1] == 1
        ):
            return log_pi
        raise ValueError(
            "log_pi must have shape [batch] or [batch, 1], "
            f"got shape={getattr(log_pi, 'shape', None)}, expected batch_size={batch_size}"
        )

    @staticmethod
    def _validate_2d_batch(value: Any, *, name: str, batch_size: int) -> None:
        if getattr(value, "ndim", None) != 2:
            raise ValueError(f"{name} must have shape [batch, feature_dim]")
        if value.shape[0] != batch_size:
            raise ValueError(f"{name} batch size must match actions/log_pi batch size")


def _no_grad_context() -> Any:
    import torch

    return torch.no_grad()


def _is_finite(value: Any) -> bool:
    import torch

    return bool(torch.isfinite(value).all())
