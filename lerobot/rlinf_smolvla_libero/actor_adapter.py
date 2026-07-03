from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Any

from lerobot.rlinf_smolvla_libero.replay import flatten_chunk


class SmolVLASACFlowActor:
    """把 SmolVLA policy 的 SAC-Flow 相关方法包装成稳定 actor 接口。"""

    def __init__(self, policy: Any, device: Any, train_noise_std: float, rollout_noise_std: float) -> None:
        self._require_callable(policy, "sac_sample_action_chunk")
        self._require_callable(policy, "sac_encode_observation")
        self.policy = policy
        self.device = device
        self.train_noise_std = train_noise_std
        self.rollout_noise_std = rollout_noise_std

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

        flat_actions = flatten_chunk(raw_chunk)
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
