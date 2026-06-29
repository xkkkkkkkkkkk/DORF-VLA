from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Callable

from .replay import ChunkTransition, chunk_discount, discounted_chunk_reward, flatten_chunk


_BAD_CHUNK_SHAPE = "raw_chunk must have shape [1, chunk, action_dim]"


@dataclass
class ChunkRolloutResult:
    transition: ChunkTransition
    raw_rewards: list[float]
    success: bool
    truncated: bool


def execute_action_chunk(
    env: Any,
    curr_obs: dict[str, Any],
    raw_chunk: Any,
    gamma: float,
    max_chunk_steps: int | None = None,
    action_postprocessor: Callable[[Any], Any] | None = None,
    stop_on_success: bool = False,
    observation_preparer: Callable[..., Any] | None = None,
) -> ChunkRolloutResult:
    """在 dummy/LIBERO-like env 中顺序执行一个 action chunk 的前缀。"""
    _validate_raw_chunk(raw_chunk)
    vector_num_envs = _vector_num_envs(env)

    chunk_size = int(raw_chunk.shape[1])
    action_dim = int(raw_chunk.shape[2])
    if max_chunk_steps is None:
        steps_to_run = chunk_size
    else:
        if max_chunk_steps <= 0:
            raise ValueError("max_chunk_steps must be positive")
        steps_to_run = min(int(max_chunk_steps), chunk_size)

    raw_rewards: list[float] = []
    next_obs = None
    done_result = False
    truncated_result = False
    success_result = False

    for step_idx in range(steps_to_run):
        action = raw_chunk[0, step_idx]
        if action_postprocessor is not None:
            action = action_postprocessor(action)
        if vector_num_envs == 1:
            action = _batched_vector_action(action, action_dim)

        step_result = env.step(action)
        if len(step_result) != 5:
            raise ValueError("env.step(action) must return (next_obs, reward, done, truncated, info)")
        next_obs, reward, done, truncated, info = step_result

        raw_rewards.append(_scalar_float(reward))
        truncated_result = _scalar_bool(truncated)
        success_result = _info_success(info)
        done_result = _scalar_bool(done)
        if stop_on_success and success_result:
            # stop_on_success 表示 wrapper 把成功状态定义为人工终止，success 会写入 transition.done；这不同于 timeout/truncated。
            done_result = True

        if done_result or truncated_result:
            break

    horizon = len(raw_rewards)
    if horizon <= 0 or next_obs is None:
        raise ValueError("execute_action_chunk must execute at least one step")

    next_obs = _prepare_next_obs(next_obs, env, observation_preparer)

    transition = ChunkTransition(
        curr_obs=curr_obs,
        actions=flatten_chunk(raw_chunk[:, :horizon]),
        next_obs=next_obs,
        rewards=raw_rewards,
        done=done_result,
        horizon=horizon,
        discount=chunk_discount(horizon=horizon, gamma=gamma),
        chunk_reward=discounted_chunk_reward(raw_rewards, gamma=gamma),
    )
    return ChunkRolloutResult(
        transition=transition,
        raw_rewards=raw_rewards,
        success=success_result,
        truncated=truncated_result,
    )


def _validate_raw_chunk(raw_chunk: Any) -> None:
    ndim = getattr(raw_chunk, "ndim", None)
    shape = getattr(raw_chunk, "shape", None)
    if ndim != 3 or shape is None:
        raise ValueError(_BAD_CHUNK_SHAPE)
    if len(shape) != 3:
        raise ValueError(_BAD_CHUNK_SHAPE)

    batch_size, chunk_size, action_dim = int(shape[0]), int(shape[1]), int(shape[2])
    if batch_size != 1 or chunk_size <= 0 or action_dim <= 0:
        raise ValueError(_BAD_CHUNK_SHAPE)


def _vector_num_envs(env: Any) -> int | None:
    if not hasattr(env, "num_envs"):
        return None
    num_envs = int(getattr(env, "num_envs"))
    if num_envs != 1:
        raise ValueError(f"VectorEnv num_envs={num_envs} is not supported; only num_envs=1 is supported")
    return num_envs


def _batched_vector_action(action: Any, action_dim: int) -> Any:
    shape = getattr(action, "shape", None)
    if shape is not None and len(shape) == 2 and int(shape[0]) == 1:
        return action
    if hasattr(action, "reshape"):
        return action.reshape(1, action_dim)
    return [action]


def _unwrap_single(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError("VectorEnv num_envs=1 outputs must contain exactly one item")
        return value[0]

    shape = getattr(value, "shape", None)
    if shape is None:
        return value

    size = 1
    for dim in shape:
        size *= int(dim)
    if size != 1:
        return value
    if hasattr(value, "reshape"):
        return value.reshape(-1)[0]
    return value


def _scalar_float(value: Any) -> float:
    return float(_unwrap_single(value))


def _scalar_bool(value: Any) -> bool:
    return bool(_unwrap_single(value))


def _info_success(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    for key in ("success", "is_success"):
        if key in info and _scalar_bool(info[key]):
            return True

    final_info = info.get("final_info")
    final_info = _unwrap_single(final_info) if final_info is not None else None
    if isinstance(final_info, dict) and "is_success" in final_info:
        return _scalar_bool(final_info["is_success"])

    return False


def _prepare_next_obs(next_obs: Any, env: Any, observation_preparer: Callable[..., Any] | None) -> Any:
    if observation_preparer is None:
        return next_obs

    try:
        signature = inspect.signature(observation_preparer)
    except (TypeError, ValueError):
        return observation_preparer(next_obs, env)

    positional_params = [
        param
        for param in signature.parameters.values()
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_varargs = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in signature.parameters.values())
    if has_varargs or len(positional_params) >= 2:
        return observation_preparer(next_obs, env)
    return observation_preparer(next_obs)
