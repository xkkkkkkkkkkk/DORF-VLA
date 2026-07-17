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


@dataclass
class BatchedChunkRolloutResult:
    """One vector-env chunk execution, represented as one replay transition per env."""

    rollouts: list[ChunkRolloutResult]
    next_obs: dict[str, Any]


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
        if vector_num_envs == 1:
            action = _batched_vector_action(action, action_dim)
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
        # SAC critic/actor 以完整 action chunk 为动作；环境 smoke 可只执行前缀来限制预算。
        actions=flatten_chunk(raw_chunk),
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


def execute_batched_action_chunk(
    env: Any,
    curr_obs: dict[str, Any],
    raw_chunk: Any,
    gamma: float,
    max_chunk_steps: int | None = None,
    action_postprocessor: Callable[[Any], Any] | None = None,
    stop_on_success: bool = False,
    observation_preparer: Callable[..., Any] | None = None,
) -> BatchedChunkRolloutResult:
    """Execute a policy batch in a vector env and split it into independent replay items.

    Each vector slot owns an independent horizon/reward/done signal. This is important:
    a batch of four simultaneous environments is four transitions in replay, not one
    transition whose reward or terminal state has been accidentally aggregated.
    """
    batch_size, chunk_size, action_dim = _validate_batched_raw_chunk(raw_chunk)
    vector_num_envs = _vector_num_envs(env, expected_num_envs=batch_size)
    if vector_num_envs is None and batch_size != 1:
        raise ValueError("A non-vector env only supports raw_chunk batch size 1")

    if max_chunk_steps is None:
        steps_to_run = chunk_size
    else:
        if max_chunk_steps <= 0:
            raise ValueError("max_chunk_steps must be positive")
        steps_to_run = min(int(max_chunk_steps), chunk_size)

    rewards_by_env: list[list[float]] = [[] for _ in range(batch_size)]
    done_by_env = [False] * batch_size
    truncated_by_env = [False] * batch_size
    success_by_env = [False] * batch_size
    active_by_env = [True] * batch_size
    next_raw_obs: Any = None

    for step_idx in range(steps_to_run):
        action = raw_chunk[:, step_idx]
        if action_postprocessor is not None:
            action = action_postprocessor(action)
        if vector_num_envs is not None:
            action = _batched_vector_action(action, action_dim, batch_size=batch_size)

        step_result = env.step(action)
        if len(step_result) != 5:
            raise ValueError("env.step(action) must return (next_obs, reward, done, truncated, info)")
        next_raw_obs, reward, done, truncated, info = step_result
        reward_items = _batch_items(reward, batch_size, name="reward")
        done_items = _batch_items(done, batch_size, name="done")
        truncated_items = _batch_items(truncated, batch_size, name="truncated")

        for env_idx in range(batch_size):
            if not active_by_env[env_idx]:
                continue
            rewards_by_env[env_idx].append(float(reward_items[env_idx]))
            success = _info_success_at(info, env_idx, batch_size)
            terminal = bool(done_items[env_idx]) or (stop_on_success and success)
            timeout = bool(truncated_items[env_idx])
            success_by_env[env_idx] = success_by_env[env_idx] or success
            done_by_env[env_idx] = terminal
            truncated_by_env[env_idx] = timeout
            if terminal or timeout:
                active_by_env[env_idx] = False

        if not any(active_by_env):
            break

    if next_raw_obs is None or any(not rewards for rewards in rewards_by_env):
        raise ValueError("execute_batched_action_chunk must execute at least one step for every env")

    next_obs = _prepare_next_obs(next_raw_obs, env, observation_preparer)
    rollouts: list[ChunkRolloutResult] = []
    for env_idx in range(batch_size):
        raw_rewards = rewards_by_env[env_idx]
        horizon = len(raw_rewards)
        transition = ChunkTransition(
            curr_obs=_slice_batch(curr_obs, env_idx, batch_size),
            actions=flatten_chunk(raw_chunk[env_idx : env_idx + 1]),
            next_obs=_slice_batch(next_obs, env_idx, batch_size),
            rewards=raw_rewards,
            done=done_by_env[env_idx],
            horizon=horizon,
            discount=chunk_discount(horizon=horizon, gamma=gamma),
            chunk_reward=discounted_chunk_reward(raw_rewards, gamma=gamma),
        )
        rollouts.append(
            ChunkRolloutResult(
                transition=transition,
                raw_rewards=raw_rewards,
                success=success_by_env[env_idx],
                truncated=truncated_by_env[env_idx],
            )
        )
    return BatchedChunkRolloutResult(rollouts=rollouts, next_obs=next_obs)


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


def _validate_batched_raw_chunk(raw_chunk: Any) -> tuple[int, int, int]:
    ndim = getattr(raw_chunk, "ndim", None)
    shape = getattr(raw_chunk, "shape", None)
    if ndim != 3 or shape is None or len(shape) != 3:
        raise ValueError("raw_chunk must have shape [batch, chunk, action_dim]")
    batch_size, chunk_size, action_dim = (int(shape[0]), int(shape[1]), int(shape[2]))
    if batch_size <= 0 or chunk_size <= 0 or action_dim <= 0:
        raise ValueError("raw_chunk must have non-empty shape [batch, chunk, action_dim]")
    return batch_size, chunk_size, action_dim


def _vector_num_envs(env: Any, *, expected_num_envs: int = 1) -> int | None:
    if not hasattr(env, "num_envs"):
        return None
    num_envs = int(getattr(env, "num_envs"))
    if num_envs != expected_num_envs:
        raise ValueError(f"VectorEnv num_envs={num_envs} does not match raw_chunk batch size {expected_num_envs}")
    return num_envs


def _batched_vector_action(action: Any, action_dim: int, *, batch_size: int = 1) -> Any:
    shape = getattr(action, "shape", None)
    if shape is not None and len(shape) == 2 and int(shape[0]) == batch_size:
        return action
    if hasattr(action, "reshape"):
        return action.reshape(batch_size, action_dim)
    return [action] if batch_size == 1 else action


def _batch_items(value: Any, batch_size: int, *, name: str) -> list[Any]:
    if isinstance(value, (list, tuple)):
        if len(value) != batch_size:
            raise ValueError(f"VectorEnv {name} output must contain {batch_size} items")
        return list(value)
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 1:
        if int(shape[0]) != batch_size:
            raise ValueError(f"VectorEnv {name} output must have batch size {batch_size}")
        return [value[index] for index in range(batch_size)]
    if batch_size == 1:
        return [value]
    raise ValueError(f"VectorEnv {name} output must have batch size {batch_size}")


def _slice_batch(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, dict):
        return {key: _slice_batch(item, index, batch_size) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_slice_batch(item, index, batch_size) for item in value)
    if isinstance(value, list):
        if len(value) == batch_size:
            return value[index : index + 1]
        return [_slice_batch(item, index, batch_size) for item in value]
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 1 and int(shape[0]) == batch_size:
        return value[index : index + 1]
    return value


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


def _info_success_at(info: Any, index: int, batch_size: int) -> bool:
    if not isinstance(info, dict):
        return False
    for key in ("success", "is_success"):
        if key in info:
            return bool(_batch_items(info[key], batch_size, name=key)[index])

    final_info = info.get("final_info")
    if final_info is None:
        return False
    if isinstance(final_info, dict):
        for key in ("success", "is_success"):
            if key in final_info:
                return bool(_batch_items(final_info[key], batch_size, name=f"final_info.{key}")[index])
        return False
    final_item = _batch_items(final_info, batch_size, name="final_info")[index]
    if isinstance(final_item, dict):
        return bool(final_item.get("success", final_item.get("is_success", False)))
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
