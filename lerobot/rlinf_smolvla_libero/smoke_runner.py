from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from .checkpointing import assert_sac_flow_device_ready, load_sac_flow_checkpoint, save_sac_flow_checkpoint
from .config import SACFlowConfig
from .runtime_builder import build_smolvla_policy_runtime
from .trainable_scope import apply_actor_trainable_scope
from .training_loop import SACFlowOnlineLoop
from .wandb_logger import format_sac_update_metrics


@dataclass(frozen=True)
class SACFlowSmokeConfig:
    """真实 GPU smoke 的硬预算配置，不用于长训练或性能评估。"""

    device: str = "cpu"
    confirm_gpu_smoke: bool = False
    seed: int = 0
    max_train_steps: int = 2
    max_chunk_steps: int = 1
    num_updates_per_step: int = 1
    batch_size: int = 1
    min_buffer_size: int = 1
    replay_capacity: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device:
            raise ValueError(f"device must be a non-empty string, got {self.device!r}.")
        if not self.confirm_gpu_smoke:
            raise RuntimeError("Set confirm_gpu_smoke=True or pass --confirm-gpu-smoke before running SAC-Flow smoke.")

        _require_budget("max_train_steps", self.max_train_steps, maximum=2)
        _require_budget("max_chunk_steps", self.max_chunk_steps, maximum=1)
        _require_budget("num_updates_per_step", self.num_updates_per_step, maximum=1)
        _require_budget("batch_size", self.batch_size, maximum=1)
        _require_budget("min_buffer_size", self.min_buffer_size, maximum=1)
        _require_budget("replay_capacity", self.replay_capacity, maximum=16)


@dataclass(frozen=True)
class SACFlowRunConfig:
    """受控 SAC-Flow 训练配置；不带 smoke 的硬预算上限。"""

    device: str = "cpu"
    seed: int = 0
    max_train_steps: int = 100
    max_chunk_steps: int = 1
    num_updates_per_step: int = 4
    batch_size: int = 2
    min_buffer_size: int = 2
    replay_capacity: int = 64
    num_envs: int = 1
    actor_snapshot_updates: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device:
            raise ValueError(f"device must be a non-empty string, got {self.device!r}.")
        for name in (
            "max_train_steps",
            "max_chunk_steps",
            "num_updates_per_step",
            "batch_size",
            "min_buffer_size",
            "replay_capacity",
            "num_envs",
        ):
            _require_positive_integer(name, getattr(self, name))
        if any(not isinstance(item, int) or item < 1 for item in self.actor_snapshot_updates):
            raise ValueError("actor_snapshot_updates must contain positive integers")


@dataclass(frozen=True)
class SACFlowSmokeResult:
    steps: int
    checkpoint_dir: Path | None
    update_metrics: list[dict[str, float]]


def log_sac_flow_step_results(
    step_results: list[Any],
    *,
    logger: Any,
    replay_buffer: Any,
    start_step: int = 0,
) -> None:
    """把 rollout 与 SAC 更新指标写入 logger。"""
    for offset, result in enumerate(step_results):
        global_step = start_step + offset
        rollouts = getattr(result, "rollouts", ()) or (result.rollout,)
        rollout = rollouts[0]
        transition = rollout.transition
        if len(rollouts) == 1:
            raw_reward = float(sum(rollout.raw_rewards))
            chunk_reward = float(transition.chunk_reward)
            success = 1.0 if rollout.success else 0.0
            chunk_steps: float | int = int(transition.horizon)
        else:
            raw_reward = float(sum(sum(item.raw_rewards) for item in rollouts) / len(rollouts))
            chunk_reward = float(sum(item.transition.chunk_reward for item in rollouts) / len(rollouts))
            success = float(sum(1.0 if item.success else 0.0 for item in rollouts) / len(rollouts))
            chunk_steps = float(sum(item.transition.horizon for item in rollouts) / len(rollouts))
        metrics: dict[str, Any] = {
            "train/global_step": global_step,
            "env/reward": raw_reward,
            "env/discounted_return": chunk_reward,
            "env/success": success,
            "env/chunk_steps": chunk_steps,
            "train/replay_buffer/size": len(replay_buffer),
        }
        if len(rollouts) > 1:
            metrics["env/parallel_envs"] = len(rollouts)
            metrics["train/transitions_collected"] = len(rollouts)
        for update_metric in result.update_metrics:
            metrics.update(format_sac_update_metrics(update_metric))
        logger.log(metrics, step=global_step)


def prepare_policy_observation(
    *,
    raw_obs: Any,
    env: Any,
    env_preprocessor: Callable[[Any], Any],
    policy_preprocessor: Callable[[Any], Any],
    preprocess_observation_fn: Callable[[Any], Any],
    add_envs_task_fn: Callable[[Any, Any], Any],
) -> dict[str, Any]:
    """复用 LeRobot eval 的 observation 处理顺序。"""
    observation = preprocess_observation_fn(raw_obs)
    observation = add_envs_task_fn(env, observation)
    observation = env_preprocessor(observation)
    observation = policy_preprocessor(observation)
    return observation


def select_single_libero_vector_env(envs: Any, *, expected_num_envs: int = 1) -> Any:
    """从 LeRobot env factory 的嵌套返回值中选出唯一 vector env。"""
    leaves = _collect_vector_env_leaves(envs)
    if len(leaves) != 1:
        raise ValueError(f"SAC-Flow smoke requires exactly one vector env, found {len(leaves)}.")
    vector_env = leaves[0]
    num_envs = int(getattr(vector_env, "num_envs", 1))
    if num_envs != expected_num_envs:
        raise ValueError(f"SAC-Flow runner requires num_envs={expected_num_envs}, got {num_envs}.")
    return vector_env


def run_sac_flow_gpu_smoke(
    *,
    train_cfg: Any,
    smoke_cfg: SACFlowSmokeConfig,
    sac_config: SACFlowConfig | None = None,
    logger: Any | None = None,
    build_runtime_fn: Callable[..., Any] = build_smolvla_policy_runtime,
    make_env_fn: Callable[..., Any] | None = None,
    make_env_processors_fn: Callable[..., tuple[Any, Any]] | None = None,
    build_loop_components_fn: Callable[..., Mapping[str, Any]] | None = None,
    loop_cls: type = SACFlowOnlineLoop,
    save_checkpoint_fn: Callable[..., Path] = save_sac_flow_checkpoint,
    close_envs_fn: Callable[[Any], None] | None = None,
    preprocess_observation_fn: Callable[[Any], Any] | None = None,
    add_envs_task_fn: Callable[[Any, Any], Any] | None = None,
) -> SACFlowSmokeResult:
    """运行一次强预算限制的 SmolVLA/LIBERO SAC-Flow smoke。"""
    effective_config = _effective_sac_config(sac_config, smoke_cfg)
    assert_sac_flow_device_ready(effective_config.device)

    runtime = build_runtime_fn(train_cfg, validate_config=True)
    env_cfg = getattr(runtime.train_cfg, "env", getattr(train_cfg, "env", None))
    policy_cfg = getattr(runtime.train_cfg, "policy", getattr(train_cfg, "policy", None))

    if make_env_fn is None:
        from lerobot.envs.factory import make_env as make_env_fn
    if make_env_processors_fn is None:
        from lerobot.envs.factory import make_env_pre_post_processors as make_env_processors_fn
    if preprocess_observation_fn is None:
        from lerobot.envs.utils import preprocess_observation as preprocess_observation_fn
    if add_envs_task_fn is None:
        from lerobot.envs.utils import add_envs_task as add_envs_task_fn
    if close_envs_fn is None:
        from lerobot.envs.utils import close_envs as close_envs_fn

    envs = make_env_fn(env_cfg, n_envs=1, use_async_envs=False)
    try:
        env = select_single_libero_vector_env(envs, expected_num_envs=1)
        env_preprocessor, action_postprocessor = make_env_processors_fn(env_cfg, policy_cfg)

        raw_obs = _reset_vector_env(env, seed=smoke_cfg.seed)
        initial_obs = prepare_policy_observation(
            raw_obs=raw_obs,
            env=env,
            env_preprocessor=env_preprocessor,
            policy_preprocessor=runtime.preprocessor,
            preprocess_observation_fn=preprocess_observation_fn,
            add_envs_task_fn=add_envs_task_fn,
        )

        if build_loop_components_fn is None:
            build_loop_components_fn = build_default_loop_components
        components = dict(
            build_loop_components_fn(
                runtime=runtime,
                sac_config=effective_config,
                initial_obs=initial_obs,
            )
        )

        loop = loop_cls(
            actor=components["actor"],
            env=env,
            replay_buffer=components["replay_buffer"],
            trainer=components["trainer"],
            config=effective_config,
            action_postprocessor=lambda action: postprocess_env_action(
                action,
                policy_postprocessor=runtime.postprocessor,
                env_postprocessor=action_postprocessor,
            ),
            observation_preparer=lambda raw_next_obs, next_env=env: prepare_policy_observation(
                raw_obs=raw_next_obs,
                env=next_env,
                env_preprocessor=env_preprocessor,
                policy_preprocessor=runtime.preprocessor,
                preprocess_observation_fn=preprocess_observation_fn,
                add_envs_task_fn=add_envs_task_fn,
            ),
            max_chunk_steps=smoke_cfg.max_chunk_steps,
            stop_on_success=True,
        )
        step_results = loop.run(initial_obs, num_steps=smoke_cfg.max_train_steps)
        if logger is not None:
            log_sac_flow_step_results(
                step_results,
                logger=logger,
                replay_buffer=components["replay_buffer"],
            )

        checkpoint_dir = save_checkpoint_fn(
            output_dir=_output_dir(runtime.train_cfg, train_cfg),
            step=smoke_cfg.max_train_steps,
            policy=runtime.policy,
            policy_preprocessor=runtime.preprocessor,
            policy_postprocessor=runtime.postprocessor,
            q_network=components["q_network"],
            target_q_network=components["target_q_network"],
            temperature=components["temperature"],
            config=effective_config,
            trainer=components["trainer"],
            replay_buffer=components["replay_buffer"],
            extra_state={"smoke_steps": smoke_cfg.max_train_steps},
        )
        metrics = [metric for result in step_results for metric in result.update_metrics]
        return SACFlowSmokeResult(
            steps=smoke_cfg.max_train_steps,
            checkpoint_dir=checkpoint_dir,
            update_metrics=metrics,
        )
    finally:
        close_envs_fn(envs)


def run_sac_flow_training_run(
    *,
    train_cfg: Any,
    run_cfg: SACFlowRunConfig,
    sac_config: SACFlowConfig | None = None,
    logger: Any | None = None,
    build_runtime_fn: Callable[..., Any] = build_smolvla_policy_runtime,
    make_env_fn: Callable[..., Any] | None = None,
    make_env_processors_fn: Callable[..., tuple[Any, Any]] | None = None,
    build_loop_components_fn: Callable[..., Mapping[str, Any]] | None = None,
    loop_cls: type = SACFlowOnlineLoop,
    save_checkpoint_fn: Callable[..., Path] = save_sac_flow_checkpoint,
    resume_checkpoint: str | Path | None = None,
    override_actor_lr_on_resume: bool = False,
    load_checkpoint_fn: Callable[..., dict[str, Any]] = load_sac_flow_checkpoint,
    close_envs_fn: Callable[[Any], None] | None = None,
    preprocess_observation_fn: Callable[[Any], Any] | None = None,
    add_envs_task_fn: Callable[[Any, Any], Any] | None = None,
) -> SACFlowSmokeResult:
    """运行受控 SAC-Flow 训练，用于 short-run/baseline-run。"""
    effective_config = _effective_sac_config_from_run(sac_config, run_cfg)
    assert_sac_flow_device_ready(effective_config.device)
    if resume_checkpoint is None:
        _seed_sac_flow_rng(run_cfg.seed)

    runtime = build_runtime_fn(train_cfg, validate_config=True)
    env_cfg = getattr(runtime.train_cfg, "env", getattr(train_cfg, "env", None))
    policy_cfg = getattr(runtime.train_cfg, "policy", getattr(train_cfg, "policy", None))

    if make_env_fn is None:
        from lerobot.envs.factory import make_env as make_env_fn
    if make_env_processors_fn is None:
        from lerobot.envs.factory import make_env_pre_post_processors as make_env_processors_fn
    if preprocess_observation_fn is None:
        from lerobot.envs.utils import preprocess_observation as preprocess_observation_fn
    if add_envs_task_fn is None:
        from lerobot.envs.utils import add_envs_task as add_envs_task_fn
    if close_envs_fn is None:
        from lerobot.envs.utils import close_envs as close_envs_fn

    envs = make_env_fn(env_cfg, n_envs=run_cfg.num_envs, use_async_envs=False)
    try:
        env = select_single_libero_vector_env(envs, expected_num_envs=run_cfg.num_envs)
        env_preprocessor, action_postprocessor = make_env_processors_fn(env_cfg, policy_cfg)

        raw_obs = _reset_vector_env(env, seed=run_cfg.seed)
        initial_obs = prepare_policy_observation(
            raw_obs=raw_obs,
            env=env,
            env_preprocessor=env_preprocessor,
            policy_preprocessor=runtime.preprocessor,
            preprocess_observation_fn=preprocess_observation_fn,
            add_envs_task_fn=add_envs_task_fn,
        )

        if build_loop_components_fn is None:
            build_loop_components_fn = build_default_loop_components
        components = dict(
            build_loop_components_fn(
                runtime=runtime,
                sac_config=effective_config,
                initial_obs=initial_obs,
            )
        )
        start_step = 0
        if resume_checkpoint is not None:
            payload = load_checkpoint_fn(
                checkpoint_dir=resume_checkpoint,
                q_network=components["q_network"],
                target_q_network=components["target_q_network"],
                temperature=components["temperature"],
                trainer=components["trainer"],
                replay_buffer=components["replay_buffer"],
                map_location=effective_config.device,
                restore_rng=True,
            )
            start_step = _checkpoint_global_step(payload)
            if override_actor_lr_on_resume:
                _override_optimizer_lr(components["trainer"].actor_optimizer, effective_config.actor_lr)

        loop = loop_cls(
            actor=components["actor"],
            env=env,
            replay_buffer=components["replay_buffer"],
            trainer=components["trainer"],
            config=effective_config,
            action_postprocessor=lambda action: postprocess_env_action(
                action,
                policy_postprocessor=runtime.postprocessor,
                env_postprocessor=action_postprocessor,
            ),
            observation_preparer=lambda raw_next_obs, next_env=env: prepare_policy_observation(
                raw_obs=raw_next_obs,
                env=next_env,
                env_preprocessor=env_preprocessor,
                policy_preprocessor=runtime.preprocessor,
                preprocess_observation_fn=preprocess_observation_fn,
                add_envs_task_fn=add_envs_task_fn,
            ),
            max_chunk_steps=run_cfg.max_chunk_steps,
            stop_on_success=True,
            update_callback=_build_actor_snapshot_callback(
                requested_updates=run_cfg.actor_snapshot_updates,
                start_step=start_step,
                runtime=runtime,
                train_cfg=train_cfg,
                sac_config=effective_config,
                components=components,
                save_checkpoint_fn=save_checkpoint_fn,
            ),
        )
        step_results = loop.run(initial_obs, num_steps=run_cfg.max_train_steps)
        if logger is not None:
            log_sac_flow_step_results(
                step_results,
                logger=logger,
                replay_buffer=components["replay_buffer"],
                start_step=start_step,
            )

        completed_step = start_step + run_cfg.max_train_steps
        checkpoint_dir = save_checkpoint_fn(
            output_dir=_output_dir(runtime.train_cfg, train_cfg),
            step=completed_step,
            policy=runtime.policy,
            policy_preprocessor=runtime.preprocessor,
            policy_postprocessor=runtime.postprocessor,
            q_network=components["q_network"],
            target_q_network=components["target_q_network"],
            temperature=components["temperature"],
            config=effective_config,
            trainer=components["trainer"],
            replay_buffer=components["replay_buffer"],
            extra_state={
                "train_steps": run_cfg.max_train_steps,
                "global_step": completed_step,
                "resumed_from": str(resume_checkpoint) if resume_checkpoint is not None else None,
            },
        )
        metrics = [metric for result in step_results for metric in result.update_metrics]
        return SACFlowSmokeResult(
            steps=completed_step,
            checkpoint_dir=checkpoint_dir,
            update_metrics=metrics,
        )
    finally:
        close_envs_fn(envs)


def build_default_loop_components(*, runtime: Any, sac_config: SACFlowConfig, initial_obs: dict[str, Any]) -> dict[str, Any]:
    """创建真实 actor、critic、trainer 和 replay；维度从一次 policy 采样中推断。"""
    import copy

    import torch

    from .actor_adapter import SmolVLASACFlowActor
    from .critic import EntropyTemperature, MultiQHead
    from .replay import ChunkReplayBuffer
    from .trainer import SACFlowTrainer

    device = torch.device(sac_config.device)
    policy = move_policy_to_device(runtime.policy, device)
    trainable_audit = configure_actor_trainable_scope(policy, sac_config)
    reference_policy = None
    if sac_config.kl_penalty_coef > 0.0:
        # Anchor each fresh/resumed phase to the exact policy loaded at its start.
        # Keep the reference weights frozen while allowing gradients through its
        # trajectory inputs back to the live policy.
        reference_policy = copy.deepcopy(policy).eval()
        for parameter in reference_policy.parameters():
            parameter.requires_grad_(False)
    actor = SmolVLASACFlowActor(
        policy=policy,
        device=device,
        train_noise_std=sac_config.noise_std_train,
        rollout_noise_std=sac_config.noise_std_rollout,
        reference_policy=reference_policy,
    )
    with torch.no_grad():
        flat_actions, _, obs_features, _ = actor.sample_chunk(initial_obs, train=False)

    obs_dim = int(obs_features.shape[1])
    action_dim = int(flat_actions.shape[1])
    q_network = MultiQHead(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=sac_config.hidden_dim,
        num_q_heads=sac_config.num_q_heads,
    ).to(device)
    target_q_network = copy.deepcopy(q_network).to(device)
    temperature = EntropyTemperature(sac_config.initial_alpha).to(device)
    replay_buffer = ChunkReplayBuffer(capacity=sac_config.replay_capacity, seed=0)
    trainer = SACFlowTrainer(
        actor=actor,
        q_network=q_network,
        target_q_network=target_q_network,
        config=sac_config,
        temperature=temperature,
    )
    return {
        "actor": actor,
        "replay_buffer": replay_buffer,
        "trainer": trainer,
        "q_network": q_network,
        "target_q_network": target_q_network,
        "temperature": temperature,
        "trainable_audit": trainable_audit,
        "reference_policy": reference_policy,
    }


def _checkpoint_global_step(payload: Mapping[str, Any]) -> int:
    step = int(payload.get("step", -1))
    if step < 0:
        raise RuntimeError(f"SAC-Flow checkpoint has invalid step {step}.")
    return step


def _build_actor_snapshot_callback(
    *,
    requested_updates: tuple[int, ...],
    start_step: int,
    runtime: Any,
    train_cfg: Any,
    sac_config: SACFlowConfig,
    components: Mapping[str, Any],
    save_checkpoint_fn: Callable[..., Path],
) -> Callable[[dict[str, float], int], None] | None:
    if not requested_updates:
        return None
    pending = set(requested_updates)
    actor_update_count = 0

    def callback(metrics: dict[str, float], collection_step: int) -> None:
        nonlocal actor_update_count
        if "actor_loss" not in metrics:
            return
        actor_update_count += 1
        if actor_update_count not in pending:
            return
        pending.remove(actor_update_count)
        checkpoint_step = start_step + collection_step + 1
        save_checkpoint_fn(
            output_dir=_output_dir(runtime.train_cfg, train_cfg),
            step=checkpoint_step,
            policy=runtime.policy,
            policy_preprocessor=runtime.preprocessor,
            policy_postprocessor=runtime.postprocessor,
            q_network=components["q_network"],
            target_q_network=components["target_q_network"],
            temperature=components["temperature"],
            config=sac_config,
            trainer=components["trainer"],
            replay_buffer=components["replay_buffer"],
            extra_state={
                "global_step": checkpoint_step,
                "actor_update_count": actor_update_count,
                "snapshot": True,
            },
        )

    return callback


def _seed_sac_flow_rng(seed: int) -> None:
    """Seed local stochastic components for a fresh SAC-Flow run only."""
    import random

    random.seed(seed)
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
    except ImportError:
        return
    np.random.seed(seed)


def _override_optimizer_lr(optimizer: Any, learning_rate: float) -> None:
    """Apply an explicitly requested actor LR after optimizer-state restoration."""
    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate


def configure_actor_trainable_scope(
    policy: Any,
    sac_config: SACFlowConfig,
    *,
    apply_scope_fn: Callable[..., Any] = apply_actor_trainable_scope,
) -> Any:
    """按 SAC 配置应用 actor 参数训练范围。"""
    return apply_scope_fn(policy, scope=sac_config.actor_train_scope)


def move_policy_to_device(policy: Any, device: Any) -> Any:
    """返回目标设备上的 policy；不修改 frozen runtime 容器。"""
    to_device = getattr(policy, "to", None)
    if callable(to_device):
        return to_device(device)
    return policy


def postprocess_env_action(
    action: Any,
    *,
    policy_postprocessor: Callable[[Any], Any],
    env_postprocessor: Callable[[Any], Any],
) -> Any:
    """按 LeRobot eval 顺序把 policy action 转成 env.step 可用动作。"""
    from lerobot.utils.constants import ACTION

    action = policy_postprocessor(action)
    action_transition = {ACTION: action}
    action_transition = env_postprocessor(action_transition)
    action = action_transition[ACTION]
    if callable(getattr(action, "to", None)):
        action = action.to("cpu")
    if callable(getattr(action, "numpy", None)):
        action = action.numpy()
    return action


def _effective_sac_config(sac_config: SACFlowConfig | None, smoke_cfg: SACFlowSmokeConfig) -> SACFlowConfig:
    base = sac_config if sac_config is not None else SACFlowConfig()
    return replace(
        base,
        device=smoke_cfg.device,
        num_updates_per_step=smoke_cfg.num_updates_per_step,
        batch_size=smoke_cfg.batch_size,
        min_buffer_size=smoke_cfg.min_buffer_size,
        replay_capacity=smoke_cfg.replay_capacity,
    )


def _effective_sac_config_from_run(sac_config: SACFlowConfig | None, run_cfg: SACFlowRunConfig) -> SACFlowConfig:
    base = sac_config if sac_config is not None else SACFlowConfig()
    return replace(
        base,
        device=run_cfg.device,
        num_updates_per_step=run_cfg.num_updates_per_step,
        batch_size=run_cfg.batch_size,
        min_buffer_size=run_cfg.min_buffer_size,
        replay_capacity=run_cfg.replay_capacity,
    )


def _collect_vector_env_leaves(value: Any) -> list[Any]:
    if isinstance(value, Mapping):
        leaves: list[Any] = []
        for item in value.values():
            leaves.extend(_collect_vector_env_leaves(item))
        return leaves
    if hasattr(value, "reset") and (hasattr(value, "step") or hasattr(value, "num_envs")):
        return [value]
    return []


def _reset_vector_env(env: Any, *, seed: int) -> Any:
    reset_result = env.reset(seed=seed)
    if isinstance(reset_result, tuple):
        if len(reset_result) != 2:
            raise ValueError("env.reset(seed=...) must return observation or (observation, info).")
        return reset_result[0]
    return reset_result


def _output_dir(runtime_train_cfg: Any, train_cfg: Any) -> Path:
    output_dir = getattr(runtime_train_cfg, "output_dir", None) or getattr(train_cfg, "output_dir", None)
    if output_dir is None:
        output_dir = Path("outputs") / "sac_flow_smoke"
    return Path(output_dir)


def _require_budget(name: str, value: int, *, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value > maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}], got {value!r}.")


def _require_positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
