from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from .checkpointing import (
    assert_sac_flow_device_ready,
    capture_rng_state,
    load_sac_flow_checkpoint,
    restore_rng_state,
    save_sac_flow_checkpoint,
)
from .config import SACFlowConfig
from .intervention import FixedSlotInterventionCollector
from .libero_adapter import _info_success_at
from .runtime_builder import build_smolvla_policy_runtime
from .replay import collate_transitions
from .trainable_scope import apply_actor_trainable_scope
from .training_loop import SACFlowCollectionStats, SACFlowOnlineLoop, summarize_rollouts
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
    save_checkpoint: bool = False

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
        if not isinstance(self.save_checkpoint, bool):
            raise ValueError(f"save_checkpoint must be a bool, got {self.save_checkpoint!r}.")


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
    save_checkpoint: bool = False
    # A positive value enables a collection-only, independent Bellman check
    # after training.  These transitions never enter the training replay.
    heldout_num_steps: int = 0
    heldout_seed: int = 2000
    target_valid_pairs: int = 0
    # Root-cause diagnostics are opt-in because they perform additional actor
    # forwards/backwards over a bounded replay snapshot, but never update
    # actor/critic parameters.
    root_cause_diagnostics: bool = False
    root_cause_max_transitions_per_task: int = 32
    root_cause_gradient_repeats: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device:
            raise ValueError(f"device must be a non-empty string, got {self.device!r}.")
        for name in (
            "max_train_steps",
            "num_updates_per_step",
            "batch_size",
            "min_buffer_size",
            "replay_capacity",
            "num_envs",
        ):
            _require_positive_integer(name, getattr(self, name))
        _require_budget("max_chunk_steps", self.max_chunk_steps, maximum=1)
        if any(not isinstance(item, int) or item < 1 for item in self.actor_snapshot_updates):
            raise ValueError("actor_snapshot_updates must contain positive integers")
        if isinstance(self.heldout_num_steps, bool) or not isinstance(self.heldout_num_steps, int) or self.heldout_num_steps < 0:
            raise ValueError("heldout_num_steps must be a non-negative integer")
        if isinstance(self.heldout_seed, bool) or not isinstance(self.heldout_seed, int):
            raise ValueError("heldout_seed must be an integer")
        if isinstance(self.target_valid_pairs, bool) or not isinstance(self.target_valid_pairs, int) or self.target_valid_pairs < 0:
            raise ValueError("target_valid_pairs must be a non-negative integer")
        if not isinstance(self.root_cause_diagnostics, bool):
            raise ValueError("root_cause_diagnostics must be a bool")
        _require_positive_integer(
            "root_cause_max_transitions_per_task",
            self.root_cause_max_transitions_per_task,
        )
        _require_positive_integer("root_cause_gradient_repeats", self.root_cause_gradient_repeats)
        if not isinstance(self.save_checkpoint, bool):
            raise ValueError(f"save_checkpoint must be a bool, got {self.save_checkpoint!r}.")


@dataclass(frozen=True)
class SACFlowSmokeResult:
    steps: int
    checkpoint_dir: Path | None
    update_metrics: list[dict[str, float]]
    collection_stats: dict[str, float] = field(default_factory=dict)
    heldout_metrics: dict[str, float] = field(default_factory=dict)
    critic_metrics: dict[str, float] = field(default_factory=dict)


def log_sac_flow_step_results(
    step_results: list[Any],
    *,
    logger: Any,
    replay_buffer: Any,
    start_step: int = 0,
    include_update_metrics: bool = True,
) -> None:
    """把 rollout 与 SAC 更新指标写入 logger。"""
    collection_stats = SACFlowCollectionStats()
    for offset, result in enumerate(step_results):
        rollouts = getattr(result, "rollouts", ()) or (result.rollout,)
        collection_stats.update(tuple(rollouts))
        global_step = start_step + offset
        metrics = _format_collection_metrics(
            result,
            global_step=global_step,
            replay_size=len(replay_buffer),
        )
        metrics.update(collection_stats.metrics(prefix="env"))
        if include_update_metrics:
            for update_metric in result.update_metrics:
                metrics.update(format_sac_update_metrics(update_metric))
        logger.log(metrics, step=global_step)


def _format_collection_metrics(result: Any, *, global_step: int, replay_size: int) -> dict[str, Any]:
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
        "train/replay_buffer/size": replay_size,
    }
    if len(rollouts) > 1:
        metrics["env/parallel_envs"] = len(rollouts)
        metrics["train/transitions_collected"] = len(rollouts)
    return metrics


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


def select_libero_vector_env(envs: Any, *, expected_num_envs_per_task: int = 1) -> Any:
    """Select one vector env or combine multiple LIBERO task envs.

    ``make_env`` returns one vector env per task id.  The previous training
    path called ``select_single_libero_vector_env`` and therefore either ran a
    genuinely single-task experiment or failed before collection when several
    task ids were requested.  This adapter makes a multi-task replay experiment
    explicit: every child has ``expected_num_envs_per_task`` slots and the
    returned object exposes their concatenation as one vector environment.
    """
    leaves = _collect_vector_env_leaves(envs)
    if not leaves:
        raise ValueError("SAC-Flow training requires at least one vector env.")
    for vector_env in leaves:
        num_envs = int(getattr(vector_env, "num_envs", 1))
        if num_envs != expected_num_envs_per_task:
            raise ValueError(
                "SAC-Flow training requires "
                f"num_envs={expected_num_envs_per_task} per task, got {num_envs}."
            )
    if len(leaves) == 1:
        return leaves[0]
    return MultiTaskVectorEnv(leaves)


class MultiTaskVectorEnv:
    """Minimal synchronous adapter over one LIBERO vector env per task."""

    def __init__(self, vector_envs: list[Any]) -> None:
        if not vector_envs:
            raise ValueError("vector_envs must contain at least one child.")
        self.vector_envs = list(vector_envs)
        self._child_num_envs = [
            int(getattr(vector_env, "num_envs", 1))
            for vector_env in self.vector_envs
        ]
        self.num_envs = sum(self._child_num_envs)
        self.envs = [
            child_env
            for vector_env in self.vector_envs
            for child_env in getattr(vector_env, "envs", [vector_env])
        ]

    def reset(self, seed: int | None = None, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observations: list[Any] = []
        infos: list[Any] = []
        for child_index, vector_env in enumerate(self.vector_envs):
            child_seed = None if seed is None else int(seed) + child_index
            reset_result = vector_env.reset(seed=child_seed, **kwargs)
            if isinstance(reset_result, tuple):
                if len(reset_result) != 2:
                    raise ValueError("child vector env reset must return observation or (observation, info)")
                observation, info = reset_result
            else:
                observation, info = reset_result, {}
            observations.append(observation)
            infos.append(info)
        return _concatenate_batched_values(observations), _concatenate_batched_values(infos)

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, Any]:
        action_batches = _split_batched_value(actions, self._child_num_envs)
        results = [
            vector_env.step(child_actions)
            for vector_env, child_actions in zip(self.vector_envs, action_batches, strict=True)
        ]
        if any(len(result) != 5 for result in results):
            raise ValueError("child vector env step must return (obs, reward, done, truncated, info)")
        observations, rewards, dones, truncateds, infos = zip(*results, strict=True)
        return (
            _concatenate_batched_values(list(observations)),
            _concatenate_batched_values(list(rewards)),
            _concatenate_batched_values(list(dones)),
            _concatenate_batched_values(list(truncateds)),
            _merge_multitask_infos(list(infos), self._child_num_envs),
        )

    def call(self, method_name: str, *args: Any, **kwargs: Any) -> list[Any]:
        values: list[Any] = []
        for vector_env in self.vector_envs:
            child_values = vector_env.call(method_name, *args, **kwargs)
            if isinstance(child_values, (list, tuple)):
                values.extend(child_values)
            else:
                values.append(child_values)
        return values

    def close(self) -> None:
        for vector_env in self.vector_envs:
            close = getattr(vector_env, "close", None)
            if callable(close):
                close()


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

        checkpoint_dir = None
        critic_metrics = _evaluate_current_critic_gate(
            components=components,
            config=effective_config,
            device=effective_config.device,
        )
        if logger is not None and critic_metrics:
            logger.log(critic_metrics, step=None)
        if smoke_cfg.save_checkpoint:
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
                extra_state={
                    "smoke_steps": smoke_cfg.max_train_steps,
                    "critic_metrics": dict(critic_metrics),
                },
            )
        metrics = [metric for result in step_results for metric in result.update_metrics]
        return SACFlowSmokeResult(
            steps=smoke_cfg.max_train_steps,
            checkpoint_dir=checkpoint_dir,
            update_metrics=metrics,
            collection_stats=summarize_rollouts(step_results).metrics(prefix="env"),
            critic_metrics=critic_metrics,
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
        vector_env_leaves = _collect_vector_env_leaves(envs)
        env = select_libero_vector_env(
            envs,
            expected_num_envs_per_task=run_cfg.num_envs,
        )
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
        intervention_collector = None
        if effective_config.critic_intervention_fraction > 0.0:
            intervention_collector = FixedSlotInterventionCollector(
                num_tasks=len(vector_env_leaves),
                slots_per_task=run_cfg.num_envs,
                intervention_fraction=effective_config.critic_intervention_fraction,
                noise_std=effective_config.critic_intervention_noise_std,
                noise_stds=effective_config.critic_intervention_noise_stds or None,
                seed=run_cfg.seed,
                task_indices=_vector_env_task_indices(vector_env_leaves),
                pair_actions=effective_config.critic_intervention_pairing,
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

        logged_collection_steps: set[int] = set()
        logged_collection_stats = SACFlowCollectionStats()

        def log_collection_step(result: Any, collection_step: int) -> None:
            if logger is None:
                return
            rollouts = getattr(result, "rollouts", ()) or (result.rollout,)
            logged_collection_stats.update(tuple(rollouts))
            collection_metrics = logged_collection_stats.metrics(prefix="env")
            logger.log(
                {
                    **_format_collection_metrics(
                        result,
                        global_step=start_step + collection_step,
                        replay_size=len(components["replay_buffer"]),
                    ),
                    **collection_metrics,
                },
                step=None,
            )
            logged_collection_steps.add(collection_step)

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
            update_callback=_build_training_update_callback(
                requested_updates=run_cfg.actor_snapshot_updates,
                start_step=start_step,
                runtime=runtime,
                train_cfg=train_cfg,
                sac_config=effective_config,
                components=components,
                save_checkpoint_fn=save_checkpoint_fn,
                logger=logger,
            ),
            step_callback=log_collection_step if logger is not None else None,
            intervention_collector=intervention_collector,
            collection_seed=run_cfg.seed,
        )
        stop_fn = None
        if run_cfg.target_valid_pairs > 0:
            verified_pair_count = getattr(components["replay_buffer"], "verified_pair_count", None)
            if callable(verified_pair_count):
                stop_fn = lambda _result, _step: verified_pair_count(
                    min_length_gap=effective_config.critic_pairwise_min_length_gap
                ) >= run_cfg.target_valid_pairs
        step_results = loop.run(
            initial_obs,
            num_steps=run_cfg.max_train_steps,
            stop_fn=stop_fn,
        )
        if logger is not None and len(logged_collection_steps) != len(step_results):
            for offset, result in enumerate(step_results):
                if offset in logged_collection_steps:
                    continue
                logger.log(
                    {
                        **_format_collection_metrics(
                            result,
                            global_step=start_step + offset,
                            replay_size=len(components["replay_buffer"]),
                        ),
                        **summarize_rollouts(step_results[: offset + 1]).metrics(prefix="env"),
                    },
                    step=None,
                )

        completed_step = start_step + run_cfg.max_train_steps
        collection_stats = summarize_rollouts(step_results)
        heldout_metrics: dict[str, float] = {}
        if run_cfg.heldout_num_steps > 0:
            heldout_rng_state = capture_rng_state()
            try:
                heldout_metrics = _run_independent_heldout_check(
                    loop=loop,
                    env=env,
                    seed=run_cfg.heldout_seed,
                    num_steps=run_cfg.heldout_num_steps,
                    prepare_initial_obs=lambda raw_obs: prepare_policy_observation(
                        raw_obs=raw_obs,
                        env=env,
                        env_preprocessor=env_preprocessor,
                        policy_preprocessor=runtime.preprocessor,
                        preprocess_observation_fn=preprocess_observation_fn,
                        add_envs_task_fn=add_envs_task_fn,
                    ),
                    trainer=components["trainer"],
                    device=effective_config.device,
                )
            finally:
                restore_rng_state(heldout_rng_state)
            if logger is not None:
                logger.log(heldout_metrics, step=None)

        critic_metrics = _evaluate_current_critic_gate(
            components=components,
            config=effective_config,
            device=effective_config.device,
            include_root_cause_diagnostics=run_cfg.root_cause_diagnostics,
            root_cause_max_transitions_per_task=run_cfg.root_cause_max_transitions_per_task,
            root_cause_gradient_repeats=run_cfg.root_cause_gradient_repeats,
        )
        if logger is not None and critic_metrics:
            logger.log(critic_metrics, step=None)

        checkpoint_dir = None
        if run_cfg.save_checkpoint:
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
                    "collection_stats": collection_stats.state_dict(),
                    "heldout_metrics": dict(heldout_metrics),
                    "critic_metrics": dict(critic_metrics),
                },
            )
        metrics = [metric for result in step_results for metric in result.update_metrics]
        return SACFlowSmokeResult(
            steps=completed_step,
            checkpoint_dir=checkpoint_dir,
            update_metrics=metrics,
            collection_stats=collection_stats.metrics(prefix="env"),
            heldout_metrics=heldout_metrics,
            critic_metrics=critic_metrics,
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
        critic_action_steps=1,
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


def _build_training_update_callback(
    *,
    requested_updates: tuple[int, ...],
    start_step: int,
    runtime: Any,
    train_cfg: Any,
    sac_config: SACFlowConfig,
    components: Mapping[str, Any],
    save_checkpoint_fn: Callable[..., Path],
    logger: Any | None,
) -> Callable[[dict[str, float], int], None] | None:
    """Log every optimizer update immediately and optionally save actor milestones."""
    snapshot_callback = _build_actor_snapshot_callback(
        requested_updates=requested_updates,
        start_step=start_step,
        runtime=runtime,
        train_cfg=train_cfg,
        sac_config=sac_config,
        components=components,
        save_checkpoint_fn=save_checkpoint_fn,
    )
    if snapshot_callback is None and logger is None:
        return None

    def callback(metrics: dict[str, float], collection_step: int) -> None:
        if snapshot_callback is not None:
            snapshot_callback(metrics, collection_step)
        if logger is None:
            return
        payload = format_sac_update_metrics(metrics)
        payload["train/global_step"] = start_step + collection_step
        # Do not reuse collection_step as WandB's internal step: several
        # critic updates occur per collection step and would overwrite each
        # other. The explicit update counters are the chart x-axes.
        logger.log(payload, step=None)

    return callback


def _run_independent_heldout_check(
    *,
    loop: SACFlowOnlineLoop,
    env: Any,
    seed: int,
    num_steps: int,
    prepare_initial_obs: Callable[[Any], dict[str, Any]],
    trainer: Any,
    device: Any,
    evaluation_batch_size: int = 16,
) -> dict[str, float]:
    """Collect fresh episodes and evaluate the frozen critic on them.

    The loop is deliberately switched to ``update=False`` and
    ``store_in_replay=False``.  This makes the split independent in both data
    membership and optimizer updates, unlike taking 20% of an already-trained
    replay buffer.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}.")
    if evaluation_batch_size <= 0:
        raise ValueError(f"evaluation_batch_size must be positive, got {evaluation_batch_size}.")

    # The environment seed alone does not control stochastic SmolVLA action
    # sampling. The caller preserves and restores the training RNG state around
    # this function, so the held-out split can use a fully independent seed.
    _seed_sac_flow_rng(seed)
    raw_obs = _reset_vector_env(env, seed=seed)
    initial_obs = prepare_initial_obs(raw_obs)
    loop.reset_episode_tracking()
    reset_intervention_rng = getattr(loop, "reset_intervention_rng", None)
    if callable(reset_intervention_rng):
        reset_intervention_rng(seed)
    original_step_callback = loop.step_callback
    original_update_callback = loop.update_callback
    loop.step_callback = None
    loop.update_callback = None
    try:
        step_results = loop.run(
            initial_obs,
            num_steps=num_steps,
            store_in_replay=False,
            update=False,
        )
    finally:
        loop.step_callback = original_step_callback
        loop.update_callback = original_update_callback

    stats = summarize_rollouts(step_results)
    metrics = stats.metrics(prefix="heldout")
    rollouts = [
        rollout
        for result in step_results
        for rollout in (getattr(result, "rollouts", ()) or (result.rollout,))
    ]
    transitions = [rollout.transition for rollout in rollouts]
    if not transitions:
        raise RuntimeError("independent held-out collection produced no transitions")

    bellman_batches: list[dict[str, float]] = []
    for start in range(0, len(transitions), evaluation_batch_size):
        batch = collate_transitions(
            transitions[start : start + evaluation_batch_size],
            device=device,
        )
        bellman_batches.append(trainer.evaluate_bellman_error(batch, train=True))

    total_samples = sum(item["sample_count"] for item in bellman_batches)
    for key in ("bellman_mse", "bellman_abs_error", "q_mean", "target_q_mean", "q_head_span"):
        metrics[f"heldout/{key}"] = float(
            sum(item[key] * item["sample_count"] for item in bellman_batches) / total_samples
        )
    metrics["heldout/q_head_span_max"] = float(
        max(item["q_head_span_max"] for item in bellman_batches)
    )
    metrics["heldout/bellman_batches"] = float(len(bellman_batches))
    return metrics


def _evaluate_current_critic_gate(
    *,
    components: Mapping[str, Any],
    config: SACFlowConfig,
    device: Any,
    evaluation_batch_size: int = 16,
    include_root_cause_diagnostics: bool = False,
    root_cause_max_transitions_per_task: int = 32,
    root_cause_gradient_repeats: int = 3,
) -> dict[str, float]:
    """Run the replay-action gate when the concrete replay exposes a snapshot."""
    replay_buffer = components.get("replay_buffer")
    items_fn = getattr(replay_buffer, "items", None)
    if not callable(items_fn):
        return {}
    transitions = list(items_fn())
    if not transitions:
        return {}

    from .critic_diagnostics import (
        evaluate_actor_gradient_compatibility,
        evaluate_critic_action_gate,
        evaluate_critic_root_causes,
        evaluate_intervention_coverage,
        evaluate_pairwise_action_ranking,
    )

    metrics = evaluate_critic_action_gate(
        actor=components["actor"],
        q_network=components["q_network"],
        transitions=transitions,
        device=device,
        agg=config.agg_q,
        batch_size=evaluation_batch_size,
        seed=0,
        max_transitions_per_task=root_cause_max_transitions_per_task,
    )
    output = {f"train/critic_gate/{key}": value for key, value in metrics.items()}
    output.update(
        {
            f"train/intervention/{key}": value
            for key, value in evaluate_intervention_coverage(transitions).items()
        }
    )
    output.update(
        {
            f"train/critic_gate/{key}": value
            for key, value in evaluate_pairwise_action_ranking(
                actor=components["actor"],
                q_network=components["q_network"],
                transitions=transitions,
                device=device,
                agg=config.agg_q,
                margin=config.critic_pairwise_margin,
                min_length_gap=config.critic_pairwise_min_length_gap,
                batch_size=evaluation_batch_size,
            ).items()
        }
    )
    if not include_root_cause_diagnostics:
        return output

    root_cause_metrics = evaluate_critic_root_causes(
        actor=components["actor"],
        q_network=components["q_network"],
        transitions=transitions,
        device=device,
        agg=config.agg_q,
        batch_size=evaluation_batch_size,
        seed=0,
        max_transitions_per_task=root_cause_max_transitions_per_task,
    )
    output.update(
        {
            f"train/critic_root_cause/{key}": value
            for key, value in root_cause_metrics.items()
            if isinstance(value, (int, float))
        }
    )

    actor_gradient_metrics = evaluate_actor_gradient_compatibility(
        actor=components["actor"],
        q_network=components["q_network"],
        transitions=transitions,
        device=device,
        agg=config.actor_agg_q,
        kl_penalty_coef=config.kl_penalty_coef,
        include_kl=config.kl_penalty_coef > 0.0,
        repeats=root_cause_gradient_repeats,
        # Full SmolVLA actor gradients retain substantially more activation
        # memory than critic input diagnostics. Keep this probe small enough to
        # coexist with the frozen reference policy on a single 48 GB GPU.
        max_transitions_per_task=min(root_cause_max_transitions_per_task, 4),
        seed=0,
    )
    output.update(
        {
            f"train/actor_gradient/{key}": value
            for key, value in actor_gradient_metrics.items()
        }
    )
    return output


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


def _vector_env_task_indices(vector_envs: list[Any]) -> tuple[int, ...]:
    task_indices: list[int] = []
    for fallback, vector_env in enumerate(vector_envs):
        children = list(getattr(vector_env, "envs", ()))
        task_index = getattr(children[0], "task_id", fallback) if children else fallback
        task_indices.append(int(task_index))
    return tuple(task_indices)


def _split_batched_value(value: Any, batch_sizes: list[int]) -> list[Any]:
    if len(batch_sizes) == 1:
        return [value]
    total = sum(batch_sizes)
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 1 and int(shape[0]) == total:
        outputs = []
        start = 0
        for batch_size in batch_sizes:
            outputs.append(value[start : start + batch_size])
            start += batch_size
        return outputs
    if isinstance(value, (list, tuple)) and len(value) == total:
        outputs = []
        start = 0
        for batch_size in batch_sizes:
            outputs.append(value[start : start + batch_size])
            start += batch_size
        return outputs
    raise ValueError(f"batched value must have leading batch size {total}")


def _concatenate_batched_values(values: list[Any]) -> Any:
    if not values:
        return values
    first = values[0]
    if isinstance(first, dict):
        keys = set(first)
        if any(set(value) != keys for value in values[1:] if isinstance(value, dict)):
            raise ValueError("child vector env dictionaries must have matching keys")
        return {
            key: _concatenate_batched_values([value[key] for value in values])
            for key in keys
        }
    if isinstance(first, tuple):
        return tuple(
            _concatenate_batched_values([value[index] for value in values])
            for index in range(len(first))
        )
    if isinstance(first, list):
        if all(isinstance(value, list) and len(value) == len(first) for value in values):
            return [
                _concatenate_batched_values([value[index] for value in values])
                for index in range(len(first))
            ]
        output: list[Any] = []
        for value in values:
            output.extend(value if isinstance(value, list) else [value])
        return output
    shape = getattr(first, "shape", None)
    if shape is not None and hasattr(first, "dtype"):
        try:
            import numpy as np

            return np.concatenate(values, axis=0)
        except (ImportError, TypeError, ValueError):
            pass
    try:
        import numpy as np

        return np.concatenate([np.asarray(value) for value in values], axis=0)
    except (ImportError, TypeError, ValueError):
        return values


def _merge_multitask_infos(infos: list[Any], batch_sizes: list[int]) -> dict[str, Any]:
    """Expose only the per-slot success signal across heterogeneous task infos."""
    if len(infos) != len(batch_sizes):
        raise ValueError("info values and batch sizes must have matching lengths")
    success_values: list[bool] = []
    for info, batch_size in zip(infos, batch_sizes, strict=True):
        success_values.extend(
            _info_success_at(info, index, batch_size)
            for index in range(batch_size)
        )
    try:
        import numpy as np

        success_array: Any = np.asarray(success_values, dtype=np.bool_)
    except ImportError:
        success_array = success_values
    return {"is_success": success_array}


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
