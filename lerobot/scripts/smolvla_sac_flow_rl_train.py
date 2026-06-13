#!/usr/bin/env python

"""RLinf-style SAC-flow post-training for SmolVLA.

This is the first real rewrite path away from the weighted-FM surrogate:

    rollout chunk decisions -> typed replay buffer -> critic/Q updates
    -> actor SAC updates -> alpha updates -> target Q soft updates

The implementation stays inside the existing LeRobot / SmolVLA stack, but its
training semantics now match RLinf's flow-policy SAC loop much more closely
than the earlier return-weighted FM script.
"""

import dataclasses
import logging
import os
import time
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import pad_vector
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.scripts.smolvla_fm_rl_train import (
    RAW_POLICY_ACTION,
    RAW_POLICY_ACTION_STEP,
    configure_dataset_for_fm_rl,
    env_float,
    env_int,
    rollout_with_policy_chunks,
)
from lerobot.scripts.smolvla_fm_rl_utils import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
)
from lerobot.scripts.smolvla_sac_flow_utils import (
    ChunkDecisionStats,
    EntropyTemperature,
    MLPQEnsemble,
    TypedReplayBuffer,
    chunk_discount_factor,
    chunk_reward_sum,
    flatten_action_chunk,
    freeze_module,
    min_q_value,
    soft_update_module,
    unfreeze_module,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, init_logging


REQUIRED_OBS_KEYS = [
    "observation.state",
    "observation.images.image",
    "observation.images.image2",
    OBS_LANGUAGE_TOKENS,
    OBS_LANGUAGE_ATTENTION_MASK,
]


def extract_policy_obs_at_timestep(obs_dict: dict[str, Any], env_idx: int, time_idx: int) -> dict[str, torch.Tensor]:
    missing = [key for key in REQUIRED_OBS_KEYS if key not in obs_dict]
    if missing:
        raise KeyError(f"Rollout observations are missing policy-ready keys: {missing}")
    return {
        key: obs_dict[key][env_idx : env_idx + 1, time_idx].detach().cpu()
        for key in REQUIRED_OBS_KEYS
    }


def validate_transition_action_shape(policy: PreTrainedPolicy, flat_action: torch.Tensor) -> None:
    expected_dim = policy.config.chunk_size * policy.config.max_action_dim
    if flat_action.ndim != 2:
        raise RuntimeError(
            f"Expected flattened replay action shape [1, {expected_dim}], got {tuple(flat_action.shape)}."
        )
    if flat_action.shape[0] != 1 or flat_action.shape[1] != expected_dim:
        raise RuntimeError(
            f"Expected flattened replay action shape [1, {expected_dim}], got {tuple(flat_action.shape)}."
        )


def pad_policy_action_chunk(policy: PreTrainedPolicy, action_chunk: torch.Tensor) -> torch.Tensor:
    """Pad a raw policy chunk back to SmolVLA's max_action_dim layout."""

    padded = pad_vector(action_chunk.unsqueeze(0), policy.config.max_action_dim)
    return padded.detach().cpu()


def build_chunk_transitions_from_rollout(
    *,
    rollout_data: dict[str, Any],
    policy: PreTrainedPolicy,
    gamma: float,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Compress step-level rollout traces into chunk-decision SAC transitions."""

    obs_dict = rollout_data["observation"]
    rewards = rollout_data["reward"].float()
    dones = rollout_data["done"].bool()
    successes = rollout_data.get("success")
    raw_policy_actions = rollout_data[RAW_POLICY_ACTION].float()
    raw_policy_action_steps = rollout_data[RAW_POLICY_ACTION_STEP].long()

    batch_size, total_steps = rewards.shape[:2]
    transitions: list[dict[str, Any]] = []

    for env_idx in range(batch_size):
        decision_starts = [
            step_idx
            for step_idx in range(total_steps)
            if int(raw_policy_action_steps[env_idx, step_idx].item()) == 0
        ]
        if not decision_starts:
            continue

        for decision_idx, start_step in enumerate(decision_starts):
            end_step = decision_starts[decision_idx + 1] if decision_idx + 1 < len(decision_starts) else total_steps
            decision_rewards = rewards[env_idx, start_step:end_step].tolist()
            if not decision_rewards:
                continue

            stats = ChunkDecisionStats()
            for local_step, reward in enumerate(decision_rewards):
                terminal = bool(dones[env_idx, start_step + local_step].item())
                stats.add_step(reward=float(reward), terminal=terminal)

            action_chunk = pad_policy_action_chunk(policy, raw_policy_actions[env_idx, start_step])
            state = extract_policy_obs_at_timestep(obs_dict, env_idx, start_step)
            next_state = extract_policy_obs_at_timestep(obs_dict, env_idx, end_step)
            flat_action = flatten_action_chunk(action_chunk)
            validate_transition_action_shape(policy, flat_action)

            transitions.append(
                {
                    "state": state,
                    "action": flat_action,
                    "next_state": next_state,
                    "reward": chunk_reward_sum(decision_rewards, gamma=gamma),
                    "done": bool(dones[env_idx, end_step - 1].item()),
                    "truncated": False,
                    "discount": chunk_discount_factor(num_steps=stats.horizon, gamma=gamma),
                    "horizon": stats.horizon,
                    "raw_reward_sum": stats.raw_reward_sum,
                }
            )

    flat_success = 0.0
    if successes is not None:
        flat_success = float(successes.bool().any(dim=1).float().mean().item())
    returns = rewards.sum(dim=1)
    stats = {
        "rollout/return_mean": float(returns.mean().item()),
        "rollout/return_std": float(returns.std().item()) if returns.numel() > 1 else 0.0,
        "rollout/success_rate": flat_success,
        "rollout/chunk_transitions": float(len(transitions)),
    }
    return transitions, stats


def save_sac_extra_state(
    checkpoint_dir: Path,
    *,
    q_network,
    target_q_network,
    temperature,
) -> None:
    extra_state_path = checkpoint_dir / "training_state" / "smolvla_sac_flow_state.pt"
    torch.save(
        {
            "q_network": q_network.state_dict(),
            "target_q_network": target_q_network.state_dict(),
            "temperature": temperature.state_dict(),
        },
        extra_state_path,
    )


def load_sac_extra_state(
    checkpoint_dir: Path,
    *,
    q_network,
    target_q_network,
    temperature,
) -> None:
    extra_state_path = checkpoint_dir / "training_state" / "smolvla_sac_flow_state.pt"
    if not extra_state_path.exists():
        logging.warning("No SAC-flow extra state found at %s; keeping fresh critic/alpha state.", extra_state_path)
        return
    payload = torch.load(extra_state_path, map_location="cpu")
    q_network.load_state_dict(payload["q_network"])
    target_q_network.load_state_dict(payload["target_q_network"])
    temperature.load_state_dict(payload["temperature"])


def zero_if_nan(value: float) -> float:
    return 0.0 if value != value else value


def require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Encountered non-finite values in {name}.")


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    init_logging()

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = torch.device(cfg.policy.device if cfg.policy.device is not None else "cuda")
    is_main_process = True

    # RLinf SAC-flow style hyperparameters.
    train_n_envs = env_int("SMOLVLA_SAC_FLOW_N_ENVS", 1)
    replay_capacity = env_int("SMOLVLA_SAC_FLOW_REPLAY_CAPACITY", 200)
    min_buffer_size = env_int("SMOLVLA_SAC_FLOW_MIN_BUFFER_SIZE", 8)
    updates_per_rollout = env_int("SMOLVLA_SAC_FLOW_NUM_UPDATES_PER_ROLLOUT", 32)
    critic_actor_ratio = env_int("SMOLVLA_SAC_FLOW_CRITIC_ACTOR_RATIO", 4)
    critic_hidden_dim = env_int("SMOLVLA_SAC_FLOW_CRITIC_HIDDEN_DIM", 512)
    num_q_heads = env_int("SMOLVLA_SAC_FLOW_NUM_Q_HEADS", 10)
    gamma = env_float("SMOLVLA_SAC_FLOW_GAMMA", 0.99)
    tau = env_float("SMOLVLA_SAC_FLOW_TAU", 0.005)
    initial_alpha = env_float("SMOLVLA_SAC_FLOW_INITIAL_ALPHA", 0.01)
    rollout_noise_std = env_float("SMOLVLA_SAC_FLOW_ROLLOUT_NOISE_STD", 0.02)
    train_noise_std = env_float("SMOLVLA_SAC_FLOW_TRAIN_NOISE_STD", 0.30)
    critic_lr = env_float("SMOLVLA_SAC_FLOW_CRITIC_LR", cfg.optimizer.lr)
    alpha_lr = env_float("SMOLVLA_SAC_FLOW_ALPHA_LR", 3e-4)
    batch_size = env_int("SMOLVLA_SAC_FLOW_BATCH_SIZE", max(1, int(cfg.batch_size)))
    rl_eval_every = cfg.eval_freq

    logging.info(pformat(cfg.to_dict()))
    logging.info(
        "SmolVLA SAC-flow rewrite: n_envs=%s replay_capacity=%s min_buffer=%s updates_per_rollout=%s "
        "critic_actor_ratio=%s num_q_heads=%s gamma=%s tau=%s batch_size=%s",
        train_n_envs,
        replay_capacity,
        min_buffer_size,
        updates_per_rollout,
        critic_actor_ratio,
        num_q_heads,
        gamma,
        tau,
        batch_size,
    )

    wandb_logger = WandBLogger(cfg) if cfg.wandb.enable and cfg.wandb.project else None
    configure_dataset_for_fm_rl(cfg, is_main_process=is_main_process)
    dataset = make_dataset(cfg)

    logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        policy = policy.wrap_with_peft(peft_cli_overrides=dataclasses.asdict(cfg.peft))

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    policy.to(device)
    actor_optimizer, actor_scheduler = make_optimizer_and_scheduler(cfg, policy)
    obs_feature_dim = policy.model.vlm_with_expert.config.text_config.hidden_size
    action_flat_dim = policy.config.chunk_size * policy.config.max_action_dim
    q_network = MLPQEnsemble(
        obs_dim=obs_feature_dim,
        action_dim=action_flat_dim,
        hidden_dim=critic_hidden_dim,
        num_q_heads=num_q_heads,
    ).to(device)
    target_q_network = MLPQEnsemble(
        obs_dim=obs_feature_dim,
        action_dim=action_flat_dim,
        hidden_dim=critic_hidden_dim,
        num_q_heads=num_q_heads,
    ).to(device)
    target_q_network.load_state_dict(q_network.state_dict())
    temperature = EntropyTemperature(initial_alpha=initial_alpha).to(device)
    target_entropy = env_float(
        "SMOLVLA_SAC_FLOW_TARGET_ENTROPY",
        -float(policy.config.chunk_size * policy.config.action_feature.shape[0]),
    )

    critic_optimizer = torch.optim.AdamW(q_network.parameters(), lr=critic_lr, weight_decay=cfg.optimizer.weight_decay)
    alpha_optimizer = torch.optim.Adam([temperature.log_alpha], lr=alpha_lr)
    optimizer_dict = {
        "actor": actor_optimizer,
        "critic": critic_optimizer,
        "alpha": alpha_optimizer,
    }

    step = 0
    if cfg.resume:
        if cfg.checkpoint_path is not None:
            step, optimizer_dict, actor_scheduler = load_training_state(
                cfg.checkpoint_path, optimizer_dict, actor_scheduler
            )
            actor_optimizer = optimizer_dict["actor"]
            critic_optimizer = optimizer_dict["critic"]
            alpha_optimizer = optimizer_dict["alpha"]
            load_sac_extra_state(
                cfg.checkpoint_path,
                q_network=q_network,
                target_q_network=target_q_network,
                temperature=temperature,
            )
        else:
            logging.warning(colored("resume=True but checkpoint_path is None; starting from step 0.", "yellow"))

    envs = make_env(cfg.env, n_envs=train_n_envs, use_async_envs=cfg.eval.use_async_envs)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    suite_name = list(envs.keys())[0]
    suite_task_ids = sorted(envs[suite_name].keys())
    if not suite_task_ids:
        raise ValueError(f"No tasks found in suite {suite_name!r}.")

    replay_buffer = TypedReplayBuffer(capacity=replay_capacity)
    num_learnable_params = sum(param.numel() for param in policy.parameters() if param.requires_grad)
    num_total_params = sum(param.numel() for param in policy.parameters())
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
    logging.info("Rollout suite '%s' task ids: %s", suite_name, suite_task_ids)

    while step < cfg.steps:
        task_id = suite_task_ids[step % len(suite_task_ids)]
        active_env = envs[suite_name][task_id]

        policy.eval()
        rollout_start = time.perf_counter()
        with torch.no_grad():
            rollout_data = rollout_with_policy_chunks(
                env=active_env,
                policy=policy,
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                return_observations=True,
            )
        rollout_s = time.perf_counter() - rollout_start

        transitions, rollout_stats = build_chunk_transitions_from_rollout(
            rollout_data=rollout_data,
            policy=policy,
            gamma=gamma,
        )
        for transition in transitions:
            replay_buffer.add(transition)

        update_metrics: list[dict[str, float]] = []
        update_start = time.perf_counter()
        if len(replay_buffer) >= min_buffer_size:
            for update_idx in range(updates_per_rollout):
                batch = replay_buffer.sample(batch_size=batch_size, device=device)

                with torch.no_grad():
                    next_actions, next_log_prob, next_obs_features = policy.sac_sample_action_chunk(
                        batch["next_state"],
                        train=True,
                        rollout_noise_std=rollout_noise_std,
                        train_noise_std=train_noise_std,
                    )
                    next_actions_flat = flatten_action_chunk(next_actions)
                    require_finite_tensor("next_actions_flat", next_actions_flat)
                    require_finite_tensor("next_log_prob", next_log_prob)
                    require_finite_tensor("next_obs_features", next_obs_features)
                    next_q_values = target_q_network(next_obs_features.detach(), next_actions_flat)
                    require_finite_tensor("next_q_values", next_q_values)
                    next_q = min_q_value(next_q_values)
                    target_value = batch["reward"] + (1.0 - batch["done"]) * batch["discount"] * (
                        next_q - temperature.alpha.detach() * next_log_prob
                    )
                    require_finite_tensor("target_value", target_value)

                    obs_features = policy.sac_encode_observation(batch["state"]).detach()
                    require_finite_tensor("obs_features", obs_features)

                critic_optimizer.zero_grad(set_to_none=True)
                critic_q_values = q_network(obs_features, batch["action"])
                require_finite_tensor("critic_q_values", critic_q_values)
                critic_target = target_value.unsqueeze(0).expand_as(critic_q_values)
                critic_loss = torch.nn.functional.mse_loss(critic_q_values, critic_target)
                require_finite_tensor("critic_loss", critic_loss)
                critic_loss.backward()
                critic_grad_norm = float(torch.nn.utils.clip_grad_norm_(q_network.parameters(), cfg.optimizer.grad_clip_norm))
                critic_optimizer.step()

                actor_loss_value = 0.0
                alpha_loss_value = 0.0
                actor_grad_norm = 0.0
                mean_entropy = 0.0
                mean_log_prob = 0.0

                if update_idx % critic_actor_ratio == 0:
                    actor_optimizer.zero_grad(set_to_none=True)
                    freeze_module(q_network)
                    sampled_actions, log_prob, obs_features = policy.sac_sample_action_chunk(
                        batch["state"],
                        train=True,
                        rollout_noise_std=rollout_noise_std,
                        train_noise_std=train_noise_std,
                    )
                    sampled_actions_flat = flatten_action_chunk(sampled_actions)
                    require_finite_tensor("sampled_actions_flat", sampled_actions_flat)
                    require_finite_tensor("actor_log_prob", log_prob)
                    require_finite_tensor("actor_obs_features", obs_features)
                    q_pi = q_network(obs_features, sampled_actions_flat)
                    require_finite_tensor("q_pi", q_pi)
                    actor_loss = (temperature.alpha.detach() * log_prob - min_q_value(q_pi)).mean()
                    require_finite_tensor("actor_loss", actor_loss)
                    actor_loss.backward()
                    actor_grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.optimizer.grad_clip_norm)
                    )
                    actor_optimizer.step()
                    if actor_scheduler is not None:
                        actor_scheduler.step()
                    unfreeze_module(q_network)

                    alpha_optimizer.zero_grad(set_to_none=True)
                    alpha_loss = -(temperature.alpha * (log_prob.detach() + target_entropy)).mean()
                    require_finite_tensor("alpha_loss", alpha_loss)
                    alpha_loss.backward()
                    alpha_optimizer.step()

                    actor_loss_value = float(actor_loss.detach().item())
                    alpha_loss_value = float(alpha_loss.detach().item())
                    mean_entropy = float((-log_prob).mean().detach().item())
                    mean_log_prob = float(log_prob.mean().detach().item())

                soft_update_module(q_network, target_q_network, tau=tau)

                update_metrics.append(
                    {
                        "train/critic_loss": float(critic_loss.detach().item()),
                        "train/critic_grad_norm": zero_if_nan(critic_grad_norm),
                        "train/actor_loss": actor_loss_value,
                        "train/actor_grad_norm": zero_if_nan(actor_grad_norm),
                        "train/alpha_loss": alpha_loss_value,
                        "train/alpha": float(temperature.alpha.detach().item()),
                        "train/entropy": mean_entropy,
                        "train/log_prob": mean_log_prob,
                        "train/q_mean": float(critic_q_values.mean().detach().item()),
                    }
                )
        update_s = time.perf_counter() - update_start

        mean_update_metrics = {}
        if update_metrics:
            keys = update_metrics[0].keys()
            mean_update_metrics = {
                key: float(sum(metric[key] for metric in update_metrics) / len(update_metrics))
                for key in keys
            }

        log_metrics = {
            "train/step": step,
            "train/task_id": task_id,
            "train/replay_buffer_size": len(replay_buffer),
            "train/rollout_time_s": rollout_s,
            "train/update_time_s": update_s,
            "train/rollout_task_count": len(suite_task_ids),
            "train/replay_min_ready": float(len(replay_buffer) >= min_buffer_size),
            "train/action_flat_dim": float(action_flat_dim),
            "train/obs_feature_dim": float(obs_feature_dim),
            **rollout_stats,
            **mean_update_metrics,
        }

        logging.info(
            "SAC-flow step=%s task=%s return=%.3f success=%.3f chunks=%.1f replay=%s critic=%.4f actor=%.4f alpha=%.4f",
            step,
            task_id,
            rollout_stats["rollout/return_mean"],
            rollout_stats["rollout/success_rate"],
            rollout_stats["rollout/chunk_transitions"],
            len(replay_buffer),
            mean_update_metrics.get("train/critic_loss", 0.0),
            mean_update_metrics.get("train/actor_loss", 0.0),
            mean_update_metrics.get("train/alpha", float(temperature.alpha.detach().item())),
        )

        if wandb_logger:
            wandb_logger.log_dict(log_metrics, step, mode="train")

        is_save_step = cfg.save_freq > 0 and step % cfg.save_freq == 0
        if is_save_step:
            checkpoint_dir = get_step_checkpoint_dir(Path(cfg.output_dir), cfg.steps, step)
            logging.info("Checkpoint policy after SAC-flow step %s at %s", step, checkpoint_dir)
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                step=step,
                cfg=cfg,
                policy=policy,
                optimizer=optimizer_dict,
                scheduler=actor_scheduler,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
            )
            save_sac_extra_state(
                checkpoint_dir,
                q_network=q_network,
                target_q_network=target_q_network,
                temperature=temperature,
            )
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        is_eval_step = rl_eval_every > 0 and step > 0 and step % rl_eval_every == 0
        if is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info("Eval policy at SAC-flow step %s", step)
            with torch.no_grad():
                eval_info = eval_policy_all(
                    envs=envs,
                    policy=policy,
                    env_preprocessor=env_preprocessor,
                    env_postprocessor=env_postprocessor,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    n_episodes=cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                    max_parallel_tasks=cfg.env.max_parallel_tasks,
                )
            if wandb_logger:
                wandb_logger.log_dict(eval_info, step, mode="eval")

        step += 1

    close_envs(envs)
    logging.info("End of SmolVLA SAC-flow post-training")


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
