#!/usr/bin/env python

"""Clean SmolVLA flow-matching RL post-training entry point.

This script is deliberately separate from ``tlofVLA_train.py``.  It implements
the RLinf-aligned baseline first:

    rollout group -> true return -> group-normalized advantage -> weighted FM update

No DORF reward model, critic, stage gate, or offline expert reward is trained in
this file.  DORF should be added back only after this baseline is behaviorally
validated.
"""

from __future__ import annotations

import dataclasses
import glob
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all, rollout
from lerobot.scripts.smolvla_fm_rl_utils import (
    OBS_LANGUAGE_ATTENTION_MASK,
    OBS_LANGUAGE_TOKENS,
    build_weighted_fm_batch,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, has_method, init_logging


def compute_grad_norm(module: torch.nn.Module) -> float:
    grad_norms = [param.grad.detach().norm(2) for param in module.parameters() if param.grad is not None]
    if not grad_norms:
        return 0.0
    return torch.norm(torch.stack(grad_norms), 2).item()


def get_policy_chunk_size(policy: PreTrainedPolicy) -> int:
    for attr in ("chunk_size", "n_action_steps"):
        value = getattr(policy.config, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError("Cannot infer SmolVLA action chunk size from policy.config.")


def move_tensor_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def configure_dataset_for_fm_rl(cfg: TrainPipelineConfig, *, is_main_process: bool) -> None:
    """Apply the minimal dataset setup needed by the current SmolVLA baseline.

    FM-RL policy updates are online, but this LeRobot stack still needs the
    offline dataset metadata for normalization stats and processor construction.
    The existing DORF script uses a local LIBERO snapshot to avoid expensive HF
    downloads and full-parquet materialization. Reuse that convention when the
    path exists, while still allowing CLI ``dataset.root=...`` to override it.
    """

    if cfg.dataset is None:
        raise ValueError("TrainPipelineConfig.dataset is required for metadata/normalization.")

    candidate_roots: list[Path] = []
    if cfg.dataset.root:
        candidate_roots.append(Path(cfg.dataset.root))
    env_root = os.environ.get("LEROBOT_LIBERO_ROOT") or os.environ.get("SMOLVLA_FM_RL_DATA_ROOT")
    if env_root:
        candidate_roots.append(Path(env_root))
    candidate_roots.append(Path("/root/autodl-fs/hf_libero_full"))

    selected_root = next((root for root in candidate_roots if root.exists()), None)
    if selected_root is None:
        if is_main_process:
            logging.warning(
                "No local LIBERO root found in %s; falling back to cfg.dataset as provided.",
                [str(root) for root in candidate_roots],
            )
        return

    parquet_files = sorted(glob.glob(str(selected_root / "data" / "chunk-000" / "*.parquet")))
    if not parquet_files:
        if is_main_process:
            logging.warning(
                "Local dataset root %s exists but contains no data/chunk-000/*.parquet; "
                "falling back to cfg.dataset as provided.",
                selected_root,
            )
        return

    requested_episodes = getattr(cfg.dataset, "episodes", None)
    cfg.dataset.root = selected_root
    cfg.dataset.revision = None
    cfg.dataset.streaming = True
    # Some local LeRobot dataset variants read this dynamic flag.
    cfg.dataset.sequence_padding = True

    if requested_episodes is None:
        cfg.dataset.episodes = None
    if is_main_process:
        logging.info(
            "Using local streaming dataset root %s for metadata/normalization (%s parquet shards).",
            selected_root,
            len(parquet_files),
        )


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an int, got {raw!r}.") from exc


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {raw!r}.") from exc


def env_choice(name: str, default: str, choices: set[str]) -> str:
    raw = os.environ.get(name)
    value = default if raw is None or raw == "" else raw
    if value not in choices:
        raise ValueError(f"Environment variable {name} must be one of {sorted(choices)}, got {value!r}.")
    return value


def build_initial_observation_batch(
    obs_dict: dict[str, torch.Tensor],
    indices: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    required_keys = [
        "observation.state",
        "observation.images.image",
        "observation.images.image2",
        OBS_LANGUAGE_TOKENS,
        OBS_LANGUAGE_ATTENTION_MASK,
    ]
    missing = [key for key in required_keys if key not in obs_dict]
    if missing:
        raise KeyError(f"Rollout observations are missing policy-ready keys: {missing}")

    batch = {
        "observation.state": obs_dict["observation.state"][indices, 0],
        "observation.images.image": obs_dict["observation.images.image"][indices, 0],
        "observation.images.image2": obs_dict["observation.images.image2"][indices, 0],
        OBS_LANGUAGE_TOKENS: obs_dict[OBS_LANGUAGE_TOKENS][indices, 0],
        OBS_LANGUAGE_ATTENTION_MASK: obs_dict[OBS_LANGUAGE_ATTENTION_MASK][indices, 0],
    }
    return move_tensor_batch(batch, device)


@torch.no_grad()
def sample_synthetic_action_chunks(
    policy: PreTrainedPolicy,
    obs_dict: dict[str, torch.Tensor],
    indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Sample FM supervision labels from the current policy at fixed initial observations."""

    was_training = policy.training
    policy.eval()
    policy.reset()
    batch = build_initial_observation_batch(obs_dict, indices, device)
    actions = policy.predict_action_chunk(batch)
    policy.reset()
    if was_training:
        policy.train()
    return actions.detach()


def build_entries_from_rollout(
    *,
    rollout_data: dict[str, Any],
    policy: PreTrainedPolicy,
    device: torch.device,
    group_id: int,
    label_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Convert one rollout result into weighted-FM candidate entries.

    ``label_mode='rollout_prefix'`` trains on the initial action prefix returned
    by ``lerobot_eval.rollout``. Use it only if those actions are confirmed to be
    in the same normalized action space expected by ``policy.forward``.

    ``label_mode='synthetic'`` is the safe default for the current SmolVLA
    baseline: rollout provides only the return signal, while labels are sampled
    in policy/Fm space at the fixed initial observation via
    ``predict_action_chunk``.
    """

    if "observation" not in rollout_data:
        raise KeyError(f"rollout_data has no 'observation' key; available keys={list(rollout_data.keys())}")
    if "action" not in rollout_data:
        raise KeyError(f"rollout_data has no 'action' key; available keys={list(rollout_data.keys())}")
    if "reward" not in rollout_data:
        raise KeyError(f"rollout_data has no 'reward' key; available keys={list(rollout_data.keys())}")

    obs_dict = rollout_data["observation"]
    rollout_actions = rollout_data["action"].to(device).float()
    rewards = rollout_data["reward"].to(device).float()
    returns = rewards.sum(dim=1)
    batch_size = int(rollout_actions.shape[0])
    indices = torch.arange(batch_size, device=device)
    chunk_size = get_policy_chunk_size(policy)

    if label_mode == "rollout_prefix":
        action_labels = rollout_actions[:, :chunk_size]
    elif label_mode == "synthetic":
        action_labels = sample_synthetic_action_chunks(policy, obs_dict, indices, device)
    else:
        raise ValueError(f"Unsupported label_mode={label_mode!r}; use 'rollout_prefix' or 'synthetic'.")

    entries: list[dict[str, Any]] = []
    for idx in range(batch_size):
        action_seq = action_labels[idx]
        valid_len = min(int(action_seq.shape[0]), chunk_size)
        padded_actions = torch.zeros(
            (chunk_size, action_seq.shape[-1]),
            dtype=action_seq.dtype,
            device=action_seq.device,
        )
        padded_actions[:valid_len] = action_seq[:valid_len]
        action_is_pad = torch.ones(chunk_size, dtype=torch.bool, device=action_seq.device)
        action_is_pad[:valid_len] = False

        entry = {
            "action": padded_actions.detach().cpu(),
            "actions_id_pad": action_is_pad.detach().cpu(),
            "observation.state": obs_dict["observation.state"][idx, 0].detach().cpu(),
            "observation.images.image": obs_dict["observation.images.image"][idx, 0].detach().cpu(),
            "observation.images.image2": obs_dict["observation.images.image2"][idx, 0].detach().cpu(),
            OBS_LANGUAGE_TOKENS: obs_dict[OBS_LANGUAGE_TOKENS][idx, 0].detach().cpu(),
            OBS_LANGUAGE_ATTENTION_MASK: obs_dict[OBS_LANGUAGE_ATTENTION_MASK][idx, 0].detach().cpu(),
            "trajectory_return": float(returns[idx].item()),
            "group_id": group_id,
        }
        entries.append(entry)

    success_rate = 0.0
    if "success" in rollout_data:
        success = rollout_data["success"].to(device)
        trajectory_success = success.bool().any(dim=1).float() if success.ndim > 1 else success.float()
        success_rate = float(trajectory_success.mean().item())
    else:
        success_rate = float((returns > 0.5).float().mean().item())

    stats = {
        "rollout_return_mean": float(returns.mean().item()),
        "rollout_return_std": float(returns.std().item()) if returns.numel() > 1 else 0.0,
        "rollout_success_rate": success_rate,
        "rollout_trajectory_count": float(batch_size),
    }
    return entries, stats


def weighted_fm_update(
    *,
    policy: PreTrainedPolicy,
    optimizer: Optimizer,
    accelerator: Accelerator,
    batch: dict[str, Any],
    weights: torch.Tensor,
    micro_batch_size: int,
    grad_clip_norm: float,
    lr_scheduler=None,
    lock=None,
) -> dict[str, float]:
    """Run one weighted flow-matching optimizer step over a logical RL batch."""

    policy.train()
    optimizer.zero_grad()

    batch_size = int(weights.shape[0])
    selected_weight_sum = float(weights.sum().item())
    weighted_loss_sum = 0.0
    micro_grad_norm_sum = 0.0
    micro_batches = 0
    last_output_dict: dict[str, Any] = {}

    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        micro_batch = {
            key: value[start:end]
            for key, value in batch.items()
        }
        micro_batch = move_tensor_batch(micro_batch, accelerator.device)
        micro_weights = weights[start:end].to(accelerator.device)

        with accelerator.autocast():
            per_sample_loss, output_dict = policy.forward(micro_batch, reduction="none")
            micro_weights = micro_weights.to(per_sample_loss.device)
            loss_weighted_sum = (per_sample_loss * micro_weights).sum()
            loss = loss_weighted_sum / (selected_weight_sum + 1e-6)

        accelerator.backward(loss)
        weighted_loss_sum += float(loss_weighted_sum.detach().item())
        micro_grad_norm_sum += compute_grad_norm(policy)
        micro_batches += 1
        last_output_dict = output_dict

    if grad_clip_norm > 0:
        final_grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
        final_grad_norm = float(final_grad_norm.item())
    else:
        final_grad_norm = compute_grad_norm(policy)

    with lock if lock is not None else nullcontext():
        optimizer.step()
    if lr_scheduler is not None:
        lr_scheduler.step()
    optimizer.zero_grad()

    unwrapped = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
    if has_method(unwrapped, "update"):
        unwrapped.update()

    fm_loss = weighted_loss_sum / max(1e-8, selected_weight_sum)
    return {
        "policy/fm_loss": fm_loss,
        "policy/loss": fm_loss,
        "policy/loss_weighted_sum": weighted_loss_sum,
        "policy/loss_denominator": selected_weight_sum,
        "policy/micro_batches": float(micro_batches),
        "policy/micro_grad_norm": micro_grad_norm_sum / max(1, micro_batches),
        "policy/final_grad_norm": final_grad_norm,
        "policy/grad_norm": final_grad_norm,
        "policy/raw_forward_loss": float(last_output_dict.get("loss", fm_loss)),
    }


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    cfg.validate()

    # RLinf-aligned FM-RL baseline hyperparameters. Keep these in one block so
    # ablations are explicit and do not get mixed with future DORF knobs.
    rl_group_size = env_int("SMOLVLA_FM_RL_GROUP_SIZE", 8)
    rl_weight_beta = env_float("SMOLVLA_FM_RL_WEIGHT_BETA", 0.5)
    rl_min_weight = env_float("SMOLVLA_FM_RL_MIN_WEIGHT", 0.05)
    rl_max_weight = env_float("SMOLVLA_FM_RL_MAX_WEIGHT", 5.0)
    rl_accumulation_steps = env_int("SMOLVLA_FM_RL_ACCUMULATION_STEPS", 4)
    # alternative: "rollout_prefix" if rollout actions are confirmed policy-space labels
    rl_label_mode = env_choice("SMOLVLA_FM_RL_LABEL_MODE", "synthetic", {"synthetic", "rollout_prefix"})
    rl_eval_every = cfg.eval_freq

    if rl_group_size <= 0:
        raise ValueError(f"SMOLVLA_FM_RL_GROUP_SIZE must be positive, got {rl_group_size}.")
    if rl_accumulation_steps <= 0:
        raise ValueError(
            f"SMOLVLA_FM_RL_ACCUMULATION_STEPS must be positive, got {rl_accumulation_steps}."
        )
    if rl_min_weight <= 0.0 or rl_max_weight < rl_min_weight:
        raise ValueError(
            "Require 0 < SMOLVLA_FM_RL_MIN_WEIGHT <= SMOLVLA_FM_RL_MAX_WEIGHT; "
            f"got {rl_min_weight} and {rl_max_weight}."
        )

    if accelerator is None:
        accelerator = Accelerator(step_scheduler_with_optimizer=False, cpu=(cfg.policy.device == "cpu"))
    init_logging(accelerator=accelerator)
    is_main_process = accelerator.is_main_process

    if accelerator.num_processes != 1:
        raise RuntimeError(
            "smolvla_fm_rl_train.py currently supports one process. "
            "This keeps rollout grouping and environment ownership explicit; extend it before distributed use."
        )

    if is_main_process:
        logging.info(pformat(cfg.to_dict()))
        logging.info(
            "SmolVLA FM-RL baseline: group_size=%s beta=%s weight_clip=[%s,%s] label_mode=%s",
            rl_group_size,
            rl_weight_beta,
            rl_min_weight,
            rl_max_weight,
            rl_label_mode,
        )

    wandb_logger = WandBLogger(cfg) if cfg.wandb.enable and cfg.wandb.project and is_main_process else None

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if cfg.env is None:
        raise ValueError("FM-RL post-training requires cfg.env; pass an environment config such as LIBERO.")

    if is_main_process:
        logging.info("Creating dataset for metadata and normalization stats")
    configure_dataset_for_fm_rl(cfg, is_main_process=is_main_process)
    dataset = make_dataset(cfg)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        policy = policy.wrap_with_peft(peft_cli_overrides=dataclasses.asdict(cfg.peft))

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats
    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta
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

    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    step = 0
    if cfg.resume:
        if cfg.checkpoint_path is not None:
            step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
        else:
            logging.warning(colored("resume=True but checkpoint_path is None; starting from step 0.", "yellow"))

    if is_main_process:
        logging.info("Creating rollout environments with n_envs=%s", rl_group_size)
    envs = make_env(cfg.env, n_envs=rl_group_size, use_async_envs=cfg.eval.use_async_envs)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    suite_name = list(envs.keys())[0]
    suite_task_ids = sorted(envs[suite_name].keys())
    if not suite_task_ids:
        raise ValueError(f"No tasks found in suite {suite_name!r}.")

    policy, optimizer, lr_scheduler = accelerator.prepare(policy, optimizer, lr_scheduler)

    num_learnable_params = sum(param.numel() for param in policy.parameters() if param.requires_grad)
    num_total_params = sum(param.numel() for param in policy.parameters())
    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
        logging.info("Rollout suite '%s' task ids: %s", suite_name, suite_task_ids)

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "rollout_s": AverageMeter("roll_s", ":.3f"),
        "rollout_return_mean": AverageMeter("ret", ":.3f"),
        "rollout_return_std": AverageMeter("rstd", ":.3f"),
        "rollout_success_rate": AverageMeter("succ", ":.3f"),
        "weight_mean": AverageMeter("w_mu", ":.3f"),
        "weight_max": AverageMeter("w_max", ":.3f"),
        "weight_min": AverageMeter("w_min", ":.3f"),
        "advantage_mean": AverageMeter("a_mu", ":.3f"),
        "advantage_std": AverageMeter("a_std", ":.3f"),
        "policy_micro_batches": AverageMeter("mb", ":.1f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    for _ in range(step, cfg.steps):
        step_start = time.perf_counter()
        task_id = suite_task_ids[step % len(suite_task_ids)]
        active_env = envs[suite_name][task_id]

        policy.eval()
        rollout_start = time.perf_counter()
        with torch.no_grad(), accelerator.autocast():
            rollout_data = rollout(
                env=active_env,
                policy=accelerator.unwrap_model(policy, keep_fp32_wrapper=True),
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                return_observations=True,
            )
        rollout_s = time.perf_counter() - rollout_start

        entries, rollout_stats = build_entries_from_rollout(
            rollout_data=rollout_data,
            policy=accelerator.unwrap_model(policy, keep_fp32_wrapper=True),
            device=device,
            group_id=step,
            label_mode=rl_label_mode,
        )
        if len(entries) != rl_group_size:
            logging.warning(
                "Expected %s rollout entries but got %s. Check env n_envs/grouping.",
                rl_group_size,
                len(entries),
            )

        fm_batch, weights, weight_stats = build_weighted_fm_batch(
            entries,
            beta=rl_weight_beta,
            min_weight=rl_min_weight,
            max_weight=rl_max_weight,
        )
        micro_batch_size = max(1, min(cfg.batch_size, len(entries)))
        if rl_accumulation_steps > 0:
            micro_batch_size = max(1, min(micro_batch_size, (len(entries) + rl_accumulation_steps - 1) // rl_accumulation_steps))

        update_start = time.perf_counter()
        policy_stats = weighted_fm_update(
            policy=policy,
            optimizer=optimizer,
            accelerator=accelerator,
            batch=fm_batch,
            weights=weights,
            micro_batch_size=micro_batch_size,
            grad_clip_norm=cfg.optimizer.grad_clip_norm,
            lr_scheduler=lr_scheduler,
        )
        update_s = time.perf_counter() - update_start

        train_tracker.loss = policy_stats["policy/fm_loss"]
        train_tracker.grad_norm = policy_stats["policy/final_grad_norm"]
        train_tracker.lr = optimizer.param_groups[0]["lr"]
        train_tracker.update_s = update_s
        train_tracker.rollout_s = rollout_s
        train_tracker.rollout_return_mean = rollout_stats["rollout_return_mean"]
        train_tracker.rollout_return_std = rollout_stats["rollout_return_std"]
        train_tracker.rollout_success_rate = rollout_stats["rollout_success_rate"]
        train_tracker.weight_mean = weight_stats["weight_mean"]
        train_tracker.weight_max = weight_stats["weight_max"]
        train_tracker.weight_min = weight_stats["weight_min"]
        train_tracker.advantage_mean = weight_stats["advantage_mean"]
        train_tracker.advantage_std = weight_stats["advantage_std"]
        train_tracker.policy_micro_batches = policy_stats["policy/micro_batches"]
        train_tracker.step()

        output_dict = {
            "train/rl_step_s": time.perf_counter() - step_start,
            "train/rollout_task_id": task_id,
            "train/rollout_task_index": step % len(suite_task_ids),
            "train/rollout_task_count": len(suite_task_ids),
            "train/rollout_s": rollout_s,
            "train/update_s": update_s,
            "train/rollout_return_mean": rollout_stats["rollout_return_mean"],
            "train/rollout_return_std": rollout_stats["rollout_return_std"],
            "train/rollout_success_rate": rollout_stats["rollout_success_rate"],
            "train/rollout_trajectory_count": rollout_stats["rollout_trajectory_count"],
            "train/rl_group_size": len(entries),
            "train/rl_weight_beta": rl_weight_beta,
            "train/rl_label_mode_rollout_prefix": float(rl_label_mode == "rollout_prefix"),
            "train/rl_label_mode_synthetic": float(rl_label_mode == "synthetic"),
            "train/policy_weight_mean": weight_stats["weight_mean"],
            "train/policy_weight_max": weight_stats["weight_max"],
            "train/policy_weight_min": weight_stats["weight_min"],
            "train/policy_advantage_mean": weight_stats["advantage_mean"],
            "train/policy_advantage_std": weight_stats["advantage_std"],
            "train/policy_micro_batches": policy_stats["policy/micro_batches"],
            **{f"train/{key}": value for key, value in policy_stats.items()},
        }

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        if is_log_step and is_main_process:
            logging.info(
                "step=%s task=%s return=%.3f success=%.3f fm_loss=%.4f w_mean=%.3f grad=%.3f",
                step,
                task_id,
                rollout_stats["rollout_return_mean"],
                rollout_stats["rollout_success_rate"],
                policy_stats["policy/fm_loss"],
                weight_stats["weight_mean"],
                policy_stats["policy/final_grad_norm"],
            )
            if wandb_logger:
                wandb_logger.log_dict({**train_tracker.to_dict(), **output_dict}, step)
            train_tracker.reset_averages()

        is_saving_step = cfg.save_freq > 0 and (step % cfg.save_freq == 0 or step == cfg.steps - 1)
        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                logging.info("Checkpoint policy after FM-RL step %s at %s", step, checkpoint_dir)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

        is_eval_step = rl_eval_every > 0 and step % rl_eval_every == 0
        if is_eval_step and is_main_process:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info("Eval policy at FM-RL step %s", step)
            with torch.no_grad(), accelerator.autocast():
                eval_info = eval_policy_all(
                    envs=envs,
                    policy=accelerator.unwrap_model(policy, keep_fp32_wrapper=True),
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

    if is_main_process:
        logging.info("End of SmolVLA FM-RL post-training")
        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
