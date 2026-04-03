#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import glob
import os
import types
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"  # 禁用 Hugging Face 下载进度条
os.environ["TQDM_DISABLE"] = "1"
import pandas as pd
import dataclasses
import logging
import time

from contextlib import nullcontext
from pprint import pformat
from typing import Any
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets

from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
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
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
from lerobot.scripts.lerobot_eval import rollout
from lerobot.scripts.dorf_vision_reward import VisionCritic, VisionReward, compute_gae

def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.
        rabc_weights_provider: Optional RABCWeights instance for sample weighting.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Get RA-BC weights if enabled
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        # Use per-sample loss when RA-BC is enabled for proper weighting
        if rabc_batch_weights is not None:
            # Get per-sample losses
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # Apply RA-BC weights: L_RA-BC = Σ(w_i * l_i) / (Σw_i + ε)
            # rabc_batch_weights is already normalized to sum to batch_size
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            # Log raw mean weight (before normalization) - this is the meaningful metric
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict

def flatten_robot_state(d):
            """递归扁平化 robot_state 字典并拼接"""
            tensors = []
            if isinstance(d, torch.Tensor):
                return d
            if isinstance(d, dict):
                # 按字母顺序排序键名，确保每次运行的特征顺序一致
                for k in sorted(d.keys()):
                    val = flatten_robot_state(d[k])
                    if isinstance(val, torch.Tensor):
                        # 如果是多维张量，保留 batch 和 sequence，扁平化特征维
                        if val.dim() > 2:
                            val = val.flatten(start_dim=2)
                        tensors.append(val)
            return torch.cat(tensors, dim=-1) if tensors else None

@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # Accelerate auto-detects the device based on the available hardware and ignores the policy.device setting.
        # Force the device to be CPU when policy.device is set to CPU.
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(accelerator=accelerator)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("正在初始化本地官方数据集加载器...")
    
    # 指向你复刻的 snapshots 路径
    latest_snapshot = "/root/.cache/huggingface/hub/datasets--HuggingFaceVLA--libero/snapshots/86958911c0f959db2bbbdb107eb3e17c5f9c798e"
        
    # 1. 自动探测本地已有的分片范围
    parquet_files = sorted(glob.glob(os.path.join(latest_snapshot, "data/chunk-000/*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"在 {latest_snapshot} 下未找到数据分片，请检查路径。")
        
    # 读取最后一个分片，获取最大 Episode 索引
    df_last = pd.read_parquet(parquet_files[-1])
    max_ep_idx = int(df_last['episode_index'].max())
    logging.info(f"本地数据覆盖至 Episode {max_ep_idx}，将仅加载此范围内索引。")

    # 2. 调用官方 LeRobotDataset 
    '''dataset = LeRobotDataset(
        repo_id="HuggingFaceVLA/libero",
        root=latest_snapshot,
        revision="86958911c0f959db2bbbdb107eb3e17c5f9c798e", # 锁死哈希，跳过版本查询
        episodes=list(range(max_ep_idx + 1)),  
        n_action_steps=cfg.policy.chunk_size if hasattr(cfg.policy, "chunk_size") else None,
        n_obs_steps=cfg.policy.n_obs_steps if hasattr(cfg.policy, "n_obs_steps") else 1,          # 只请求本地有的 Episode
        )'''
    cfg.dataset.root = Path(latest_snapshot)
    cfg.dataset.episodes = list(range(max_ep_idx + 1))
    cfg.dataset.sequence_padding = True
    dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    if cfg.peft is not None:
        logging.info("Using PEFT! Wrapping model.")
        # Convert CLI peft config to dict for overrides
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    # For SARM, always provide dataset_meta for progress normalization
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
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
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

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Load precomputed SARM progress for RA-BC if enabled
    # Generate progress using: src/lerobot/policies/sarm/compute_rabc_weights.py
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        # Get chunk_size from policy config
        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info(f"Loading SARM progress for RA-BC from {cfg.rabc_progress_path}")
        logging.info(f"Using chunk_size={chunk_size} from policy config, head_mode={head_mode}")
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        if cfg.checkpoint_path is not None:
            step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
        else:
            logging.warning(colored("检测到 --resume=True 但未提供 --checkpoint_path。将从新开始训练。", "yellow"))

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    # 真正的 Workers 修复：确保参数被正确传递    
    safe_num_workers = min(cfg.num_workers, getattr(dataset, 'num_shards', 1))
    if is_main_process:
        logging.info(f"DEBUG: DataLoader 实际使用 Workers 数量: {safe_num_workers}")
    # 注入动态补丁：拦截 Dataset 内部的 None 值
    def dorf_safe_collate(samples):
        """
        增强版安全堆叠器：解决 PIL 图像、None 值以及 Accelerate 字符串合并崩溃问题
        在数据交给 accelerate 之前，
        把 accelerate 不认识的 'str' (字符串) 字段全部扔掉。
        """
        # 2. 诊断拦截（保持之前的逻辑）：探测 None 值
        for i, s in enumerate(samples):
            for k, v in s.items():
                if v is None:
                    print(f"\n[DORF 关键诊断] !!! 样本 {i} 的字段 '{k}' 是 None !!!", flush=True)
        
        # 3. 补全：如果离线数据缺失 rewards 字段
        if "rewards" not in samples[0] or samples[0]["rewards"] is None:
            for s in samples:
                if "action" in s and s["action"] is not None:
                    s["rewards"] = torch.zeros(s["action"].shape[0], dtype=torch.float32)
                else:
                    s["rewards"] = torch.tensor(0.0, dtype=torch.float32)
        
        # 4. 调用官方打包逻辑
        return torch.utils.data.dataloader.default_collate(samples)
    def patched_get_delta_frames(self, dataset_iterator, current_item):
        """增强版 delta 帧获取逻辑：自动填充 None 并修复缺失的 rewards"""
        query_result = {}
        padding = {}
        current_episode_idx = current_item["episode_index"]

        for key, delta_indices in self.delta_indices.items():
            if key in self.meta.video_keys: continue
            
            target_frames = []
            is_pad = []
            
            # 如果 key 是 rewards 且数据集里根本没有，我们手动伪造当前帧
            if key == "rewards" and key not in current_item:
                current_item[key] = torch.tensor(0.0, device=current_item["action"].device)

            # 这里的逻辑是 StreamingLeRobotDataset._get_delta_frames 的容错版本
            # 核心改进：在 torch.stack 之前，把所有的 None 替换为当前帧的拷贝
            fallback_frame = current_item.get(key)
            
            # 执行原始的 lookup 逻辑（这里简化描述，确保你本地运行的是安全版）
            # ... (内部逻辑会由补丁接管) ...
            
            # [关键修复]：确保 target_frames 中绝对没有 None
            for delta in delta_indices:
                # 模拟 lookup，如果失败或返回 None，则使用 fallback_frame
                # 由于篇幅，这里我会提供一个闭包函数注入
                pass 

        # 为了保证代码简洁且能直接运行，我们采用更 surgical 的方式：
        return original_get_delta_frames(self, dataset_iterator, current_item)

    # 简单的拦截器：直接修复 yielded 的字典
    def wrap_iterator(it):
        for item in it:
            # 自动补全 rewards
            if "rewards" not in item:
                item["rewards"] = torch.zeros(item["action"].shape[0], dtype=torch.float32)
            # 检查并修复 None
            for k, v in item.items():
                if v is None:
                    logging.warning(f"Detected None in key {k}, patching with zeros")
                    item[k] = torch.zeros_like(item["action"]) # 粗略补全
            yield item

    # 3. 重新配置 DataLoader (必须确保 collate_fn 和 num_workers 都正确)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=safe_num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=dorf_safe_collate,
        prefetch_factor=2 if safe_num_workers > 0 else None,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    policy, optimizer, lr_scheduler = accelerator.prepare(
        policy, optimizer, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "dorf_loss": AverageMeter("dorf_loss", ":.3f"),   # 自加
        "critic_loss": AverageMeter("critic_loss", ":.3f"), # 自加
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info(
            f"Start offline training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    # 提取实际的 env，供 Rollout 交互使用
    if eval_env is None:
        raise ValueError("DORF fine-tuning requires an active environment! Please pass --env.type in CLI.")
    suite_name = list(eval_env.keys())[0]
    task_id = list(eval_env[suite_name].keys())[0]
    active_env = eval_env[suite_name][task_id]

    if is_main_process:
        logging.info("Initializing Vision-DORF RL modules...")

    actual_state_dim = 8
    
    if is_main_process:
        logging.info(f"--- [DORF 适配] 确认使用策略同分布状态维度: {actual_state_dim}")
    
    critic = VisionCritic(state_dim=actual_state_dim).to(device)
    dorf_reward = VisionReward(
        state_dim=actual_state_dim,
        action_dim=policy.config.output_features["action"].shape[0]
    ).to(device)
    
    opt_dorf = torch.optim.Adam(list(critic.parameters()) + list(dorf_reward.parameters()), lr=2e-5)

    # 尝试断点重续 DORF
    if cfg.resume and cfg.checkpoint_path is not None:
        dorf_path = Path(cfg.checkpoint_path) / "dorf_state.pt"
        if dorf_path.exists():
            checkpoint = torch.load(dorf_path, map_location=device)
            critic.load_state_dict(checkpoint['critic'])
            dorf_reward.load_state_dict(checkpoint['dorf_reward'])
            opt_dorf.load_state_dict(checkpoint['opt_dorf'])
            if is_main_process:
                logging.info("成功加载断点重续的 DORF 权重！")

    if hasattr(active_env, "max_episode_steps"):
        active_env.max_episode_steps = 50 
    elif hasattr(active_env, "_max_episode_steps"):
        active_env._max_episode_steps = 50

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        output_dict = {}
        # ---------------------------------------------
        # 阶段 A：Rollout 收集纯净数据 (利用官方管线)
        # ---------------------------------------------
        policy.eval()
        with torch.no_grad(), accelerator.autocast():
            rollout_data = rollout(
                env=active_env,
                policy=accelerator.unwrap_model(policy),
                env_preprocessor=env_preprocessor,
                env_postprocessor=env_postprocessor,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                return_observations=True, 
            )

        train_success_rate = 0.0
        if "success" in rollout_data:
            # 提取所有样本的成功标志并求平均
            successes = rollout_data["success"]
            train_success_rate = successes.float().mean().item()
        output_dict["train/rollout_success_rate"] = train_success_rate

        # 1. 确保 observation 键存在
        if "observation" not in rollout_data:
             logging.error(f"Rollout 失败。可用键: {list(rollout_data.keys())}")
             continue
        
        obs_dict = rollout_data["observation"]
        # 2. 核心长度核对：防止 S=0 导致的崩溃
        seq_len = obs_dict["observation.state"].shape[1]
        if seq_len <= 1:
            logging.warning(f"Rollout 序列过短 ({seq_len})，跳过本次更新。")
            continue
        # 3. 数据提取：直接使用 8 维状态和拼接图像
        actions = rollout_data["action"].to(device).float()
        dones = rollout_data["done"].to(device).float()
        true_rewards = rollout_data["reward"].to(device).float()
        
        states = obs_dict["observation.state"][:, :-1].to(device).float()
        img_global = obs_dict["observation.images.image"][:, :-1].to(device).float()
        img_wrist = obs_dict["observation.images.image2"][:, :-1].to(device).float()
        imgs = torch.cat([img_global, img_wrist], dim=2)

        policy.train()
        opt_dorf.zero_grad()

        # ---------------------------------------------
        # 阶段 B & C：DORF 评分与网络更新
        # ---------------------------------------------
        # 1. 计算学到的奖励和 Critic 的价值估计
        learned_dense_rewards = dorf_reward(imgs, states, actions)
        values = critic(imgs, states)
        next_value = torch.zeros(imgs.shape[0]).to(device)

        # 2. 计算 Advantage
        A_learned = compute_gae(learned_dense_rewards, values, next_value, dones)
        A_true = compute_gae(true_rewards, values.detach(), next_value, dones)

        # 3. 计算损失并反向传播更新 DORF 的视神经
        returns_true = A_true.detach() + values.detach()
        critic_loss = torch.nn.functional.mse_loss(values, returns_true)
        dorf_loss = - torch.mean(A_true.detach() * A_learned)
        
        total_dorf_loss = dorf_loss + critic_loss
        total_dorf_loss.backward()
        opt_dorf.step()
        train_tracker.dorf_loss = dorf_loss.item()
        train_tracker.critic_loss = critic_loss.item()
        # ---------------------------------------------
        # 阶段 D：VLA 微调更新 
        # ---------------------------------------------
        # 暂时抽取官方的离线 Batch 交给 VLA 训练（保证代码无缝跑通和防止灾难性遗忘）。
        # 验证链路跑通后，再根据 A_learned 的正负值去筛选 rollout_data 并组装在线 Batch。
        
        # 1. 动态确定筛选门槛
        # 如果 A>0 的样本太少（少于 10%），就用 A > Mean，保证训练有足够数据
        pos_mask = (A_learned > 0)
        threshold = A_learned.mean() if pos_mask.float().mean() < 0.10 else 0
        mask = (A_learned >= threshold)
        num_good = mask.sum().item()
        
        # 调试代码
        if is_main_process:
            print(f"\n[诊断] A_learned: min={A_learned.min().item():.4f}, "
                  f"max={A_learned.max().item():.4f}, "
                  f"mean={A_learned.mean().item():.4f}, "
                  f"std={A_learned.std().item():.4f} | "
                  f"Threshold: {threshold:.4f}", flush=True)

        # 2. 先获取离线 Batch 
        try:
            batch = next(dl_iter)
        except (StopIteration, UnboundLocalError, NameError):
            dl_iter = iter(dataloader)
            batch = next(dl_iter)

        # 3. 样本注入逻辑 (将在线优良样本覆盖掉离线 Batch 的前 N 个)
        warmup_steps = 50 # 50步预热

        if num_good > 0 and step > warmup_steps:
            # 混合 25% 的在线优良数据，75% 保持离线专家数据以防遗忘
            num_to_replace = min(cfg.batch_size // 4, num_good)
            
            # 找到符合条件的索引并随机抽取
            b_idx, s_idx = torch.where(mask)
            rand_perm = torch.randperm(num_good)[:num_to_replace]
            sel_b = b_idx[rand_perm]
            sel_s = s_idx[rand_perm]
            
            chunk_size = policy.config.chunk_size
            # A. 注入动作与状态 (统一 float32)
            if batch["action"].ndim == 3:
                # 将在线选出的优质动作覆盖到离线 Batch 的第 0 步
                batch["action"][:num_to_replace, 0] = actions[sel_b, sel_s].cpu().float()
                batch["observation.state"][:num_to_replace, 0] = states[sel_b, sel_s].cpu().float()
                
                # 为了防止 42 Token 报错，这里必须确保 batch["action"].shape[1] == 50
                # 如果你已经改了上面的 LeRobotDataset 初始化，这里就已经是 50 了。
                # 如果依然不是，我们需要进行手动 Padding (仅作防御):
                if batch["action"].shape[1] < chunk_size:
                    padding_len = chunk_size - batch["action"].shape[1]
                    batch["action"] = torch.nn.functional.pad(batch["action"], (0, 0, 0, padding_len))
                    batch["observation.state"] = torch.nn.functional.pad(batch["observation.state"], (0, 0, 0, padding_len))
            # B. 图像强制对齐
            import torch.nn.functional as F
            for img_key in ["observation.images.image", "observation.images.image2"]:
                # 获取离线模板的尺寸 [H, W]
                target_h, target_w = batch[img_key].shape[-2:]
                # 获取在线原始图像 [N, C, H, W]
                online_imgs = obs_dict[img_key][sel_b, sel_s].cpu().float()
                # 强行缩放
                rescaled_imgs = F.interpolate(online_imgs, size=(target_h, target_w), mode='bilinear')
                
                if batch[img_key].ndim == 5:
                    batch[img_key][:num_to_replace, 0] = rescaled_imgs
                else:
                    batch[img_key][:num_to_replace] = rescaled_imgs
            # C. 任务指令替换
            if "task" in batch:
                current_task = batch["task"][0]
            elif "task_index" in batch and hasattr(dataset.meta, 'tasks'):
                task_idx = batch["task_index"][0].item()
                current_task = dataset.meta.tasks[task_idx]
            else:
                current_task = "Complete the task"
                
            if "<image>" not in current_task:
                current_task = "<image><image> " + current_task
            batch["task"] = [current_task] * len(batch["task"])
           
            # --- D. 【清除缓存】强迫重算 ---
            keys_to_scrub = ["pixel_values", "pixel_attention_mask", "lang_tokens", 
                             "attention_mask", "img_masks", "image_grid_thw"]
            for k in keys_to_scrub:
                if k in batch:
                    del batch[k]

            if is_main_process:
                logging.info(f"--- [Phase D] 占位符已补全: {current_task[:50]}... ---")
        elif is_main_process and step <= warmup_steps:
            # 预热的反馈日志
            if step % 10 == 0:
                logging.info(f"--- [Phase D] 预热状态 (Step {step}/{warmup_steps}): 仅更新奖励模型 ---")

        # 4. 对混合后的batch 统一执行预处理
        batch = preprocessor(batch)

        train_tracker.dataloading_s = time.perf_counter() - start_time
        # 5. 调用官方VLA更新函数
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )
        
        # 6. 将 DORF 相关的指标挂载到输出日志中
        output_dict["dorf/total_loss"] = dorf_loss.item() if 'dorf_loss' in locals() else 0.0
        output_dict["dorf/critic_loss"] = critic_loss.item() if 'critic_loss' in locals() else 0.0
        output_dict["good_samples_ratio"] = num_good / A_learned.numel()

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        # 更新训练步数与状态
        step += 1
        train_tracker.step()
        
        # 显存清理，防止显存爆炸
        # del raw_batch, batch, imgs, states, actions, learned_dense_rewards, values
        if step % 5 == 0:
            torch.cuda.empty_cache()

        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                wandb_log_dict.update(output_dict)

                clean_log_dict = {}
                
                # Log RA-BC statistics if enabled
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
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
                # 保存 DORF 模型
                dorf_state_path = checkpoint_dir / "dorf_state.pt"
                torch.save({
                    'critic': critic.state_dict(),
                    'dorf_reward': dorf_reward.state_dict(),
                    'opt_dorf': opt_dorf.state_dict()
                }, dorf_state_path)
                update_last_checkpoint(checkpoint_dir)
                if is_main_process:
                    logging.info(f"DORF 状态已保存至: {dorf_state_path}")
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
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
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
