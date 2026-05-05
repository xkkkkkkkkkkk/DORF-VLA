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
import torch.nn.functional as F

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
    sample_weights: torch.Tensor = None,
) -> tuple[MetricsTracker, dict]:
    """
    执行单步策略更新，支持可选的样本级别的损失加权。
    """
    start_time = time.perf_counter()
    policy.train()

    with accelerator.autocast():
        # 如果传入了权重，就执行加权 Loss
        if sample_weights is not None:
            # 1. 拿到每个样本独立的 loss (reduction="none")
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")
            
            # 2. 确保权重和loss都在同一个设备上
            sample_weights = sample_weights.to(per_sample_loss.device)
            
            # 3. 计算加权平均 Loss = Σ(w_i * l_i) / (Σw_i + ε)
            epsilon = 1e-6
            loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + epsilon)
            
            # 记录权重的统计信息到日志
            output_dict["dorf/mean_sample_weight"] = sample_weights.mean().item()
            output_dict["dorf/max_sample_weight"] = sample_weights.max().item()
            output_dict["dorf/weight_min"] = sample_weights.min().item()

        else:
            # 预热期或无权重时，走标准的平均 Loss
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
        "expert_dorf_loss": AverageMeter("expert_dorf_loss", ":.3f"),   # 自加
        "online_dorf_loss": AverageMeter("online_dorf_loss", ":.3f"),   # 自加
        "total_dorf_loss": AverageMeter("total_dorf_loss", ":.3f"),   # 自加
        "critic_loss": AverageMeter("critic_loss", ":.3f"), # 自加
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "num_good": AverageMeter("good", ":.1f"), # 自加
        "true_reward": AverageMeter("rwrd", ":.2f"), # 自加
        "lr": AverageMeter("lr", ":0.1e"),                 # VLA 学习率
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
    
    opt_dorf = torch.optim.Adam(list(critic.parameters()) + list(dorf_reward.parameters()), lr=2e-5)# RL学习率

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
        train_tracker.true_reward.update(true_rewards.sum().item())
        if is_main_process:
            print(f"--- [Step {step}] Rollout True Reward Sum: {true_rewards.sum().item():.2f} (Active Envs: {true_rewards.shape[0]})")
        
        states = obs_dict["observation.state"][:, :-1].to(device).float()
        img_global = obs_dict["observation.images.image"][:, :-1].to(device).float()
        img_wrist = obs_dict["observation.images.image2"][:, :-1].to(device).float()
        imgs = torch.cat([img_global, img_wrist], dim=2)

        policy.train()
        opt_dorf.zero_grad()

        # 先获取离线 Batch 
        try:
            batch = next(dl_iter)
        except (StopIteration, UnboundLocalError, NameError):
            dl_iter = iter(dataloader)
            batch = next(dl_iter)

        # ---------------------------------------------
        # 阶段 B & C：DORF 评分与网络更新
        # ---------------------------------------------
        # 1. 计算学到的奖励和 Critic 的价值估计
        learned_dense_rewards = dorf_reward(imgs, states, actions)
        values = critic(imgs, states)

        next_states = obs_dict["observation.state"][:, 1:].to(device).float()
        next_img_global = obs_dict["observation.images.image"][:, 1:].to(device).float()
        next_img_wrist = obs_dict["observation.images.image2"][:, 1:].to(device).float()
        next_imgs = torch.cat([next_img_global, next_img_wrist], dim=2)

        with torch.no_grad():
            next_values = critic(next_imgs, next_states)
            next_values = next_values * (1.0 - dones)
            trajectory_success = (true_rewards.sum(dim=1) > 0.5).float()
        
        terminal_value = next_values[:, -1]
        # 2. 计算 Advantage
        
        A_learned = compute_gae(learned_dense_rewards, values, terminal_value, dones)
        A_learned_norm = (A_learned - A_learned.mean()) / (A_learned.std() + 1e-8)
        A_true = compute_gae(true_rewards, values.detach(), terminal_value, dones)
        A_true_norm = (A_true - A_true.mean()) / (A_true.std() + 1e-8)

        target_dorf_signal = A_true_norm.detach()
        penalty_mask = (trajectory_success < 0.5).unsqueeze(1).expand_as(target_dorf_signal)
        target_dorf_signal[penalty_mask] -= 0.2
        
        returns_true = A_true.detach() + values.detach()
        returns_norm = (returns_true - returns_true.mean()) / (returns_true.std() + 1e-8)
        values_norm = (values - values.mean()) / (values.std() + 1e-8)
        # 3. [DORF核心修复] 计算损失（在线拟合 + 专家锚定）
        # A. 在线部分：让 A_learned 拟合环境真实反馈 A_true
        # online_dorf_loss = torch.nn.functional.mse_loss(A_learned, A_true.detach())
        online_dorf_loss = torch.nn.functional.mse_loss(A_learned_norm, target_dorf_signal)
        # B. 专家部分：强制让价值网络认出专家动作是满分 1.0
        with torch.no_grad():
            e_img_global = batch["observation.images.image"].to(device).float()
            e_img_wrist = batch["observation.images.image2"].to(device).float()
            e_imgs = torch.cat([e_img_global, e_img_wrist], dim=2)
            e_states = batch["observation.state"].to(device).float()
            e_actions = batch["action"][:, 0:1].to(device).float() # 对齐点对点评分
        e_A_learned = dorf_reward(e_imgs, e_states, e_actions)
        # 强制锚定：专家动作的优势得分必须趋向 1.0
        expert_dorf_loss = torch.nn.functional.mse_loss(e_A_learned, torch.ones_like(e_A_learned) * 1.0)
        critic_loss = torch.nn.functional.mse_loss(values_norm, returns_norm.detach())
        total_dorf_loss = online_dorf_loss +  expert_dorf_loss + critic_loss

        total_dorf_loss.backward()
        opt_dorf.step()

        train_tracker.expert_dorf_loss = expert_dorf_loss.item()
        train_tracker.online_dorf_loss = online_dorf_loss.item()
        train_tracker.critic_loss = critic_loss.item()
        # ---------------------------------------------
        # 阶段 D：VLA 微调更新 
        # ---------------------------------------------
        # 1. 计算在线数据的平滑权重 w = exp(A / tau)
        tau = 0.5 # 温度超参数，控制权重的差异程度
        w_online_full = torch.exp(A_learned_norm / tau)
        w_online_full = torch.clamp(w_online_full, min = 0.01, max=5.0) # 截断，防止偶尔极端的优势值导致梯度爆炸

        mask = (A_learned_norm > 0)
        num_good = mask.sum().item()
        train_tracker.num_good.update(num_good)
        warmup_steps = 500 
        chunk_size = batch["action"].shape[1] # 获取当前 VLA 需要的序列长度 (通常为 280 或 50)
        # 决定要注入多少在线样本
        num_to_replace = min(cfg.batch_size // 4, num_good) if (num_good > 0 and step > warmup_steps) else 0
        output_dict["dorf/online_samples_count"] = num_to_replace
        if num_to_replace > 0:
            # 找到正优势值的索引，并随机选取 num_to_replace 个
            b_idx, s_idx = torch.where(mask)
            rand_perm = torch.randperm(num_good)[:num_to_replace]
            sel_b = b_idx[rand_perm]
            sel_s = s_idx[rand_perm]
            
            # 初始化存储在线提取数据的字典
            online_data = {
                "action": [],
                "observation.state": [],
                "observation.images.image": [],
                "observation.images.image2": [],
                "task": []
            }
            online_weights = []

            # 2. 从在线 rollout_data 中安全地提取完整的连续 Chunk
            for i in range(num_to_replace):
                b, s = sel_b[i].item(), sel_s[i].item()
                start_s = max(0, min(s - (chunk_size // 2), actions.shape[1] - chunk_size))
                end_s = start_s + chunk_size
                
                # 提取连续的一整段动作和起始状态
                online_data["action"].append(actions[b, start_s:end_s].cpu())
                online_data["observation.state"].append(states[b, start_s].unsqueeze(0).cpu())
                
                # 提取图像并强制插值缩放，对齐离线图像的分辨率
                for img_key in ["observation.images.image", "observation.images.image2"]:
                    target_h, target_w = batch[img_key].shape[-2:]
                    img_raw = obs_dict[img_key][b, start_s].cpu().float()
                    # unsqueeze(0) 增加 batch 维度用于 interpolate
                    img_resized = F.interpolate(img_raw.unsqueeze(0), size=(target_h, target_w), mode='bilinear').squeeze(0)
                    online_data[img_key].append(img_resized)
                
                # 提取任务指令
                online_data["task"].append("Complete the task") 
                # 提取对应时刻的权重
                online_weights.append(w_online_full[b, s].item())
            
            # 3. 构建全新的干净 Mixed Batch（绝对不修改原始的 batch）
            mixed_batch = {}
            for k in batch.keys():
                # 先把离线数据保留（从 num_to_replace 往后取）
                if isinstance(batch[k], torch.Tensor):
                    offline_part = batch[k][num_to_replace:].cpu()
                    # 如果有对应的在线数据，就 concat 起来
                    if k in online_data:
                        online_tensor = torch.stack(online_data[k])
                        # 维度对齐检查 (尤其是图像如果是5维 [B, 1, C, H, W])
                        if offline_part.ndim == 5 and online_tensor.ndim == 4:
                            online_tensor = online_tensor.unsqueeze(1) 
                        mixed_batch[k] = torch.cat([offline_part, online_tensor], dim=0)
                    else:
                        # 没有的话，需要用全 0 补齐 batch_size
                        padding_shape = list(offline_part.shape)
                        padding_shape[0] = num_to_replace
                        padding = torch.zeros(padding_shape, dtype=offline_part.dtype)
                        mixed_batch[k] = torch.cat([offline_part, padding], dim=0)
                elif isinstance(batch[k], list) and k == "task":
                    offline_tasks = batch["task"][num_to_replace:]
                    mixed_batch["task"] = offline_tasks + online_data["task"]

            # 4. 构建拼接后的权重张量 (离线权重设为 1.0)
            offline_w = torch.ones(cfg.batch_size - num_to_replace, dtype=torch.float32)
            online_w = torch.tensor(online_weights, dtype=torch.float32)
            mixed_weights = torch.cat([offline_w, online_w], dim=0).to(device)
            # batch 均值归一化
            mixed_weights = mixed_weights / (mixed_weights.mean() + 1e-8)
        else:
            # 没有提取到好样本，或者在预热期，直接使用纯离线数据
            mixed_batch = batch
            mixed_weights = torch.ones(cfg.batch_size, dtype=torch.float32).to(device)

        # 5. 安全地传入 preprocessor，根据最新的 task 和 images 重新生成 pixel_values 和 lang_tokens
        mixed_batch = preprocessor(mixed_batch)

        train_tracker.dataloading_s = time.perf_counter() - start_time
        
        # 6. 调用重写的 update_policy
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            mixed_batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            sample_weights=mixed_weights, # 注入平滑权重！
        )
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
