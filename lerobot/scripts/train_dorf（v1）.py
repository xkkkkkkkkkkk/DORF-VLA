import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_HTTP2"] = "1"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from reward_model import Reward  # 导入你的 DORF 评估器
from lerobot.scripts.lerobot_eval import make_env
from lerobot.policies.factory import make_policy
from types import SimpleNamespace
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from transformers import AutoProcessor
from torch.utils.tensorboard import SummaryWriter
from lerobot.envs.factory import make_env as lerobot_make_env
from lerobot.envs.configs import LiberoEnv
import torchvision.transforms.functional as TF
    
# ==========================================
#  1.轻量级视觉编码器 (Asymmetric 特征提取)
# ==========================================
class SimpleVisionEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # 强制将任意分辨率压成 4x4
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, output_dim)
        )
    def forward(self, img):
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        # 确保图片是 [B, C, H, W] 格式
        if img.ndim == 5: # 如果带有时间维度 [B, T, C, H, W]
            img = img[:, -1, ...] # 取最新的一帧
        # 将通道调整到正确位置 (LeRobot的图像可能是 HWC，需要 permute)
        if img.shape[-1] == 3: 
            img = img.permute(0, 3, 1, 2)
        return self.cnn(img)

# ==========================================
# 2.Critic 网络 (计算 Advantage)
# ==========================================
class Critic(nn.Module):
    def __init__(self, vision_dim=128, state_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim + state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, vision_features, robot_state):
        x = torch.cat([vision_features, robot_state], dim=-1)
        return self.net(x).squeeze(-1)

def compute_gae(rewards, values, next_value, dones, gamma=0.99, gae_lambda=0.95):
    """标准的 GAE 优势计算"""
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        nextnonterminal = 1.0 - dones[t]
        nextvalues = next_value if t == len(rewards) - 1 else values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages

# ==========================================
# 主流程：DORF + Filtered BC 微调
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("使用 LeRobot 官方 Wrapper 初始化环境...")
    # 使用官方配置类，它会自动处理各种底层包裹逻辑
    env_cfg = LiberoEnv(
        task="libero_goal",
        episode_length=600,
    )
    # 这样直接出来的 envs，不仅支持多线程，而且自带完美的数据字典！
    all_tasks = lerobot_make_env(env_cfg, n_envs=6)
    
    if isinstance(all_tasks, dict):
        first_key = list(all_tasks.keys())[0]
        # 拿到里面真正装有 10 个物理环境的那个大背包！
        all_tasks = all_tasks[first_key]
    else:
        all_tasks = all_tasks

    # 1. 初始化大模型 Actor
    print("加载 smolVLA...")
    policy = SmolVLAPolicy.from_pretrained("HuggingFaceVLA/smolvla_libero").to(device)
    policy.train() 

    # 加载分词器，用于将自然语言指令转化为模型认识的 Tokens
    print("加载自然语言分词器...")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")
    tokenizer = processor.tokenizer
    
    # 2. 初始化 DORF 系统 (Vision + Critic + Reward)
    vision_encoder = SimpleVisionEncoder(output_dim=128).to(device)
    critic = Critic(vision_dim=128, state_dim=8).to(device)
    # DORF 奖励网络 (接收 128维视觉 + 8维本体状态 = 136维 state)
    dorf_reward = Reward(state_dim=136, action_dim=7, hidden_dim=128, encode_dim=64).to(device)
    
    # 优化器
    opt_policy = optim.AdamW(policy.parameters(), lr=1e-5)
    opt_dorf = optim.AdamW(list(vision_encoder.parameters()) + list(critic.parameters()) + list(dorf_reward.parameters()), lr=1e-4)
    
    # ---------------------------------------------
    # 数据翻译：将 Gym 格式转为 smolVLA 格式
    # ---------------------------------------------
    def get_robosuite_obs(env_layer):
        # 如果当前层有这个方法，说明到底了，直接调用
        if hasattr(env_layer, '_get_observations'):
            return env_layer._get_observations()
        # 如果没有，尝试继续往下剥 (Gym 标准是用 .env，部分框架用 ._env)
        elif hasattr(env_layer, 'env'):
            return get_robosuite_obs(env_layer.env)
        elif hasattr(env_layer, '_env'):
            return get_robosuite_obs(env_layer._env)
        else:
            raise RuntimeError("剥壳失败：找不到底层的物理核心！")
        
    def get_real_language_instruction(env_layer):
        if hasattr(env_layer, 'language_instruction'):
            return env_layer.language_instruction
        elif hasattr(env_layer, 'task') and hasattr(env_layer.task, 'language'):
            return env_layer.task.language
        elif hasattr(env_layer, '_env'):
            return get_real_language_instruction(env_layer._env)
        elif hasattr(env_layer, 'env'):
            return get_real_language_instruction(env_layer.env)
        else:
            raise RuntimeError("无法在底层物理引擎中找到真实的语言指令！")

    def convert_obs(raw_obs, envs_instance, tokenizer, device):
        vla_obs = {}
        # 1. 扁平化提取并转换原生 obs 中的嵌套字典数据 (解决 dict 无法转 tensor 的问题)
        def format_img(img_array):
            img_t = torch.from_numpy(img_array) if isinstance(img_array, np.ndarray) else torch.tensor(img_array)
            # 动态适配通道: 将 (H, W, C) 或 (Batch, H, W, C) 转换为 PyTorch 标准的通道在前格式
            if img_t.ndim >= 3 and img_t.shape[-1] == 3: 
                dims = list(range(img_t.ndim))
                dims[-3], dims[-2], dims[-1] = dims[-1], dims[-3], dims[-2]
                img_t = img_t.permute(*dims)
            # 缩放到 0-1 的基础浮点数
            if img_t.max() > 2.0 or img_t.dtype == torch.uint8:
                img_t = img_t.float() / 255.0
            return img_t.to(device)

        pixels = raw_obs.get("pixels", {})
        if "image" in pixels:
            vla_obs["observation.images.image"] = format_img(pixels["image"])
        if "image2" in pixels:
            vla_obs["observation.images.image2"] = format_img(pixels["image2"])

        state_obj = raw_obs.get("robot_state", {})
        if isinstance(state_obj, dict):
            joints_val = state_obj.get('joints', {}).get('pos', [])
            gripper_val = state_obj.get('gripper', {}).get('qpos', [])
            j_t = torch.from_numpy(joints_val) if isinstance(joints_val, np.ndarray) else torch.tensor(joints_val)
            g_t = torch.from_numpy(gripper_val) if isinstance(gripper_val, np.ndarray) else torch.tensor(gripper_val)
            state_t = torch.cat([j_t, g_t[..., :1]], dim=-1)
        else:
            state_t = torch.from_numpy(state_obj) if isinstance(state_obj, np.ndarray) else torch.tensor(state_obj)
        vla_obs["observation.state"] = state_t.float().to(device)

        # 2. 提取语言指令字符串列表
        tasks = []
        for i in range(envs_instance.num_envs if hasattr(envs_instance, 'num_envs') else len(envs_instance.envs)):
            single_env = envs_instance.envs[i].unwrapped
            raw_text = get_real_language_instruction(single_env)
            tasks.append(raw_text if raw_text and raw_text.strip() != "" else "solve the task")
        vla_obs["task"] = tasks

        # 3. 恢复手动分词 (为绕过 select_action 的硬编码 bug)
        batch_instructions = []
        tokenizer.padding_side = "left" 
        
        image_token = "<image>"
        if hasattr(tokenizer, "image_token") and tokenizer.image_token:
            image_token = tokenizer.image_token
            
        has_image2 = "observation.images.image2" in vla_obs
        image_placeholders = f"{image_token}{image_token}" if has_image2 else f"{image_token}"
        
        for raw_text in tasks:
            formatted_text = (
                f"<|im_start|>user\n"
                f"{image_placeholders}What action should the robot take to {raw_text}?<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            batch_instructions.append(formatted_text)

        token_outputs = tokenizer(
            batch_instructions, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=128, 
            truncation=True,
            add_special_tokens=False
        )

        vla_obs['observation.language.tokens'] = token_outputs["input_ids"].to(device)
        vla_obs['observation.language.attention_mask'] = token_outputs["attention_mask"].bool().to(device)
        
        return vla_obs
        '''vla_obs = {}

        pixels = raw_obs["pixels"]
        
        # 1. 处理图像 (自带安全装甲：自动识别 numpy/Tensor，自动转换 0-255 和通道维度)
        # 动态提取大模型预训练时死记硬背的视觉参数
        try:
            target_size = processor.image_processor.size['height']
            image_mean = processor.image_processor.image_mean
            image_std = processor.image_processor.image_std
        except:
            # 兜底：SmolVLM / SigLIP 的绝对标准物理参数
            target_size = 384
            image_mean = [0.5, 0.5, 0.5]
            image_std = [0.5, 0.5, 0.5]


        def format_img(img_array):
            img_t = torch.from_numpy(img_array) if not isinstance(img_array, torch.Tensor) else img_array
            # 探针证实输入是 (Batch, H, W, C) = (6, 256, 256, 3)
            # 模型需要 (Batch, C, H, W) = (6, 3, 256, 256)
            if img_t.ndim == 4 and img_t.shape[-1] == 3:
                img_t = img_t.permute(0, 3, 1, 2)
            # 归一化到 0-1
            if img_t.max() > 2.0 or img_t.dtype == torch.uint8:
                img_t = img_t.float() / 255.0
            # 为大模型植入丢失的 image_transforms
            # 强行缩放到大模型视网膜的绝对分辨率 
            img_t = TF.resize(img_t, [target_size, target_size], antialias=True)
            # 强行打上预训练时的特征滤镜 
            img_t = TF.normalize(img_t, mean=image_mean, std=image_std)    
            return img_t.to(device)

        # 探针证实键名就是 'image' 和 'image2'
        vla_obs["observation.images.image"] = format_img(pixels["image"])
        if "image2" in pixels:
            vla_obs["observation.images.image2"] = format_img(pixels["image2"])

        # 2. 处理状态 (直接精准提取官方字典里的 pos 和 qpos)
        state_obj = raw_obs["robot_state"]
        
        # --- train_dorf.py 的 convert_obs 函数中 ---

        if isinstance(state_obj, dict):
            joints_val = state_obj['joints']['pos']
            gripper_val = state_obj['gripper']['qpos']
            
            j_t = torch.from_numpy(joints_val) if not isinstance(joints_val, torch.Tensor) else joints_val
            g_t = torch.from_numpy(gripper_val) if not isinstance(gripper_val, torch.Tensor) else gripper_val
            
            # 完美的 8 维状态：关节(7) + 夹爪(1)
            state = torch.cat([j_t, g_t[..., :1]], dim=-1)
        else:
            state = torch.from_numpy(state_obj) if not isinstance(state_obj, torch.Tensor) else state_obj

        vla_obs["observation.state"] = state.float().to(device)

        # 3. 处理语言指令 (无视底层渲染引擎，直接暴力硬拼接！)
        batch_instructions = []

        tokenizer.padding_side = "left"

        # 核心探查：获取模型真正认识的图像占位符 ID
        image_token = "<image>"
        if hasattr(tokenizer, "image_token") and tokenizer.image_token :
            image_token = tokenizer.image_token

        image_token_id = tokenizer.convert_tokens_to_ids(image_token)

        has_image2 = "observation.images.image2" in vla_obs
        image_placeholders = f"{image_token}{image_token}" if has_image2 else f"{image_token}"
        
        for i in range(envs_instance.num_envs if hasattr(envs_instance, 'num_envs') else len(envs_instance.envs)):
            single_env = envs_instance.envs[i].unwrapped
            raw_text = get_real_language_instruction(single_env)
            
            # 兜底防空锁：如果环境提取出来的真是空的，强制给它一个动作指令防崩溃
            if not raw_text or raw_text.strip() == "":
                raw_text = "solve the task"
                
            formatted_text = (
                f"<|im_start|>user\n"
                f"{image_placeholders}What action should the robot take to {raw_text}?<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            batch_instructions.append(formatted_text)
            
            # 同时打印出原始文字和拼接后的文字
            if i == 0 and not hasattr(convert_obs, "token_printed"):
                print("\n" + "="*40)
                print(f"[文本对齐核查] 最终喂给大模型的指令:")
                print(f"原始环境文字 (raw_text): '{raw_text}'")
                print(f"最终拼装 Prompt:\n{formatted_text}")
                print("="*40 + "\n")
                convert_obs.text_printed = True

            # 🚨 终极探针：直接透视大模型视网膜的接入点
            if i == 0 and not hasattr(convert_obs, "token_printed"):
                print("\n" + "="*40)
                print("🗣️ [Token 级视觉神经核查]")
                test_tokens = tokenizer(formatted_text, add_special_tokens=False)["input_ids"]
                print(f"模型预期的 Image Token ID: {image_token_id}")
                print(f"实际分词后的 ID 序列: {test_tokens}")
                
                if image_token_id is None or image_token_id not in test_tokens:
                    print("❌ 致命错误：分词器未能识别图像占位符！视觉特征将被完全丢弃！")
                else:
                    print("✅ 占位符 ID 匹配成功！视觉与语言通道已彻底联通！")
                print("="*40 + "\n")
                convert_obs.token_printed = True

        # 执行 Tokenize，将 max_length 从 73 放大到 128
        token_outputs = tokenizer(
            batch_instructions, 
            return_tensors="pt", 
            padding="max_length", 
            max_length=128, 
            truncation=True,
            add_special_tokens= False
        )
        vla_obs['observation.language.tokens'] = token_outputs["input_ids"].to(device)
        vla_obs['observation.language.attention_mask'] = token_outputs["attention_mask"].bool().to(device)
        
        return vla_obs'''

    # 打印当前交互任务
    print("\n" + "="*40)
    print("探查当前 Libero 任务分配...")
    try:
        sample_env = envs.envs[0].unwrapped
        # 提取任务名
        task_name = "未找到任务名"
        if hasattr(sample_env, 'task_name'):
            task_name = sample_env.task_name
        elif hasattr(sample_env, 'task') and hasattr(sample_env.task, 'name'):
            task_name = sample_env.task.name
            
        # 提取自然语言指令 (利用你之前写的提取函数)
        instruction = get_real_language_instruction(sample_env)
        
        print(f"🔹 内部任务名称: {task_name}")
        print(f"🔹 设定的语言指令: '{instruction}'")
    except Exception as e:
        print(f"探查任务信息时遇到阻碍: {e}")
    print("="*40 + "\n")

    # ==========================================
    # 断点重续：加载检测
    # ==========================================
    checkpoint_path = "dorf_vla_checkpoint_latest.pt"
    start_iteration = 1  # 默认从第 1 轮开始
    
    if os.path.exists(checkpoint_path):
        print(f"\n 发现断点文件 '{checkpoint_path}'，正在加载...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 恢复所有模型权重
        policy.load_state_dict(checkpoint['policy'])
        dorf_reward.load_state_dict(checkpoint['dorf_reward'])
        critic.load_state_dict(checkpoint['critic'])
        vision_encoder.load_state_dict(checkpoint['vision_encoder'])
        
        # 恢复优化器状态 (保留学习率等动量信息)
        opt_policy.load_state_dict(checkpoint['opt_policy'])
        opt_dorf.load_state_dict(checkpoint['opt_dorf'])
        
        # 恢复进度
        start_iteration = checkpoint['iteration'] + 1
        print(f" 成功加载断点！从第 {start_iteration} 轮继续训练！\n")
    else:
        print("\n 未发现历史断点，将从新训练 (第 1 轮)。\n")

    # 初始化 TensorBoard，日志存放在 tb_logs 文件夹
    tb_writer = SummaryWriter(log_dir="./tb_logs/dorf_vla_run1")

    for iteration in range(start_iteration, 501):
        print(f"\n========== [第 {iteration} 轮开始] ==========")
        task_list = list(all_tasks.values())
        num_tasks = len(task_list)
        current_task_idx = iteration % num_tasks
        envs = task_list[current_task_idx]
        print(f"🎯 任务调度：本轮分配任务 ID -> {current_task_idx} / {num_tasks-1}")

        # ---------------------------------------------
        # 阶段 A：Rollout 收集数据
        # ---------------------------------------------
        print("📍 [探针 1] 开始在环境中收集交互数据 (Env Rollout)...")
        policy.eval()
        total_env_reward = 0.0
        print("\n" + "="*40)
        obs, _ = envs.reset()

        # 完成一轮任务后清空action chunking
        if hasattr(policy, 'reset'):
            policy.reset()

        last_step_img = None

        batch_data = {"img": [], "state": [], "action": [], "true_reward": [], "done": [], "task": [], "lang_tokens": [], "lang_mask": []}

        with torch.no_grad():
            for step in range(600):
                vla_obs = convert_obs(obs, envs, tokenizer, device)

                # 距离探针，检测视觉特征前后差异
                current_img = vla_obs["observation.images.image"][0] # 取 Batch 中第一个环境
                if last_step_img is not None:
                    # 计算相邻两帧图像的像素差异
                    img_diff = torch.norm(current_img - last_step_img).item()
                    if step % 100 == 0:
                        print(f"[视觉流监控] Step {step} 像素变化量: {img_diff:.4f}")
                    
                    if img_diff < 1e-6 and step > 10:
                        print("警告：检测到图像静止！obs = next_obs 可能在底层逻辑中未生效。")
                last_step_img = current_img.clone()

                # 动作预测 (利用 smolVLA 内部的 Queue 机制正常执行)
                actions = policy.select_action(vla_obs)
                if hasattr(actions, "detach"):
                    actions_np = actions.detach().cpu().numpy()
                else:
                    actions_np = actions

                # 探针：在第 150 步截获机械臂的动作
                if step == 150:
                    print("\n" + "="*40)
                    print(f"[探针] 模型在第 150 步输出的物理动作 (第一台机械臂):")
                    print(actions_np[0])
                    print("="*40 + "\n")

                next_obs, true_reward, done, _, _ = envs.step(actions_np)
                total_env_reward += true_reward.sum().item()
                
                # 存入列表 (需要把 Tensor 搬到 CPU 并解绑，防止显存泄漏)
                batch_data["img"].append(vla_obs['observation.images.image'].cpu().clone())
                batch_data["state"].append(vla_obs['observation.state'].cpu().clone())
                batch_data["lang_tokens"].append(vla_obs['observation.language.tokens'].cpu().clone())
                batch_data["lang_mask"].append(vla_obs['observation.language.attention_mask'].cpu().clone())
                batch_data["action"].append(actions.cpu().clone())
                batch_data["true_reward"].append(torch.tensor(true_reward, dtype=torch.float32))
                batch_data["done"].append(torch.tensor(done, dtype=torch.float32))
                
                obs = next_obs
                if done.all(): break
                
        # 转换并搬回 GPU
        imgs = torch.stack(batch_data["img"]).to(device)       # [Steps, B, ...]
        states = torch.stack(batch_data["state"]).to(device)   # [Steps, B, 8]
        actions = torch.stack(batch_data["action"]).to(device) # [Steps, B, 7]
        true_rewards = torch.stack(batch_data["true_reward"]).to(device)
        dones = torch.stack(batch_data["done"]).to(device)
        flat_tasks = [t for step_tasks in batch_data["task"] for t in step_tasks]
        # ---------------------------------------------
        # 阶段 B：DORF 计算 Advantage与更新DORF系统
        # ---------------------------------------------
            # 提取整个轨迹的视觉特征
        print(f"本轮 1200 步探索中，真实物理环境给出的总奖励为: {total_env_reward}")
        print("📍 [探针 2] 数据收集完毕，开始梯度累加并进行DORF评分更新...")
        flat_imgs = imgs.view(-1, *imgs.shape[2:]) 
        flat_states = states.view(-1, states.shape[-1])
        flat_actions = actions.view(-1, actions.shape[-1])

        opt_dorf.zero_grad()
        num_envs = imgs.shape[1]
        # 用于收集脱离计算图的 A_learned，供阶段 D (VLA微调) 筛选优质数据使用
        A_learned_list = []
        # 记录累加的 Loss 用于最后打印
        sum_dorf_loss = 0.0
        sum_critic_loss = 0.0

        for env_idx in range(num_envs):
            # 剥离单个环境的 600 步数据
            env_imgs = imgs[:, env_idx]           # [600, 3, 384, 384]
            env_states = states[:, env_idx]       # [600, 8]
            env_actions = actions[:, env_idx]     # [600, 7]
            env_dones = dones[:, env_idx]         # [600]
            env_true_rewards = true_rewards[:, env_idx] # [600]
            
            # 1. 提取视觉特征 
            env_v_features = vision_encoder(env_imgs)
            env_combined_state = torch.cat([env_v_features, env_states], dim=-1)
            
            # 2. 计算稠密奖励和价值 (需展平末端维度防报错)
            env_dense_rewards = dorf_reward(env_combined_state, env_actions).squeeze(-1)
            env_values = critic(env_v_features, env_states).squeeze(-1)
            env_next_value = torch.zeros(1).to(device)
            
            # 3. 计算优势 
            env_A_learned = compute_gae(env_dense_rewards, env_values, env_next_value, env_dones)
            # 保存该环境的 A_learned 到列表中 (必须 detach 剥离图)，供下阶段使用
            A_learned_list.append(env_A_learned.detach())
            
            # A_true 是物理标签，计算时脱离 critic 的梯度图
            env_A_true = compute_gae(env_true_rewards, env_values.detach(), env_next_value, env_dones)
            
            # 4. 计算 Critic MSE 损失
            env_returns_true = env_A_true.detach() + env_values.detach()
            critic_loss = nn.functional.mse_loss(env_values, env_returns_true)
            
            # 5. 计算 DORF Loss
            dorf_loss = - torch.mean(env_A_true.detach() * env_A_learned)
            
            # 6. 损失联合 (除以 num_envs 保持总梯度的步长一致性)
            total_env_loss = (dorf_loss + critic_loss) / num_envs
            # 反向传播计算梯度，并立刻释放600高清图
            total_env_loss.backward()
            
            sum_dorf_loss += dorf_loss.item() / num_envs
            sum_critic_loss += critic_loss.item() / num_envs
            
        # 循环 6 次，梯度在 opt_dorf 中累加完毕后，统一执行一次网络权重更新
        opt_dorf.step()
        # 将各环境的 A_learned 重新拼接回 [Steps, Envs] 的形状，以完美兼容阶段 D
        A_learned = torch.stack(A_learned_list, dim=1)
        print(f"VLA dorf_Loss: {sum_dorf_loss:.4f}, critic_Loss: {sum_critic_loss:.4f}")
       # ---------------------------------------------
        # 阶段 C：更新 smolVLA (Filtered BC)
        # ---------------------------------------------
        print("📍 [探针 3] DORF 更新完毕，开始检查是否需要微调 VLA 大模型...")
        # warmup_iterations = 0 # 调试使用
        warmup_iterations = 20  # 设置预热期
        
        if iteration <= warmup_iterations:
            print(f"[warm-up 阶段 {iteration}/{warmup_iterations}] 正在预热 DORF 奖励模型，暂时跳过 VLA 微调。")
        else:
            # 过滤：只保留 Advantage > 0 (表现好) 的动作数据来微调 VLA
            flat_A_learned = A_learned.view(-1)
            # 监控：输出DORF给每批动作打的分
            mean_adv = flat_A_learned.mean().item()
            max_adv = flat_A_learned.max().item()
            print(f"📈 DORF 优势值打分 -> 平均: {mean_adv:.4f}, 最高: {max_adv:.4f}")

            good_indices = torch.where(flat_A_learned > 0)[0]
            
            if total_env_reward == 0.0:
                print("[防御启动]本轮真实物理奖励为 0，DORF选出的样本为假动作,跳过微调！")
                # 强行把好样本的篮子清空！
                good_indices = torch.tensor([], dtype=torch.long, device=flat_A_learned.device)

            # 强行塞入样本，调试使用
            '''if len(good_indices) == 0:
                print("[Debug] 强行征用前 16 个样本进行管线连通性测试")
                good_indices = torch.arange(16, device=flat_A_learned.device)'''

            if len(good_indices) > 256:
                # 随机打乱并只截取前 256 个
                good_indices = good_indices[torch.randperm(len(good_indices))][:256]
            
            if len(good_indices) > 0:
                print(f"本轮共发现 {len(good_indices)} 个优质样本，开始分批微调 VLA...")
                policy.train()
                MICRO_BATCH_SIZE = 8 
                good_indices = good_indices[torch.randperm(len(good_indices))]

                total_loss = 0.0
                update_steps = 0

                # 切割样本，分batch喂给模型
                for start_idx in range(0, len(good_indices), MICRO_BATCH_SIZE):
                    chunk_indices = good_indices[start_idx : start_idx + MICRO_BATCH_SIZE]

                    opt_policy.zero_grad()
                
                    # 把优质数据打包成 LeRobot 要求的 batch 格式
                    batch_size = len(chunk_indices)
                    device = flat_imgs.device
                
                    # 输入端 Mask： 122(49 Vision + 73 Language)
                    vision_mask = torch.ones((batch_size, 49), dtype=torch.bool, device=device)
                    lang_mask = flat_lang_masks[chunk_indices] 
                    full_att_mask = torch.cat([vision_mask, lang_mask], dim=1) 
                    
                    # 输出端的 Action Chunking 依然补齐到 50 步
                    action_chunk_size = 50
                    real_action = flat_actions[chunk_indices].unsqueeze(1)
                    action_dim = real_action.shape[-1]
                
                    padded_action = torch.zeros((batch_size, action_chunk_size - 1, action_dim), dtype=real_action.dtype, device=device)
                    full_actions = torch.cat([real_action, padded_action], dim=1) 
                    
                    # 输出端的 Action Mask (独立存放，喂给 action_is_pad)
                    action_valid_pad = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
                    action_invalid_pad = torch.ones((batch_size, action_chunk_size - 1), dtype=torch.bool, device=device)
                    full_action_is_pad = torch.cat([action_valid_pad, action_invalid_pad], dim=1) 
                    
                    # 图片和状态的补齐标签
                    dummy_is_pad = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)

                    # 把独立的数据打包喂给 smolVLA
                    good_batch = {
                        "observation.images.image": flat_imgs[chunk_indices].unsqueeze(1),
                        "observation.images.image_is_pad": dummy_is_pad, 
                        "observation.state": flat_states[chunk_indices].unsqueeze(1),
                        "observation.state_is_pad": dummy_is_pad,        
                        "action": full_actions,                           
                        "action_is_pad": full_action_is_pad,              
                        "task": [flat_tasks[idx.item()] for idx in chunk_indices],
                        "observation.language.tokens": flat_lang_tokens[chunk_indices],
                        "observation.language.attention_mask": flat_lang_masks[chunk_indices]
                    }
                    # 调用原生 forward 计算 BC 损失
                    output_dict = policy.forward(good_batch)

                    if isinstance(output_dict, dict):
                        loss = output_dict.get('loss', output_dict)
                    elif isinstance(output_dict, tuple):
                        loss = output_dict[0]  # 大模型返回元组时，第一个元素绝对是 loss！
                    else:
                        loss = output_dict
                
                    loss.backward()
                    opt_policy.step()

                    total_loss += loss.item()
                    update_steps += 1

                avg_loss = total_loss / update_steps
                print(f"微调 VLA结束。分 {update_steps} 批次更新， 平均 BC Loss: {avg_loss:.4f}")
                # 把数据画到 TensorBoard 上
                tb_writer.add_scalar("Training/BC_Loss", avg_loss, iteration)
                tb_writer.add_scalar("Metrics/Good_Samples", len(good_indices), iteration)
                tb_writer.add_scalar("Metrics/Max_Advantage", max_adv, iteration)
            else:
                print("这一轮没有发现优质样本，跳过 VLA 更新。")
        
        print(f"📍 [探针 4] VLA 检查/更新完毕，第 {iteration} 轮结束！")

        # ---------------------------------------------
        # 阶段 D：保存断点 (Checkpointing)
        # ---------------------------------------------
        # 我们设定每 5 轮保存一次，兼顾安全性和硬盘 IO 效率
        if iteration % 5 == 0:
            print(f" 正在保存第 {iteration} 轮的检查点...")
            torch.save({
                'iteration': iteration,
                'policy': policy.state_dict(),
                'dorf_reward': dorf_reward.state_dict(),
                'critic': critic.state_dict(),
                'vision_encoder': vision_encoder.state_dict(),
                'opt_policy': opt_policy.state_dict(),
                'opt_dorf': opt_dorf.state_dict(),
            }, checkpoint_path)
            print(f" 第 {iteration} 轮检查点已稳妥保存至 {checkpoint_path}！")

if __name__ == "__main__":
    main()