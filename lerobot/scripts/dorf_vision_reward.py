import torch
import torch.nn as nn

class VisionCritic(nn.Module):
    """能够处理视觉特征和机器人状态的 Critic 网络"""
    def __init__(self, visual_feature_dim=512, state_dim=8, hidden_dim=256):
        super().__init__()
        # 简单的视觉降维编码器 (你可以根据需要替换为更强的 CNN/ViT)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(50176, visual_feature_dim) # 假设输入是 384x384
        )
        
        # 融合视觉和状态特征计算 Value
        self.value_net = nn.Sequential(
            nn.Linear(visual_feature_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, img, state):
        B, S, C, H, W = img.shape
        # 融合 Batch 和 Sequence 维度处理图像
        flat_img = img.view(B * S, C, H, W)
        v_feat = self.vision_encoder(flat_img)
        # 使用显式维度而非 -1，防止在 S=0 时出现歧义报错
        feature_dim = v_feat.shape[-1]
        v_feat = v_feat.view(B, S, feature_dim)
        
        combined_feat = torch.cat([v_feat, state], dim=-1)
        value = self.value_net(combined_feat)
        return value.squeeze(-1) # [B, S]

class VisionReward(nn.Module):
    """能够处理视觉特征、状态和动作的 Reward 网络 (对应 DORF 思想)"""
    def __init__(self, visual_feature_dim=512, state_dim=8, action_dim=7, hidden_dim=256):
        super().__init__()
        # 复用 Critic 的视觉编码器结构 (为了简单起见，这里独立训练)
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(50176, visual_feature_dim)
        )
        
        self.reward_net = nn.Sequential(
            nn.Linear(visual_feature_dim + state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, img, state, action):
        B, S, C, H, W = img.shape
        flat_img = img.view(B * S, C, H, W)
        v_feat = self.vision_encoder(flat_img)
        # 使用显式维度而非 -1，防止在 S=0 时出现歧义报错
        feature_dim = v_feat.shape[-1]
        v_feat = v_feat.view(B, S, feature_dim)
        
        combined_feat = torch.cat([v_feat, state, action], dim=-1)
        reward = self.reward_net(combined_feat)
        return reward.squeeze(-1) # [B, S]

def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """计算广义优势估计 (Generalized Advantage Estimation)"""
    B, S = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae_lam = 0
    
    # 扩展 values 以包含 next_value
    next_values = torch.cat([values[:, 1:], next_value.unsqueeze(1).expand(B, 1)], dim=1)
    
    for t in reversed(range(S)):
        next_non_terminal = 1.0 - dones[:, t].float()
        delta = rewards[:, t] + gamma * next_values[:, t] * next_non_terminal - values[:, t]
        advantages[:, t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        
    return advantages