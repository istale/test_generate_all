!pip install stable-baselines3

import os
import subprocess
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO

# ==========================
# 1. 训练奖励模型（Reward Model）- 让 AI 生成更符合用户风格的布局
# ==========================

class RewardModel(nn.Module):
    def __init__(self, input_dim):
        super(RewardModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出一个评分
        )

    def forward(self, x):
        return self.fc(x)

# 初始化 Reward Model
reward_model = RewardModel(input_dim=20).to("cuda")
optimizer = optim.AdamW(reward_model.parameters(), lr=5e-4)

# 训练数据示例（用户布局风格数据）
user_style_data = [
    ([0.8, 0.7, 0.9, 0.5, 0.6, 0.8, 0.9, 0.7, 0.6, 0.5, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.7, 0.6, 0.8], 1.0),  # 高质量布局
    ([0.3, 0.4, 0.2, 0.5, 0.6, 0.4, 0.3, 0.5, 0.2, 0.1, 0.4, 0.2, 0.1, 0.3, 0.5, 0.2, 0.1, 0.3, 0.4, 0.2], 0.2)   # 低质量布局
]

# 训练 Reward Model
for epoch in range(5):  # 实际训练可增加轮次
    total_loss = 0
    for features, label in user_style_data:
        inputs = torch.tensor(features, dtype=torch.float32).to("cuda")
        target = torch.tensor([label], dtype=torch.float32).to("cuda")

        optimizer.zero_grad()
        output = reward_model(inputs)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

torch.save(reward_model.state_dict(), "reward_model.pth")  # 保存奖励模型

# ==========================
# 2. 训练强化学习（PPO）- 生成完整布局后评估
# ==========================

# 调用 EDA 工具（Calibre）进行 DRC 检查
def run_calibre_drc(layout_gds):
    """
    运行 Calibre 进行 DRC 检查，并返回违规数
    """
    calibre_command = f"calibre -drc -hier -runset my_drc_rules.runset -design {layout_gds}"
    result = subprocess.run(calibre_command, shell=True, capture_output=True, text=True)

    drc_results_file = "calibre_drc.results"
    if os.path.exists(drc_results_file):
        with open(drc_results_file, "r") as f:
            drc_violations = sum(1 for _ in f if "DRC ERROR" in _)
    else:
        drc_violations = 0  # 无错误

    return drc_violations

# 计算填充率
def calculate_density(layout_area, total_area):
    return (layout_area / total_area) * 10  # 归一化到 0-10 分

# 计算风格相似性奖励
def get_style_reward(layout_features):
    input_tensor = torch.tensor(layout_features, dtype=torch.float32).to("cuda")
    with torch.no_grad():
        reward = reward_model(input_tensor).item()
    return reward  

# 计算完整布局后的奖励
def compute_reward(layout_gds, layout_area, total_area, layout_features):
    drc_violations = run_calibre_drc(layout_gds)
    drc_reward = 10 if drc_violations == 0 else -5 * drc_violations
    density_reward = calculate_density(layout_area, total_area)
    style_reward = get_style_reward(layout_features)

    return drc_reward + density_reward + style_reward

# 定义 RL 训练环境
class BatchLayoutEnv(gym.Env):
    def __init__(self, width, height):
        super(BatchLayoutEnv, self).__init__()
        self.width = width
        self.height = height
        self.total_area = width * height
        self.layout_gds = "current_layout.gds"  # 存储当前布局的 GDSII 文件
        self.action_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(20,), dtype=np.float32)

    def step(self, action):
        # 生成完整布局并保存 GDS 文件
        layout_area, layout_features = self.generate_layout_gds(action)

        # 计算 DRC 奖励 + 风格奖励 + 密度奖励
        reward = compute_reward(self.layout_gds, layout_area, self.total_area, layout_features)
        done = True  # 生成完整布局后终止

        return np.array(layout_features), reward, done, {}

    def generate_layout_gds(self, action):
        """
        这里应该使用 OpenROAD / KLayout / Magic / 自定义 GDS 生成工具
        action: RL 生成的布局参数（如矩形排列方式）
        """
        layout_area = np.random.uniform(0.5, 1.0) * self.total_area  # 伪随机填充率
        layout_features = list(action) + [np.random.uniform(0.5, 1.0) for _ in range(10)]  # 伪随机风格特征
        return layout_area, layout_features

    def reset(self):
        return np.random.uniform(0, 1, size=(20,))

# 训练强化学习模型
env = BatchLayoutEnv(width=100, height=100)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("batch_layout_ppo")