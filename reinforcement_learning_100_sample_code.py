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
# 1. 训练奖励模型（Reward Model）
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

# ==========================
# 2. 定义批量 DRC 评估环境
# ==========================

class BatchLayoutEnv(gym.Env):
    def __init__(self, width, height, batch_size=100):
        super(BatchLayoutEnv, self).__init__()
        self.width = width
        self.height = height
        self.total_area = width * height
        self.batch_size = batch_size  # 设定批量处理数量
        self.layout_gds_list = [f"layout_{i}.gds" for i in range(batch_size)]  # 100 组 GDS
        self.action_space = spaces.Box(low=0, high=1, shape=(batch_size, 10), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(batch_size, 20), dtype=np.float32)

    def step(self, actions):
        # 生成 100 组完整布局并保存 GDS 文件
        layout_areas, layout_features_list = self.generate_layout_batch(actions)

        # 批量运行 DRC 计算违规情况
        drc_violations_list = run_batch_calibre_drc(self.layout_gds_list)

        # 计算每个布局的奖励
        rewards = np.array([
            compute_reward(self.layout_gds_list[i], layout_areas[i], self.total_area, layout_features_list[i], drc_violations_list[i])
            for i in range(self.batch_size)
        ])

        done = True  # 生成完整布局后终止

        return np.array(layout_features_list), rewards, done, {}

    def generate_layout_batch(self, actions):
        """
        生成 GDSII 文件（批量处理）
        """
        layout_areas = []
        layout_features_list = []
        for i in range(self.batch_size):
            layout_area = np.random.uniform(0.5, 1.0) * self.total_area  # 伪随机填充率
            layout_features = list(actions[i]) + [np.random.uniform(0.5, 1.0) for _ in range(10)]  # 伪随机风格特征
            layout_areas.append(layout_area)
            layout_features_list.append(layout_features)

        return layout_areas, layout_features_list

    def reset(self):
        return np.random.uniform(0, 1, size=(self.batch_size, 20))

# ==========================
# 3. 计算 DRC & 奖励
# ==========================

def run_batch_calibre_drc(layout_gds_list):
    """
    批量运行 DRC，返回每个布局的违规数
    """
    drc_violations_list = []
    for layout_gds in layout_gds_list:
        calibre_command = f"calibre -drc -hier -runset my_drc_rules.runset -design {layout_gds}"
        result = subprocess.run(calibre_command, shell=True, capture_output=True, text=True)

        drc_results_file = "calibre_drc.results"
        if os.path.exists(drc_results_file):
            with open(drc_results_file, "r") as f:
                drc_violations = sum(1 for _ in f if "DRC ERROR" in _)
        else:
            drc_violations = 0  # 无错误

        drc_violations_list.append(drc_violations)

    return drc_violations_list

def compute_reward(layout_gds, layout_area, total_area, layout_features, drc_violations):
    """
    计算完整布局后的奖励
    """
    drc_reward = 10 if drc_violations == 0 else -5 * drc_violations
    density_reward = (layout_area / total_area) * 10  # 归一化填充率
    style_reward = get_style_reward(layout_features)

    return drc_reward + density_reward + style_reward

def get_style_reward(layout_features):
    """
    计算风格相似性奖励
    """
    input_tensor = torch.tensor(layout_features, dtype=torch.float32).to("cuda")
    with torch.no_grad():
        reward = reward_model(input_tensor).item()
    return reward  

# ==========================
# 4. 训练 RL 模型
# ==========================

# 创建环境
env = BatchLayoutEnv(width=100, height=100, batch_size=100)

# 训练强化学习模型（批量评估）
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50000)
model.save("batch_layout_ppo")