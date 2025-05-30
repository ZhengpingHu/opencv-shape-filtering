#!/usr/bin/env python3
# test_gp_agent.py

import argparse
import numpy as np
import torch
import torch.nn as nn
from env_yolo_kalman import FeatureEnv

# -------------------------------------------------------------------
# 网络结构定义（必须和训练时保持一致）
# -------------------------------------------------------------------
INPUT_DIM  = 8
H1_DIM     = 32
H2_DIM     = 32
OUTPUT_DIM = 4

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM,  H1_DIM)
        self.fc2 = nn.Linear(H1_DIM,     H2_DIM)
        self.fc3 = nn.Linear(H2_DIM,     OUTPUT_DIM)
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

# -------------------------------------------------------------------
# 主流程：加载检测模型、加载策略模型、跑若干回合
# -------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--yolo',    required=True,
                   help="YOLO 检测模型权重文件 (.pt)")
    p.add_argument('--policy',  required=True,
                   help="GP 训练得到的策略模型文件 best_model.pt")
    p.add_argument('--title',    default="LunarLander-v3",
                   help="环境窗口标题关键字")
    p.add_argument('--fps',      type=float, default=5.0,
                   help="环境帧率")
    p.add_argument('--gravity',  type=float, default=-3.5,
                   help="重力加速度")
    p.add_argument('--episodes', type=int,   default=10,
                   help="测试回合数")
    p.add_argument('--debug',    action='store_true',
                   help="打印每步调试信息")
    args = p.parse_args()

    # 初始化策略网络并加载权重
    policy_net = Net()
    policy_net.load_state_dict(torch.load(args.policy).state_dict())
    policy_net.eval()

    rewards = []
    for ep in range(1, args.episodes + 1):
        # 只把 --yolo 权重传给 FeatureEnv 用于目标检测
        env = FeatureEnv(
            model_path=args.yolo,
            title=args.title,
            fps=args.fps,
            gravity=args.gravity,
            launch_env=args.debug
        )

        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            x = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                out = policy_net(x)
            action = int(torch.argmax(out).item())

            state, r, done = env.step(action)
            total_reward += r

            if args.debug:
                print(f"\r[Ep{ep}] Step reward={r:.3f} cum={total_reward:.3f}", end="")

        if args.debug:
            print()  # 换行
        env.close()

        print(f"Episode {ep:2d}: total_reward = {total_reward:.3f}")
        rewards.append(total_reward)

    rewards = np.array(rewards, dtype=np.float32)
    print(f"\nAverage reward over {args.episodes} episodes: {rewards.mean():.3f}")

if __name__ == "__main__":
    main()
