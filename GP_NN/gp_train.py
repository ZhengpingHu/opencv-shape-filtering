#!/usr/bin/env python3
# gp_train.py

import argparse
import random
import sys
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from deap import base, creator, tools, algorithms
from env_yolo_kalman import FeatureEnv

# -------------------------------------------------------------------
# 1. 基础超参
# -------------------------------------------------------------------
BASE_INPUT_DIM = 8
H1_DIM         = 32
H2_DIM         = 32
OUTPUT_DIM     = 4

# 下面两个在 main() 中根据滑窗大小 K 动态重写
INPUT_DIM = BASE_INPUT_DIM
D = (
    INPUT_DIM * H1_DIM + H1_DIM +
    H1_DIM * H2_DIM     + H2_DIM +
    H2_DIM * OUTPUT_DIM + OUTPUT_DIM
)

# -------------------------------------------------------------------
# 2. 策略网络定义
# -------------------------------------------------------------------
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
# 3. 解码基因并选动作
# -------------------------------------------------------------------
def decode_and_act(ind, state):
    idx = 0
    def get_w(r, c):
        nonlocal idx
        size = r*c
        w = np.array(ind[idx:idx+size], dtype=np.float32).reshape(r, c)
        idx += size
        return w
    def get_b(n):
        nonlocal idx
        b = np.array(ind[idx:idx+n], dtype=np.float32)
        idx += n
        return b

    W1 = get_w(H1_DIM, INPUT_DIM);  b1 = get_b(H1_DIM)
    W2 = get_w(H2_DIM, H1_DIM);     b2 = get_b(H2_DIM)
    W3 = get_w(OUTPUT_DIM, H2_DIM); b3 = get_b(OUTPUT_DIM)

    h1 = np.tanh(W1.dot(state) + b1)
    h2 = np.tanh(W2.dot(h1)   + b2)
    out = W3.dot(h2) + b3
    return int(np.argmax(out))

# -------------------------------------------------------------------
# 4. 构造评价函数：滑动窗口 + 恒定渲染
# -------------------------------------------------------------------
def make_evaluate(model_path, title, fps, gravity, K, alpha):
    def evaluate(individual):
        env = FeatureEnv(
            model_path=model_path,
            title=title,
            fps=fps,
            gravity=gravity,
            launch_env=True
        )

        buf = deque([env.reset()] * K, maxlen=K)
        total_env_r = 0.0
        shaping_r = 0.0
        done = False

        env.same_cnt  = 0
        env.last_act  = -1
        env.phi       = 0.0
        env.gating    = False
        env.gate_cnt  = 0
        env.gate_ok   = True
        env.angle_thr = 0.5
        env.vel_thr   = 0.5
        env.vy_lim    = 0.6
        env.same_thr  = 15
        env.speed_pen = 0.2
        env.repeat_pen = 0.4
        env.pos_gate  = 2.0
        env.neg_gate  = 2.0

        while not done:
            window = np.concatenate(buf)
            action = decode_and_act(individual, window)
            state, reward, done = env.step(action)
            buf.append(state)
            total_env_r += reward

            # shaping
            env.same_cnt = env.same_cnt + 1 if action == env.last_act else 1
            env.last_act = action

            if action == 2:
                shaping_r += 0.2
            elif action in (1, 3):
                shaping_r += 0.1

            x, y, vx, vy, angle, ang_vel, pad_x, pad_y = state
            phi2 = -np.hypot(x - pad_x, y - pad_y)
            shaping_r += (0.99 * phi2 - env.phi)
            env.phi = phi2

            if abs(vy) > env.vy_lim:
                shaping_r -= env.speed_pen
            if env.same_cnt >= env.same_thr:
                shaping_r -= env.repeat_pen

            if not env.gating and action in (1, 2, 3):
                env.gating = True
                env.gate_cnt = 10
                env.gate_ok = True

            if env.gating:
                if abs(angle) > env.angle_thr or abs(vy) > env.vel_thr:
                    env.gate_ok = False
                env.gate_cnt -= 1
                if env.gate_cnt <= 0:
                    shaping_r += (env.pos_gate if env.gate_ok else -env.neg_gate)
                    env.gating = False

        env.close()
        final_reward = (1 - alpha) * total_env_r + alpha * shaping_r
        return (final_reward,)

    return evaluate


# -------------------------------------------------------------------
# 5. 主流程：遗传算法演化
# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   required=True,
                        help="YOLO 检测模型 .pt 权重")
    parser.add_argument('--title',   default="LunarLander-v3",
                        help="OpenCV 窗口标题")
    parser.add_argument('--fps',     type=float, default=5.0,
                        help="帧率 (fps)")
    parser.add_argument('--gravity', type=float, default=-3.5,
                        help="重力加速度")
    parser.add_argument('--pop',     type=int,   default=50,
                        help="种群大小")
    parser.add_argument('--gen',     type=int,   default=20,
                        help="进化代数")
    parser.add_argument('--cxpb',    type=float, default=0.5,
                        help="交叉概率")
    parser.add_argument('--mutpb',   type=float, default=0.2,
                        help="变异概率")
    parser.add_argument('--alpha', type=float, default=0.1,
                    help="自定义 reward 的权重系数 alpha")
    args = parser.parse_args()

    # 5.1 计算滑动窗口大小 K = 2 秒的帧数
    K = int(2 * args.fps)

    # 5.2 根据 K 重写 INPUT_DIM 与基因长度 D
    global INPUT_DIM, D
    INPUT_DIM = BASE_INPUT_DIM * K
    D = (
        INPUT_DIM * H1_DIM + H1_DIM +
        H1_DIM * H2_DIM     + H2_DIM +
        H2_DIM * OUTPUT_DIM + OUTPUT_DIM
    )

    # 5.3 注册 DEAP 组件
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list,   fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1.0, 1.0)
    toolbox.register("individual",
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.attr_float,
                     n=D)
    toolbox.register("population",
                     tools.initRepeat,
                     list,
                     toolbox.individual)
    toolbox.register("evaluate",
                     make_evaluate(
                         args.model,
                         args.title,
                         args.fps,
                         args.gravity,
                         K,
                         args.alpha
                     ))
    toolbox.register("mate",   tools.cxTwoPoint)
    toolbox.register("mutate",
                     tools.mutGaussian,
                     mu=0, sigma=0.2, indpb=0.05)
    toolbox.register("select",
                     tools.selTournament,
                     tournsize=3)

    # 5.4 初始化种群 & 统计器
    pop = toolbox.population(n=args.pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # 5.5 评估第 0 代，逐个打印
    print("Evaluating generation 0:", flush=True)
    for i, ind in enumerate(pop, start=1):
        print(f"  Pop {i}:", flush=True)
        fit = toolbox.evaluate(ind)
        ind.fitness.values = fit
        print(f"    Reward = {fit[0]:.3f}", flush=True)
    rec = stats.compile(pop)
    print(f"Gen  0: max={rec['max']:.3f}, avg={rec['avg']:.3f}", flush=True)

    # 5.6 进化循环：每代都逐个打印
    for gen in range(1, args.gen + 1):
        print(f"\nEvaluating generation {gen}:", flush=True)
        offspring = toolbox.select(pop, len(pop))
        offspring = algorithms.varAnd(
            offspring,
            toolbox,
            cxpb=args.cxpb,
            mutpb=args.mutpb
        )
        pop = offspring  # 用 offspring 更新 pop
        for i, ind in enumerate(pop, start=1):
            print(f"  Pop {i}:", flush=True)
            fit = toolbox.evaluate(ind)
            ind.fitness.values = fit
            print(f"    Reward = {fit[0]:.3f}", flush=True)
        hof.update(pop)
        rec = stats.compile(pop)
        print(f"Gen {gen:2d}: max={rec['max']:.3f}, avg={rec['avg']:.3f}", flush=True)

    # 5.7 保存最优模型
    best = hof[0]
    net = Net()
    idx = 0

    # 填充 fc1
    size = H1_DIM * INPUT_DIM
    net.fc1.weight.data = torch.tensor(
        best[idx:idx+size], dtype=torch.float32
    ).view(H1_DIM, INPUT_DIM)
    idx += size
    net.fc1.bias.data = torch.tensor(
        best[idx:idx+H1_DIM], dtype=torch.float32
    )
    idx += H1_DIM

    # 填充 fc2
    size = H2_DIM * H1_DIM
    net.fc2.weight.data = torch.tensor(
        best[idx:idx+size], dtype=torch.float32
    ).view(H2_DIM, H1_DIM)
    idx += size
    net.fc2.bias.data = torch.tensor(
        best[idx:idx+H2_DIM], dtype=torch.float32
    )
    idx += H2_DIM

    # 填充 fc3
    size = OUTPUT_DIM * H2_DIM
    net.fc3.weight.data = torch.tensor(
        best[idx:idx+size], dtype=torch.float32
    ).view(OUTPUT_DIM, H2_DIM)
    idx += size
    net.fc3.bias.data = torch.tensor(
        best[idx:idx+OUTPUT_DIM], dtype=torch.float32
    )

    torch.save(net, "best_model.pt")
    print(f"\n[INFO] 最优模型已保存为 best_model.pt，适应度 = {best.fitness.values[0]:.3f}", flush=True)


if __name__ == "__main__":
    main()
