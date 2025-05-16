#!/usr/bin/env python3
# gp_train.py

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from deap import base, creator, tools, algorithms
from env_yolo_kalman import FeatureEnv

# -------------------------------------------------------------------
# 网络超参
# -------------------------------------------------------------------
INPUT_DIM  = 8
H1_DIM     = 32
H2_DIM     = 32
OUTPUT_DIM = 4

# 基因长度 D
D = (INPUT_DIM*H1_DIM + H1_DIM
   + H1_DIM*H2_DIM + H2_DIM
   + H2_DIM*OUTPUT_DIM + OUTPUT_DIM)

# -------------------------------------------------------------------
# PyTorch 网络定义（用于最终保存）
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
# 从基因解码动作
# -------------------------------------------------------------------
def decode_and_act(ind, state):
    idx = 0
    def get_w(r, c):
        nonlocal idx
        size = r*c
        w = np.array(ind[idx:idx+size], dtype=np.float32).reshape(r,c)
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
    h2 = np.tanh(W2.dot(h1) + b2)
    out = W3.dot(h2) + b3
    return int(np.argmax(out))

# -------------------------------------------------------------------
# 构造评价函数
# -------------------------------------------------------------------
def make_evaluate(model_path, title, fps, gravity, debug):
    def evaluate(individual):
        env = FeatureEnv(
            model_path=model_path,
            title=title,
            fps=fps,
            gravity=gravity,
            launch_env=debug
        )
        total_reward = 0.0
        state = env.reset()
        done = False
        while not done:
            action = decode_and_act(individual, state)
            state, r, done = env.step(action)
            total_reward += r
            if debug:
                print(f"\r[Step] r={r:.3f} cum={total_reward:.3f}", end="")
        if debug:
            print()
        env.close()
        return (total_reward,)
    return evaluate

# -------------------------------------------------------------------
# 主流程：遗传演化
# -------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',   required=True, help="YOLO .pt 权重")
    p.add_argument('--title',   default="LunarLander-v3", help="窗口标题关键字")
    p.add_argument('--fps',     type=float, default=5.0)
    p.add_argument('--gravity', type=float, default=-3.5)
    p.add_argument('--pop',     type=int,   default=50)
    p.add_argument('--gen',     type=int,   default=20)
    p.add_argument('--cxpb',    type=float, default=0.5)
    p.add_argument('--mutpb',   type=float, default=0.2)
    p.add_argument('--debug',   action='store_true')
    args = p.parse_args()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list,   fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1.0, 1.0)
    toolbox.register("individual",
                     tools.initRepeat,
                     creator.Individual,
                     toolbox.attr_float,
                     n=D)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate",
                     make_evaluate(args.model,
                                   args.title,
                                   args.fps,
                                   args.gravity,
                                   args.debug))
    toolbox.register("mate",   tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian,
                     mu=0, sigma=0.2, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=args.pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # 评估 Gen 0
    fits = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit
    rec = stats.compile(pop)
    print(f"Gen 0: max={rec['max']:.3f}, avg={rec['avg']:.3f}")

    for gen in range(1, args.gen+1):
        offspring = toolbox.select(pop, len(pop))
        offspring = algorithms.varAnd(offspring,
                                     toolbox,
                                     cxpb=args.cxpb,
                                     mutpb=args.mutpb)
        fits = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits):
            ind.fitness.values = fit
        pop[:] = offspring
        hof.update(pop)
        rec = stats.compile(pop)
        print(f"Gen {gen}: max={rec['max']:.3f}, avg={rec['avg']:.3f}")

    # 保存最优模型
    best = hof[0]
    net = Net()
    idx = 0

    # W1 & b1
    size = H1_DIM*INPUT_DIM
    net.fc1.weight.data = torch.tensor(best[idx:idx+size], dtype=torch.float32).view(H1_DIM, INPUT_DIM)
    idx += size
    net.fc1.bias.data   = torch.tensor(best[idx:idx+H1_DIM], dtype=torch.float32)
    idx += H1_DIM

    # W2 & b2
    size = H2_DIM*H1_DIM
    net.fc2.weight.data = torch.tensor(best[idx:idx+size], dtype=torch.float32).view(H2_DIM, H1_DIM)
    idx += size
    net.fc2.bias.data   = torch.tensor(best[idx:idx+H2_DIM], dtype=torch.float32)
    idx += H2_DIM

    # W3 & b3
    size = OUTPUT_DIM*H2_DIM
    net.fc3.weight.data = torch.tensor(best[idx:idx+size], dtype=torch.float32).view(OUTPUT_DIM, H2_DIM)
    idx += size
    net.fc3.bias.data   = torch.tensor(best[idx:idx+OUTPUT_DIM], dtype=torch.float32)

    torch.save(net, "best_model.pt")
    print("[INFO] 最优模型已保存为 best_model.pt，适应度 =", best.fitness.values[0])

if __name__ == "__main__":
    main()
