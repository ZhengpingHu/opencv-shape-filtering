#!/usr/bin/env python3
# gp_train.py

import argparse
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from deap import base, creator, tools, algorithms
from env_yolo_kalman import FeatureEnv

# -------------------------------
# 网络结构设置
# -------------------------------
BASE_INPUT_DIM = 8
OUTPUT_DIM = 4
H1_DIM, H2_DIM = 32, 32

# 占位，稍后动态设置
INPUT_DIM = None
D = None


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, H1_DIM)
        self.fc2 = nn.Linear(H1_DIM, H2_DIM)
        self.fc3 = nn.Linear(H2_DIM, OUTPUT_DIM)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


def decode_and_act(ind, input_vec):
    idx = 0

    def get_w(r, c):
        nonlocal idx
        size = r * c
        w = np.array(ind[idx:idx + size], dtype=np.float32).reshape(r, c)
        idx += size
        return w

    def get_b(n):
        nonlocal idx
        b = np.array(ind[idx:idx + n], dtype=np.float32)
        idx += n
        return b

    W1 = get_w(H1_DIM, INPUT_DIM)
    b1 = get_b(H1_DIM)
    W2 = get_w(H2_DIM, H1_DIM)
    b2 = get_b(H2_DIM)
    W3 = get_w(OUTPUT_DIM, H2_DIM)
    b3 = get_b(OUTPUT_DIM)

    h1 = np.tanh(W1.dot(input_vec) + b1)
    h2 = np.tanh(W2.dot(h1) + b2)
    out = W3.dot(h2) + b3
    return int(np.argmax(out))


def make_evaluate(model_path, title, fps, gravity, K, alpha, render):
    def evaluate(individual):
        env = FeatureEnv(model_path, title, fps, gravity, render)
        buf = deque([env.reset()] * K, maxlen=K)
        act_buf = deque([[0]*OUTPUT_DIM] * K, maxlen=K)
        total_env_r, shaping_r = 0.0, 0.0
        done = False

        env.same_cnt = 0
        env.last_act = -1
        env.phi = 0.0
        env.gating = False
        env.gate_cnt = 0
        env.gate_ok = True
        env.angle_thr = 0.5
        env.vel_thr = 0.5
        env.vy_lim = 0.4
        env.same_thr = 15
        env.speed_pen = 0.5
        env.repeat_pen = 0.4
        env.pos_gate = 2.0
        env.neg_gate = 2.0

        while not done:
            window = np.concatenate(buf)
            past_acts = np.concatenate(act_buf)
            input_vec = np.concatenate([window, past_acts])
            action = decode_and_act(individual, input_vec)

            one_hot = [0] * OUTPUT_DIM
            one_hot[action] = 1
            act_buf.append(one_hot)

            state, reward, done = env.step(action)
            buf.append(state)
            total_env_r += reward

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
                env.gating, env.gate_cnt, env.gate_ok = True, 10, True

            if env.gating:
                if abs(angle) > env.angle_thr or abs(vy) > env.vel_thr:
                    env.gate_ok = False
                env.gate_cnt -= 1
                if env.gate_cnt <= 0:
                    shaping_r += (env.pos_gate if env.gate_ok else -env.neg_gate)
                    env.gating = False

        env.close()
        return ((1 - alpha) * total_env_r + alpha * shaping_r,)
    return evaluate


def main():
    global INPUT_DIM, D

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--title', default="LunarLander-v3")
    parser.add_argument('--fps', type=float, default=5.0)
    parser.add_argument('--gravity', type=float, default=-3.5)
    parser.add_argument('--pop', type=int, default=50)
    parser.add_argument('--gen', type=int, default=20)
    parser.add_argument('--cxpb', type=float, default=0.5)
    parser.add_argument('--mutpb', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--continue_from', default=None)
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    K = int(2 * args.fps)
    INPUT_DIM = (BASE_INPUT_DIM + OUTPUT_DIM) * K
    D = INPUT_DIM * H1_DIM + H1_DIM + H1_DIM * H2_DIM + H2_DIM + H2_DIM * OUTPUT_DIM + OUTPUT_DIM

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=D)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", make_evaluate(
        args.model, args.title, args.fps, args.gravity, K, args.alpha, args.render
    ))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    if args.continue_from:
        print(f"[INFO] 正在加载模型参数: {args.continue_from}", flush=True)
        net = torch.load(args.continue_from)
        genome = []
        genome += net.fc1.weight.data.cpu().numpy().flatten().tolist()
        genome += net.fc1.bias.data.cpu().numpy().tolist()
        genome += net.fc2.weight.data.cpu().numpy().flatten().tolist()
        genome += net.fc2.bias.data.cpu().numpy().tolist()
        genome += net.fc3.weight.data.cpu().numpy().flatten().tolist()
        genome += net.fc3.bias.data.cpu().numpy().tolist()
        assert len(genome) == D
        pop = [creator.Individual([x + np.random.normal(0, 0.01) for x in genome])
               for _ in range(args.pop)]
    else:
        pop = toolbox.population(n=args.pop)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    print("Evaluating generation 0:")
    for i, ind in enumerate(pop, start=1):
        print(f"  Pop {i}:")
        fit = toolbox.evaluate(ind)
        ind.fitness.values = fit
        print(f"    Reward = {fit[0]:.3f}")
    rec = stats.compile(pop)
    print(f"Gen  0: max={rec['max']:.3f}, avg={rec['avg']:.3f}")

    for gen in range(1, args.gen + 1):
        print(f"\nEvaluating generation {gen}:")
        offspring = toolbox.select(pop, len(pop))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=args.cxpb, mutpb=args.mutpb)
        pop = offspring
        for i, ind in enumerate(pop, start=1):
            print(f"  Pop {i}:")
            fit = toolbox.evaluate(ind)
            ind.fitness.values = fit
            print(f"    Reward = {fit[0]:.3f}")
        hof.update(pop)
        rec = stats.compile(pop)
        print(f"Gen {gen:2d}: max={rec['max']:.3f}, avg={rec['avg']:.3f}")

    best = hof[0]
    net = Net()
    idx = 0
    net.fc1.weight.data = torch.tensor(best[idx:idx + H1_DIM * INPUT_DIM]).view(H1_DIM, INPUT_DIM)
    idx += H1_DIM * INPUT_DIM
    net.fc1.bias.data = torch.tensor(best[idx:idx + H1_DIM])
    idx += H1_DIM
    net.fc2.weight.data = torch.tensor(best[idx:idx + H2_DIM * H1_DIM]).view(H2_DIM, H1_DIM)
    idx += H2_DIM * H1_DIM
    net.fc2.bias.data = torch.tensor(best[idx:idx + H2_DIM])
    idx += H2_DIM
    net.fc3.weight.data = torch.tensor(best[idx:idx + OUTPUT_DIM * H2_DIM]).view(OUTPUT_DIM, H2_DIM)
    idx += OUTPUT_DIM * H2_DIM
    net.fc3.bias.data = torch.tensor(best[idx:idx + OUTPUT_DIM])

    save_name = args.save_path or f"best_model_{time.strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(net, save_name)
    print(f"\n[INFO] 最优模型已保存为 {save_name}，适应度 = {best.fitness.values[0]:.3f}")


if __name__ == "__main__":
    main()
