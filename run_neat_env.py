#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import argparse

import neat
import numpy as np
import gymnasium as gym
from neat.reporting import BaseReporter

class EarlyStop(Exception):
    """用来触发提前终止进化"""
    pass

class StatsReporter(BaseReporter):
    """在每代评估后检查平均 fitness，并在达到阈值时抛出 EarlyStop"""
    def __init__(self, avg_thr):
        self.avg_thr = avg_thr
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species_set, best_genome):
        fits = [g.fitness for g in population.values() if g.fitness is not None]
        if not fits:
            return
        avg_fit = sum(fits) / len(fits)
        print(f"Gen {self.generation}: avg fitness = {avg_fit:.3f}")
        if self.avg_thr is not None and avg_fit >= self.avg_thr:
            raise EarlyStop(f"avg fitness {avg_fit:.3f} ≥ {self.avg_thr}")

    # 下面两个方法是 BaseReporter 的抽象方法，这里不做处理
    def complete_extinction(self): pass
    def found_solution(self, config, generation, best): pass


class DirectFeatureEnv:
    """
    直接从 LunarLander-v3 环境读取 8 维特征，并在 reset 时初始化 shaping 状态。
    特征顺序：x, y, vx, vy, angle, angular_velocity, pad_x, pad_y
    """
    def __init__(self, gravity, render=False, fps=5.0):
        mode = "human" if render else "rgb_array"
        self.env0 = gym.make(
            "LunarLander-v3",
            render_mode=mode,
            gravity=gravity
        )
        self.env = self.env0.unwrapped
        # shaping 参数
        self.vy_lim      = 2.0
        self.same_thr    = 100
        self.speed_pen   = 50.0
        self.repeat_pen  = 300.0
        self.angle_thr   = 0.1
        self.vel_thr     = 1.0
        self.gate_frames = max(1, int(0.3 * fps))
        self.pos_gate    = 0.3
        self.neg_gate    = 0.3

    def reset(self):
        obs, info = self.env0.reset()
        # 初始化 shaping 状态
        x, y = obs[0], obs[1]
        pad_x = getattr(self.env, "helipad_x", 0.0)
        pad_y = getattr(self.env, "helipad_y", 0.0)
        self.phi      = -np.hypot(x - pad_x, y - pad_y)
        self.gating   = False
        self.gate_cnt = 0
        self.gate_ok  = True
        self.last_act = None
        self.same_cnt = 0
        return self._make_state(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env0.step(action)
        state = self._make_state(obs)
        done = terminated or truncated
        return state, reward, done

    def _make_state(self, obs):
        pad_x = getattr(self.env, "helipad_x", 0.0)
        pad_y = getattr(self.env, "helipad_y", 0.0)
        return np.array([
            obs[0], obs[1], obs[2], obs[3],
            obs[4], obs[5], pad_x, pad_y
        ], dtype=np.float32)

    def close(self):
        self.env0.close()


def eval_genomes(genomes, config):
    """对每个 genome 运行一个 episode，fitness = env_reward + alpha * shaped_reward"""
    env       = eval_genomes.env
    max_steps = eval_genomes.max_steps
    alpha     = eval_genomes.alpha

    for gid, genome in genomes:
        net    = neat.nn.FeedForwardNetwork.create(genome, config)
        state  = env.reset()
        env_r  = 0.0
        shaped = 0.0

        for _ in range(max_steps):
            out    = net.activate(state.tolist())
            act    = int(np.argmax(out))
            state, r, done = env.step(act)
            env_r += r

            # —— shaping reward —— #
            env.same_cnt = env.same_cnt + 1 if act == env.last_act else 1
            env.last_act = act

            # 发动机启动奖励
            if act == 2:
                shaped += 0.2
            elif act in (1, 3):
                shaped += 0.1

            # 势能 shaping
            x, y, vx, vy, angle, ang_vel, pad_x, pad_y = state
            phi2 = -np.hypot(x - pad_x, y - pad_y)
            shaped += (0.99 * phi2 - env.phi)
            env.phi = phi2

            # 超速 & 重复惩罚
            if abs(vy) > env.vy_lim:
                shaped -= env.speed_pen
            if env.same_cnt >= env.same_thr:
                shaped -= env.repeat_pen

            # 门控奖励/惩罚
            if not env.gating and act in (1, 2, 3):
                env.gating   = True
                env.gate_cnt = env.gate_frames
                env.gate_ok  = True

            if env.gating:
                if abs(angle) > env.angle_thr or abs(vy) > env.vel_thr:
                    env.gate_ok = False
                env.gate_cnt -= 1
                if env.gate_cnt <= 0:
                    shaped += (env.pos_gate if env.gate_ok else -env.neg_gate)
                    env.gating = False

            if done:
                break

        genome.fitness = env_r + alpha * shaped
        print(f"  Genome {gid}: env={env_r:.3f}, shaped={shaped:.3f}, total={genome.fitness:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="neat-config.ini",
                        help="NEAT 配置文件路径（num_inputs=8）")
    parser.add_argument("--gens",          type=int,   default=500,
                        help="最大进化世代数")
    parser.add_argument("--steps",         type=int,   default=300,
                        help="每个 genome 的最大步数")
    parser.add_argument("--gravity",       type=float, default=-3.5,
                        help="LunarLander 环境重力")
    parser.add_argument("--alpha",         type=float, default=0.05,
                        help="shaping 奖励缩放系数")
    parser.add_argument("--render",        action="store_true",
                        help="是否以 human 模式渲染环境")
    parser.add_argument("--avg_threshold", type=float, default=None,
                        help="平均 fitness 阈值，达到即提前停止")
    parser.add_argument("--resume",        default=None,
                        help="（可选）checkpoint 前缀，用于恢复训练")
    args = parser.parse_args()

    # 准备环境
    wrapper = DirectFeatureEnv(
        gravity=args.gravity,
        render=args.render,
        fps=5.0
    )
    eval_genomes.env       = wrapper
    eval_genomes.max_steps = args.steps
    eval_genomes.alpha     = args.alpha

    # 加载 NEAT 配置
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config
    )

    # 创建或恢复种群
    if args.resume and os.path.exists(f"{args.resume}.pkl"):
        pop = neat.Checkpointer.restore_checkpoint(args.resume)
        print(f"[INFO] Restored checkpoint at generation {pop.generation}")
    else:
        pop = neat.Population(config)

    # 添加 Reporter（包括 early‐stop）
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(StatsReporter(avg_thr=args.avg_threshold))
    # pop.add_reporter(neat.Checkpointer(
    #     generation_interval=10,
    #     filename_prefix="neat-shape-checkpoint-"
    # ))

    # 运行进化
    try:
        winner = pop.run(eval_genomes, args.gens)
    except EarlyStop as e:
        print(f"[INFO] EarlyStop: {e}")
        winner = max(pop.population.values(), key=lambda g: g.fitness)

    # 存盘最终 checkpoint
    # cp = neat.Checkpointer(filename_prefix="neat-shape-checkpoint-")
    # cp.save_checkpoint(config, pop.population, pop.species, pop.generation)
    # print(f"[INFO] final checkpoint at generation {pop.generation}")

    # 保存最佳基因组
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("[INFO] saved best_genome.pkl")

    # 保存最佳网络
    best_net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open("best_network.pkl", "wb") as f:
        pickle.dump(best_net, f)
    print("[INFO] saved best_network.pkl")

    wrapper.close()
