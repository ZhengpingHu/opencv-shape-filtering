#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import argparse
import numpy as np
import neat
from env_yolo_kalman import FeatureEnv

class EarlyStop(Exception):
    pass

class StatsReporter:
    def __init__(self, avg_thr=None):
        self.avg_thr    = avg_thr
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        fits = [g.fitness for g in population.values() if g.fitness is not None]
        if not fits:
            return
        mx, av = max(fits), sum(fits) / len(fits)
        print(f"Gen {self.generation:2d}: max={mx:.3f}, avg={av:.3f}")
        if self.avg_thr is not None and av >= self.avg_thr:
            raise EarlyStop(f"avg {av:.3f} ≥ {self.avg_thr}")

    def info(self, msg):
        print(msg)

    def species_stagnant(self, sid, species):
        print(f"Species {sid} is stagnated: removing it")

    def complete_extinction(self):
        print("All species extinct.")

    def end_generation(self, config, population, species):
        pass

    def complete(self):
        pass

    def found_solution(self, config, generation, best):
        pass

def eval_genomes(genomes, config):
    env       = eval_genomes.env
    max_steps = eval_genomes.max_steps
    alpha     = eval_genomes.alpha
    fps       = eval_genomes.fps

    # 奖惩参数
    vy_lim, same_thr      = 2.0, 100
    speed_pen, repeat_pen = 50.0, 300.0
    angle_thr, vel_thr    = 0.1, 1.0
    gate_frames           = max(1, int(0.3 * fps))
    pos_gate, neg_gate    = 0.3, 0.3

    for gid, genome in sorted(genomes, key=lambda x: x[0]):
        net    = neat.nn.FeedForwardNetwork.create(genome, config)
        env_r  = 0.0
        shaped = 0.0

        state     = env.reset()
        last_act  = None
        same_cnt  = 0
        dx, dy    = state[0] - state[6], state[1] - state[7]
        phi       = -np.hypot(dx, dy)
        gating    = False
        gate_cnt  = 0
        gate_ok   = True

        for _ in range(max_steps):
            out = net.activate(state.tolist())
            act = int(np.argmax(out))

            same_cnt = same_cnt + 1 if act == last_act else 1
            last_act = act

            if act in (1, 2, 3) and not gating:
                gating, gate_cnt, gate_ok = True, gate_frames, True

            nxt, rwd, done = env.step(act)
            env_r += rwd

            # 发动机启动奖励
            if act == 2:
                shaped += 0.2
            elif act in (1, 3):
                shaped += 0.1

            # 势能 shaping
            dx2, dy2 = nxt[0] - nxt[6], nxt[1] - nxt[7]
            phi2     = -np.hypot(dx2, dy2)
            shaped  += (0.99 * phi2 - phi)
            phi       = phi2

            # 超速 & 重复惩罚
            if abs(nxt[3]) > vy_lim:
                shaped -= speed_pen
            if same_cnt >= same_thr:
                shaped -= repeat_pen

            # 门控奖励/惩罚
            if gating:
                if abs(nxt[4]) > angle_thr or abs(nxt[3]) > vel_thr:
                    gate_ok = False
                gate_cnt -= 1
                if gate_cnt <= 0:
                    shaped += (pos_gate if gate_ok else -neg_gate)
                    gating = False

            state = nxt
            if done:
                break

        genome.fitness = env_r + alpha * shaped
        print(f"  Genome {gid}: env={env_r:.3f}, shaped={shaped:.3f}, comp={genome.fitness:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="neat-config.ini")
    parser.add_argument("--model",         required=True)
    parser.add_argument("--fps",           type=float, default=5.0)
    parser.add_argument("--gravity",       type=float, default=-3.5)
    parser.add_argument("--render",        action="store_true")
    parser.add_argument("--steps",         type=int,   default=200)
    parser.add_argument("--gens",          type=int,   default=50)
    parser.add_argument("--alpha",         type=float, default=0.01)
    parser.add_argument("--avg_threshold", type=float, default=None)
    parser.add_argument("--resume",        default=None)
    args = parser.parse_args()

    # 环境 & eval 绑定
    env = FeatureEnv(
        args.model,
        title      = "LunarLander-NEAT",
        fps        = args.fps,
        gravity    = args.gravity,
        launch_env = args.render
    )
    eval_genomes.env       = env
    eval_genomes.max_steps = args.steps
    eval_genomes.alpha     = args.alpha
    eval_genomes.fps       = args.fps

    # 载入配置
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config
    )

    # 恢复或新建种群
    if args.resume and os.path.exists(args.resume):
        pop = neat.Checkpointer.restore_checkpoint(args.resume)
        # —— 补齐至 pop_size —— 
        existing = pop.population
        next_key = max(existing.keys()) + 1
        while len(pop.population) < config.pop_size:
            g = config.genome_type(next_key)
            g.configure_new(config.genome_config)
            pop.population[next_key] = g
            next_key += 1
        pop.species.speciate(config, pop.population, pop.generation)
        print(f"[INFO] Restored and filled to pop_size={config.pop_size}")
    else:
        pop = neat.Population(config)

    # 添加 Reporter
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(StatsReporter(avg_thr=args.avg_threshold))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(neat.Checkpointer(
        generation_interval=10,
        filename_prefix="neat-checkpoint-"
    ))

    # 开始演化
    try:
        winner = pop.run(eval_genomes, args.gens)
    except EarlyStop as e:
        print(f"[INFO] EarlyStop: {e}")
        # 若提前终止，用当前种群中 fitness 最优者作为 winner
        winner = max(pop.population.values(), key=lambda g: g.fitness)

    # 最后强制存盘 checkpoint
    cp = neat.Checkpointer(filename_prefix="neat-checkpoint-")
    cp.save_checkpoint(config, pop.population, pop.species, pop.generation)
    print(f"[INFO] final checkpoint={pop.generation}")

    # 保存最佳基因组
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("[INFO] saved best_genome.pkl")

    # 保存对应的 NEAT 网络
    best_net = neat.nn.FeedForwardNetwork.create(winner, config)
    with open("best_network.pkl", "wb") as f:
        pickle.dump(best_net, f)
    print("[INFO] saved best_network.pkl")

    env.close()
