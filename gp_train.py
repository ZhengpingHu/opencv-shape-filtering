#!/usr/bin/env python3
# gp_train.py

import argparse
import random
import numpy as np
from deap import base, creator, tools, algorithms
from env_yolo_kalman import FeatureEnv

# -----------------------------------------------------------------------------
# hyperparameter
# -----------------------------------------------------------------------------
INPUT_DIM  = 6
H1_DIM     = 32
H2_DIM     = 32
OUTPUT_DIM = 4
# total hyper
D = INPUT_DIM*H1_DIM + H1_DIM + H1_DIM*H2_DIM + H2_DIM + H2_DIM*OUTPUT_DIM + OUTPUT_DIM

# -----------------------------------------------------------------------------
# mapping to MLP
# -----------------------------------------------------------------------------
def decode_and_act(ind, state):
    idx = 0
    def get_w(shape):
        nonlocal idx
        size = shape[0]*shape[1]
        w = np.array(ind[idx:idx+size]).reshape(shape)
        idx += size
        return w
    def get_b(length):
        nonlocal idx
        b = np.array(ind[idx:idx+length])
        idx += length
        return b

    W1 = get_w((H1_DIM, INPUT_DIM));  b1 = get_b(H1_DIM)
    W2 = get_w((H2_DIM, H1_DIM));     b2 = get_b(H2_DIM)
    W3 = get_w((OUTPUT_DIM, H2_DIM)); b3 = get_b(OUTPUT_DIM)

    x = state
    h1 = np.tanh(W1.dot(x) + b1)
    h2 = np.tanh(W2.dot(h1) + b2)
    out = W3.dot(h2) + b3
    return int(np.argmax(out))

# -----------------------------------------------------------------------------
# calculate the reward
# -----------------------------------------------------------------------------
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
        step = 0
        while not done:
            action = decode_and_act(individual, state)
            state, r, done = env.step(action)
            total_reward += r
            step += 1
            if debug:
                print(f"\r[Ep Step {step}] step_reward={r:.3f}  cum_reward={total_reward:.3f}", end="")
        if debug:
            print()
        env.close()
        return (total_reward,)
    return evaluate

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',   required=True, help="YOLO .pt model path")
    p.add_argument('--title',   default="LunarLander-v3", help="window frame")
    p.add_argument('--fps',     type=float, default=5.0,   help="fps")
    p.add_argument('--gravity', type=float, default=-3.5,  help="gravity")
    p.add_argument('--pop',     type=int,   default=50,    help="population")
    p.add_argument('--gen',     type=int,   default=20,    help="generation")
    p.add_argument('--cxpb',    type=float, default=0.5,   help="crossover rate")
    p.add_argument('--mutpb',   type=float, default=0.2,   help="mutation rate")
    p.add_argument('--debug',   action='store_true',
                   help="display reward")
    args = p.parse_args()

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.uniform, -1.0, 1.0)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=D)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate",
                     make_evaluate(args.model,
                                   args.title,
                                   args.fps,
                                   args.gravity,
                                   args.debug))
    toolbox.register("mate",    tools.cxTwoPoint)
    toolbox.register("mutate",  tools.mutGaussian,
                     mu=0, sigma=0.2, indpb=0.05)
    toolbox.register("select",  tools.selTournament, tournsize=3)

    # evo population
    pop = toolbox.population(n=args.pop)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # first evoluation
    fits = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fits):
        ind.fitness.values = fit
    record = stats.compile(pop)
    print(f"Gen 0: max_reward={record['max']:.3f}, avg_reward={record['avg']:.3f}")

    # evaluate loop
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

        record = stats.compile(pop)
        print(f"Gen {gen}: max_reward={record['max']:.3f}, avg_reward={record['avg']:.3f}")

    # save the best
    best = hof[0]
    np.savetxt("best_weights.txt", best)
    print("Best fitness:", best.fitness.values[0])

if __name__ == "__main__":
    main()
