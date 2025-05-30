#!/usr/bin/env python3
# replay_neat_yolo.py

import argparse
import pickle
import neat
import numpy as np
from env_yolo_kalman import FeatureEnv

def main():
    parser = argparse.ArgumentParser(
        description="Replay a trained NEAT genome in the YOLO+Kalman LunarLander environment"
    )
    parser.add_argument("--config", default="neat-config.ini",
                        help="path to the NEAT config file")
    parser.add_argument("--genome", default="best_genome.pkl",
                        help="path to the pickled best genome")
    parser.add_argument("--model", required=True,
                        help="path to the YOLO .pt weights file")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="render fps for YOLO window")
    parser.add_argument("--gravity", type=float, default=-3.5,
                        help="gravity setting for LunarLander")
    parser.add_argument("--episodes", type=int, default=1,
                        help="number of episodes to replay")
    args = parser.parse_args()

    # Load genome
    with open(args.genome, 'rb') as f:
        genome = pickle.load(f)

    # Load NEAT config and build network
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        args.config
    )
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    # Create environment with YOLO+Kalman and rendering
    env = FeatureEnv(
        model_path = args.model,
        title      = "LunarLander Replay",
        fps        = args.fps,
        gravity    = args.gravity,
        launch_env = True
    )

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        print(f"Starting episode {ep}")

        while not done:
            # Forward pass through the network
            output = net.activate(state.tolist())  # 4 outputs
            action = int(np.argmax(output))         # discrete action: 0-3

            # Step the environment
            state, reward, done = env.step(action)
            total_reward += reward

        print(f"Episode {ep} finished, total reward = {total_reward:.3f}")

    env.close()

if __name__ == "__main__":
    main()
