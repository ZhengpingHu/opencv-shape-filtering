# The function reuse from the GP course
# and the default lunar lander environment function.
import gymnasium as gym
import numpy as np


# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human", gravity=-3.5)

# Reset the environment to generate the first observation
# random seed could be 42, removed
observation, info = env.reset()

for _ in range(50):
    # this is where you would insert your policy
    
    print(type(env.action_space.sample()))
    #action = env.action_space.sample()
    action = 1
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)
    observation, info = env.reset()

    #  We hope the env could be different this time, so same env step been commented.
    #   observation, info = env.reset(seed=42)

env.close()