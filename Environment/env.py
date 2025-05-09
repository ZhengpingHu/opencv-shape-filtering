# env.py
import gymnasium as gym
import time

def main():
    env = gym.make("LunarLander-v3", render_mode="human", gravity=-3.5)
    try:
        while True:
            obs, info = env.reset()
            done = False
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    main()
