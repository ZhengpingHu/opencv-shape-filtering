#!/usr/bin/env python3
# env_headless.py

import gymnasium as gym
import time

def main():
    # render_mode="rgb_array" 不弹 GUI 窗口，step() 返回 frame
    env = gym.make("LunarLander-v3", render_mode="rgb_array", gravity=-3.5)
    obs, info = env.reset()
    try:
        while True:
            # action = env.action_space.sample()  # 随机，或者自行传入
            action = 0
            obs, reward, terminated, truncated, info = env.step(action)
            frame = info['rgb_array']  # ndarray H×W×3
            # 让主脚本来显示，而这里仅模拟数据流
            time.sleep(0.02)
            if terminated or truncated:
                obs, info = env.reset()
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    main()
