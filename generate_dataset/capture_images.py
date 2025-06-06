import os
import json
import numpy as np
import gymnasium as gym
import cv2
from tqdm import tqdm

import fixed_env

import random
IMG_DIR = "images"
os.makedirs(IMG_DIR, exist_ok=True)

# 设置采样
NUM_IMAGES = 2000
ANGLES = np.linspace(-np.pi / 2, np.pi / 2, NUM_IMAGES)

# 固定位置（根据你的FixedLander设定）

GRAVITY = -3.5
ENV_NAME = "FixedLander-v3"

def main():
    labels = {}
    for idx, angle in tqdm(enumerate(ANGLES), total=NUM_IMAGES, desc="Capturing"):
        # FIXED_X = 600.0 / 30.0 / 2     # VIEWPORT_W / SCALE / 2
        FIXED_X = 10.0 + random.uniform(-9.5, 9.5)
        FIXED_HEIGHT = random.uniform(2.0, 6.0)
        FIXED_Y = (0.25 * 400 / 30.0) + FIXED_HEIGHT        
        env = gym.make(
            ENV_NAME,
            render_mode="rgb_array",
            gravity=GRAVITY,
            init_x=FIXED_X,
            init_y=FIXED_Y,
            init_angle=angle
        )
        obs, _ = env.reset()
        frame = env.render()

        filename = f"{idx:04d}.png"
        cv2.imwrite(os.path.join(IMG_DIR, filename), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        labels[filename] = {
            "init_x": FIXED_X,
            "init_y": FIXED_Y,
            "init_angle": float(angle)
        }

        env.close()

    with open("labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    print("Dataset collection complete.")

if __name__ == "__main__":
    main()
