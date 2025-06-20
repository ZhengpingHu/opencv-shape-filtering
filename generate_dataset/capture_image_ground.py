#!/usr/bin/env python3
import os
import json
import random
import gymnasium as gym
import cv2
from tqdm import tqdm

# ====================
# 参数配置
# ====================
TOTAL_IMAGES = 2000    # 目标总图片数
START_INDEX = 2001     # 图片编号起始

def main():
    # 计算每个状态的配额
    per = TOTAL_IMAGES // 3
    rem = TOTAL_IMAGES % 3
    quotas = {1: per, 2: per, 3: per + rem}

    # 创建分类文件夹和初始化标签结构
    IMG_DIR_BASE = "images"
    os.makedirs(IMG_DIR_BASE, exist_ok=True)
    for s in [1, 2, 3]:
        os.makedirs(os.path.join(IMG_DIR_BASE, f"status{s}"), exist_ok=True)

    LABELS_PATHS = {s: f"labels_status{s}.json" for s in [1, 2, 3]}
    labels = {1: {}, 2: {}, 3: {}}
    counts = {1: 0, 2: 0, 3: 0}

    # 初始化环境
    env = gym.make('LunarLander-v3', render_mode='rgb_array')
    obs, _ = env.reset()

    image_counter = START_INDEX
    pbar = tqdm(total=TOTAL_IMAGES, desc="Capturing images")

    # 持续采集直到总数满足
    while sum(counts.values()) < TOTAL_IMAGES:
        action = random.randint(0, 3)
        obs, reward, terminated, truncated, _ = env.step(action)

        # 检测接地状态
        left_contact = bool(obs[6])
        right_contact = bool(obs[7])
        status = None
        if not left_contact and right_contact:
            status = 1
        elif left_contact and not right_contact:
            status = 2
        elif left_contact and right_contact:
            status = 3

        # 满足条件且未超配额时截取
        if status and counts[status] < quotas[status]:
            x, y = float(obs[0]), float(obs[1])
            angle = float(obs[4])

            frame = env.render()
            filename = f"{image_counter:04d}.png"
            folder = os.path.join(IMG_DIR_BASE, f"status{status}")
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            labels[status][filename] = {
                "init_x": x,
                "init_y": y,
                "init_angle": angle,
                "status": status
            }

            counts[status] += 1
            image_counter += 1
            pbar.update(1)

        # 如果 Episode 结束，重置环境
        if terminated or truncated:
            obs, _ = env.reset()

    pbar.close()
    env.close()

    # 写入各自标签文件
    for s in [1, 2, 3]:
        with open(LABELS_PATHS[s], 'w') as f:
            json.dump(labels[s], f, indent=2)

    print(f"Dataset collection complete. Totals: {counts}")


if __name__ == '__main__':
    main()
