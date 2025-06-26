import os, json, random
from collections import defaultdict
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import gymnasium as gym

# === 配置 ===
PER_CLASS = 500
SCALE = 30.0
IMG_W, IMG_H = 600, 400
BBOX_W, BBOX_H = 100, 100
SPLIT = {'train': 0.7, 'val': 0.1, 'test': 0.2}
ROOT = "D:/Git/opencv-shape-filtering/generate_dataset"
ROOT_LANDER = f"{ROOT}/pose_lander"
ROOT_TERRAIN = f"{ROOT}/pose_terrain"
VIS_COUNT = 100
random.seed(42)

# === 工具函数 ===
def world_to_pixel(x, y): return int(x * SCALE), int(IMG_H - y * SCALE)
def norm_xy(x, y): return x / IMG_W, y / IMG_H
def get_leg_tip(leg, center):
    verts = [leg.GetWorldPoint(v) for v in leg.fixtures[0].shape.vertices]
    dists = [np.linalg.norm(np.array(v) - np.array(center)) for v in verts]
    return verts[int(np.argmax(dists))]
def get_leg_top(leg, center):
    verts = [leg.GetWorldPoint(v) for v in leg.fixtures[0].shape.vertices]
    dists = [np.linalg.norm(np.array(v) - np.array(center)) for v in verts]
    return verts[int(np.argmin(dists))]
def get_touchdown_class(legs):
    leg_left = min(legs, key=lambda l: l.position[0])
    leg_right = max(legs, key=lambda l: l.position[0])
    l, r = leg_left.ground_contact, leg_right.ground_contact
    return 0 if not l and not r else 1 if not l and r else 2 if l and not r else 3
def make_dirs(root, split): 
    for d in ['images', 'labels', 'meta']:
        os.makedirs(f"{root}/{split}/{d}", exist_ok=True)
os.makedirs("./visuals", exist_ok=True)

# === 使用标准 LunarLander 环境 ===
env = gym.make("LunarLander-v3", render_mode="rgb_array")
per_class_data = {i: [] for i in range(4)}

print("[INFO] Collecting samples…")
with tqdm(total=PER_CLASS * 4, desc="Collecting samples…") as pbar:
    while any(len(v) < PER_CLASS for v in per_class_data.values()):
        obs, _ = env.reset()
        done = False
        step = 0

        while not done:
            action = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc

            if step % 5 == 0:
                try:
                    lander = env.unwrapped.lander
                    legs = env.unwrapped.legs
                    cls = get_touchdown_class(legs)
                    if len(per_class_data[cls]) < PER_CLASS:
                        frame = env.render()
                        cx, cy = lander.position
                        angle = lander.angle
                        tip0 = get_leg_tip(legs[0], (cx, cy))
                        tip1 = get_leg_tip(legs[1], (cx, cy))
                        top0 = get_leg_top(legs[0], (cx, cy))
                        top1 = get_leg_top(legs[1], (cx, cy))

                        # 本体上下移动修正
                        structure = [
                            (-0.433, 0.283 + 0.1*2),
                            ( 0.433, 0.283 + 0.1*2),
                            ( 0.433, -0.183 + 0.1 - 0.067*2),
                            (-0.433, -0.183 + 0.1 - 0.067*2)
                        ]
                        def rotate_point(cx, cy, angle, dx, dy):
                            cos_a, sin_a = np.cos(angle), np.sin(angle)
                            x = cx + cos_a * dx - sin_a * dy
                            y = cy + sin_a * dx + cos_a * dy
                            return x, y

                        corners = [rotate_point(cx, cy, angle, dx, dy) for dx, dy in structure]
                        points = [*corners, (cx, cy), top0, tip0, top1, tip1]

                        kpts = []
                        skip = False
                        for x, y in points:
                            px, py = world_to_pixel(x, y)
                            nx, ny = px / IMG_W, py / IMG_H
                            if not (0 <= nx <= 1 and 0 <= ny <= 1):
                                skip = True
                                break
                            kpts.extend([nx, ny, 2])

                        if skip:
                            continue

                        terrain_world = [v for poly in env.unwrapped.sky_polys for v in poly[:2]]
                        terrain_pixels = [world_to_pixel(x, y) for x, y in terrain_world]

                        per_class_data[cls].append((frame.copy(), cls, (cx, cy), kpts, terrain_pixels))
                        pbar.update(1)
                except Exception as e:
                    print("[WARN] Skip one frame:", e)
            step += 1

env.close()

# === 数据分割 ===
splits = {'train': [], 'val': [], 'test': []}
for cls, items in per_class_data.items():
    random.shuffle(items)
    n = len(items)
    n_train = int(n * SPLIT['train'])
    n_val = int(n * SPLIT['val'])
    splits['train'].extend(items[:n_train])
    splits['val'].extend(items[n_train:n_train + n_val])
    splits['test'].extend(items[n_train + n_val:])

# === 保存 Lander 数据 ===
global_id = 0
index_map = []
for split_name, items in splits.items():
    make_dirs(ROOT_LANDER, split_name)

    for img, cls, center, kpts_lander, terrain_pixels in tqdm(items, desc=f"Saving {split_name} lander"):
        name = f"{global_id:05d}"
        Image.fromarray(img).save(f"{ROOT_LANDER}/{split_name}/images/{name}.png")

        cx_pix, cy_pix = world_to_pixel(*center)
        x_center, y_center = cx_pix / IMG_W, cy_pix / IMG_H
        bbox_w, bbox_h = BBOX_W / IMG_W, BBOX_H / IMG_H

        with open(f"{ROOT_LANDER}/{split_name}/labels/{name}.txt", "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f} " + " ".join(f"{x:.6f}" for x in kpts_lander))

        index_map.append((name, cls, split_name))
        global_id += 1

# === 保存 Terrain 数据 ===
for split_name, items in splits.items():
    make_dirs(ROOT_TERRAIN, split_name)

    for i, (img, cls, center, kpts_lander, terrain_pixels) in enumerate(items):
        name = f"{i:05d}"
        Image.fromarray(img).save(f"{ROOT_TERRAIN}/{split_name}/images/{name}.png")

        terrain_contour = sorted(terrain_pixels, key=lambda p: p[0])
        xs = np.linspace(0, IMG_W - 1, 11).astype(int)

        def find_nearest_y(x_target):
            candidates = [y for x, y in terrain_contour if abs(x - x_target) <= 2]
            if not candidates:
                candidates = [y for x, y in terrain_contour if abs(x - x_target) <= 10]
            return min(candidates) if candidates else IMG_H - 1

        kpts_terrain = []
        valid = True
        for x in xs:
            y = find_nearest_y(x)
            nx, ny = norm_xy(x, y)
            if not (0 <= nx <= 1 and 0 <= ny <= 1):
                valid = False
                break
            kpts_terrain.extend([nx, ny, 2])

        if not valid:
            continue

        with open(f"{ROOT_TERRAIN}/{split_name}/labels/{name}.txt", "w") as f:
            f.write(f"0 0.5 0.5 1.0 1.0 " + " ".join(f"{x:.6f}" for x in kpts_terrain))

# === 可视化 ===
sampled_vis = random.sample(index_map, VIS_COUNT)
for name, cls, split in sampled_vis:
    img_path = f"{ROOT_LANDER}/{split}/images/{name}."
    label_path = f"{ROOT_LANDER}/{split}/labels/{name}.txt"
    img = np.array(Image.open(img_path))
    vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    with open(label_path, 'r') as f:
        line = f.readline().strip().split()
        xc, yc, bw, bh = map(float, line[1:5])
        pts = list(map(float, line[5:]))
    x0 = int((xc - bw/2) * IMG_W)
    y0 = int((yc - bh/2) * IMG_H)
    x1 = int((xc + bw/2) * IMG_W)
    y1 = int((yc + bh/2) * IMG_H)
    cv2.rectangle(vis, (x0, y0), (x1, y1), (0,255,0), 1)
    points = []
    for i in range(0, len(pts), 3):
        px = int(pts[i] * IMG_W)
        py = int(pts[i+1] * IMG_H)
        cv2.circle(vis, (px, py), 4, (0,255,0), -1)
        points.append((px, py))

    if len(points) == 9:
        for i in range(4):
            cv2.line(vis, points[i], points[(i+1)%4], (255,0,255), 1)
            cv2.line(vis, points[i], points[4], (255,0,255), 1)
        cv2.line(vis, points[4], points[5], (255,0,255), 1)
        cv2.line(vis, points[4], points[7], (255,0,255), 1)
        cv2.line(vis, points[5], points[6], (255,0,255), 1)
        cv2.line(vis, points[7], points[8], (255,0,255), 1)

    cv2.imwrite(f"./visuals/vis_{name}.png", vis)

# === YAML ===
with open(f"{ROOT_LANDER}/dataset.yaml", "w") as f:
    f.write(f"""
train: {ROOT_LANDER}/train/images
val: {ROOT_LANDER}/val/images
test: {ROOT_LANDER}/test/images

nc: 1
names: ['lander']
kpt_shape: [9, 3]
skeleton:
  - [0, 1]
  - [1, 2]
  - [2, 3]
  - [3, 0]
  - [0, 4]
  - [1, 4]
  - [2, 4]
  - [3, 4]
  - [4, 5]
  - [4, 7]
  - [5, 6]
  - [7, 8]
""")

with open(f"{ROOT_TERRAIN}/dataset.yaml", "w") as f:
    f.write(f"""
train: {ROOT_TERRAIN}/train/images
val: {ROOT_TERRAIN}/val/images
test: {ROOT_TERRAIN}/test/images

nc: 1
names: ['terrain']
kpt_shape: [11, 3]
skeleton: []
""")

print("\n🚀 数据集生成完成，包括 lander + terrain，已根据 label 进行可视化重环。")
