import os, json, random
from collections import defaultdict
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
import gymnasium as gym

# === 配置 ===
PER_CLASS = 1000
SCALE = 30.0
IMG_W, IMG_H = 600, 400
BBOX_W, BBOX_H = 80, 80
SPLIT = {'train': 0.7, 'val': 0.1, 'test': 0.2}
ROOT_LANDER = "./pose_lander"
ROOT_TERRAIN = "./pose_terrain"
VIS_COUNT = 100
random.seed(42)

def world_to_pixel(x, y): return int(x * SCALE), int(IMG_H - y * SCALE)
def norm_xy(x, y): return x / IMG_W, y / IMG_H
def get_leg_tip(leg, center):
    verts = [leg.GetWorldPoint(v) for v in leg.fixtures[0].shape.vertices]
    dists = [np.linalg.norm(np.array(v) - np.array(center)) for v in verts]
    return verts[int(np.argmax(dists))]
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

print("[INFO] Sampling with standard LunarLander-v3 environment and random actions...")
with tqdm(total=PER_CLASS * 4, desc="Sampling") as pbar:
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
                        tip0 = get_leg_tip(legs[0], (cx, cy))
                        tip1 = get_leg_tip(legs[1], (cx, cy))
                        terrain_world = [v for poly in env.unwrapped.sky_polys for v in poly[:2]]
                        terrain_pixels = [world_to_pixel(x, y) for x, y in terrain_world]
                        per_class_data[cls].append((frame.copy(), cls, (cx, cy), tip0, tip1, terrain_world, terrain_pixels))
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

# === 数据保存 ===
global_id = 0
index_map = []
for split_name, items in splits.items():
    make_dirs(ROOT_LANDER, split_name)
    make_dirs(ROOT_TERRAIN, split_name)

    for img, cls, center, tip0, tip1, terrain_world, terrain_pixels in tqdm(items, desc=f"Saving {split_name}"):
        name = f"{global_id:05d}"
        Image.fromarray(img).save(f"{ROOT_LANDER}/{split_name}/images/{name}.jpg")
        Image.fromarray(img).save(f"{ROOT_TERRAIN}/{split_name}/images/{name}.jpg")

        # === Lander ===
        cx_pix, cy_pix = world_to_pixel(*center)
        x_center, y_center = cx_pix / IMG_W, cy_pix / IMG_H
        bbox_w, bbox_h = BBOX_W / IMG_W, BBOX_H / IMG_H
        def kpt_pix(xw, yw):
            xp, yp = world_to_pixel(xw, yw)
            return [xp / IMG_W, yp / IMG_H]
        kpts_lander = kpt_pix(*center) + kpt_pix(*tip0) + kpt_pix(*tip1)

        # === Terrain ===
        terrain_contour = sorted(terrain_pixels, key=lambda p: p[0])
        xs = np.linspace(0, IMG_W - 1, 11).astype(int)
        def find_nearest_y(x_target):
            candidates = [y for x, y in terrain_contour if abs(x - x_target) <= 2]
            if not candidates:
                candidates = [y for x, y in terrain_contour if abs(x - x_target) <= 10]
            return min(candidates) if candidates else IMG_H - 1
        kpts_terrain = []
        for x in xs:
            y = find_nearest_y(x)
            kpts_terrain += list(norm_xy(x, y))

        with open(f"{ROOT_LANDER}/{split_name}/labels/{name}.txt", "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f} " + " ".join(f"{x:.6f}" for x in kpts_lander))
        with open(f"{ROOT_TERRAIN}/{split_name}/labels/{name}.txt", "w") as f:
            f.write(f"0 0.5 0.5 1.0 1.0 " + " ".join(f"{x:.6f}" for x in kpts_terrain))

        index_map.append((name, cls, split_name))
        global_id += 1

# === 可视化 ===
sampled_vis = random.sample(index_map, VIS_COUNT)
for name, cls, split in sampled_vis:
    img_path = f"{ROOT_LANDER}/{split}/images/{name}.jpg"
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
    for i in range(0, len(pts), 2):
        px = int(pts[i] * IMG_W)
        py = int(pts[i+1] * IMG_H)
        cv2.circle(vis, (px, py), 4, (0,255,0), -1)
    cv2.imwrite(f"./visuals/vis_{name}.jpg", vis)

# === YAML ===
with open(f"{ROOT_LANDER}/dataset.yaml", "w") as f:
    f.write(f"""
train: {ROOT_LANDER}/train/images
val: {ROOT_LANDER}/val/images
test: {ROOT_LANDER}/test/images

nc: 1
names: ['lander']
kpt_shape: [3, 2]
""")
with open(f"{ROOT_TERRAIN}/dataset.yaml", "w") as f:
    f.write(f"""
train: {ROOT_TERRAIN}/train/images
val: {ROOT_TERRAIN}/val/images
test: {ROOT_TERRAIN}/test/images

nc: 1
names: ['terrain']
kpt_shape: [11, 2]
""")

print("\n✅ 数据集生成完成，可视化已使用 label 数据进行重环。")
