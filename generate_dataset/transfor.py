#!/usr/bin/env python3
import json
import os
import math

# --------------------
# 常量配置
# --------------------
IMAGE_W, IMAGE_H = 600, 400
SCALE = 30.0           # 世界单位到像素的比例

# 状态逆归一化参数
half_w = (IMAGE_W / SCALE) / 2    # = 10.0 世界单位
half_h = (IMAGE_H / SCALE) / 2    # ≈ 6.667 世界单位
pad_y  = (IMAGE_H / SCALE) / 4    # ≈ 3.333 世界单位

# Lander 近似尺寸（像素）
LANDER_W, LANDER_H = 70, 50
# 半宽半高
dx, dy = LANDER_W / 2, LANDER_H / 2

# 用比例矫正质心和可视中心之间的系统性偏差（非硬编码）
OFFSET_X_RATIO = -0.05     # 暂设为 0，如有观察性偏差再微调
OFFSET_Y_RATIO = -0.3   # Lander 视觉中心偏下，建议微调 10~15%
offset_x = OFFSET_X_RATIO * LANDER_W
offset_y = OFFSET_Y_RATIO * LANDER_H

# 输入/输出
INPUT_JSON = 'labels_status3.json'
OUTPUT_DIR = 'labels'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 读取 JSON 数据
with open(INPUT_JSON, 'r') as f:
    data = json.load(f)

# 对每个样本生成旋转后的 OBB
for filename, info in data.items():
    obs_x = info['init_x']
    obs_y = info['init_y']
    theta = info['init_angle']
    status = info.get('status', 0)

    # 1) 逆归一化：归一化状态 → 世界坐标（米）
    world_x = obs_x * half_w + half_w
    world_y = obs_y * half_h + pad_y

    # 2) 世界坐标 → 像素坐标，再转换到图像坐标系（y 向下）
    cx = world_x * SCALE + offset_x
    cy = IMAGE_H - (world_y * SCALE) + offset_y

    # 3) 旋转框顶点（顺时针，基于中心(cx,cy)）
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    corners = [(-dx, -dy), (dx, -dy), (dx, dy), (-dx, dy)]

    norm = []
    for px_rel, py_rel in corners:
        # 顺时针旋转公式（屏幕坐标 y 向下）
        rx = cx + px_rel * cos_t + py_rel * sin_t
        ry = cy - px_rel * sin_t + py_rel * cos_t
        # 归一化到 [0,1]
        norm.append(rx / IMAGE_W)
        norm.append(ry / IMAGE_H)

    # 4) 写入 YOLO-OBB 标签文件
    out_txt = os.path.join(OUTPUT_DIR, filename.replace('.png', '.txt'))
    with open(out_txt, 'w') as fw:
        fw.write(f"{status} " + " ".join(f"{v:.6f}" for v in norm))

print("✅ transfor_rotated.py 已调整：考虑视觉中心与质心系统偏移，提升全局对齐效果。")