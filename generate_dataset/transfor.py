import json
import os
import math
import cv2

# 图像尺寸（分辨率固定）
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 400

# Lander 实际渲染出来的宽高估算（单位：像素）
LANDER_WIDTH = 60
LANDER_HEIGHT = 40

# 读取 labels.json
with open('labels.json', 'r') as f:
    label_data = json.load(f)

# 输出目录
output_dir = 'labels'
os.makedirs(output_dir, exist_ok=True)

for filename, info in label_data.items():
    x = info['init_x'] * 30.0  # 坐标转换回像素（Box2D world coord → pixels）
    y = IMAGE_HEIGHT - (info['init_y'] * 30.0)  # 注意 y 要反转（gym坐标原点在左下角）
    theta = info['init_angle']

    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    dx = LANDER_WIDTH / 2
    dy = LANDER_HEIGHT / 2

    # ⚠️ 顺时针旋转四个点（YOLO-OBB需要顺时针）
    corners = [
        (+dx, -dy),  # 右上
        (+dx, +dy),  # 右下
        (-dx, +dy),  # 左下
        (-dx, -dy),  # 左上
    ]

    rotated = []
    for px, py in corners:
        rx = x + px * cos_t - py * sin_t
        ry = y + px * sin_t + py * cos_t
        rotated.extend([rx / IMAGE_WIDTH, ry / IMAGE_HEIGHT])

    # 写入 txt（class_id=0）
    label_name = os.path.splitext(filename)[0] + '.txt'
    label_path = os.path.join(output_dir, label_name)
    with open(label_path, 'w') as f:
        f.write("0 " + " ".join(f"{p:.6f}" for p in rotated))

print("✅ 所有标签已更新完成！")
