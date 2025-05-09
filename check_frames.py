import os
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ========== 配置 ==========
filename = "frames_gray.h5"         # 要读取的 H5 文件名
save_dir = "selected_frames"        # 筛选后帧保存目录
script_dir = os.path.dirname(os.path.abspath(__file__))
h5_path = os.path.join(script_dir, filename)

# ========== 路径确认 ==========
print(f"当前脚本目录: {script_dir}")
print(f"读取文件路径: {h5_path}")
assert os.path.exists(h5_path), f"❌ 文件未找到: {h5_path}"

# ========== 创建保存目录 ==========
os.makedirs(save_dir, exist_ok=True)

# ========== 打开 H5 文件 ==========
with h5py.File(h5_path, "r") as h5f:
    keys = list(h5f.keys())
    if not keys:
        raise ValueError("❌ H5 文件中未包含任何 dataset。")

    dataset_key = keys[0]
    print(f"✔ 使用数据集键: '{dataset_key}'")

    dataset = h5f[dataset_key]
    total_frames = len(dataset)
    print(f"✔ 总帧数: {total_frames}")

    # 启用交互显示
    plt.ion()

    for i in range(total_frames):
        frame = dataset[i]

        # 若数据非uint8，则先转换（例如float32范围[0,1]）
        if frame.dtype != np.uint8:
            frame = (frame * 255).clip(0, 255).astype(np.uint8)

        # 从原始数组直接创建图像（保持原样）
        img = Image.fromarray(frame)

        # 显示图像
        plt.imshow(img, cmap='gray' if frame.ndim == 2 else None)
        plt.title(f"Frame {i}")
        plt.axis("off")
        plt.draw()
        plt.pause(0.001)

        # 用户交互
        ans = input(f"是否保存帧 {i}? (y/n/q): ").strip().lower()
        if ans == "y":
            out_path = os.path.join(save_dir, f"frame_{i:05d}.png")
            img.save(out_path)
            print(f"✔ 已保存 {out_path}")
        elif ans == "q":
            print("✅ 已中止。")
            break

        plt.clf()
