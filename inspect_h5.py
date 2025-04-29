#!/usr/bin/env python3
import h5py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

h5_path = "frames_gray.h5"  # 改成你的文件名

with h5py.File(h5_path, "r") as hf:
    print("Keys:", list(hf.keys()))
    dset = hf["frames"]
    shape = dset.shape
    print("Dataset shape:", shape)    # 可能是 (N,H,W) 或 (N,H,W,C)
    print("Dtype:", dset.dtype)

    N = shape[0]
    # 随机取 3 帧
    idxs = np.random.choice(N, size=3, replace=False)
    for idx in idxs:
        img = dset[idx]   # numpy array
        print(f"\nFrame {idx}: shape={img.shape}, min={img.min()}, max={img.max()}, mean={img.mean():.3f}")

        plt.figure(figsize=(4,4))
        if img.ndim == 2:
            # 灰度
            plt.imshow(img, cmap="gray", vmin=0, vmax=255)
        elif img.ndim == 3 and img.shape[2] == 1:
            plt.imshow(img[:,:,0], cmap="gray", vmin=0, vmax=255)
        else:
            # 彩色
            plt.imshow(img)
        plt.title(f"Frame {idx}")
        plt.axis("off")

    plt.show()

