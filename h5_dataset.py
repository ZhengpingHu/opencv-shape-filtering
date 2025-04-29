import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class H5Dataset(Dataset):
    def __init__(self, h5_path, transform=None, normalize=True):
        """
        h5_path: .h5 文件路径，内部应包含 'frames' dataset
        transform: 可选 torchvision.transforms
        normalize: 如果 True，将 uint8 [0,255] 归一化到 [0,1]
        """
        self.h5_path = h5_path
        self.transform = transform
        self.normalize = normalize
        # 延迟打开：这里只读取长度
        with h5py.File(self.h5_path, "r") as hf:
            self.length = hf["frames"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 每次打开文件，避免多进程冲突
        with h5py.File(self.h5_path, "r") as hf:
            img = hf["frames"][idx]  # numpy array, shape=(H,W) or (H,W,C)
        # 灰度扩通道
        if img.ndim == 2:
            img = img[:, :, np.newaxis]  # (H,W,1)
        # 转为 float
        img = img.astype(np.float32)
        if self.normalize:
            img /= 255.0
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        tensor = torch.from_numpy(img)  # shape=(C,H,W)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor
