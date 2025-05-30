# data_loader.py
import os
from torch.utils.data import Dataset
from PIL import Image
import torch

class FrameDataset(Dataset):
    def __init__(self, frames_dir: str, manifest_path: str,
                 transform=None):
        self.frames_dir = frames_dir
        with open(manifest_path, 'r') as f:
            self.filenames = [line.strip() for line in f if line.strip()]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        path = os.path.join(self.frames_dir, fname)
        img = Image.open(path).convert('L')
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        return img, idx
