#!/usr/bin/env python3
"""
eval_metrics.py

在 HDF5 上批量评估 CVAE 重建性能，随机抽 n_samples 张帧，
并用 Resize(64×64) + ToTensor([0,1]) 保证与训练一致。
"""
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

from model import Encoder, Decoder, reparameterize

def compute_metrics(h5_path, enc_path, dec_path, device='cpu', n_samples=500):
    # 1) 加载模型
    enc = Encoder(1,16).to(device).eval()
    dec = Decoder(1,16).to(device).eval()
    enc.load_state_dict(torch.load(enc_path, map_location=device))
    dec.load_state_dict(torch.load(dec_path, map_location=device))

    # 2) 打开 HDF5 并取随机 idx
    with h5py.File(h5_path, 'r') as f:
        N = f['frames'].shape[0]
        idxs = np.random.choice(N, size=min(n_samples, N), replace=False)

        # 3) 预处理管线：PIL.Image → Resize(64×64) → Tensor([0,1])
        preprocess = transforms.Compose([
            transforms.Resize((64,64)),  # PIL.Image -> PIL.Image
            transforms.ToTensor(),       # PIL.Image -> Tensor C×64×64
        ])

        mses = []
        for i in tqdm(idxs, desc="Evaluating", ncols=80):
            arr = f['frames'][i]         # ndarray H×W 或 H×W×1
            # 转 PIL（squeeze 掉多余通道维）
            img = Image.fromarray(arr.squeeze())
            # Resize & ToTensor
            tensor = preprocess(img).unsqueeze(0).to(device)  # 1×1×64×64

            with torch.no_grad():
                mu, logvar = enc(tensor)
                z = reparameterize(mu, logvar)
                recon = dec(z)

            orig = tensor.squeeze().cpu().numpy()
            rec  = recon.squeeze().cpu().numpy()
            mse = np.mean((orig - rec)**2)
            mses.append(mse)

    mses = np.array(mses)
    psnr = 10 * np.log10(1.0 / (mses + 1e-8))
    print(f"Avg MSE over {len(mses)} samples: {mses.mean():.6f}")
    print(f"Avg PSNR: {psnr.mean():.2f} dB")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Evaluate CVAE: MSE & PSNR on random samples")
    p.add_argument("--h5",    required=True, help="Path to frames.h5")
    p.add_argument("--enc",   required=True, help="Path to encoder .pth")
    p.add_argument("--dec",   required=True, help="Path to decoder .pth")
    p.add_argument("--device",default="cpu",  help="cuda or cpu")
    p.add_argument("--n",     type=int, default=500,
                   help="Number of random samples to evaluate")
    args = p.parse_args()

    compute_metrics(args.h5, args.enc, args.dec, args.device, args.n)

