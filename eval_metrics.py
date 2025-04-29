#!/usr/bin/env python3
import h5py
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

from model import Encoder, Decoder, reparameterize

def compute_metrics(h5_path, enc_path, dec_path, device='cpu', n_samples=500):
    enc = Encoder(1,16).to(device).eval()
    dec = Decoder(1,16).to(device).eval()
    enc.load_state_dict(torch.load(enc_path, map_location=device))
    dec.load_state_dict(torch.load(dec_path, map_location=device))

    with h5py.File(h5_path, 'r') as f:
        N = f['frames'].shape[0]
        idxs = np.random.choice(N, size=min(n_samples, N), replace=False)
        preprocess = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
        ])

        mses = []
        for i in tqdm(idxs, desc="Evaluating", ncols=80):
            arr = f['frames'][i]
            img = Image.fromarray(arr.squeeze())
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

