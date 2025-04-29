#!/usr/bin/env python3
# test the trained cvae model.

import os
import random
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from model_improved import Encoder, Decoder, reparameterize

preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

def load_cvae(enc_path, dec_path, in_ch, latent_dim, device='cpu'):
    encoder = Encoder(in_ch, latent_dim).to(device)
    decoder = Decoder(in_ch, latent_dim).to(device)
    encoder.load_state_dict(torch.load(enc_path, map_location=device), strict=True)
    decoder.load_state_dict(torch.load(dec_path, map_location=device), strict=True)
    encoder.eval()
    decoder.eval()
    return encoder, decoder

def reconstruct_and_show(h5_path, encoder, decoder,
                         in_ch, latent_dim, device='cpu', save_dir=None):
    # random frame from dataset.
    with h5py.File(h5_path, 'r') as hf:
        N   = hf['frames'].shape[0]
        idx = random.randrange(N)
        arr = hf['frames'][idx]
    img    = Image.fromarray(arr.squeeze())
    tensor = preprocess(img).unsqueeze(0).to(device)  # 1×C×64×64
    with torch.no_grad():
        mu, logvar = encoder(tensor)
        z          = reparameterize(mu, logvar)
        recon      = decoder(z)
    orig_np  = tensor.squeeze().cpu().numpy()
    recon_np = recon.squeeze().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(6,3))
    axes[0].imshow(orig_np,  cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Original idx={idx}")
    axes[0].axis('off')
    axes[1].imshow(recon_np, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        outpath = os.path.join(save_dir, f"recon_{idx}.png")
        fig.savefig(outpath, dpi=150)
        print(f"Saved comparison image to {outpath}")
    else:
        plt.show()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Test improved CVAE reconstruction")
    p.add_argument("--h5",        required=True,
                   help="Path to frames_gray.h5")
    p.add_argument("--enc",       required=True,
                   help="Path to enc_epXXX.pth")
    p.add_argument("--dec",       required=True,
                   help="Path to dec_epXXX.pth")
    p.add_argument("--in_ch",     type=int, default=1,
                   help="Input channel count (e.g. 1)")
    p.add_argument("--latent",    type=int, default=64,
                   help="Latent dimension (e.g. 64)")
    p.add_argument("--device",    default="cpu",
                   help="Device: cuda or cpu")
    p.add_argument("--outdir",    default=None,
                   help="If set, save comparison images there")
    args = p.parse_args()

    encoder, decoder = load_cvae(
        args.enc, args.dec, args.in_ch, args.latent, device=args.device
    )
    reconstruct_and_show(
        args.h5, encoder, decoder,
        args.in_ch, args.latent,
        device=args.device,
        save_dir=args.outdir
    )

