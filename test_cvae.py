#!/usr/bin/env python3
# test the trained cvae model.

import os
import random
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from PIL import Image

from model_improved import Encoder, Decoder, reparameterize

preprocess = transforms.Compose([

    transforms.Resize((64, 64)),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

def load_cvae(enc_path, dec_path, in_ch, latent, device):
    encoder = Encoder(in_ch, latent).to(device)
    decoder = Decoder(in_ch, latent).to(device)
    encoder.load_state_dict(torch.load(enc_path, map_location=device), strict=True)
    decoder.load_state_dict(torch.load(dec_path, map_location=device), strict=True)
    encoder.eval(); decoder.eval()
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
        
def reconstruct_h5(path, encoder, decoder, device):
    with h5py.File(path, 'r') as hf:
        N = hf['frames'].shape[0]
        idx = random.randrange(N)
        arr = hf['frames'][idx]
    img = Image.fromarray(arr.squeeze())
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        mu, logvar = encoder(tensor)
        z = reparameterize(mu, logvar)
        recon = decoder(z)
    return tensor.squeeze().cpu().numpy(), recon.squeeze().cpu().numpy(), idx

def reconstruct_mnist(root, encoder, decoder, device):
    ds = datasets.MNIST(root=root, train=False, download=True, transform=preprocess)
    idx = random.randrange(len(ds))
    tensor, _ = ds[idx]
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        mu, logvar = encoder(tensor)
        z = reparameterize(mu, logvar)
        recon = decoder(z)
    return tensor.squeeze().cpu().numpy(), recon.squeeze().cpu().numpy(), idx

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["h5","mnist"], default="h5")
    p.add_argument("--h5",        help="path to frames_gray.h5")
    p.add_argument("--mnist_root", default="./data")
    p.add_argument("--enc",       required=True)
    p.add_argument("--dec",       required=True)
    p.add_argument("--in_ch",     type=int, default=1)
    p.add_argument("--latent",    type=int, default=64)
    p.add_argument("--device",    default="cpu")
    p.add_argument("--outdir",    default=None)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder, decoder = load_cvae(args.enc, args.dec, args.in_ch, args.latent, device)

    if args.dataset == "h5":
        orig, recon, idx = reconstruct_h5(args.h5, encoder, decoder, device)
        title = f"Original idx={idx}"
    else:
        orig, recon, idx = reconstruct_mnist(args.mnist_root, encoder, decoder, device)
        title = f"MNIST idx={idx}"

    fig, axes = plt.subplots(1, 2, figsize=(6,3))
    axes[0].imshow(orig, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(title); axes[0].axis("off")
    axes[1].imshow(recon, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Reconstruction"); axes[1].axis("off")
    plt.tight_layout()

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)
        out = os.path.join(args.outdir, f"recon_{args.dataset}_{idx}.png")
        fig.savefig(out, dpi=150)
        print(f"Saved comparison to {out}")
    else:
        plt.show()

