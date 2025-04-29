#!/usr/bin/env python3
# train_improved.py

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from tqdm import tqdm

from model_improved import Encoder, Decoder, reparameterize
from h5_dataset      import H5Dataset
from sampler         import DiversitySampler

# SSIM Loss
from pytorch_msssim import ssim

# VGG Perceptual Loss
import torchvision.models as models
# We'll initialize VGG after parsing args to get device

def perceptual_loss(x, y, vgg):
    # x,y: [B,1,64,64] → repeat to 3 channels
    xx = vgg(x.repeat(1,3,1,1))
    yy = vgg(y.repeat(1,3,1,1))
    return F.l1_loss(xx, yy, reduction='mean')

def train_cvae(
    h5_path, save_dir,
    batch_size=32, epochs=100, lr=1e-3, max_beta=1.0,
    device='cuda', workers=4, fp16=False
):
    # 1) load dataset & split
    ds = H5Dataset(h5_path, normalize=True, transform=None)
    N = len(ds)
    idxs = np.random.permutation(N)
    split = int(0.1 * N)
    val_ds = Subset(ds, idxs[:split])
    tr_ds  = Subset(ds, idxs[split:])

    tr_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        sampler=DiversitySampler(tr_ds, batch_size, num_segments=8),
        num_workers=workers,
        pin_memory=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )

    # 2) model, optimizer, scheduler
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(1, 64).to(device)
    decoder = Decoder(1, 64).to(device)
    if fp16:
        encoder.half()
        decoder.half()

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # load VGG for perceptual loss
    vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
    for p in vgg.parameters():
        p.requires_grad = False

    # 3) augmentation pipeline
    aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    logs = {'tr_loss': [], 'val_loss': []}

    # 4) main training loop
    for epoch in range(1, epochs + 1):
        beta = max_beta * (epoch / epochs)  # linear KL annealing

        encoder.train(); decoder.train()
        pbar = tqdm(tr_loader, desc=f"Train Epoch {epoch}/{epochs}", ncols=100, leave=False)
        train_loss = 0.0

        for batch in pbar:
            # batch: ndarray H×W or H×W×1
            x = torch.stack([aug(b.squeeze()) for b in batch], dim=0).to(device)
            if fp16:
                x = x.half()

            mu, logvar = encoder(x)
            z = reparameterize(mu, logvar)
            recon = decoder(z)

            # losses
            l1 = F.l1_loss(recon, x, reduction='mean')
            ssim_l = 1 - ssim(recon, x, data_range=1.0, size_average=True)
            perc_l = perceptual_loss(recon, x, vgg)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = l1 + 0.5 * ssim_l + 0.1 * perc_l + beta * kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{train_loss/(pbar.n+1):.4f}"})

        scheduler.step()
        logs['tr_loss'].append(train_loss / len(tr_ds))

        # --- validation ---
        encoder.eval(); decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val   Epoch {epoch}/{epochs}", ncols=100, leave=False):
                x = torch.stack([aug(b.squeeze()) for b in batch], dim=0).to(device)
                mu, logvar = encoder(x)
                recon = decoder(reparameterize(mu, logvar))
                l1 = F.l1_loss(recon, x, reduction='mean')
                ssim_l = 1 - ssim(recon, x, data_range=1.0, size_average=True)
                perc_l = perceptual_loss(recon, x, vgg)
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                val_loss += (l1 + 0.5*ssim_l + 0.1*perc_l + beta*kld).item()
        logs['val_loss'].append(val_loss / len(val_ds))

        print((
            f"Epoch {epoch}/{epochs}  "
            f"TrainLoss {logs['tr_loss'][-1]:.4f}  "
            f"ValLoss {logs['val_loss'][-1]:.4f}"
        ))

        # save checkpoints
        os.makedirs(save_dir, exist_ok=True)
        torch.save(encoder.state_dict(), os.path.join(save_dir, f"enc_ep{epoch:03d}.pth"))
        torch.save(decoder.state_dict(), os.path.join(save_dir, f"dec_ep{epoch:03d}.pth"))

    # 5) plot losses
    plt.plot(logs['tr_loss'], label='train')
    plt.plot(logs['val_loss'], label='val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    print(f"Saved loss curve to {save_dir}/loss_curve.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Improved CVAE Training")
    p.add_argument("--h5",       required=True, help="path to frames_gray.h5")
    p.add_argument("--save_dir", default="models2", help="checkpoints dir")
    p.add_argument("--batch",    type=int,   default=32)
    p.add_argument("--epochs",   type=int,   default=100)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--max_beta", type=float, default=1.0,
                   help="final weight for KLD")
    p.add_argument("--device",   default="cuda")
    p.add_argument("--workers",  type=int,   default=4)
    p.add_argument("--fp16",     action="store_true",
                   help="use float16 for model and data")
    args = p.parse_args()

    train_cvae(
        args.h5, args.save_dir,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        max_beta=args.max_beta,
        device=args.device,
        workers=args.workers,
        fp16=args.fp16
    )

