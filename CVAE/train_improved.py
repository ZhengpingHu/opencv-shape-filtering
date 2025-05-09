#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms
from tqdm import tqdm

from model_improved import Encoder, Decoder, reparameterize
from pytorch_msssim import ssim
import torchvision.models as models

from h5_dataset import H5Dataset
from sampler import DiversitySampler

def perceptual_loss(x, y, vgg):
    x3 = x.repeat(1,3,1,1)
    y3 = y.repeat(1,3,1,1)
    fx = vgg(x3)
    fy = vgg(y3)
    return F.l1_loss(fx, fy, reduction='mean')

def train_cvae(dataset, h5_path, mnist_root, save_dir,
               batch_size=32, epochs=100, lr=1e-3, max_beta=1.0,
               device='cuda', workers=4, fp16=False):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # dataload
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
        ])
        full = datasets.MNIST(root=mnist_root, train=True, download=True, transform=transform)
        n = len(full)
        vlen = int(0.1 * n)
        tlen = n - vlen
        tr_ds, val_ds = random_split(full, [tlen, vlen])
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                               num_workers=workers, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                num_workers=workers, pin_memory=True)
    else:
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

    # models
    encoder = Encoder(1, 64).to(device)
    decoder = Decoder(1, 64).to(device)
    if fp16:
        encoder.half()
        decoder.half()

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # vgg for loss
    vgg = models.vgg16(pretrained=True).features[:16].to(device).eval()
    if fp16:
        vgg.half()
    for p in vgg.parameters():
        p.requires_grad = False

    # data enhance for h5
    aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    logs = {'tr_loss': [], 'val_loss': []}
    size_tr = len(tr_ds)

    # train
    for epoch in range(1, epochs + 1):
        beta = max_beta * (epoch / epochs)
        encoder.train(); decoder.train()
        pbar = tqdm(tr_loader, desc=f"Train {epoch}/{epochs}", ncols=100, leave=False)
        tloss = 0.0
        for batch in pbar:
            # get dataset [B,1,64,64]
            if dataset == 'h5':
                x = torch.stack([aug(b.squeeze()) for b in batch], dim=0).to(device)
            else:
                x = batch[0].to(device)
            if fp16:
                x = x.half()

            mu, logvar = encoder(x)
            z = reparameterize(mu, logvar)
            recon = decoder(z)

            # loss
            l1 = F.l1_loss(recon, x, reduction='mean')
            ssim_l = 1 - ssim(recon, x, data_range=1.0, size_average=True)
            perc_l = perceptual_loss(recon, x, vgg)
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = l1 + 0.5 * ssim_l + 0.1 * perc_l + beta * kld

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tloss += loss.item()
            pbar.set_postfix({'loss': f"{tloss/(pbar.n+1):.4f}"})
        scheduler.step()
        logs['tr_loss'].append(tloss / size_tr)

        # recursive
        encoder.eval(); decoder.eval()
        vloss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Val   {epoch}/{epochs}", ncols=100, leave=False):
                if dataset == 'h5':
                    x = torch.stack([aug(b.squeeze()) for b in batch], dim=0).to(device)
                else:
                    x = batch[0].to(device)
                if fp16:
                    x = x.half()

                mu, logvar = encoder(x)
                recon = decoder(reparameterize(mu, logvar))
                l1 = F.l1_loss(recon, x, reduction='mean')
                ssim_l = 1 - ssim(recon, x, data_range=1.0, size_average=True)
                perc_l = perceptual_loss(recon, x, vgg)
                kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                vloss += (l1 + 0.5*ssim_l + 0.1*perc_l + beta*kld).item()
        logs['val_loss'].append(vloss / len(val_ds))

        print(f"Epoch {epoch}/{epochs} TrainLoss {logs['tr_loss'][-1]:.4f} ValLoss {logs['val_loss'][-1]:.4f}")

        os.makedirs(save_dir, exist_ok=True)
        torch.save(encoder.state_dict(), os.path.join(save_dir, f"enc_ep{epoch:03d}.pth"))
        torch.save(decoder.state_dict(), os.path.join(save_dir, f"dec_ep{epoch:03d}.pth"))

    # loss curve
    plt.plot(logs['tr_loss'], label='train')
    plt.plot(logs['val_loss'], label='val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Improved CVAE Training")
    p.add_argument("--dataset", choices=["h5","mnist"], default="h5")
    p.add_argument("--h5", help="path to frames_gray.h5")
    p.add_argument("--mnist_root", default="./data")
    p.add_argument("--save_dir",   default="models2")
    p.add_argument("--batch",      type=int,   default=32)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--max_beta",   type=float, default=1.0)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--workers",    type=int,   default=4)
    p.add_argument("--fp16",       action="store_true")
    args = p.parse_args()

    train_cvae(
        args.dataset,
        args.h5,
        args.mnist_root,
        args.save_dir,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        max_beta=args.max_beta,
        device=args.device,
        workers=args.workers,
        fp16=args.fp16
    )

