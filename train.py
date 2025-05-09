#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from tqdm import tqdm

from model import Encoder, Decoder, reparameterize, cvae_loss
from h5_dataset import H5Dataset
from sampler import DiversitySampler

def train_cvae(
    h5_path, save_dir,
    batch_size=32, epochs=50, lr=1e-3, beta=1.0,
    device='cuda', num_workers=2, fp16=False
):
    # 1) split dataset.
    dataset = H5Dataset(h5_path, normalize=True, transform=None)
    N = len(dataset)
    idxs = np.arange(N)
    np.random.shuffle(idxs)
    split = int(0.1 * N)
    val_idxs, train_idxs = idxs[:split], idxs[split:]

    train_set = Subset(dataset, train_idxs)
    val_set   = Subset(dataset, val_idxs)

    train_sampler = DiversitySampler(train_set,
                                     batch_size=batch_size,
                                     num_segments=8)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              num_workers=num_workers,
                              pin_memory=True,
                              prefetch_factor=2)
    val_loader = DataLoader(val_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)

    # 2) model.
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    encoder = Encoder(1, 64).to(device)
    decoder = Decoder(1, 64).to(device)
    if fp16:
        encoder.half(); decoder.half()

    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr
    )
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    os.makedirs(save_dir, exist_ok=True)

    # train log
    logs = {
        'train_loss': [], 'train_recon': [], 'train_kld': [],
        'val_loss': [],   'val_recon':   [], 'val_kld':   []
    }

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((64,64)),
        transforms.ToTensor(),
    ])

    for epoch in range(1, epochs+1):
        # train
        encoder.train(); decoder.train()
        tl, tr, tk = 0.0, 0.0, 0.0
        pbar = tqdm(train_loader,
                    desc=f"[Train] Epoch {epoch}/{epochs}",
                    ncols=100, leave=False)
        for batch in pbar:
            # Preprocessing：resize & to tensor
            imgs = torch.stack([transform(img.squeeze()) for img in batch], dim=0)
            imgs = imgs.to(device)
            if fp16: imgs = imgs.half()

            mu, logvar = encoder(imgs)
            z = reparameterize(mu, logvar)
            recon = decoder(z)

            loss, recon_loss, kld = cvae_loss(recon, imgs, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tl += loss.item()
            tr += recon_loss.item()
            tk += kld.item()
            pbar.set_postfix({
                'loss': f"{tl/((pbar.n+1)*batch_size):.4f}"
            })

        scheduler.step()
        logs['train_loss'].append(tl/len(train_set))
        logs['train_recon'].append(tr/len(train_set))
        logs['train_kld'].append(tk/len(train_set))

        # eval
        encoder.eval(); decoder.eval()
        vl, vr, vk = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader,
                              desc=f"[ Valid] Epoch {epoch}/{epochs}",
                              ncols=100, leave=False):
                imgs = torch.stack([transform(img.squeeze()) for img in batch], dim=0)
                imgs = imgs.to(device)
                if fp16: imgs = imgs.half()

                mu, logvar = encoder(imgs)
                z = reparameterize(mu, logvar)
                recon = decoder(z)

                loss, recon_loss, kld = cvae_loss(recon, imgs, mu, logvar, beta)
                vl += loss.item()
                vr += recon_loss.item()
                vk += kld.item()

        logs['val_loss'].append(vl/len(val_set))
        logs['val_recon'].append(vr/len(val_set))
        logs['val_kld'].append(vk/len(val_set))

        # print conslog.
        print(
            f"Epoch {epoch}/{epochs}  "
            f"Train Loss {logs['train_loss'][-1]:.4f}  "
            f"Val Loss {logs['val_loss'][-1]:.4f}"
        )

        # save model
        torch.save(encoder.state_dict(),
                   os.path.join(save_dir, f"enc_ep{epoch:02d}.pth"))
        torch.save(decoder.state_dict(),
                   os.path.join(save_dir, f"dec_ep{epoch:02d}.pth"))

    # 4) result curve
    epochs_range = range(1, epochs+1)
    plt.figure(figsize=(10,6))
    plt.plot(epochs_range, logs['train_loss'], label='Train Total')
    plt.plot(epochs_range, logs['val_loss'],   label='Val Total')
    plt.plot(epochs_range, logs['train_recon'], '--', label='Train Recon')
    plt.plot(epochs_range, logs['val_recon'],   '--', label='Val Recon')
    plt.plot(epochs_range, logs['train_kld'],   ':', label='Train KLD')
    plt.plot(epochs_range, logs['val_kld'],     ':', label='Val KLD')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    print(f"Saved training curves to {save_dir}/training_curves.png")
    plt.close()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Enhanced CVAE Training Script")
    p.add_argument("--h5",       required=True, help="Path to frames.h5")
    p.add_argument("--save_dir", default="models", help="Where to save checkpoints")
    p.add_argument("--batch",    type=int,   default=32)
    p.add_argument("--epochs",   type=int,   default=50)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--beta",     type=float, default=1.0, help="KLD weight")
    p.add_argument("--device",   default="cuda")
    p.add_argument("--workers",  type=int,   default=2)
    p.add_argument("--fp16",     action="store_true",
                   help="Use float16 to save memory")
    args = p.parse_args()

    train_cvae(
        args.h5, args.save_dir,
        batch_size=args.batch,
        epochs=args.epochs,
        lr=args.lr,
        beta=args.beta,
        device=args.device,
        num_workers=args.workers,
        fp16=args.fp16
    )