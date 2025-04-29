import torch
import torch.nn as nn
import torch.nn.functional as F

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def cvae_loss(recon, x, mu, logvar, beta=1.0):
    # L1 重构损失 + β·KLD
    recon_loss = F.l1_loss(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld

class Encoder(nn.Module):
    def __init__(self, in_ch=1, latent_dim=64):
        super().__init__()
        # 输入 1×64×64
        self.enc_conv1 = nn.Conv2d(in_ch, 64, kernel_size=4, stride=2, padding=1)   # → 64×32×32
        self.enc_conv2 = nn.Conv2d(64,    128, kernel_size=4, stride=2, padding=1)  # →128×16×16
        self.enc_conv3 = nn.Conv2d(128,   256, kernel_size=4, stride=2, padding=1)  # →256× 8× 8
        self.fc_mu     = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, out_ch=1, latent_dim=64):
        super().__init__()
        self.fc          = nn.Linear(latent_dim, 256 * 8 * 8)
        self.dec_deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)  # →128×16×16
        self.dec_deconv2 = nn.ConvTranspose2d(128,  64, kernel_size=4, stride=2, padding=1)  # → 64×32×32
        self.dec_deconv3 = nn.ConvTranspose2d(64,   out_ch, kernel_size=4, stride=2, padding=1) # →1×64×64

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(x.size(0), 256, 8, 8)
        x = F.relu(self.dec_deconv1(x))
        x = F.relu(self.dec_deconv2(x))
        x = torch.sigmoid(self.dec_deconv3(x))
        return x

