import torch
import torch.nn as nn
import torch.nn.functional as F

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(ch)
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)

class Encoder(nn.Module):
    def __init__(self, in_ch=1, latent_dim=64):
        super().__init__()
        # 三阶段：64→128→256
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1), nn.BatchNorm2d(64),
            ResBlock(64)
        )  # 64×32×32
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128),
            ResBlock(128)
        )  # 128×16×16
        self.block3 = nn.Sequential(
            nn.Conv2d(128,256, 4, 2, 1), nn.BatchNorm2d(256),
            ResBlock(256)
        )  # 256×8×8
        self.fc_mu     = nn.Linear(256*8*8, latent_dim)
        self.fc_logvar = nn.Linear(256*8*8, latent_dim)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)

class Decoder(nn.Module):
    def __init__(self, out_ch=1, latent_dim=64):
        super().__init__()
        self.fc        = nn.Linear(latent_dim, 256*8*8)
        self.deconv1   = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128),
            ResBlock(128)
        )  # 128×16×16
        self.deconv2   = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64),
            ResBlock(64)
        )  # 64×32×32
        self.deconv3   = nn.ConvTranspose2d(64, out_ch, 4,2,1)  # 1×64×64

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(x.size(0),256,8,8)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = torch.sigmoid(self.deconv3(x))
        return x
