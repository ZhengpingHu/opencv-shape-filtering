import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# References:
# 1. Kullback–Leibler divergence. Wiki. https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
# Define the CVAE model.

class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=16):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim

        # Convolutional Layer
        self.enc_conv1 = nn.Conv2d(in_channels, 16, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)

        # Fully connected output layer 
        # Maybe add dropout in the future.
        self.fc_mu = nn.Linear(32 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(32 * 8 * 8, latent_dim)

    def forward(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=16):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        self.fc = nn.Linear(latent_dim, 32 * 8 * 8)
        self.dec_deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.dec_deconv2 = nn.ConvTranspose2d(16, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(x.size(0), 32, 8, 8)
        x = F.relu(self.dec_deconv1(x))
        x = torch.sigmoid(self.dec_deconv2(x))
        return x


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class CVAE(nn.Module):
    # In the training period, we keep both encoder and decoder to self_training.
    # But only keep encoder for next training period.
    def __init__(self, in_channels=3, latent_dim=16):
        super(CVAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


def cvae_loss(recon_x, x, mu, logvar):
    # Default loss function: MSE
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    # Kullback–Leibler divergence (1) see the reference
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


# Training period

def train_cvae(model, dataloader, epochs=5, lr=1e-3, device='cuda'):
    # Normalization to [0,1]
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_data in dataloader:
            if isinstance(batch_data, (tuple, list)):
                batch_data = batch_data[0]

            batch_data = batch_data.to(device)

            recon, mu, logvar = model(batch_data)
            loss = cvae_loss(recon, batch_data, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print("CVAE training finished!")


# Model save the load

def save_cvae(model, path="cvae_model.pth"):
    # Save the entire model.
    torch.save(model.state_dict(), path)

def load_cvae(model, path="cvae_model.pth", device='cpu'):
    # Load the model.
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


# Reasoning process

@torch.no_grad()
def encode_image(model, frame_bgr, device='cpu'):
    # still the same input image, the gray channel img.
    import cv2
    import numpy as np
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    import torch
    tensor_img = torch.tensor(frame_rgb, dtype=torch.float32).permute(2,0,1) / 255.0
    tensor_img = tensor_img.unsqueeze(0).to(device)
    mu, logvar = model(tensor_img)
    z = reparameterize(mu, logvar)
    return z[0]


@torch.no_grad()
def reconstruct(cvae_model, frame_bgr, device='cpu'):
    import cv2
    import numpy as np
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor_img = torch.tensor(frame_rgb, dtype=torch.float32).permute(2,0,1) / 255.0
    tensor_img = tensor_img.unsqueeze(0).to(device)
    recon, mu, logvar = cvae_model(tensor_img)
    # return to numpy
    recon_clamped = recon.clamp(0,1)[0].cpu().permute(1,2,0).numpy()
    recon_bgr = cv2.cvtColor((recon_clamped * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return recon_bgr
