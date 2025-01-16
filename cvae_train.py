import gymnasium as gym
import numpy as np
import cv2
from mss import mss
from PIL import Image
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from support import subprocess
import os

def getWindow(window_name: str):
    # modify by linux 
    try:
        cmd_search = ["xdotool", "search", "--onlyvisible", "--name", window_name]
        window_ids = subprocess.check_output(cmd_search).decode("utf-8").strip().split("\n")
        if not window_ids or window_ids[0].strip() == "":
            print(f"[getWindow] No window found with name: {window_name}")
            return None
        window_id = window_ids[0].strip()

        cmd_info = ["xwininfo", "-id", window_id]
        info_output = subprocess.check_output(cmd_info).decode("utf-8").strip().split("\n")

        left, top, width, height = None, None, None, None
        for line in info_output:
            line = line.strip()
            if line.startswith("Absolute upper-left X:"):
                left = int(line.split(":")[1])
            elif line.startswith("Absolute upper-left Y:"):
                top = int(line.split(":")[1])
            elif line.startswith("Width:"):
                width = int(line.split(":")[1])
            elif line.startswith("Height:"):
                height = int(line.split(":")[1])

        if None in [left, top, width, height]:
            print("[getWindow] Failed to parse xwininfo output.")
            return None
        return (left, top, width, height)

    except subprocess.CalledProcessError as e:
        print(f"[getWindow] Error: {e}")
        return None


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


# Encoder setting
class Encoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=16):
        super(Encoder, self).__init__()
        # passage 8*16
        self.enc_conv1 = nn.Conv2d(in_channels, 8, kernel_size=4, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1)
        # input 64x64 => conv1(8,32,32) => conv2(16,16,16)
        self.fc_mu = nn.Linear(16*16*16, latent_dim)
        self.fc_logvar = nn.Linear(16*16*16, latent_dim)

    def forward(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, out_channels=1, latent_dim=16):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 16*16*16)
        self.dec_deconv1 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.dec_deconv2 = nn.ConvTranspose2d(8, out_channels, kernel_size=4, stride=2, padding=1)
        # at last: B,1,64,64

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(x.size(0), 16, 16, 16) 
        x = F.relu(self.dec_deconv1(x))
        x = torch.sigmoid(self.dec_deconv2(x))
        return x


class CVAE(nn.Module):
    def __init__(self, in_channels=1, latent_dim=16):
        super(CVAE, self).__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(out_channels=1, latent_dim=latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar


def cvae_loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld


# Single channel grayscale
class FrameDataset(Dataset):
    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_bgr = self.frames[idx]
        # Single gray channel
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # resize
        frame_gray = cv2.resize(frame_gray, (64, 64), interpolation=cv2.INTER_AREA)
        # reshape 64,64

        # normalization
        tensor_img = torch.tensor(frame_gray, dtype=torch.float32).unsqueeze(0) / 255.0
        return tensor_img


def main():
    # 1. Get lunar lander v3 (the newest one window)
    # 2. Get the window (only for linux)
    # 3. Using loop for the screenshot by opencv
    # 4. Dataloader, build the CVAE, training
    # Possible overfitting and running out of memory in step 4
    # Waiting for evolave
    # 5. After training, save the parameter matrix.
    print(os.environ["DISPLAY"])
    env = gym.make("LunarLander-v3", render_mode="human", gravity=-3.5)
    env.reset()
    time.sleep(3)
    window_name = "window"
    geometry = getWindow(window_name)
    if geometry is None:
        print(f"Failed to find or parse window '{window_name}'.")
        return
    left, top, width, height = geometry
    print(f"Window geometry: left={left}, top={top}, w={width}, h={height}")


    frames = []
    num_epis = 30
    max_steps = 50
    ep_num = 50
    sct = mss()

    for ep in range(num_epis):
        env.reset()
        for step in range(max_steps):
            action = env.action_space.sample() # random actions here, for the training robutness.
            obs, reward, terminated, truncated, info = env.step(action)

            monitor = {
                'top': top,
                'left': left,
                'width': width,
                'height': height
            }
            sct_img = sct.grab(monitor)
            img_rgb = Image.frombytes('RGB', (width, height), sct_img.rgb)
            frame_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
            frames.append(frame_bgr)

            if terminated or truncated:
                break

    env.close()
    print(f"Collected frames: {len(frames)}")
    # Build dataset and datasetloader. (Could be evolve)
    dataset = FrameDataset(frames)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Build gray channel CVAE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cvae = CVAE(in_channels=1, latent_dim=16).to(device)
    optimizer = optim.Adam(cvae.parameters(), lr=1e-3)

    epochs = 50
    for epoch in range(epochs):
        cvae.train()
        total_loss = 0
        for batch in dataloader:
            batch = batch.to(device)  # B,1,64,64
            recon, mu, logvar = cvae(batch) # B,1,64,64
            loss = cvae_loss_function(recon, batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {avg_loss:.4f}")

    # Save the trained encoder
    torch.save(cvae.encoder.state_dict(), "encoder_only.pth")
    print("Saved encoder_only.pth")


if __name__ == "__main__":
    main()
