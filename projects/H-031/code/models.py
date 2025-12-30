import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# -------------------------
# model
# -------------------------

LATENT1 = 48
LATENT2 = 12

class EncSto(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 2), nn.ReLU(),
            nn.Linear(2, LATENT1)
        )

    def forward(self, s):
        return self.net(s)                             # (B,32)

class DecSto(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT1, 2), nn.ReLU(),
            nn.Linear(2, 1)
        )

    def forward(self, z):
        return self.net(z)                               # (B,1)

class EncShared(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT1, 24), nn.ReLU(),
            nn.Linear(24, LATENT2)
        )

    def forward(self, z):
        return self.net(z)                             # (B,3)

class DecShared(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT2, 24), nn.ReLU(),
            nn.Linear(24, LATENT1)
        )

    def forward(self, z):
        return self.net(z)                             # (B,32)

class EncImg(nn.Module):
    def __init__(self, img_hw):
        super().__init__()
        self.img_hw = img_hw

        self.net = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=2, padding=1), nn.ReLU(),   # H/2, W/2
            nn.Conv2d(4, 16, 3, stride=2, padding=1), nn.ReLU(), # H/4, W/4
            nn.Conv2d(16, 64, 3, stride=2, padding=1), nn.ReLU() # H/8, W/8
        )

        # compute feature-map size dynamically
        h, w = img_hw
        h = (h + 7) // 8   # equivalent to ceil(h / 8)
        w = (w + 7) // 8

        self.fc = nn.Sequential(
            nn.Linear(64 * h * w, 64), nn.ReLU(),
            nn.Linear(64, LATENT1)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)
        
class DecImg(nn.Module):
    def __init__(self, img_hw):
        super().__init__()
        self.img_hw = img_hw

        # encoder downsamples by factor 8
        start_h = int(np.ceil(img_hw[0] / 8))
        start_w = int(np.ceil(img_hw[1] / 8))
        self.start_h, self.start_w = start_h, start_w

        # latent → feature map (match encoder channels = 8)
        self.fc = nn.Sequential(
            nn.Linear(LATENT1, 64), nn.ReLU(),
            nn.Linear(64, 64 * start_h * start_w), nn.ReLU(),        
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(64, 16, 4, stride=2, padding=1), nn.ReLU(),  # ×2
            nn.ConvTranspose2d(16, 4, 4, stride=2, padding=1), nn.ReLU(),  # ×4
            nn.ConvTranspose2d(4, 1, 4, stride=2, padding=1),              # ×8
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 64, self.start_h, self.start_w)
        x = self.net(x)
        return x[:, :, :self.img_hw[0], :self.img_hw[1]]

class UniversalTranslator(nn.Module):
    def __init__(self, img_hw):
        super().__init__()
        self.enc_img = EncImg(img_hw)
        self.enc_sto = EncSto()
        self.enc_shared = EncShared()
        self.dec_shared = DecShared()
        self.dec_img = DecImg(img_hw)
        self.dec_sto = DecSto()

    def forward(self, img, sto):
        # -------- image path --------
        z_img1 = self.enc_img(img)
        z_img2  = self.enc_shared(z_img1)
        z_img2b = self.dec_shared(z_img2)

        img_i2i = self.dec_img(z_img2b)
        sto_i2s = self.dec_sto(z_img2b)

        # -------- stoich path --------
        z_sto1 = self.enc_sto(sto)
        z_sto2  = self.enc_shared(z_sto1)
        z_sto2b = self.dec_shared(z_sto2)

        img_s2i = self.dec_img(z_sto2b)
        sto_s2s = self.dec_sto(z_sto2b)

        return {
            "img_i2i": img_i2i,
            "sto_i2s": sto_i2s,
            "img_s2i": img_s2i,
            "sto_s2s": sto_s2s,
            "z_img2": z_img2,
            "z_sto2": z_sto2,
        }

# -------------------------
# dataset
# -------------------------

class PairedDataset(Dataset):
    def __init__(self, images_np, stoich_np, indices, img_hw, augment=False):
        self.images = images_np
        self.stoich = stoich_np
        self.indices = np.array(indices, dtype=int)
        self.img_hw = img_hw
        self.augment = augment

    def __len__(self):
        return len(self.indices)        

    def __getitem__(self, i):
        idx = self.indices[i]
        img = torch.tensor(self.images[idx], dtype=torch.float32).unsqueeze(0)  # (1,H,W)
        # downsample
        img = F.interpolate(img.unsqueeze(0), size=self.img_hw, mode="bilinear", align_corners=False).squeeze(0)
        sto = torch.tensor(self.stoich[idx], dtype=torch.float32)               # (1,)

        if self.augment:
            sto = sto + 0.0001 * torch.abs(sto).clamp_min(1e-6) * torch.randn_like(sto)
        return img, sto

