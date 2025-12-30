# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: py4dstem
#     language: python
#     name: py4dstem
# ---

# %% [markdown]
# ## Train encoder on 4dSTEM data.

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
from pathlib import Path
from tqdm.notebook import tqdm

import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F

import data
import model

# %%
d_figs = Path('/nfs/pic.es/user/e/eriksen/proj/hackathon/figs')

# %%
# Location of the input data.
d = Path('/pnfs/pic.es/data/incaem/microscopy-hackathon/4dSTEMdata/1_1mrad_233nm_fov_256_256_12pA_CL460mm')
path = d / 'scan_x256_y256.raw'

# Where to store the trained model.
d_models = Path('/data/incaem/common/eriksen/hackathon/models')
ver = 5

# If including the rotational loss.
rot_loss = True
lbl = 'rotloss' if rot_loss else 'normal'

# Number of epochs
Nepochs = 10

# Number of features describing the material (of 16 total).
Ncore = 10

# %%
ds = data.get_dataset(path).cuda()
loader = DataLoader(ds, batch_size=32, shuffle=True)

# %%
net = model.ConvAutoencoder(in_channels=1, latent_ch=4, out_activation='none').cuda()
optimizer = optim.Adam(net.parameters(), lr=1e-4)


# %%

# %%
def loss_recon(bimgs, pred):
    return (bimg - pred).clip(-10, 10).abs().mean()


# %%

# %%
net = net.train()

for epoch in range(Nepochs):
    L = []
    for bimg in tqdm(loader):
        optimizer.zero_grad()
        pred, z = net(bimg)

        loss_recon1 = loss_recon(bimg, pred)

        angle = 260*np.random.random()
        bimg_rot = F.rotate(bimg, angle)
        pred_rot, z_rot = net(bimg_rot)
        
        loss_recon2 = loss_recon(bimg_rot, pred_rot)

        if rot_loss:
            zcore = z[:,:Ncore]
            zcore = z_rot[:,:Ncore]
            loss_z = (z - z_rot).abs().mean()
            loss = loss_recon1 + loss_recon2 + 50*loss_z
        else:
            loss = loss_recon1 + loss_recon2

    
        loss.backward()
        optimizer.step()
    
        L.append(loss.item())
    
    epoch_loss = sum(L) / len(L)
    
    print(epoch, epoch_loss)

# %%
# Storing the state.
torch.save(net.state_dict(), d_models / f'model_angle_{lbl}_v{ver}.pt')

# %%
