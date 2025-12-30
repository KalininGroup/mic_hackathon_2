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

# %%
from matplotlib import pyplot as plt
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
import hdbscan
from matplotlib.colors import LogNorm

import torch
from torch.utils.data import DataLoader

import data
import model


# %%

# %%
def load_network(ver, rot_loss):
    """Load the trained network.
       ver: Model version.
       rot_loss: If trained including the rotation loss. 
    """
    
    lbl = 'rotloss' if rot_loss else 'normal'
    net = model.ConvAutoencoder(in_channels=1, latent_ch=4, out_activation='none').cuda()
    
    state_dict = torch.load(d_models / f'model_angle_{lbl}_v{ver}.pt')
    net.load_state_dict(state_dict)
    
    return net


# %%
d_models = Path('/data/incaem/common/eriksen/hackathon/models')


# %%
# Location of the input data.
d = Path('/pnfs/pic.es/data/incaem/microscopy-hackathon/4dSTEMdata/1_1mrad_233nm_fov_256_256_12pA_CL460mm')
path = d / 'scan_x256_y256.raw'

# %%
ds = data.get_dataset(path).cuda()
loader = DataLoader(ds, batch_size=128, shuffle=False)


# %%
def find_features(net, loader):
    """Find the features predicted by the network.
       net: Auto-encoder network.
       loader: Data loader.
    """
    
    # By now also reconstructing the image
    net = net.eval()
    
    predL = []
    reconL = []
    for bimgs in tqdm(loader):
        with torch.no_grad():
            pred = net.enc(bimgs)
            
        tmp = pred.reshape((len(pred), -1))
        predL.append(tmp.cpu())
    
    comb = torch.cat(predL)

    return comb


# %%
def cluster_samples(inp, Nmin=60):
    """Cluster the feature vectors.
       inp: Input features.
       Nmin: Minimum cluster size in HDBScan.
    """
    
    clusterer = hdbscan.HDBSCAN(min_cluster_size=Nmin)
    cluster_id_map = clusterer.fit_predict(inp)
    cluster_id_map = cluster_id_map.reshape((256, 256))
    
    return cluster_id_map


# %% [markdown]
# ## Estimate the clusters.

# %%
ver = 5
net1 = load_network(ver, rot_loss=False)
net2 = load_network(ver, rot_loss=True)

# %%
feat1 = find_features(net1, loader)
feat2 = find_features(net2, loader)

# %%
# %%time
Nmin = 60
cluster1 = cluster_samples(feat1, Nmin=Nmin)
cluster2 = cluster_samples(feat2, Nmin=Nmin)

# %% [markdown]
# ## Create a map of the classes.

# %%
# Estimating the variance. Just loading again to have on a different format.
imgs_in = data.load_data(path)
imgs = torch.Tensor(imgs_in)

# Plotting the variance to compare.
std = imgs.std(dim=[2,3])
X = std**2
S = pd.Series(X.flatten())
vmax = S.quantile(0.99)

# %%
fig, A = plt.subplots(ncols=3, sharey=True)
fig.set_size_inches([20, 5.5])

fs = 20
ax = A[0]
ax.imshow(cluster1, origin='lower')
ax.set_title('Default', fontsize=fs)

ax = A[1]
ax.imshow(cluster2, origin='lower')
ax.set_title('Rotation-invariant latent constraint', fontsize=fs)

ax = A[2]
ax.imshow(X, vmax=vmax, origin='lower')
ax.set_title('Variance in reciprocal space', fontsize=fs)
#plt.savefig('figs/clusters_v6.pdf')

# %% [markdown]
# ## Plotting the diffraction pattern for different classes.

# %%
cl_list = set(cluster2.flatten())

# %%
# These will change if running again... Yes, fun!
#cl_list = [0, 4, -1]

cl_list = [1, 2, -1]

# %%
fig, A = plt.subplots(nrows=2, ncols=len(cl_list), sharex='row', sharey='row')
fig.set_size_inches([20, 10])

for i,cl_id in tqdm(enumerate(cl_list)):
    mask = torch.Tensor(1*(cluster2 == cl_id))
    tmp = (mask[:,:,None,None]*imgs).sum(dim=[0,1]) / mask.sum()
    
    ax = A[0,i]
    ax.imshow(1*(mask == 1), origin='lower', cmap="Blues", alpha=0.7) #, alpha=0.4)
    #ax.set_title(f'ID: {cl_id}')
    
    ax = A[1,i]
    ax.imshow(tmp, origin='lower', norm=LogNorm(), cmap='plasma')

plt.savefig('figs/cluster_regions_v3.pdf')

# %%
