#!/usr/bin/env python
# encoding: UTF8

import numpy as np
import torch

def load_data(path):
    """Loading the binary 4DSTEM data."""
    
    A = np.fromfile(path, dtype=np.float32)
    
    per_frame = 128*128 + 2*128
    #nx, ny = 315, 315
    nx, ny = 256, 256
    assert nx*ny * per_frame == len(A), 'Inconsistent numbers'
    
    # The metadata is appended at the end of each frame.
    hmm = A.reshape((nx*ny, per_frame))
    hmm = hmm[:,:128**2]
    
    data = hmm.flatten().reshape((nx, ny, 128, 128))
    
    return data

def get_dataset(path):
    """Load the data, normalize and convert into a PyTorch dataset."""

    
    imgs_in = load_data(path)

    # Hardcoding the values here... should be estimated per dataset.
    med = 12257.0205
    sig68 = 17031.10986328125

    data_norm = (imgs_in - med) / sig68
    data_norm = torch.Tensor(data_norm)
    ds = torch.Tensor(data_norm.reshape((-1, 128, 128))).unsqueeze(1)

    return ds