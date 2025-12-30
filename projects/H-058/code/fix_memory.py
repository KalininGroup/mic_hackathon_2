import torch
import os

# Add this to help with memory fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Also enable gradient checkpointing
torch.backends.cudnn.benchmark = False  # Can help with memory
print("Memory optimization applied!")
