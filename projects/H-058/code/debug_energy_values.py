"""
Debug the actual energy values in our dataset
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import numpy as np
from torch.utils.data import DataLoader

from vantascope.data.dft_graphene_loader import DFTGrapheneDataset, collate_dft_batch
from vantascope.utils.logging import logger

def analyze_energy_distribution():
    """Analyze the actual energy distribution in our dataset."""
    logger.info("🔍 Analyzing energy distribution...")
    
    # Load a sample of the dataset
    dataset = DFTGrapheneDataset(
        data_path="data/train",
        split='train',
        max_samples=100,  # Sample 100 files
        grid_size=512
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False,
        collate_fn=collate_dft_batch,
        num_workers=0
    )
    
    all_energies = []
    
    for batch in dataloader:
        energies = batch['energy'].numpy().flatten()
        all_energies.extend(energies)
    
    all_energies = np.array(all_energies)
    
    logger.info(f"📊 Energy Statistics:")
    logger.info(f"   Count: {len(all_energies)}")
    logger.info(f"   Min: {all_energies.min():.2f} eV")
    logger.info(f"   Max: {all_energies.max():.2f} eV")
    logger.info(f"   Mean: {all_energies.mean():.2f} eV")
    logger.info(f"   Std: {all_energies.std():.2f} eV")
    logger.info(f"   Median: {np.median(all_energies):.2f} eV")
    
    # Show some sample values
    logger.info(f"📋 Sample values:")
    for i, energy in enumerate(all_energies[:10]):
        logger.info(f"   Sample {i+1}: {energy:.2f} eV")
    
    return all_energies.mean(), all_energies.std()

if __name__ == "__main__":
    mean, std = analyze_energy_distribution()
    print(f"\nRecommended normalization:")
    print(f"energy_mean = {mean:.1f}")
    print(f"energy_std = {std:.1f}")
