"""
PyTorch Dataset classes for AFM reconstruction
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, Optional, List
import h5py

class AFMDataset(Dataset):
    """Dataset for AFM tip and surface reconstruction"""
    
    def __init__(self, data_list: List[Dict], config, transform=None):
        """
        Args:
            data_list: List of dicts with 'tip', 'surface', 'image', etc.
            config: Configuration object
            transform: Optional transform to apply
        """
        self.data_list = data_list
        self.config = config
        self.transform = transform
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        sample = self.data_list[idx]
        
        # Convert to tensors
        image = torch.from_numpy(sample['image']).unsqueeze(0)  # [1, H, W]
        surface = torch.from_numpy(sample['surface']).unsqueeze(0)  # [1, H, W]
        tip = torch.from_numpy(sample['tip']).unsqueeze(0)  # [1, H, W, D]
        
        # Normalize
        image = self._normalize(image)
        surface = self._normalize(surface)
        tip = self._normalize(tip)
        
        data_dict = {
            'image': image.float(),
            'surface': surface.float(),
            'tip': tip.float(),
            'tip_type': sample['tip_type'],
            'surface_type': sample['surface_type']
        }
        
        if 'clean_image' in sample:
            clean_image = torch.from_numpy(sample['clean_image']).unsqueeze(0)
            clean_image = self._normalize(clean_image)
            data_dict['clean_image'] = clean_image.float()
        
        if self.transform:
            data_dict = self.transform(data_dict)
        
        return data_dict
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to [0, 1] range"""
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min > 1e-6:
            x = (x - x_min) / (x_max - x_min)
        return x


class CachedAFMDataset(Dataset):
    """Memory-efficient dataset using HDF5 caching"""
    
    def __init__(self, hdf5_path: str, config, transform=None):
        """
        Args:
            hdf5_path: Path to HDF5 file
            config: Configuration object
            transform: Optional transform
        """
        self.hdf5_path = hdf5_path
        self.config = config
        self.transform = transform
        
        # Open file to get length
        with h5py.File(hdf5_path, 'r') as f:
            self.length = len(f['images'])
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """Load single sample from HDF5"""
        with h5py.File(self.hdf5_path, 'r') as f:
            image = torch.from_numpy(f['images'][idx]).unsqueeze(0)
            surface = torch.from_numpy(f['surfaces'][idx]).unsqueeze(0)
            tip = torch.from_numpy(f['tips'][idx]).unsqueeze(0)
            
            # Get metadata
            tip_type = f['tip_types'][idx].decode('utf-8')
            surface_type = f['surface_types'][idx].decode('utf-8')
        
        # Normalize
        image = self._normalize(image)
        surface = self._normalize(surface)
        tip = self._normalize(tip)
        
        data_dict = {
            'image': image.float(),
            'surface': surface.float(),
            'tip': tip.float(),
            'tip_type': tip_type,
            'surface_type': surface_type
        }
        
        if self.transform:
            data_dict = self.transform(data_dict)
        
        return data_dict
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize to [0, 1] range"""
        x_min = x.min()
        x_max = x.max()
        if x_max - x_min > 1e-6:
            x = (x - x_min) / (x_max - x_min)
        return x


def save_dataset_hdf5(dataset: List[Dict], filepath: str):
    """Save dataset to HDF5 format for efficient loading"""
    import h5py
    
    print(f"Saving dataset to {filepath}...")
    
    with h5py.File(filepath, 'w') as f:
        # Get dimensions from first sample
        n_samples = len(dataset)
        img_shape = dataset[0]['image'].shape
        surf_shape = dataset[0]['surface'].shape
        tip_shape = dataset[0]['tip'].shape
        
        # Create datasets
        f.create_dataset('images', shape=(n_samples, *img_shape), dtype=np.float32)
        f.create_dataset('surfaces', shape=(n_samples, *surf_shape), dtype=np.float32)
        f.create_dataset('tips', shape=(n_samples, *tip_shape), dtype=np.float32)
        
        if 'clean_image' in dataset[0]:
            f.create_dataset('clean_images', shape=(n_samples, *img_shape), dtype=np.float32)
        
        # String datasets for metadata
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('tip_types', shape=(n_samples,), dtype=dt)
        f.create_dataset('surface_types', shape=(n_samples,), dtype=dt)
        
        # Fill datasets
        for i, sample in enumerate(dataset):
            if i % 1000 == 0:
                print(f"  Saved {i}/{n_samples} samples")
            
            f['images'][i] = sample['image']
            f['surfaces'][i] = sample['surface']
            f['tips'][i] = sample['tip']
            f['tip_types'][i] = sample['tip_type']
            f['surface_types'][i] = sample['surface_type']
            
            if 'clean_image' in sample:
                f['clean_images'][i] = sample['clean_image']
    
    print(f"✓ Saved {n_samples} samples to {filepath}")


def create_dataloaders(config, use_cached: bool = False, cache_dir: str = './data/cache'):
    """
    Create train, validation, and test dataloaders
    
    Args:
        config: Configuration object
        use_cached: Whether to use cached HDF5 datasets
        cache_dir: Directory for cached datasets
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from data.simulator import AFMSimulator
    
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    train_cache = cache_path / 'train.h5'
    val_cache = cache_path / 'val.h5'
    test_cache = cache_path / 'test.h5'
    
    if use_cached and all([train_cache.exists(), val_cache.exists(), test_cache.exists()]):
        print("Loading cached datasets...")
        train_dataset = CachedAFMDataset(str(train_cache), config)
        val_dataset = CachedAFMDataset(str(val_cache), config)
        test_dataset = CachedAFMDataset(str(test_cache), config)
        
    else:
        print("Generating datasets from scratch...")
        simulator = AFMSimulator(config)
        
        # Generate datasets
        train_data = simulator.generate_dataset(config.data.train_size, 'train')
        val_data = simulator.generate_dataset(config.data.val_size, 'val')
        test_data = simulator.generate_dataset(config.data.test_size, 'test')
        
        # Save to HDF5 for future use
        save_dataset_hdf5(train_data, str(train_cache))
        save_dataset_hdf5(val_data, str(val_cache))
        save_dataset_hdf5(test_data, str(test_cache))
        
        # Create PyTorch datasets
        train_dataset = AFMDataset(train_data, config)
        val_dataset = AFMDataset(val_data, config)
        test_dataset = AFMDataset(test_data, config)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # For detailed evaluation
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    print(f"✓ Created dataloaders:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


class DataAugmentation:
    """Data augmentation for AFM images"""
    
    def __init__(self, config):
        self.config = config
    
    def __call__(self, sample: Dict) -> Dict:
        """Apply random augmentations"""
        # Random rotation (90 degree increments)
        if np.random.rand() < 0.5:
            k = np.random.randint(1, 4)
            sample['image'] = torch.rot90(sample['image'], k, dims=[1, 2])
            sample['surface'] = torch.rot90(sample['surface'], k, dims=[1, 2])
        
        # Random flip
        if np.random.rand() < 0.5:
            sample['image'] = torch.flip(sample['image'], dims=[1])
            sample['surface'] = torch.flip(sample['surface'], dims=[1])
        
        if np.random.rand() < 0.5:
            sample['image'] = torch.flip(sample['image'], dims=[2])
            sample['surface'] = torch.flip(sample['surface'], dims=[2])
        
        # Add small random noise
        if np.random.rand() < 0.3:
            noise = torch.randn_like(sample['image']) * 0.02
            sample['image'] = sample['image'] + noise
        
        return sample


def test_dataset():
    """Test dataset creation"""
    import matplotlib.pyplot as plt
    from config import get_config
    
    config = get_config()
    
    # Create small test dataset
    config.data.train_size = 100
    config.data.val_size = 20
    config.data.test_size = 20
    
    train_loader, val_loader, test_loader = create_dataloaders(
        config, use_cached=False
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    
    print("Batch shapes:")
    print(f"  Image: {batch['image'].shape}")
    print(f"  Surface: {batch['surface'].shape}")
    print(f"  Tip: {batch['tip'].shape}")
    
    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i in range(4):
        axes[0, i].imshow(batch['image'][i, 0].cpu().numpy(), cmap='afmhot')
        axes[0, i].set_title(f"Image {i}")
        axes[0, i].axis('off')
        
        axes[1, i].imshow(batch['surface'][i, 0].cpu().numpy(), cmap='viridis')
        axes[1, i].set_title(f"Surface {i}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_test.png', dpi=150, bbox_inches='tight')
    print("Saved dataset_test.png")

if __name__ == '__main__':
    test_dataset()
