"""
physics_diffusion/data/dataloaders.py and simulators.py

Data loading and physics-based artifact simulation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import torchvision.transforms as transforms


# ============================================================================
# Artifact Simulators
# ============================================================================

class ArtifactSimulator:
    """Base class for artifact simulation."""
    
    def __init__(self, modality: str):
        self.modality = modality
    
    def simulate(self, clean_image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Simulate artifacts on clean image.
        
        Args:
            clean_image: [C, H, W] clean image
            
        Returns:
            noisy_image: Degraded image
            params: Dictionary of artifact parameters used
        """
        raise NotImplementedError


class AFMSimulator(ArtifactSimulator):
    """AFM tip convolution and scan artifact simulator."""
    
    def __init__(
        self,
        tip_radius_range: Tuple[float, float] = (5.0, 20.0),
        tip_aspect_range: Tuple[float, float] = (1.5, 3.0),
        noise_level: float = 0.05
    ):
        super().__init__('AFM')
        self.tip_radius_range = tip_radius_range
        self.tip_aspect_range = tip_aspect_range
        self.noise_level = noise_level
    
    def generate_tip_kernel(self, radius: float, aspect: float) -> torch.Tensor:
        """Generate AFM tip kernel."""
        size = int(radius * 3) | 1  # Odd size
        ax = torch.linspace(-(size-1)/2, (size-1)/2, size)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        r_eff = torch.sqrt((xx/radius)**2 + (yy/(radius*aspect))**2)
        kernel = torch.clamp(1.0 - r_eff, min=0.0)
        kernel = kernel / (kernel.sum() + 1e-8)
        
        return kernel
    
    def simulate(self, clean_image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Simulate AFM tip convolution."""
        # Random tip parameters
        radius = np.random.uniform(*self.tip_radius_range)
        aspect = np.random.uniform(*self.tip_aspect_range)
        
        # Generate and apply kernel
        kernel = self.generate_tip_kernel(radius, aspect)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
        
        C, H, W = clean_image.shape
        noisy = torch.zeros_like(clean_image)
        
        for c in range(C):
            img_c = clean_image[c:c+1].unsqueeze(0)  # [1, 1, H, W]
            conv = torch.nn.functional.conv2d(img_c, kernel, padding='same')
            noisy[c] = conv[0, 0]
        
        # Add noise
        noise = torch.randn_like(noisy) * self.noise_level
        noisy = noisy + noise
        
        params = {
            'tip_radius': radius,
            'tip_aspect': aspect,
            'noise_level': self.noise_level
        }
        
        return noisy, params


class TEMSimulator(ArtifactSimulator):
    """TEM CTF and noise simulator."""
    
    def __init__(
        self,
        defocus_range: Tuple[float, float] = (1000.0, 5000.0),  # nm
        voltage_kv: float = 300.0,
        cs_mm: float = 2.7,
        amp_contrast: float = 0.1,
        pixel_size: float = 1.0,  # Angstroms
        noise_level: float = 0.1
    ):
        super().__init__('TEM')
        self.defocus_range = defocus_range
        self.voltage_kv = voltage_kv
        self.cs_mm = cs_mm
        self.amp_contrast = amp_contrast
        self.pixel_size = pixel_size
        self.noise_level = noise_level
    
    def compute_ctf(self, shape: Tuple[int, int], defocus: float) -> torch.Tensor:
        """Compute CTF for given defocus."""
        H, W = shape
        
        # Frequency grid
        freq_y = torch.fft.fftfreq(H) / self.pixel_size
        freq_x = torch.fft.rfftfreq(W) / self.pixel_size
        ky, kx = torch.meshgrid(freq_y, freq_x, indexing='ij')
        k = torch.sqrt(kx**2 + ky**2)
        
        # Wavelength
        wavelength = 12.2643 / np.sqrt(
            self.voltage_kv * 1000 * (1 + self.voltage_kv * 1000 * 0.978e-6)
        )
        
        # Phase aberration
        defocus_m = defocus * 1e-9
        cs_m = self.cs_mm * 1e-3
        wavelength_m = wavelength * 1e-10
        k_ang = k * 1e10
        
        chi = (np.pi * wavelength_m * defocus_m * k_ang**2 + 
               0.5 * np.pi * cs_m * wavelength_m**3 * k_ang**4)
        
        # CTF
        A = self.amp_contrast
        ctf = -torch.sqrt(torch.tensor(1 - A**2)) * torch.sin(chi) - A * torch.cos(chi)
        
        return ctf
    
    def simulate(self, clean_image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Simulate TEM CTF modulation."""
        defocus = np.random.uniform(*self.defocus_range)
        
        C, H, W = clean_image.shape
        ctf = self.compute_ctf((H, W), defocus)
        
        noisy = torch.zeros_like(clean_image)
        
        for c in range(C):
            # Apply CTF in Fourier space
            fft_img = torch.fft.rfft2(clean_image[c])
            fft_modulated = fft_img * ctf
            noisy[c] = torch.fft.irfft2(fft_modulated, s=(H, W))
        
        # Add Poisson noise
        noisy = torch.poisson(torch.clamp(noisy * 100, 0)) / 100.0
        noisy = noisy + torch.randn_like(noisy) * self.noise_level
        
        params = {
            'defocus': defocus,
            'voltage_kv': self.voltage_kv,
            'cs_mm': self.cs_mm
        }
        
        return noisy, params


class SEMSimulator(ArtifactSimulator):
    """SEM beam blur and charging simulator."""
    
    def __init__(
        self,
        beam_fwhm_range: Tuple[float, float] = (2.0, 5.0),
        charging_strength_range: Tuple[float, float] = (0.05, 0.2),
        noise_level: float = 0.05
    ):
        super().__init__('SEM')
        self.beam_fwhm_range = beam_fwhm_range
        self.charging_strength_range = charging_strength_range
        self.noise_level = noise_level
    
    def generate_beam_kernel(self, fwhm: float) -> torch.Tensor:
        """Generate Gaussian beam kernel."""
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        size = int(fwhm * 4) | 1
        
        ax = torch.linspace(-(size-1)/2, (size-1)/2, size)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def simulate(self, clean_image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Simulate SEM artifacts."""
        fwhm = np.random.uniform(*self.beam_fwhm_range)
        charging = np.random.uniform(*self.charging_strength_range)
        
        # Beam blur
        kernel = self.generate_beam_kernel(fwhm)
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        C, H, W = clean_image.shape
        blurred = torch.zeros_like(clean_image)
        
        for c in range(C):
            img_c = clean_image[c:c+1].unsqueeze(0)
            conv = torch.nn.functional.conv2d(img_c, kernel, padding='same')
            blurred[c] = conv[0, 0]
        
        # Charging (edge brightening)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-2, -1)
        
        edges = torch.zeros_like(clean_image)
        for c in range(C):
            img_c = clean_image[c:c+1].unsqueeze(0)
            gx = torch.nn.functional.conv2d(img_c, sobel_x, padding=1)
            gy = torch.nn.functional.conv2d(img_c, sobel_y, padding=1)
            edges[c] = torch.sqrt(gx**2 + gy**2)[0, 0]
        
        noisy = blurred + charging * edges
        noisy = noisy + torch.randn_like(noisy) * self.noise_level
        
        params = {
            'beam_fwhm': fwhm,
            'charging_strength': charging
        }
        
        return noisy, params


class STEMSimulator(ArtifactSimulator):
    """STEM probe convolution and scan distortion simulator."""
    
    def __init__(
        self,
        probe_size_range: Tuple[float, float] = (3.0, 8.0),
        distortion_strength: float = 0.05,
        noise_level: float = 0.1
    ):
        super().__init__('STEM')
        self.probe_size_range = probe_size_range
        self.distortion_strength = distortion_strength
        self.noise_level = noise_level
    
    def simulate(self, clean_image: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Simulate STEM artifacts."""
        probe_size = np.random.uniform(*self.probe_size_range)
        
        # Simple Gaussian probe (Airy disk approximation)
        sigma = probe_size / 2.355
        size = int(probe_size * 4) | 1
        
        ax = torch.linspace(-(size-1)/2, (size-1)/2, size)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        
        C, H, W = clean_image.shape
        noisy = torch.zeros_like(clean_image)
        
        for c in range(C):
            img_c = clean_image[c:c+1].unsqueeze(0)
            conv = torch.nn.functional.conv2d(img_c, kernel, padding='same')
            noisy[c] = conv[0, 0]
        
        # Scan distortion (random line shifts)
        max_shift = int(self.distortion_strength * W)
        if max_shift > 0:
            for h in range(H):
                if np.random.random() < 0.1:
                    shift = np.random.randint(-max_shift, max_shift + 1)
                    noisy[:, h] = torch.roll(noisy[:, h], shift, dims=-1)
        
        # Poisson noise
        noisy = torch.poisson(torch.clamp(noisy * 50, 0)) / 50.0
        noisy = noisy + torch.randn_like(noisy) * self.noise_level
        
        params = {
            'probe_size': probe_size,
            'distortion_strength': self.distortion_strength
        }
        
        return noisy, params


# ============================================================================
# Dataset Classes
# ============================================================================

class SimulatedMicroscopyDataset(Dataset):
    """Dataset with simulated artifacts."""
    
    def __init__(
        self,
        clean_image_dir: Path,
        modalities: List[str] = ['AFM', 'TEM', 'SEM', 'STEM'],
        image_size: int = 256,
        num_samples: Optional[int] = None
    ):
        self.clean_image_dir = Path(clean_image_dir)
        self.modalities = modalities
        self.image_size = image_size
        
        # Get all clean images
        self.image_paths = list(self.clean_image_dir.glob('*.png'))
        self.image_paths.extend(self.clean_image_dir.glob('*.jpg'))
        self.image_paths.extend(self.clean_image_dir.glob('*.tif'))
        
        if num_samples:
            self.image_paths = self.image_paths[:num_samples]
        
        # Create simulators
        self.simulators = {
            'AFM': AFMSimulator(),
            'TEM': TEMSimulator(),
            'SEM': SEMSimulator(),
            'STEM': STEMSimulator()
        }
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths) * len(self.modalities)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Determine image and modality
        img_idx = idx // len(self.modalities)
        mod_idx = idx % len(self.modalities)
        modality = self.modalities[mod_idx]
        
        # Load and transform clean image
        img_path = self.image_paths[img_idx]
        clean_img = Image.open(img_path).convert('L')
        clean_tensor = self.transform(clean_img)
        
        # Simulate artifacts
        simulator = self.simulators[modality]
        noisy_tensor, physics_params = simulator.simulate(clean_tensor)
        
        return {
            'clean_images': clean_tensor,
            'noisy_images': noisy_tensor,
            'modality_ids': torch.tensor(mod_idx),
            'physics_params': physics_params,
            'image_path': str(img_path)
        }


def create_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders.
    
    Args:
        data_dir: Directory with clean images
        batch_size: Batch size
        num_workers: Number of data loading workers
        train_split: Fraction for training
        val_split: Fraction for validation
        test_split: Fraction for testing
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create full dataset
    full_dataset = SimulatedMicroscopyDataset(data_dir)
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = int(total_size * val_split)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# Testing
if __name__ == '__main__':
    print("Testing artifact simulators...")
    
    # Create dummy clean image
    clean = torch.rand(1, 256, 256)
    
    for modality in ['AFM', 'TEM', 'SEM', 'STEM']:
        print(f"\n{modality}:")
        
        if modality == 'AFM':
            sim = AFMSimulator()
        elif modality == 'TEM':
            sim = TEMSimulator()
        elif modality == 'SEM':
            sim = SEMSimulator()
        else:
            sim = STEMSimulator()
        
        noisy, params = sim.simulate(clean)
        
        print(f"  Clean range: [{clean.min():.3f}, {clean.max():.3f}]")
        print(f"  Noisy range: [{noisy.min():.3f}, {noisy.max():.3f}]")
        print(f"  Parameters: {params}")
    
    print("\n✓ All simulators working!")
