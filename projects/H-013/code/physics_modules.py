"""
physics_diffusion/models/physics_modules.py

Differentiable physics operators for microscopy artifact simulation.
Each operator implements forward imaging model for a specific modality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

class PhysicsOperator(nn.Module):
    """Base class for physics operators."""
    
    def __init__(self, modality: str):
        super().__init__()
        self.modality = modality
        
    def forward(self, clean_image: torch.Tensor, params: Dict) -> torch.Tensor:
        """Apply physical imaging model to clean image."""
        raise NotImplementedError
        
    def inverse(self, noisy_image: torch.Tensor, params: Dict) -> torch.Tensor:
        """Optional: Physics-based inverse operation."""
        raise NotImplementedError


class AFMTipConvolution(PhysicsOperator):
    """AFM tip convolution operator.
    
    Models the effect of tip geometry on AFM images.
    Forward model: I_measured = I_sample ⊕ tip_shape
    where ⊕ is morphological dilation/convolution.
    """
    
    def __init__(self):
        super().__init__('AFM')
        # Learnable tip shape parameters
        self.tip_radius = nn.Parameter(torch.tensor(10.0))
        self.tip_aspect = nn.Parameter(torch.tensor(2.0))
        
    def generate_tip_kernel(self, radius: float, aspect: float, 
                           size: int = 15) -> torch.Tensor:
        """Generate tip convolution kernel from parameters.
        
        Args:
            radius: Tip radius in pixels
            aspect: Aspect ratio (height/width)
            size: Kernel size (should be odd)
            
        Returns:
            Tip kernel tensor of shape [1, 1, size, size]
        """
        # Create coordinate grid
        ax = torch.linspace(-(size-1)/2, (size-1)/2, size)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        # Elliptical tip model
        r_eff = torch.sqrt((xx/radius)**2 + (yy/(radius*aspect))**2)
        kernel = torch.clamp(1.0 - r_eff, min=0.0)
        
        # Normalize
        kernel = kernel / (kernel.sum() + 1e-8)
        
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def forward(self, clean_image: torch.Tensor, 
                params: Optional[Dict] = None) -> torch.Tensor:
        """Apply tip convolution to clean AFM image.
        
        Args:
            clean_image: [B, C, H, W] clean sample topography
            params: Optional dict with 'radius' and 'aspect'
            
        Returns:
            Convolved image [B, C, H, W]
        """
        if params is None:
            radius = self.tip_radius
            aspect = self.tip_aspect
        else:
            radius = params.get('radius', self.tip_radius)
            aspect = params.get('aspect', self.tip_aspect)
        
        # Generate kernel
        kernel = self.generate_tip_kernel(radius, aspect)
        kernel = kernel.to(clean_image.device)
        
        # Apply convolution per channel
        B, C, H, W = clean_image.shape
        output = torch.zeros_like(clean_image)
        
        for c in range(C):
            output[:, c:c+1] = F.conv2d(
                clean_image[:, c:c+1],
                kernel,
                padding='same'
            )
        
        return output
    
    def inverse(self, noisy_image: torch.Tensor, 
                params: Optional[Dict] = None) -> torch.Tensor:
        """Approximate deconvolution (Wiener filter)."""
        # Simple Wiener deconvolution
        kernel = self.generate_tip_kernel(
            self.tip_radius, self.tip_aspect
        ).to(noisy_image.device)
        
        # FFT-based deconvolution
        fft_img = torch.fft.rfft2(noisy_image)
        fft_kernel = torch.fft.rfft2(kernel, s=noisy_image.shape[-2:])
        
        # Wiener filter with regularization
        wiener_filter = torch.conj(fft_kernel) / (
            torch.abs(fft_kernel)**2 + 0.01
        )
        
        result = torch.fft.irfft2(fft_img * wiener_filter, 
                                   s=noisy_image.shape[-2:])
        
        return result


class TEMCTF(PhysicsOperator):
    """TEM Contrast Transfer Function operator.
    
    Models phase contrast in TEM/cryo-EM imaging.
    CTF(k) = -√(1-A²)·sin(χ) - A·cos(χ)
    where χ is the phase aberration function.
    """
    
    def __init__(self, voltage_kv: float = 300.0):
        super().__init__('TEM')
        self.voltage_kv = voltage_kv
        
        # Microscope parameters (learnable)
        self.defocus = nn.Parameter(torch.tensor(2000.0))  # nm
        self.cs = nn.Parameter(torch.tensor(2.7))  # mm
        self.amp_contrast = nn.Parameter(torch.tensor(0.1))
        
    def compute_ctf(self, image_shape: Tuple[int, int], 
                    pixel_size: float = 1.0) -> torch.Tensor:
        """Compute CTF in Fourier space.
        
        Args:
            image_shape: (H, W) shape of image
            pixel_size: Pixel size in Angstroms
            
        Returns:
            CTF array in Fourier space
        """
        H, W = image_shape
        
        # Create frequency grid
        freq_y = torch.fft.fftfreq(H) / pixel_size
        freq_x = torch.fft.rfftfreq(W) / pixel_size
        ky, kx = torch.meshgrid(freq_y, freq_x, indexing='ij')
        k = torch.sqrt(kx**2 + ky**2)
        
        # Wavelength (relativistic)
        wavelength = 12.2643 / torch.sqrt(
            self.voltage_kv * 1000 * (1 + self.voltage_kv * 1000 * 0.978e-6)
        )  # Angstroms
        
        # Phase aberration function χ
        defocus_m = self.defocus * 1e-9  # Convert to meters
        cs_m = self.cs * 1e-3  # Convert to meters
        wavelength_m = wavelength * 1e-10
        
        k_ang = k * 1e10  # Convert to 1/meters
        chi = (torch.pi * wavelength_m * defocus_m * k_ang**2 + 
               0.5 * torch.pi * cs_m * wavelength_m**3 * k_ang**4)
        
        # CTF formula
        A = self.amp_contrast
        ctf = -torch.sqrt(1 - A**2) * torch.sin(chi) - A * torch.cos(chi)
        
        return ctf
    
    def forward(self, clean_image: torch.Tensor,
                params: Optional[Dict] = None) -> torch.Tensor:
        """Apply CTF to clean image.
        
        Args:
            clean_image: [B, C, H, W] clean image
            params: Optional dict with 'defocus', 'cs', etc.
            
        Returns:
            CTF-modulated image
        """
        B, C, H, W = clean_image.shape
        
        # Update parameters if provided
        if params is not None:
            if 'defocus' in params:
                self.defocus.data = torch.tensor(params['defocus'])
        
        # Compute CTF
        ctf = self.compute_ctf((H, W)).to(clean_image.device)
        
        # Apply in Fourier space
        output = torch.zeros_like(clean_image)
        for b in range(B):
            for c in range(C):
                fft_img = torch.fft.rfft2(clean_image[b, c])
                fft_modulated = fft_img * ctf
                output[b, c] = torch.fft.irfft2(fft_modulated, s=(H, W))
        
        return output
    
    def inverse(self, noisy_image: torch.Tensor,
                params: Optional[Dict] = None) -> torch.Tensor:
        """CTF correction via phase flipping."""
        H, W = noisy_image.shape[-2:]
        ctf = self.compute_ctf((H, W)).to(noisy_image.device)
        
        # Phase flipping
        ctf_corrected = torch.sign(ctf)
        
        B, C = noisy_image.shape[:2]
        output = torch.zeros_like(noisy_image)
        
        for b in range(B):
            for c in range(C):
                fft_img = torch.fft.rfft2(noisy_image[b, c])
                fft_corrected = fft_img * ctf_corrected
                output[b, c] = torch.fft.irfft2(fft_corrected, s=(H, W))
        
        return output


class SEMBeamProfile(PhysicsOperator):
    """SEM beam profile and charging effects.
    
    Models electron beam convolution and charging artifacts.
    """
    
    def __init__(self):
        super().__init__('SEM')
        
        # Beam parameters
        self.beam_fwhm = nn.Parameter(torch.tensor(3.0))  # pixels
        self.charging_strength = nn.Parameter(torch.tensor(0.1))
        
    def generate_beam_kernel(self, fwhm: float, 
                             size: int = 15) -> torch.Tensor:
        """Generate Gaussian beam profile.
        
        Args:
            fwhm: Full width at half maximum
            size: Kernel size
            
        Returns:
            Gaussian kernel [1, 1, size, size]
        """
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        
        ax = torch.linspace(-(size-1)/2, (size-1)/2, size)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def forward(self, clean_image: torch.Tensor,
                params: Optional[Dict] = None) -> torch.Tensor:
        """Apply beam convolution and charging.
        
        Args:
            clean_image: [B, C, H, W] clean image
            params: Optional beam parameters
            
        Returns:
            Degraded image with beam blur and charging
        """
        # Beam convolution
        kernel = self.generate_beam_kernel(self.beam_fwhm)
        kernel = kernel.to(clean_image.device)
        
        B, C, H, W = clean_image.shape
        blurred = torch.zeros_like(clean_image)
        
        for c in range(C):
            blurred[:, c:c+1] = F.conv2d(
                clean_image[:, c:c+1],
                kernel,
                padding='same'
            )
        
        # Charging effect (bright edges)
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-2, -1)
        sobel_x = sobel_x.to(clean_image.device)
        sobel_y = sobel_y.to(clean_image.device)
        
        edges = torch.zeros_like(clean_image)
        for c in range(C):
            gx = F.conv2d(clean_image[:, c:c+1], sobel_x, padding=1)
            gy = F.conv2d(clean_image[:, c:c+1], sobel_y, padding=1)
            edges[:, c:c+1] = torch.sqrt(gx**2 + gy**2)
        
        # Add charging proportional to edge strength
        charged = blurred + self.charging_strength * edges
        
        return charged


class STEMProbeFunction(PhysicsOperator):
    """STEM probe convolution and scan distortions.
    
    Models convergent probe convolution and scan artifacts.
    """
    
    def __init__(self):
        super().__init__('STEM')
        
        # Probe parameters
        self.probe_size = nn.Parameter(torch.tensor(5.0))  # pixels
        self.scan_distortion = nn.Parameter(torch.tensor(0.05))
        
    def generate_probe_kernel(self, size: float,
                             kernel_size: int = 15) -> torch.Tensor:
        """Generate Airy disk probe function.
        
        Args:
            size: Probe FWHM in pixels
            kernel_size: Kernel size
            
        Returns:
            Airy disk kernel
        """
        from scipy.special import j1
        
        ax = torch.linspace(-(kernel_size-1)/2, (kernel_size-1)/2, kernel_size)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        r = torch.sqrt(xx**2 + yy**2)
        
        # Airy disk: (2*J1(x)/x)^2
        x = 2 * np.pi * r / size
        x = x.numpy()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            bessel = 2 * j1(x) / x
            bessel[x == 0] = 1.0
        
        kernel = torch.from_numpy(bessel**2).float()
        kernel = kernel / kernel.sum()
        
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def apply_scan_distortion(self, image: torch.Tensor) -> torch.Tensor:
        """Apply scan line distortions.
        
        Simulates flyback artifacts and drift.
        """
        B, C, H, W = image.shape
        distorted = image.clone()
        
        # Random line shifts (flyback)
        max_shift = int(self.scan_distortion * W)
        if max_shift > 0:
            for b in range(B):
                for h in range(H):
                    if torch.rand(1) < 0.1:  # 10% of lines affected
                        shift = torch.randint(-max_shift, max_shift+1, (1,)).item()
                        distorted[b, :, h] = torch.roll(distorted[b, :, h], shift, dims=-1)
        
        return distorted
    
    def forward(self, clean_image: torch.Tensor,
                params: Optional[Dict] = None) -> torch.Tensor:
        """Apply probe convolution and scan distortion.
        
        Args:
            clean_image: [B, C, H, W] clean image
            params: Optional probe parameters
            
        Returns:
            Degraded image
        """
        # Probe convolution
        kernel = self.generate_probe_kernel(self.probe_size)
        kernel = kernel.to(clean_image.device)
        
        B, C, H, W = clean_image.shape
        convolved = torch.zeros_like(clean_image)
        
        for c in range(C):
            convolved[:, c:c+1] = F.conv2d(
                clean_image[:, c:c+1],
                kernel,
                padding='same'
            )
        
        # Apply scan distortions
        distorted = self.apply_scan_distortion(convolved)
        
        return distorted


class PhysicsOperatorFactory:
    """Factory for creating physics operators."""
    
    @staticmethod
    def create(modality: str) -> PhysicsOperator:
        """Create physics operator for given modality.
        
        Args:
            modality: One of ['AFM', 'TEM', 'SEM', 'STEM']
            
        Returns:
            Appropriate PhysicsOperator instance
        """
        operators = {
            'AFM': AFMTipConvolution,
            'TEM': TEMCTF,
            'SEM': SEMBeamProfile,
            'STEM': STEMProbeFunction
        }
        
        if modality not in operators:
            raise ValueError(f"Unknown modality: {modality}")
        
        return operators[modality]()


# Example usage and testing
if __name__ == '__main__':
    # Test each operator
    batch_size = 2
    channels = 1
    height, width = 256, 256
    
    x = torch.randn(batch_size, channels, height, width)
    
    print("Testing physics operators...")
    
    for modality in ['AFM', 'TEM', 'SEM', 'STEM']:
        print(f"\n{modality}:")
        op = PhysicsOperatorFactory.create(modality)
        
        # Forward pass
        y = op.forward(x)
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        print(f"  Parameters: {sum(p.numel() for p in op.parameters())}")
        
        # Test gradient flow
        loss = y.mean()
        loss.backward()
        print(f"  Gradient flow: ✓")
