"""
Configuration settings for SE(3)-Equivariant AFM Reconstruction
Paper 3: SE(3)-Equivariant Neural Networks for Joint Blind Tip Reconstruction
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import torch

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # SE(3) Network Architecture
    num_se3_layers: int = 4
    hidden_dim: int = 64
    max_degree: int = 3  # Max spherical harmonics degree
    num_radial: int = 16  # Radial basis functions
    
    # Feature dimensions
    image_features: int = 128
    tip_latent_dim: int = 256
    surface_features: int = 64
    
    # Dropout for uncertainty
    dropout_rate: float = 0.1
    monte_carlo_samples: int = 10

@dataclass
class DataConfig:
    """Dataset configuration"""
    # Image dimensions
    image_size: Tuple[int, int] = (128, 128)
    tip_voxel_size: Tuple[int, int, int] = (32, 32, 32)
    
    # Tip types for generation
    tip_types: List[str] = field(default_factory=lambda: [
        'pyramidal', 'conical', 'spherical', 'blunt'
    ])
    
    # Surface types
    surface_types: List[str] = field(default_factory=lambda: [
        'random', 'nanoparticles', 'steps', 'periodic'
    ])
    
    # Dataset sizes
    train_size: int = 80000
    val_size: int = 10000
    test_size: int = 10000
    
    # Physics parameters
    max_tip_radius_nm: float = 20.0
    pixel_size_nm: float = 1.0
    noise_snr_db: float = 30.0
    
    # Convolution parameters
    convolution_kernel_size: int = 5

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Optimization
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    epochs: int = 300
    
    # Scheduler
    scheduler_type: str = 'cosine_warmup'
    warmup_epochs: int = 10
    t_0: int = 50  # For CosineAnnealingWarmRestarts
    t_mult: int = 2
    
    # Loss weights
    tip_weight: float = 1.0
    surface_weight: float = 1.0
    consistency_weight: float = 0.5
    smoothness_weight: float = 0.1
    uncertainty_weight: float = 0.1
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Checkpointing
    save_every: int = 10
    validate_every: int = 5
    
    # Device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    mixed_precision: bool = True

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Metrics to compute
    compute_psnr: bool = True
    compute_ssim: bool = True
    compute_ece: bool = True
    
    # Uncertainty quantification
    mc_samples: int = 20
    confidence_intervals: List[float] = field(default_factory=lambda: [0.68, 0.95, 0.99])
    
    # Baseline methods
    compare_baselines: bool = True
    baseline_methods: List[str] = field(default_factory=lambda: [
        'btr', 'wiener', 'cnn_baseline'
    ])

@dataclass
class Config:
    """Master configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Experiment tracking
    experiment_name: str = "se3_afm_reconstruction"
    save_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True

def get_config() -> Config:
    """Get default configuration"""
    return Config()

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
