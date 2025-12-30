"""
Dataset for simulated 4D-STEM dynamical processes
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ModelConfig


class Simulated4DSTEMDataset(Dataset):
    """Generate synthetic 4D-STEM data with dynamical processes"""
    
    def __init__(self, config: ModelConfig, mode: str = 'train', n_sequences: int = 1000):
        self.config = config
        self.mode = mode
        self.n_sequences = n_sequences
        self.image_size = config.image_size
        
        # Set random seed for reproducibility
        seed = 42 if mode == 'train' else (43 if mode == 'val' else 44)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        print(f"Generating {n_sequences} {mode} sequences...")
        self.data = self._generate_all_data()
        print(f"Dataset generation complete!")
        
    def _generate_all_data(self):
        """Pre-generate all sequences"""
        data = []
        process_types = ['phase_transition', 'dislocation_motion', 'beam_damage']
        
        for idx in range(self.n_sequences):
            process_type = process_types[idx % len(process_types)]
            sample = self._simulate_dynamical_process(process_type)
            data.append(sample)
            
            if (idx + 1) % 100 == 0:
                print(f"  Generated {idx + 1}/{self.n_sequences} sequences")
        
        return data
    
    def _simulate_dynamical_process(self, process_type: str = 'phase_transition'):
        """Simulate a physical process"""
        base_pattern = self._generate_base_pattern()
        
        sequences = []
        time_points = []
        physics_params = []
        
        for t in range(self.config.max_time_points):
            normalized_t = t / self.config.max_time_points
            
            if process_type == 'phase_transition':
                pattern = self._apply_phase_transition(base_pattern, normalized_t)
                physics = self._get_phase_transition_physics(normalized_t)
            elif process_type == 'dislocation_motion':
                pattern = self._apply_dislocation_motion(base_pattern, normalized_t)
                physics = self._get_dislocation_physics(normalized_t)
            else:  # beam_damage
                pattern = self._apply_beam_damage(base_pattern, normalized_t)
                physics = self._get_beam_damage_physics(normalized_t)
            
            sequences.append(pattern)
            time_points.append(normalized_t)
            physics_params.append(physics)
        
        return {
            'sequence': torch.stack(sequences, dim=0),  # [T, 1, H, W]
            'times': torch.tensor(time_points, dtype=torch.float32),  # [T]
            'physics': torch.stack(physics_params, dim=0),  # [T, P]
            'process_type': process_type
        }
    
    def _generate_base_pattern(self):
        """Generate base diffraction pattern with Bragg peaks"""
        pattern = torch.zeros(self.image_size)
        H, W = self.image_size
        
        # Add symmetric Bragg peaks
        peak_positions = [
            (H//4, W//4), (H//4, 3*W//4),
            (3*H//4, W//4), (3*H//4, 3*W//4),
            (H//2, W//2)  # Center peak
        ]
        
        for py, px in peak_positions:
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, dtype=torch.float32),
                torch.arange(W, dtype=torch.float32),
                indexing='ij'
            )
            dist = torch.sqrt((y_grid - py)**2 + (x_grid - px)**2)
            peak = torch.exp(-dist**2 / (2 * 2**2))  # Gaussian peak
            pattern += peak * np.random.uniform(0.5, 1.0)
        
        # Add thermal diffuse scattering background
        background = torch.randn(self.image_size) * 0.05
        pattern = pattern + background
        pattern = torch.clamp(pattern, 0, 1)
        
        return pattern.unsqueeze(0)  # [1, H, W]
    
    def _apply_phase_transition(self, base_pattern, t):
        """Simulate phase transition (peaks shift and intensity changes)"""
        # Sigmoid transition around t=0.5
        transition_progress = 1 / (1 + np.exp(-10 * (t - 0.5)))
        
        # Shift pattern slightly
        shift_x = int(transition_progress * 2)
        shifted = torch.roll(base_pattern, shifts=(shift_x, shift_x), dims=(1, 2))
        
        # Blend between original and shifted with intensity change
        intensity_factor = 0.8 + 0.4 * transition_progress
        result = (1 - transition_progress) * base_pattern + transition_progress * shifted
        result = result * intensity_factor
        
        return torch.clamp(result, 0, 1)
    
    def _apply_dislocation_motion(self, base_pattern, t):
        """Simulate dislocation motion (localized intensity changes)"""
        # Moving dislocation creates local intensity modulation
        disloc_pos = int(t * self.image_size[0])
        mask = torch.zeros_like(base_pattern)
        
        if disloc_pos < self.image_size[0]:
            mask[:, max(0, disloc_pos-3):min(self.image_size[0], disloc_pos+3), :] = 0.3
        
        result = base_pattern * (1 - mask) + base_pattern * 0.7 * mask
        return torch.clamp(result, 0, 1)
    
    def _apply_beam_damage(self, base_pattern, t):
        """Simulate beam damage (progressive intensity reduction)"""
        # Exponential decay
        damage_factor = np.exp(-2 * t)
        result = base_pattern * damage_factor
        
        # Add increasing noise with damage
        noise_level = 0.02 * (1 - damage_factor)
        noise = torch.randn_like(result) * noise_level
        
        return torch.clamp(result + noise, 0, 1)
    
    def _get_phase_transition_physics(self, t):
        """Physics parameters for phase transition"""
        return torch.tensor([
            np.sin(2 * np.pi * t),  # Lattice parameter modulation
            np.cos(2 * np.pi * t),  # Structure factor
            1 / (1 + np.exp(-10 * (t - 0.5))),  # Transition progress
            0.5  # Temperature factor
        ], dtype=torch.float32)
    
    def _get_dislocation_physics(self, t):
        """Physics parameters for dislocation motion"""
        return torch.tensor([
            t,  # Normalized position
            0.3,  # Burgers vector magnitude
            np.sin(np.pi * t),  # Velocity modulation
            0.2  # Strain field strength
        ], dtype=torch.float32)
    
    def _get_beam_damage_physics(self, t):
        """Physics parameters for beam damage"""
        return torch.tensor([
            np.exp(-2 * t),  # Remaining intensity
            t,  # Accumulated dose
            0.02 * (1 - np.exp(-2 * t)),  # Noise level
            1 - t  # Material integrity
        ], dtype=torch.float32)
    
    def _apply_sparse_sampling(self, sequence, times):
        """Create irregular sparse temporal sampling"""
        n_total = len(sequence)
        n_sampled = max(1, int(n_total * self.config.sparse_sampling_rate))
        
        # Create irregular sampling
        if self.config.temporal_irregularity > 0:
            # Non-uniform sampling with bias towards certain regions
            base_probs = torch.ones(n_total)
            # Add temporal bias (more samples at beginning and end)
            time_bias = torch.exp(-((torch.arange(n_total, dtype=torch.float32) - n_total/2) / (n_total/4))**2)
            base_probs = base_probs * (1 + self.config.temporal_irregularity * time_bias)
            probs = torch.softmax(base_probs, dim=0)
            sampled_indices = torch.multinomial(probs, n_sampled, replacement=False)
            sampled_indices = torch.sort(sampled_indices)[0]  # Keep temporal order
        else:
            # Uniform sampling
            step = n_total // n_sampled
            sampled_indices = torch.arange(0, n_total, step)[:n_sampled]
        
        sampled_sequence = sequence[sampled_indices]
        sampled_times = times[sampled_indices]
        
        # Add Poisson noise (electron counting statistics)
        noisy_sequence = self._add_poisson_noise(sampled_sequence)
        
        return {
            'sparse_sequence': noisy_sequence,
            'sampled_times': sampled_times,
            'sampled_indices': sampled_indices,
            'full_sequence': sequence,
            'full_times': times
        }
    
    def _add_poisson_noise(self, data):
        """Add realistic electron counting noise"""
        max_counts = self.config.max_electron_counts
        scaled = data * max_counts
        # Poisson noise
        noisy = torch.poisson(scaled) / max_counts
        return noisy
    
    def __len__(self):
        return self.n_sequences
    
    def __getitem__(self, idx):
        # Get full sequence
        sample = self.data[idx]
        full_seq = sample['sequence']  # [T, 1, H, W]
        times = sample['times']  # [T]
        physics = sample['physics']  # [T, P]
        
        # Apply sparse sampling
        sparse_data = self._apply_sparse_sampling(full_seq, times)
        
        return {
            'sparse': sparse_data['sparse_sequence'],  # [K, 1, H, W]
            'sparse_times': sparse_data['sampled_times'],  # [K]
            'full': sparse_data['full_sequence'],  # [T, 1, H, W]
            'full_times': sparse_data['full_times'],  # [T]
            'physics_params': physics,  # [T, P]
            'process_type': sample['process_type']
        }


if __name__ == "__main__":
    # Test dataset generation
    from config import ModelConfig
    
    config = ModelConfig()
    config.max_time_points = 20  # Reduce for testing
    config.image_size = (32, 32)
    
    print("Creating test dataset...")
    dataset = Simulated4DSTEMDataset(config, mode='train', n_sequences=10)
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Sparse shape: {sample['sparse'].shape}")
    print(f"Sparse times shape: {sample['sparse_times'].shape}")
    print(f"Full shape: {sample['full'].shape}")
    print(f"Full times shape: {sample['full_times'].shape}")
    print(f"Process type: {sample['process_type']}")
    
    print("\nDataset test passed!")
