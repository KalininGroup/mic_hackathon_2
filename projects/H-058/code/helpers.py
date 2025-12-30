"""
Core utility functions for VantaScope.
"""

import torch
import numpy as np
import random
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json
import yaml


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the best available device."""
    if device == "auto":
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    return torch.device(device)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save dictionary as JSON."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save dictionary as YAML."""
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


class Timer:
    """Simple context manager timer."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time:
            self.start_time.record()
        else:
            import time
            self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        if torch.cuda.is_available() and hasattr(self.start_time, 'record'):
            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()
            torch.cuda.synchronize()
            elapsed = self.start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        else:
            import time
            elapsed = time.time() - self.start_time
        
        from .logging import logger
        logger.info(f"⏱️ {self.name}: {elapsed:.3f}s")


def format_size(size_bytes: int) -> str:
    """Format byte size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"
