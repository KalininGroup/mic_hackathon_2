"""
VantaScope configuration system using Pydantic.
Clean, validated, type-safe configuration.
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Union, Any
from pydantic import BaseModel, Field
import yaml


class DatasetConfig(BaseModel):
    """Configuration for a single dataset."""
    name: str
    data_path: Path
    format: Union[Literal["numpy", "hdf5", "pyro5"], str]  # Allow custom formats
    channels: List[str]
    complexity: Literal["basic", "intermediate", "advanced"]
    source: str
    description: Optional[str] = None
    preprocessing: Optional[Dict[str, Any]] = None  # Fixed: any -> Any


class ModelConfig(BaseModel):
    """DINOv2 + Fuzzy-GAT model configuration."""
    backbone: str = "dinov2_vitb14"
    feature_dim: int = 768
    graph_hidden_dim: int = 256
    num_attention_heads: int = 8
    fuzzy_centers: int = 5
    output_classes: Optional[int] = None


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    batch_size: int = 4
    learning_rate: float = 1e-4
    epochs: int = 100
    device: str = "auto"
    seed: int = 42
    validation_split: float = 0.2


class VantaScopeConfig(BaseModel):
    """Main VantaScope configuration."""
    datasets: Dict[str, DatasetConfig]
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    
    @classmethod
    def from_yaml(cls, path: str) -> "VantaScopeConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
