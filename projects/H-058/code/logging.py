"""
VantaScope logging and experiment tracking.
Clean, structured logging for development and production.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class VantaScopeLogger:
    """Centralized logging for VantaScope."""
    
    def __init__(self, name: str = "vantascope", level: str = "INFO", log_dir: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if log_dir provided
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / f"vantascope_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            file_formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, msg: str, **kwargs): self.logger.info(msg, extra=kwargs)
    def debug(self, msg: str, **kwargs): self.logger.debug(msg, extra=kwargs)
    def warning(self, msg: str, **kwargs): self.logger.warning(msg, extra=kwargs)
    def error(self, msg: str, **kwargs): self.logger.error(msg, extra=kwargs)
    def critical(self, msg: str, **kwargs): self.logger.critical(msg, extra=kwargs)
    
    def log_experiment_start(self, config: Dict[str, Any]) -> None:
        """Log experiment configuration."""
        self.info("🚀 Starting VantaScope experiment")
        self.info(f"📊 Config: {json.dumps(config, indent=2)}")
    
    def log_model_summary(self, model, sample_input_shape: tuple) -> None:
        """Log model architecture summary."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info(f"🧠 Model Summary:")
        self.info(f"   Total parameters: {total_params:,}")
        self.info(f"   Trainable parameters: {trainable_params:,}")
        self.info(f"   Input shape: {sample_input_shape}")
    
    def log_data_summary(self, dataset_name: str, train_size: int, val_size: int) -> None:
        """Log dataset information."""
        self.info(f"📁 Dataset: {dataset_name}")
        self.info(f"   Train samples: {train_size}")
        self.info(f"   Validation samples: {val_size}")
    
    def log_training_step(self, epoch: int, step: int, metrics: Dict[str, float]) -> None:
        """Log training metrics."""
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"🔥 Epoch {epoch}, Step {step} | {metrics_str}")


class ColoredFormatter(logging.Formatter):
    """Add colors to console logging."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


# Global logger instance
logger = VantaScopeLogger()
