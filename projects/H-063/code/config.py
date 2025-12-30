import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


def _default_checkpoint(project_root: Path) -> Path:
    """
    Prefer lower-case models folder; fall back to upper-case if present.
    """
    lower = project_root / "models" / "sam2.1_hiera_base_plus.pt"
    upper = project_root / "Models" / "sam2.1_hiera_base_plus.pt"
    if lower.exists():
        return lower
    return lower if lower.parent.exists() else upper


@dataclass
class Sam2Paths:
    sam2_root: Path
    checkpoint: Path
    config: Path


@dataclass
class AppConfig:
    """
    Central place for paths and default parameters.
    Adjust SAM2 paths to match your local checkout and checkpoint.
    """

    project_root: Path = Path(__file__).resolve().parent.parent
    sam2_paths: Sam2Paths = None  # type: ignore
    default_rounds: int = 2
    default_relax_steps: int = 3
    cap_masks_total: int = 3500

    # Selection defaults
    min_quality: float = 0.30
    iou_thresh: float = 0.22
    max_instances: int = 300

    def __post_init__(self):
        # Resolve defaults with environment overrides
        sam2_root = Path(os.getenv("SAM2_ROOT", self.project_root / "sam2"))
        ckpt = Path(os.getenv("SAM2_CKPT", _default_checkpoint(self.project_root)))
        cfg = Path(os.getenv("SAM2_CFG", sam2_root / "configs/sam2.1/sam2.1_hiera_b+.yaml"))
        self.sam2_paths = Sam2Paths(sam2_root=sam2_root, checkpoint=ckpt, config=cfg)

    def device(self) -> str:
        """
        Returns the preferred torch device. Falls back to CPU if CUDA is unavailable.
        """
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
