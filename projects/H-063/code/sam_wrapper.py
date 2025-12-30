import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


class SAM2Wrapper:
    """
    Thin wrapper around SAM2ImagePredictor. Loads model once and exposes predict().
    """

    def __init__(self, sam2_root: Path, ckpt_path: Path, config_path: Path, device: Optional[str] = None):
        self._import_sam2(sam2_root)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        config_name = self._normalize_config_name(config_path, sam2_root)
        model = build_sam2(config_file=config_name, ckpt_path=str(ckpt_path), device=device)
        self.predictor = SAM2ImagePredictor(model)
        self._image_set = False

    @staticmethod
    def _import_sam2(sam2_root: Path):
        if not sam2_root.exists():
            raise RuntimeError(f"SAM2 root not found: {sam2_root}")
        sys.path.append(str(sam2_root))

    def set_image(self, image_rgb_u8: np.ndarray):
        self.predictor.set_image(image_rgb_u8)
        self._image_set = True

    def predict(self, box=None, pos_points=None, neg_points=None, multimask_output=True) -> Tuple[list, list]:
        if not self._image_set:
            raise RuntimeError("Call set_image(image) before predict().")
        pos = np.array(pos_points, dtype=np.float32) if pos_points else np.zeros((0, 2), np.float32)
        neg = np.array(neg_points, dtype=np.float32) if neg_points else np.zeros((0, 2), np.float32)

        point_coords = None
        point_labels = None
        if len(pos) or len(neg):
            point_coords = np.concatenate([pos, neg], axis=0)
            point_labels = np.concatenate([np.ones(len(pos), dtype=np.int64), np.zeros(len(neg), dtype=np.int64)])

        b = np.array(box, dtype=np.float32) if box is not None else None
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords, point_labels=point_labels, box=b, multimask_output=multimask_output
        )
        return [masks[i] for i in range(masks.shape[0])], [float(scores[i]) for i in range(scores.shape[0])]

    @staticmethod
    def _normalize_config_name(config_path: Path, sam2_root: Path) -> str:
        """
        Hydra expects a config name relative to the search path. Convert absolute paths to relative.
        """
        try:
            rel = config_path.relative_to(sam2_root)
        except Exception:
            # Fallback: just return name
            rel = Path(config_path.name)
        rel_str = str(rel)
        # Ensure configs/ prefix since Hydra search path is pkg://sam2
        if not rel_str.startswith("configs/"):
            rel_str = f"configs/{rel_str}"
        return rel_str
