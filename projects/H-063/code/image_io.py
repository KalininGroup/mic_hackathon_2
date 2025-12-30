from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageSequence


def _pil_to_numpy(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    # Handle multi-frame by taking the first for now.
    if arr.ndim == 3 and arr.shape[0] > 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    return arr


def load_image_any(source: Union[str, Path, BytesIO, bytes]) -> np.ndarray:
    """Load TIFF/JPEG/PNG/etc. Accepts path or file-like."""
    if isinstance(source, (str, Path)):
        with Image.open(source) as im:
            try:
                # If multi-frame, use first frame for now
                frame = next(ImageSequence.Iterator(im))
            except Exception:
                frame = im
            return _pil_to_numpy(frame.convert("RGB"))
    if isinstance(source, bytes):
        source = BytesIO(source)
    if isinstance(source, BytesIO):
        source.seek(0)
        with Image.open(source) as im:
            try:
                frame = next(ImageSequence.Iterator(im))
            except Exception:
                frame = im
            return _pil_to_numpy(frame.convert("RGB"))
    raise ValueError("Unsupported source type for image load")


def maybe_downsample(img: np.ndarray, target_max: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """
    Downsample image to keep max side <= target_max.
    Returns (image, scale).
    """
    if target_max is None:
        return img, 1.0
    h, w = img.shape[:2]
    mx = max(h, w)
    if mx <= target_max:
        return img, 1.0
    scale = target_max / float(mx)
    new_w, new_h = int(w * scale), int(h * scale)

    import cv2

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale

