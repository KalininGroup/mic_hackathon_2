import base64
from typing import List

import cv2
import numpy as np
from PIL import Image


def to_gray(img: np.ndarray) -> np.ndarray:
    """Convert RGB/RGBA or grayscale to a single-channel grayscale array."""
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    raise ValueError(f"Unexpected image shape: {img.shape}")


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalize any numeric dtype to uint8."""
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    x -= x.min()
    if x.max() > 0:
        x /= x.max()
    return (255.0 * x).clip(0, 255).astype(np.uint8)


def preprocess_micrograph(image: np.ndarray, clahe: bool = True, denoise: bool = True) -> np.ndarray:
    """Light microscopy preprocessing: median denoise + CLAHE."""
    g = to_uint8(to_gray(image))
    if denoise:
        g = cv2.medianBlur(g, 3)
    if clahe:
        cla = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = cla.apply(g)
    return g


def ensure_rgb_uint8(gray_u8: np.ndarray) -> np.ndarray:
    """Stack grayscale into 3-channel uint8."""
    g = gray_u8 if gray_u8.dtype == np.uint8 else to_uint8(gray_u8)
    return np.stack([g, g, g], axis=-1)


def mask_to_uint8(mask01: np.ndarray) -> np.ndarray:
    m = mask01
    if m.ndim == 3:
        m = m[..., 0]
    m = m.astype(np.float32)
    if m.max() > 1:
        m /= 255.0
    return (m > 0.5).astype(np.uint8) * 255


def cleanup_mask_u8(m_u8: np.ndarray, min_size: int = 25, hole_area: int = 25) -> np.ndarray:
    import skimage.morphology as morphology

    m = m_u8 > 0
    m = morphology.remove_small_objects(m, min_size=min_size)
    m = morphology.remove_small_holes(m, area_threshold=hole_area)
    return (m.astype(np.uint8) * 255)


def overlay_mask_rgb(image_rgb_u8: np.ndarray, mask_u8: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """Red overlay mask visualization."""
    img = image_rgb_u8.copy().astype(np.float32)
    m = (mask_u8 > 0).astype(np.float32)
    color = np.zeros_like(img)
    color[..., 0] = 255
    img = img * (1 - alpha * m[..., None]) + color * (alpha * m[..., None])
    return img.clip(0, 255).astype(np.uint8)


def draw_boundary(image_rgb_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    """Highlight boundaries in yellow."""
    out = image_rgb_u8.copy()
    edges = cv2.Canny((mask_u8 > 0).astype(np.uint8) * 255, 50, 150)
    out[edges > 0] = (255, 255, 0)
    return out


def overlay_instances(image_rgb_u8: np.ndarray, masks_u8: List[np.ndarray], alpha: float = 0.30, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = image_rgb_u8.copy().astype(np.float32)
    for m in masks_u8:
        mm = (m > 0).astype(np.float32)
        color = rng.integers(0, 256, size=(3,), dtype=np.int32).astype(np.float32)
        out = out * (1 - alpha * mm[..., None]) + color[None, None, :] * (alpha * mm[..., None])
    return out.clip(0, 255).astype(np.uint8)


def encode_png_uint8(img_u8: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_u8)
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode()


def union_mask(masks: List[np.ndarray]):
    if not masks:
        return None
    out = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        out = np.maximum(out, m.astype(np.uint8))
    return out
