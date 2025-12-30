import io
import json
import zipfile
from typing import Dict, List

import numpy as np
from PIL import Image


def masks_to_tiff_stack(masks: List[np.ndarray]) -> bytes:
    buf = io.BytesIO()
    imgs = [Image.fromarray(m.astype("uint8")) for m in masks]
    if not imgs:
        return b""
    imgs[0].save(buf, format="TIFF", save_all=True, append_images=imgs[1:])
    return buf.getvalue()


def png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()


def build_zip_export(union_mask, instance_masks, plan, history) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if union_mask is not None:
            zf.writestr("union_mask.png", png_bytes(union_mask))
        if instance_masks:
            zf.writestr("instances.tiff", masks_to_tiff_stack(instance_masks))
        zf.writestr("plan.json", json.dumps(plan, indent=2))
        summaries = [h["summary"] for h in history]
        zf.writestr("history.json", json.dumps(summaries, indent=2))
    return buf.getvalue()

