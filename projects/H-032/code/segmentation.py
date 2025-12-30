import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from skimage.segmentation import find_boundaries


OUTPUT_DIR = "static/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_segmentation(image_path):
    """
    Assumes a segmented image where:
    - background = 0
    - each object has a unique integer label
    """

    img = np.array(Image.open(image_path))

    # If RGB segmentation → convert to labels
    if img.ndim == 3:
        img = img[:, :, 0]

    labels = np.unique(img)
    labels = labels[labels != 0]

    num_objects = len(labels)

    areas = [(img == l).sum() for l in labels]
    mean_area = float(np.mean(areas)) if areas else 0.0

    return {
        "count": num_objects,
        "mean_area": round(mean_area, 2),
        "areas": areas,
        "mask": img
    }

def save_overlay(image_path, mask, filename):
    """
    Generates a clear segmentation overlay from a single segmented image.
    """

    # Load the segmented image as grayscale background
    base = np.array(Image.open(image_path).convert("L"))

    # Find object boundaries
    boundaries = find_boundaries(mask, mode="outer")

    plt.figure(figsize=(5, 5))
    plt.imshow(base, cmap="gray")
    plt.imshow(boundaries, cmap="Reds", alpha=0.9)
    plt.axis("off")

    out_path = os.path.join("static/outputs", filename)
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    return out_path
