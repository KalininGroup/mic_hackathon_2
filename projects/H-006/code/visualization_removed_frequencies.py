import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
import os

def lowpass_k_cutoff(img, cutoff_ratio=0.8):
    assert img.ndim == 2, "Image must be 2D"

    H, W = img.shape

    # FFT
    F = np.fft.fftshift(np.fft.fft2(img))

    # Frequency grid
    ky = np.fft.fftfreq(H)
    kx = np.fft.fftfreq(W)
    kx, ky = np.meshgrid(kx, ky)
    k = np.sqrt(kx**2 + ky**2)
    k = np.fft.fftshift(k)

    k_max = k.max()

    # Mask: keep low-k only
    mask = (k <= cutoff_ratio * k_max)

    F_filtered = F * mask
    img_filtered = np.fft.ifft2(np.fft.ifftshift(F_filtered)).real

    return img_filtered, mask


target_paths = [
    "/home/wdekleijne/WillemhackSTEM/WillemhackSTEM/gold200kV_109_HAADF_20_0.606_4_1.7_100_frame1.tiff"
]

input_paths = [
    "/home/wdekleijne/WillemhackSTEM/WillemhackSTEM/gold200kV_101_HAADF_20_0.606_0.2_1.7_100_frame1.tiff",
    "/home/wdekleijne/WillemhackSTEM/WillemhackSTEM/gold200kV_103_HAADF_20_0.606_0.5_1.7_100_frame1.tiff",
    "/home/wdekleijne/WillemhackSTEM/WillemhackSTEM/gold200kV_107_HAADF_20_0.606_2_1.7_100_frame1.tiff"
]

all_paths = input_paths + target_paths
cutoff_ratio = 0.6

for path in all_paths:
    # Load image
    img = tiff.imread(path).astype(np.float32)

    # Apply filter
    filtered_img, k_mask = lowpass_k_cutoff(img, cutoff_ratio=cutoff_ratio)

    # Build output filename
    base, ext = os.path.splitext(path)
    out_path = f"{base}_masked{ext}"

    # Save
    tiff.imwrite(out_path, filtered_img.astype(np.uint16))

    print(f"Saved: {out_path}")

    # Optional: show one example (comment out if not needed)
    """
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("k-space mask")
    plt.imshow(k_mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title(f"Filtered (|k| ≤ {cutoff_ratio} k_max)")
    plt.imshow(filtered_img, cmap="gray")
    plt.axis("off")

    plt.show()
    """
