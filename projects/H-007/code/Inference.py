"""Inference entry point (latest notebook logic)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import load_pair, normalize_pair
from src.unet import TinyUNetFlow
from src.util import hann2d, compute_patch_feature


@torch.no_grad()
def fm_denoise_patch_euler(model, x0_lr, steps: int = 100, t0: float = 0.0, t1: float = 1.0):
    model.eval()
    x = x0_lr.clone()
    B = x.size(0)
    dt = (t1 - t0) / float(steps)
    for k in range(steps):
        t = t0 + k * dt
        t_tensor = torch.full((B,), t, device=x.device)
        v = model(x, x0_lr, t_tensor)
        x = x + dt * v
    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Timestamp weight root containing n* folders")
    parser.add_argument("--hr", required=True, help="Path to HR npy")
    parser.add_argument("--lr", required=True, help="Path to LR npy")
    parser.add_argument("--steps_infer", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--t_ch", type=int, default=128)
    parser.add_argument("--grid", type=int, default=8, help="Patch grid split (default 8)")
    parser.add_argument("--stride_div", type=int, default=3, help="stride = patch_size // stride_div")
    parser.add_argument("--interp_order", type=int, default=0, help="Interpolation order for LR resize")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    weight_root = Path(args.weights)
    hr, lr = load_pair(args.hr, args.lr, interp_order=args.interp_order)
    hr_n, lr_n, mean, std = normalize_pair(hr, lr)

    H, W = lr_n.shape
    patch_size = min(H, W) // args.grid
    stride = patch_size // args.stride_div
    win = hann2d(patch_size).astype(np.float32)

    xs = list(range(0, H - patch_size + 1, stride))
    ys = list(range(0, W - patch_size + 1, stride))
    if xs[-1] != H - patch_size:
        xs.append(H - patch_size)
    if ys[-1] != W - patch_size:
        ys.append(W - patch_size)
    coords = [(i, j) for i in xs for j in ys]

    out_dir = weight_root / "inference"
    out_dir.mkdir(exist_ok=True)
    weight_dirs = sorted([p for p in weight_root.iterdir() if p.is_dir() and p.name.startswith("n")])

    for wdir in weight_dirs:
        wpath = wdir / "weight.pt"
        if not wpath.exists():
            continue

        print("inference with", wpath)
        model = TinyUNetFlow(base=args.base, t_ch=args.t_ch).to(device)
        model.load_state_dict(torch.load(wpath, map_location=device))
        model.eval()

        lr_tiles = []
        for (i, j) in coords:
            lr_tiles.append(lr_n[i:i + patch_size, j:j + patch_size])
        lr_tiles = torch.tensor(np.stack(lr_tiles), dtype=torch.float32, device=device).unsqueeze(1)

        sr_tiles = []
        with torch.no_grad():
            for b in range(0, len(lr_tiles), args.batch_size):
                sr = fm_denoise_patch_euler(model, lr_tiles[b:b + args.batch_size], steps=args.steps_infer)
                sr_tiles.append(sr.cpu())
        sr_tiles = torch.cat(sr_tiles, dim=0)

        out = np.zeros((H, W), dtype=np.float32)
        weight = np.zeros((H, W), dtype=np.float32)
        for (i, j), sr in zip(coords, sr_tiles):
            sr_np = sr[0].numpy()
            out[i:i + patch_size, j:j + patch_size] += sr_np * win
            weight[i:i + patch_size, j:j + patch_size] += win
        sr_full = out / (weight + 1e-8)

        gt_den = hr_n * std + mean
        lr_den = lr_n * std + mean
        sr_den = sr_full * std + mean
        vmin, vmax = np.percentile(gt_den, [1, 99])

        tag = wdir.name
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(gt_den, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        plt.title("HR (GT)")
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(lr_den, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        plt.title("LR")
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(sr_den, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        plt.title(f"SR ({tag})")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{tag}_HR_LR_SR.png", dpi=200)
        plt.close()

        grid_ps = H // args.grid
        mask_lr_feat = np.zeros((H, W), dtype=np.float32)
        mask_sr_feat = np.zeros((H, W), dtype=np.float32)

        gx = range(0, H, grid_ps)
        gy = range(0, W, grid_ps)
        for i in gx:
            for j in gy:
                gt_tile = gt_den[i:i + grid_ps, j:j + grid_ps]
                lr_tile = lr_den[i:i + grid_ps, j:j + grid_ps]
                sr_tile = sr_den[i:i + grid_ps, j:j + grid_ps]

                feat_gt = compute_patch_feature(gt_tile)
                feat_lr = compute_patch_feature(lr_tile)
                feat_sr = compute_patch_feature(sr_tile)

                rel_err_lr = np.linalg.norm(feat_lr - feat_gt) / (np.linalg.norm(feat_gt) + 1e-8)
                rel_err_sr = np.linalg.norm(feat_sr - feat_gt) / (np.linalg.norm(feat_gt) + 1e-8)

                if rel_err_lr < 0.1:
                    mask_lr_feat[i:i + grid_ps, j:j + grid_ps] = 1.0
                if rel_err_sr < 0.1:
                    mask_sr_feat[i:i + grid_ps, j:j + grid_ps] = 1.0

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(mask_lr_feat, cmap="gray", origin="lower", vmin=0, vmax=1)
        plt.title(f"LR feature err < 10% [{tag}]")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(mask_sr_feat, cmap="gray", origin="lower", vmin=0, vmax=1)
        plt.title(f"SR feature err < 10% [{tag}]")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"{tag}_feature_mask.png", dpi=200)
        plt.close()


if __name__ == "__main__":
    main()
