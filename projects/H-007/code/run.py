"""Training entry point (latest notebook logic)."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data import load_pair, normalize_pair, prepare_patch_pairs
from src.flowmatching import fm_forward_and_loss
from src.unet import TinyUNetFlow


def parse_n_list(text: str):
    return [int(x) for x in text.split(",") if x.strip()]


def train_one_setting(
    hr_patches,
    lr_patches,
    val_hr,
    val_lr,
    args,
    device,
    out_dir: Path,
):
    model = TinyUNetFlow(base=args.base, t_ch=args.t_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr_rate)
    best_val = float("inf")
    no_improve = 0

    metrics_f = open(out_dir / "metrics.csv", "w", newline="")
    mw = csv.writer(metrics_f)
    mw.writerow(["epoch", "train_loss", "val_loss", "vel", "pix", "grad", "edge"])

    train_logs, val_logs = [], []
    use_amp = bool(args.amp) and device == "cuda"

    for ep in range(args.epochs):
        model.train()
        B = hr_patches.size(0)
        lr_rep = lr_patches.repeat_interleave(args.K_t, dim=0)
        hr_rep = hr_patches.repeat_interleave(args.K_t, dim=0)
        t = torch.rand(B * args.K_t, device=device)

        loss, stats = fm_forward_and_loss(
            model,
            lr_rep,
            hr_rep,
            t,
            steps=args.steps,
            lambda_vel=args.lambda_vel,
            lambda_pix=args.lambda_pix,
            lambda_grad=args.lambda_grad,
            lambda_edge=args.lambda_edge,
            focus_pow=args.focus_pow,
            use_amp=use_amp,
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            if val_lr is not None and val_hr is not None:
                Bv = val_hr.size(0)
                lr_vrep = val_lr.repeat_interleave(args.K_t, dim=0)
                hr_vrep = val_hr.repeat_interleave(args.K_t, dim=0)
                tv = torch.rand(Bv * args.K_t, device=device)
                vloss, _ = fm_forward_and_loss(
                    model,
                    lr_vrep,
                    hr_vrep,
                    tv,
                    steps=args.steps,
                    lambda_vel=args.lambda_vel,
                    lambda_pix=args.lambda_pix,
                    lambda_grad=args.lambda_grad,
                    lambda_edge=args.lambda_edge,
                    focus_pow=args.focus_pow,
                    use_amp=False,
                )
                vloss_val = float(vloss.detach().cpu())
            else:
                vloss_val = stats["loss"]

        mw.writerow([ep, stats["loss"], vloss_val, stats["vel"], stats["pix"], stats["grad"], stats["edge"]])
        train_logs.append(stats["loss"])
        val_logs.append(vloss_val)

        if vloss_val < best_val:
            best_val = vloss_val
            no_improve = 0
            torch.save(model.state_dict(), out_dir / "weight.pt")
        else:
            no_improve += 1

        if ep % args.log_interval == 0:
            print(
                f"[ep={ep}] train={stats['loss']:.4f} "
                f"(vel={stats['vel']:.4f}, pix={stats['pix']:.4f}, "
                f"grad={stats['grad']:.4f}, edge={stats['edge']:.4f}) "
                f"val={vloss_val:.4f} (focus_pow={args.focus_pow}, dt={stats['dt']:.5f})"
            )

        if no_improve >= args.patience:
            print(f"early stop at ep {ep}")
            break

    metrics_f.close()

    plt.figure()
    plt.plot(train_logs, label="train")
    plt.plot(val_logs, label="val")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hr", required=True, help="Path to HR npy")
    parser.add_argument("--lr", required=True, help="Path to LR npy")
    parser.add_argument("--out_dir", default=None, help="Output root (default: runs/<timestamp>)")
    parser.add_argument("--n_list", default="2", help="Comma-separated patch counts, e.g., 2,4,8")
    parser.add_argument("--n_select", type=int, default=20, help="Number of patches to select")
    parser.add_argument("--grid", type=int, default=8, help="Grid split count")
    parser.add_argument("--lambda_hf", type=float, default=0.3, help="High-frequency weight")
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=800)
    parser.add_argument("--lr_rate", type=float, default=1e-4)
    parser.add_argument("--base", type=int, default=64)
    parser.add_argument("--t_ch", type=int, default=128)
    parser.add_argument("--log_interval", type=int, default=200)
    parser.add_argument("--K_t", type=int, default=4, help="Number of t samples per patch")
    parser.add_argument("--steps", type=int, default=200, help="Flow-matching dt steps")
    parser.add_argument("--lambda_vel", type=float, default=1.0)
    parser.add_argument("--lambda_pix", type=float, default=1.0)
    parser.add_argument("--lambda_grad", type=float, default=0.3)
    parser.add_argument("--lambda_edge", type=float, default=0.5)
    parser.add_argument("--focus_pow", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp")
    parser.add_argument("--interp_order", type=int, default=0, help="Interpolation order for LR resize")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    out_root = Path(args.out_dir) if args.out_dir else Path("runs") / time.strftime("%Y%m%d_%H%M%S")
    out_root.mkdir(parents=True, exist_ok=True)
    print("save dir:", out_root)

    hr, lr = load_pair(args.hr, args.lr, interp_order=args.interp_order)
    hr_n, lr_n, mean, std = normalize_pair(hr, lr)

    patches = prepare_patch_pairs(
        lr=lr_n,
        hr=hr_n,
        n_select=args.n_select,
        grid=args.grid,
        lambda_hf=args.lambda_hf,
        return_val=True,
    )

    config = {
        "epochs": args.epochs,
        "patience": args.patience,
        "lr_rate": args.lr_rate,
        "base": args.base,
        "t_ch": args.t_ch,
        "patch_size": patches["patch_size"],
        "selected_coords": patches["coords"],
        "K_t": args.K_t,
        "steps": args.steps,
        "lambda_vel": args.lambda_vel,
        "lambda_pix": args.lambda_pix,
        "lambda_grad": args.lambda_grad,
        "lambda_edge": args.lambda_edge,
        "focus_pow": args.focus_pow,
        "lambda_hf": args.lambda_hf,
        "grid": args.grid,
    }
    with open(out_root / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    n_list = parse_n_list(args.n_list)

    hr_all = torch.tensor(patches["hr_patches"]).float().unsqueeze(1).to(device)
    lr_all = torch.tensor(patches["lr_patches"]).float().unsqueeze(1).to(device)
    val_hr = val_lr = None
    if patches["val_hr"] is not None and patches["val_lr"] is not None:
        val_hr = torch.tensor(patches["val_hr"]).float().unsqueeze(1).to(device)
        val_lr = torch.tensor(patches["val_lr"]).float().unsqueeze(1).to(device)

    for n in n_list:
        print(f"=== train with {n} patches ===")
        out_dir = out_root / f"n{n}"
        out_dir.mkdir(parents=True, exist_ok=True)
        hr_t = hr_all[:n]
        lr_t = lr_all[:n]
        train_one_setting(hr_t, lr_t, val_hr, val_lr, args, device, out_dir)
        print("saved:", out_dir / "weight.pt")


if __name__ == "__main__":
    main()
