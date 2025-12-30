"""Flow-matching loss and helpers (latest notebook version)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "grad_loss",
    "edge_map",
    "build_xt_cosine",
    "fm_forward_and_loss",
]


def grad_loss(pred, target):
    gx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    gy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    gy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


_sobel = torch.tensor(
    [
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
    ],
    dtype=torch.float32,
)


def edge_map(x):
    k = _sobel.to(x.device).unsqueeze(1)
    return F.conv2d(x, k, padding=1)


def build_xt_cosine(x0_lr, x1_hr, t):
    s = 0.5 - 0.5 * torch.cos(torch.pi * t)
    s = s.view(-1, 1, 1, 1)
    return (1.0 - s) * x0_lr + s * x1_hr


def fm_forward_and_loss(
    model,
    x0_lr,
    x1_hr,
    t,
    steps: int = 50,
    lambda_vel: float = 1.0,
    lambda_pix: float = 1.0,
    lambda_grad: float = 1.0,
    lambda_edge: float = 1.0,
    focus_pow: float = 1.0,
    use_amp: bool = False,
):
    dt = 1.0 / float(steps)
    t_next = torch.clamp(t + dt, max=1.0)

    x_t = build_xt_cosine(x0_lr, x1_hr, t)
    x_tnxt = build_xt_cosine(x0_lr, x1_hr, t_next)
    v_target = (x_tnxt - x_t) / dt
    w = t.view(-1, 1, 1, 1)

    w_pix = (1.0 - w) ** focus_pow
    w_grad = (w * (1.0 - w)) ** focus_pow
    w_edge = w ** focus_pow

    with torch.cuda.amp.autocast(enabled=use_amp and x_t.is_cuda):
        v_pred = model(x_t, x0_lr, t)
        x_pred = x_t + dt * v_pred

        loss_vel = F.mse_loss(v_pred, v_target)
        loss_pix = (w_pix * F.l1_loss(x_pred, x_tnxt, reduction="none")).mean()
        loss_grad_v = grad_loss(x_pred, x_tnxt)
        loss_grad = (w_grad * loss_grad_v).mean()
        loss_edge = (w_edge * F.l1_loss(edge_map(x_pred), edge_map(x_tnxt), reduction="none")).mean()

        loss = (
            lambda_vel * loss_vel
            + lambda_pix * loss_pix
            + lambda_grad * loss_grad
            + lambda_edge * loss_edge
        )

    stats = {
        "loss": float(loss.detach().cpu()),
        "vel": float(loss_vel.detach().cpu()),
        "pix": float(loss_pix.detach().cpu()),
        "grad": float(loss_grad.detach().cpu()),
        "edge": float(loss_edge.detach().cpu()),
        "dt": dt,
        "focus_pow": focus_pow,
    }

    return loss, stats
