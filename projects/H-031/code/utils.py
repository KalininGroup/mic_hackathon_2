import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

@torch.no_grad()
def predict_from_input(
    model,
    input_data,
    input_type,        # "img" or "sto"
    IMG_HW,
    min_stoich, max_stoich,
    min_image, max_image,
    device="cpu"
):
    """
    input_data:
        - if img: raw image (H,W) from data["images"][i]
        - if sto: raw stoich scalar from data["stoich"][i]
    """

    model.eval()
    if input_type == "img":
        # ---- normalize image ----
        img = input_data.astype(np.float32)
        img = (img - min_image) / (max_image - min_image + 1e-8)

        img = torch.tensor(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        img = F.interpolate(img, IMG_HW, mode="bilinear", align_corners=False)
        img = img.to(device)

        # dummy sto (not used for encoding)
        sto = torch.zeros((1,1), device=device)

        out = model(img, sto)

        pred_img = out["img_i2i"]
        pred_sto = out["sto_i2s"]

    elif input_type == "sto":
        # ---- normalize stoich ----
        sto = np.array([[input_data]], dtype=np.float32)
        sto = (sto - min_stoich) / (max_stoich - min_stoich + 1e-8)
        sto = torch.tensor(sto).to(device)

        # dummy image (not used for encoding)
        img = torch.zeros((1,1,IMG_HW[0], IMG_HW[1]), device=device)

        out = model(img, sto)

        pred_img = out["img_s2i"]
        pred_sto = out["sto_s2s"]

    else:
        raise ValueError("input_type must be 'img' or 'sto'")

    # ---- denormalize outputs ----
    pred_img = pred_img.cpu().squeeze().numpy()
    pred_img = pred_img * (max_image - min_image) + min_image

    pred_sto = pred_sto.cpu().item()
    pred_sto = pred_sto * (max_stoich - min_stoich) + min_stoich

    return pred_img, pred_sto

