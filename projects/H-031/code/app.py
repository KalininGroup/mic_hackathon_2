import os
import urllib.request
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from models import UniversalTranslator
from utils import predict_from_input

# ===============================
# CONFIG
# ===============================
MODEL_URL  = "https://github.com/hasanjawad001/rheed-universal-translator/releases/download/v1.0/model_weights.pth"
MODEL_PATH = "outputs/model_weights.pth"
META_PATH  = "outputs/model_meta.npz"
DATA_PATH  = "inputs/rheed_stoich_data.npz"

device = "cuda" if torch.cuda.is_available() else "cpu"

def to_viridis(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (cm.viridis(img)[..., :3] * 255).astype(np.uint8)

# ===============================
# PAGE SETUP
# ===============================
st.set_page_config(
    page_title="RHEED Universal Translator",
    # layout="wide"
)

st.title("🧬 RHEED Universal Translator")
st.markdown(
"""
Translate **RHEED images ↔ stoichiometry** using a learned shared latent space.

**Capabilities**
- 🔁 RHEED image → stoichiometry (+ image reconstruction)
- 🔁 Stoichiometry → RHEED image (+ stoichiometry reconstruction)
"""
)

# ===============================
# LOAD MODEL & DATA
# ===============================
with st.spinner("Checking model weights..."):
    os.makedirs("outputs", exist_ok=True)
    if not os.path.isfile(MODEL_PATH):
        st.info("Downloading pretrained model weights (~200 MB). This may take few minutes depending on your internet speed.")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("Model downloaded successfully.")
    else:
        st.success("Model weights found.")

with st.spinner("Loading metadata..."):
    meta = np.load(META_PATH)
    min_stoich = float(meta["min_stoich"])
    max_stoich = float(meta["max_stoich"])
    min_image  = float(meta["min_image"])
    max_image  = float(meta["max_image"])
    IMG_HW     = tuple(meta["IMG_HW"])
    st.success("Metadata loaded.")

with st.spinner("Loading dataset..."):
    data = np.load(DATA_PATH)
    stoich = data["stoich"].astype(np.float32)
    images = data["images"].astype(np.float32)
    st.success(f"Dataset loaded: {len(stoich)} samples")

with st.spinner("Loading model..."):
    model = UniversalTranslator(IMG_HW).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    st.success("Model ready.")

# ===============================
# TASK SELECTION
# ===============================
st.header("🔀 Choose Translation Task")

task = st.radio(
    "Select task:",
    [
        "RHEED image → Stoichiometry",
        "Stoichiometry → RHEED image"
    ]
)

# ===============================
# TASK 1: IMAGE → STO
# ===============================
if task == "RHEED image → Stoichiometry":
    st.subheader("📸 RHEED Image → Stoichiometry")

    mode = st.radio(
        "Choose input mode:",
        ["Select from dataset", "Upload custom RHEED image"]
    )

    if mode == "Select from dataset":
        options = list(range(len(images)))

        def format_image_option(i):
            return f"Index {i} (preview below)"

        idx = st.selectbox(
            "Select an image from dataset:",
            options=options,
            index=0,
            format_func=format_image_option
        )

        input_img = images[idx]
        true_sto = float(stoich[idx])

        st.markdown("**Selected RHEED Image (preview)**")
        st.image(to_viridis(input_img), width=280)

    else:  # Upload custom image
        uploaded_file = st.file_uploader(
            "Upload a RHEED image",
            type=["png", "jpg", "jpeg", "tif", "tiff", "npy"]
        )

        input_img = None
        true_sto = None

        if uploaded_file is not None:
            if uploaded_file.name.endswith(".npy"):
                input_img = np.load(uploaded_file)
            else:
                from PIL import Image
                img_pil = Image.open(uploaded_file).convert("L")  # grayscale
                input_img = np.array(img_pil, dtype=np.float32)

            st.markdown("**Uploaded RHEED Image (preview)**")
            st.image(to_viridis(input_img), width=280)

    if input_img is not None and st.button("Run Prediction"):
        img_pred, sto_pred = predict_from_input(
            model=model,
            input_data=input_img,
            input_type="img",
            IMG_HW=IMG_HW,
            min_stoich=min_stoich,
            max_stoich=max_stoich,
            min_image=min_image,
            max_image=max_image,
            device=device
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Input RHEED Image**")
            st.image(to_viridis(input_img), use_container_width=True)
            if true_sto is not None:
                st.write(f"True stoichiometry: `{true_sto:.4f}`")

        with col2:
            st.markdown("**Reconstructed Image (I → I)**")
            st.image(to_viridis(img_pred), use_container_width=True)
            st.write(f"Predicted stoichiometry: `{sto_pred:.4f}`")

# ===============================
# TASK 2: STO → IMAGE
# ===============================
else:
    st.subheader("🧪 Stoichiometry → RHEED Image")

    mode = st.radio(
        "Choose input mode:",
        ["Select from dataset", "Enter custom value"]
    )

    if mode == "Select from dataset":
        options = list(range(len(stoich)))
        # options = np.argsort(stoich).tolist()        
        
        def format_option(i):
            return f"Index {i} | sto = {stoich[i]:.4f}"
        
        idx = st.selectbox(
            "Select stoichiometry from dataset:",
            options=options,
            index=len(options) - 1,
            format_func=format_option
        )
        input_sto = float(stoich[idx])
        true_img = images[idx]
    else:
        input_sto = st.slider(
            "Stoichiometry value:",
            min_value=float(min_stoich),
            max_value=float(max_stoich),
            value=float((min_stoich + max_stoich) / 2)
        )
        true_img = None

    if st.button("Run Prediction"):
        img_pred, sto_pred = predict_from_input(
            model=model,
            input_data=input_sto,
            input_type="sto",
            IMG_HW=IMG_HW,
            min_stoich=min_stoich,
            max_stoich=max_stoich,
            min_image=min_image,
            max_image=max_image,
            device=device
        )

        col1, col2 = st.columns(2)

        with col1:
            if true_img is not None:
                st.markdown("**True RHEED Image**")
                st.image(to_viridis(true_img), use_container_width=True)
            else:
                st.markdown("**Input Stoichiometry**")
                st.write(f"`{input_sto:.4f}`")

        with col2:
            st.markdown("**Predicted RHEED Image (Sto → I)**")
            st.image(to_viridis(img_pred), use_container_width=True)
            st.write(f"Reconstructed stoichiometry: `{sto_pred:.4f}`")

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown(
"Developed for **Microscopy Hackathon (2025)** · Universal (RHEED ↔ Stoichiometry) Translator"
)
