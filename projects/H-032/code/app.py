import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image

from segmentation import analyze_segmentation, save_overlay
from model import explain_from_stats

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -------------------------------
# Routes
# -------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/chat", methods=["POST"])
def chat():
    # ---- Get inputs ----
    image = request.files["image"]
    user_prompt = request.form["prompt"]

    # ---- Save uploaded image (TIFF allowed) ----
    original_filename = image.filename
    image_path = os.path.join(UPLOAD_FOLDER, original_filename)
    image.save(image_path)

    # ---- Convert input image to PNG for browser display ----
    base_name = os.path.splitext(original_filename)[0]
    display_image_name = base_name + ".png"
    display_image_path = os.path.join(UPLOAD_FOLDER, display_image_name)

    Image.open(image_path).convert("RGB").save(display_image_path)

    # ---- Segmentation analysis (image is assumed segmented) ----
    stats = analyze_segmentation(image_path)

    # ---- Generate overlay PNG ----
    overlay_path = save_overlay(
        image_path=image_path,
        mask=stats["mask"],
        filename=f"overlay_{base_name}.png"
    )

    # ---- Construct grounded prompt ----
    system_prompt = f"""
You are a biomedical imaging expert specializing in cellular microscopy.

Context:
- The segmented objects correspond to cell nuclei (or mitochondria).
- The image is a microscopy image (fluorescence / histology / EM).

Quantitative segmentation results:
- Number of nuclei detected: {stats['count']}
- Mean nuclear area: {stats['mean_area']} pixels

Guidelines:
- Interpret objects explicitly as nuclei
- Discuss density and size consistency
- Comment on morphology where appropriate
- Avoid medical diagnosis
- Use precise biological language

User question:
{user_prompt}
"""

    # ---- LLM explanation ----
    explanation = explain_from_stats(stats, system_prompt)

    # ---- Return response ----
    return jsonify({
        "explanation": explanation,
        "stats": {
            "count": stats["count"],
            "mean_area": stats["mean_area"]
        },
        "input_image_url": f"/uploads/{display_image_name}",
        "overlay_url": f"/{overlay_path}"
    })


if __name__ == "__main__":
    app.run(debug=True)
