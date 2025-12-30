# Mic-hackathon 2025
Machine learning–enabled SEM-driven surface morphometrics

This repository contains a lightweight prototype developed during **MIC Hackathon 2025** for **microcrack segmentation in SEM images** and **crack morphology quantification**. The pipeline combines a classical (rule-based) segmentation baseline with a small deep-learning model (ResNet18 + U-Net–style decoder) trained using weak labels derived from the classical mask.

> Note: This is a hackathon prototype intended to demonstrate an end-to-end workflow. It currently trains on a single uploaded image and uses the classical mask as a training target (weak supervision).


## Quick start (Google Colab)

1. Open the notebook in Colab.
2. Run the install cell:
3. Upload an SEM image when prompted.
4. Make sure the filename contains the FOV if you want automatic calibration, e.g.:
   - `sample_10 um_sem.png`

If the FOV is not found in the filename, the script defaults to **10.0 µm**.

---

## Local installation (optional)

### Requirements
- Python 3.9+ recommended
- TensorFlow (CPU or GPU)
- OpenCV
- scikit-image
- matplotlib
- keras-hub

## Related presentation video
[Youtube](https://youtu.be/RWNaPAT1ZDQ?si=a7bOwVTXdyPSOrQ-) 
