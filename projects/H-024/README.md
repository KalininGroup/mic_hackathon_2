# Automated Crack Growth Analysis in In-Situ TEM Tensile Testing

## Overview

This repository provides a deep learning pipeline for automated crack detection, tracking, and quantitative analysis in transmission electron microscopy (TEM) image sequences acquired during in-situ tensile testing. The methodology combines U-Net semantic segmentation with advanced image processing to extract critical fracture mechanics parameters at nanometer resolution.

## Method

### 1. Data Preprocessing
Raw TEM image sequences (.emd format) are converted to NumPy arrays with Gaussian noise filtering and intensity normalization. Frame registration is performed to correct for drift and align consecutive frames across the sequence. Approximately 50 frames are uniformly sampled from each sequence for manual annotation.

### 2. Manual Annotation
Crack regions are annotated using Napari, an interactive visualization tool. Binary masks are generated for training the segmentation model.

### 3. Model Training 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vivekdev-empa/Insitu_tensile_hackathon2025/blob/main/03_train_model_crackgrowth.ipynb)

A U-Net architecture with ResNet34 encoder (pretrained on ImageNet) is trained for binary crack segmentation. The model uses:
- **Loss function:** Combined Dice coefficient and binary cross-entropy
- **Data split:** 70% training, 15% validation, 15% test
- **Augmentation:** Random flips, rotations, brightness/contrast adjustments, and Gaussian noise

### 4. Inference and Measurement
The trained model segments cracks across entire image sequences. From the binary masks, the following fracture mechanics parameters are automatically extracted:

- **Crack tip position:** Coordinates of the leftmost crack point
- **Crack length:** Horizontal extent of the crack
- **Crack area:** Total segmented crack region
- **CTOD (Crack Tip Opening Displacement):** Vertical crack opening measured at multiple distances from the tip
- **Crack width:** Maximum vertical extent
- **Tip radius:** Measure of crack tip bluntness via circle fitting

All measurements are converted from pixels to nanometers based on image calibration (e.g., 1 pixel = 0.4 nm for the 60nm dataset).

## Repository Structure

```
├── 01_noise_removal.ipynb          # EMD to NumPy conversion and preprocessing
├── 02_annotate_crack.ipynb         # Manual annotation using Napari
├── 03_train_model_crackgrowth.ipynb # U-Net model training
├── crack_growth_1508.py            # Main analysis script
├── data/                           # Raw TEM sequences (30nm, 60nm, 120nm)
├── processed/                      # Preprocessed NumPy arrays and videos
├── annotations/                    # Manual crack annotations and train/val/test splits
├── model/                          # Trained U-Net models (.pth)
└── results/                        # Segmentation masks, measurements (CSV), and visualizations
```

## Requirements

- Python 3.8+
- PyTorch, torchvision
- segmentation-models-pytorch
- albumentations
- ncempy, opencv-python, scikit-image, scikit-learn
- napari (for annotation)
- tqdm, pandas, matplotlib

Installation:
```bash
pip install torch torchvision segmentation-models-pytorch albumentations ncempy opencv-python scikit-image scikit-learn napari[all] tqdm pandas matplotlib
```

## Usage

1. **Preprocessing:** Run `01_noise_removal.ipynb` to convert raw .emd files to .npy format
2. **Annotation:** Run `02_annotate_crack.ipynb` to manually label crack regions in ~50 frames
3. **Training:** Run `03_train_model_crackgrowth.ipynb` or use the main script to train the U-Net model
4. **Analysis:** Run `crack_growth_1508.py` to perform inference and extract measurements

Output files include:
- `*_all_predictions.npy`: Binary segmentation masks for all frames
- `*_measurements.csv`: Quantitative crack parameters (length, CTOD, tip radius, etc.)
- `*_temporal_analysis.png`: Time-series plots of crack evolution
- `*_evolution_montage.png`: Visual summary of crack propagation

## Datasets

Three sample datasets at different resolutions are included:

| Dataset | Resolution | Total Frames | Annotated Frames |
|---------|-----------|--------------|------------------|
| 30nm | 220 kx | ~1000 | 50 |
| 60nm | 105 kx | 855 | 50 |
| 120nm | 51-66 kx | ~2000 | 50 |

**Dataset:** [Add Zotero link here]

## Key Features
- Automated batch processing of entire image sequences
- Temporal tracking of crack growth dynamics and blunting behavior
