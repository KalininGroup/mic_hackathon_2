Anomaly Detection and Clustering of Atomic-Resolution STEM Images

This repository contains the code and notebooks accompanying the project
“Anomaly Detection and Clustering of Atomic-Resolution STEM Images Using Semantic Segmentation.”

The aim of this work is to develop an automated, scalable workflow for identifying, localising, and categorising defects in atomic-resolution HAADF-STEM images using machine learning, without relying on pixel-level annotations.

Project Overview

Atomic-resolution STEM experiments generate large volumes of high-dimensional image data. Manual defect identification is time-consuming, subjective, and difficult to scale, particularly when multiple defect types coexist within a single image.

This project addresses these challenges by combining:

Supervised image classification (baseline defect identification),

Weakly supervised semantic segmentation (defect localisation),

Unsupervised clustering (defect taxonomy and separation).

The framework is demonstrated on multiple material systems, including CdTe and BFO, and is designed to be adaptable to other crystalline materials.

Repository Structure

This repository currently includes three Jupyter notebooks, corresponding to the main stages of the investigation:

CNN_Image_Classification_CdTe.ipynb

Implements supervised defect classification using transfer learning.

Establishes a baseline classification performance for CdTe HAADF-STEM images.


2. Image Classification – SrTiO₃ 

CNN_Image_Classification_STO.ipynb

Trains model with same classification framework to BFO STEM data.


3. Semantic Segmentation and Defect Localisation

segmentation_usage_example.ipynb

Implements a U-Net-based semantic segmentation model.

Uses weak supervision, with pseudo-labels generated from local intensity deviations relative to bulk lattice regions.

Produces defect saliency heatmaps for each defect class within the same image.



Methodology Summary

Supervised classification is used to detect the presence of defects and establish baseline performance.

Weakly supervised semantic segmentation enables spatial localisation of multiple defects without pixel-level ground truth.

Quantitative descriptors extracted from defect heatmaps are clustered using k-means, with cluster number selection guided by silhouette metrics.

t-SNE visualisation is used to explore defect separability in feature space.

Requirements

The notebooks are written in Python and rely on standard scientific and machine-learning libraries, including:

Python ≥ 3.8

NumPy

SciPy

Matplotlib

scikit-learn

TensorFlow / Keras

OpenCV (optional, depending on preprocessing)

Exact versions can be added later if needed.

Future Work

Planned extensions of this work include:

Application to additional material systems,

Improved robustness to experimental noise,

Integration of physics-informed constraints,

Extension to fully unsupervised or generative models for defect discovery.

Citation

If you use this code or build upon this work, please cite:

Surabhi Sathish, Claudia Sosa Espada, Thomas Karagiannis
Anomaly Detection and Clustering of Atomic-Resolution STEM Images Using Semantic Segmentation

Contact

For questions or suggestions, feel free to open an issue or contact the authors.
