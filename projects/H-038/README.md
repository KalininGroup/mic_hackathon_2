# VGG16-defect-detection-in-STEM-Images
Automated region-of-interest and defect detection in large, atomic-resolution STEM images via patch-based classification using a VGG-16 convolutional neural network.

# Patch-Based Defect Detection in STEM Images using VGG-16
# Overview

This repository presents a patch-based deep-learning framework for automated defect detection and analysis in atomic-resolution scanning transmission electron microscopy (STEM) images. The workflow combines supervised learning using a VGG-16 convolutional neural network with unsupervised clustering of learned feature embeddings to enable robust defect identification, structural grouping, and spatial localization across large STEM datasets.

Large STEM images are divided into fixed-size patches, which are processed using a transfer-learned VGG-16 model to extract physically meaningful feature representations. These representations are used both for supervised defect classification and for unsupervised clustering, enabling complementary views of defect structure and facilitating downstream applications such as automated region-of-interest selection and closed-loop microscopy.

# Key Features
Patch-based analysis of STEM images (64 × 64 pixels)
ImageNet-pretrained VGG-16 for transfer learning on STEM data
Supervised classification of defect types with confidence scores
Unsupervised clustering (PCA + K-means) of learned embeddings
Silhouette score-guided cluster selection
Patch-wise spatial mapping and defect localization
Designed for automated STEM workflows

# Datasets
SrTiO₃ (STO) and CdTe Training Data
All primary training datasets for SrTiO₃ (STO) and CdTe are sourced from the following public repository:
🔗 Defect Classification in STEM Images
https://github.com/RAW-Ayyubi/defect-classification-in-stem-images
This dataset provides labeled STEM image patches corresponding to multiple defect types and bulk regions.

# Additional CdTe Dataset
For the CdTe system, a small additional dataset was created to supplement the publicly available data listed above.
The additional CdTe data can be found in this repository under: "Additional CdTe Images".
