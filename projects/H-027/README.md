# EBSD Phase Mapping Using Deep Learning

This repository contains the code and scripts developed by the VPD Group for **phase mapping in Electron Backscatter Diffraction (EBSD)** data using deep learning approaches, presented at the **Hackathon 2025**.

The project focuses on comparing **standard supervised CNN training (cross-entropy loss)** with **supervised contrastive learning** for EBSD phase classification and mapping of fcc Ni, L1₂ Ni₃Al, and L1₂ Ni₃Fe phases. The EBSD dataset was obtained from the literature [1].

---

## Project Overview

Electron Backscatter Diffraction (EBSD) is widely used for crystallographic phase identification, but conventional indexing approaches can struggle with noisy patterns, complex phase mixtures, or limited training data. Some past work has shown the potential of using deep learning to obtain composition information from EBSD using cross entropy learning [2]. Next, it is desired to test these methods on different Ni-based alloys, which have relevant high temperature and corrosion resistance applications. Additionally, contrastive learning is an alternative training method that may improve accuracy. 

In this project, we explore deep learning–based methods to:
- Classify EBSD Kikuchi patterns into crystallographic phases
- Compare cross-entropy loss versus contrastive learning
- Evaluate performance for EBSD phase mapping applications

---

## Methods Implemented

- Convolutional Neural Networks (CNNs) with EfficientNet architecture
- Supervised learning with:
  - Cross-entropy loss
  - Supervised contrastive loss 
- Data augmentation (random horizontal and vertical flips)

## Code description

- `EBSD_phaseclassification.ipynb` sorts data into train, validation, and test splits; trains contrastive learning model; visualizes results from contrastive learning and cross entropy learning
- `EBSD_efficientnet_expdata.py` trains model using cross entropy loss. Use the --validation flag to train on the training data and evaluate on validation data, and remove it to train on both the training and validation data and evaluate on test data.

## Results:

Both cross entropy and contrastive learning achieved similar test accuracies. A confusion matrix of predictions shows that Ni and Ni3Fe were often misclassified, which is likely due to the low difference in atomic number between Fe (26) and Ni (28). This suggests a physical limit to the composition information obtainable from EBSD patterns. However, the overall performance for the three phases still suggest potential for improved phase mapping Ni-based alloys over conventional techniques that only analyze band geometries.

<img width="512" height="412" alt="unnamed (18)" src="https://github.com/user-attachments/assets/1b5d7311-941a-4118-a16a-4305474b9af1" />

## Acknowledgements: 

This research was supported in part through the computational resources and staff contributions provided for the Quest high performance computing facility at Northwestern University which is jointly supported by the Office of the Provost, the Office for Research, and Northwestern University Information Technology. This work made use of Northwestern University’s NUANCE Center, which has received support from the SHyNE Resource (NSF ECCS-2025633), the International Institute for Nanotechnology (IIN), and Northwestern’s MRSEC program (NSF DMR-2308691). 

## References:

- (1) Kaufmann, K.; Zhu, C.; Rosengarten, A. S.; Vecchio, K. S. Deep Neural Network Enabled Space Group Identification in EBSD. Microscopy and Microanalysis 2020, 26 (3), 447–457. https://doi.org/10.1017/S1431927620001506. 
- (2)	Kaufmann, K.; Zhu, C.; Rosengarten, A. S.; Maryanovsky, D.; Wang, H.; Vecchio, K. S. Phase Mapping in EBSD Using Convolutional Neural Networks. Microscopy and Microanalysis 2020, 26 (3), 458–468. https://doi.org/10.1017/S1431927620001488. 	
- (3)	Musgrave, K.; Belongie, S.; Lim, S.-N. PyTorch Metric Learning. arXiv August 20, 2020. https://doi.org/10.48550/arXiv.2008.09164. 

---
