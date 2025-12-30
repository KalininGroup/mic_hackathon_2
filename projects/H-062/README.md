# Microscopy_Hackathon_2025
Dima Traboulsi, Saven Denha and Jonah Wilbur's 2025 hackathon project

This script implements the histological image registration pipeline.
The goal of the model is not dense pixel-wise alignment, but to learn a robust similarity measure between nuclei across two histological sections, which can then be used to generate high-confidence anchor correspondences for downstream geometric alignment.

The learned similarities are combined with geometric consistency checks and robust fitting to estimate global or non-rigid transforms.

It first uses an FFT based alignment to get a good rough alignment, then uses a self-supervised 2D-CNN for nuclei matching. The most consistent nuclei are then matched between images to find anchor points. The anchor points are then used to guide a more adaptive transform.

Unfortunately, all image files were too large to upload as zip files.
