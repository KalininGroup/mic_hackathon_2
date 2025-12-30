# MIC_Hackthon_Team_Nanobots
This project implements a deep learning pipeline to detect, classify, and localize atomic lattice defects from microscopy or simulation images. 
It uses a two-stage CNN approach (Normal vs Defect, followed by defect-type classification) with a Raw + FFT input representation. 
The pipeline scales to large images using sliding-window localization and expands limited datasets through pseudo-labeling and auto-sorting.
# Technologies & Models Used
- Python, NumPy, SciPy – data processing and FFT-based feature extraction  
- TensorFlow / Keras – CNN model development and training  
- Scikit-learn – dataset splitting and evaluation metrics  
- Convolutional Neural Networks (CNNs) – binary and multiclass defect classification  
- Sliding-window inference & pseudo-labeling – pixel-level localization and dataset expansion

