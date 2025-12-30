# ML-Hackathon-2025
Classifies defects in STEM images. Made as part of the Machine Learning and Microscopy Hackathon hosted at the University of Tennesse.

classifyImage.ipynb uses the pretrained models to classify defects in a provided STEM image.
monoFeatureDetecterTrainer.ipynb was used to train the models.
preprocPipelineDemo.ipynb contains an example of the preprocessing pipeline used. 

Data and project from:
https://github.com/RAW-Ayyubi/defect-classification-in-stem-images


Using this project:
Image files must be in either .tif or .jpg format. 
All desired model files must be in the correct TrainedModels directory.
The output is saved in the Output folder.
To note: Interstitial detection for CdTe Samples does not work very well, as a side effect of the preprocessing. A different model has been trained for interstitial classification in this ensamble. 


Authors:
Noah Holt, Kamal Khanal, Joshua Marvin, Ramji Subedi
