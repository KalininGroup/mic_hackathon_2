# Cryo-EM-ice-detection-Mic-hacakthon
This project presents a machine learning based approach for automatically detecting ice contamination in cryo-electron microscopy (cryo-EM) micrographs. Using expert-curated annotations from the CryoPPP dataset (EMPIAR-10061), image patches are labeled as 'BAD' or "GOOD' and used to train a convolutional Neural Network to classify if the patches are real or contaminated.

The pipeline used in this project is as follows:

## Step 1: Obtaning the data from cryoPPP (Dataset used:10061)

Each dataset will contain two folders. micrographs and ground_truth

micrographs : This folder contains all the 2D images recorded by the detector.

ground_truth : This folder contains coordinates of the false_positives and true particles.

*True particle coordinates*: points where experts say “this is a real particle”
	
*False-positive coordinates*: points where experts say “this is not a particle” (often ice/contamination/carbon)
	
**Output of Step 1:**

A list of micrographs + two sets of points per micrograph (true vs false)

## Step 2: Preprocessing the dataset

Before training the model micrographs were divided into smaller, fixed size patches (64×64 px). 

BAD patches (artifact-enriched) : Square patches centered on false positive coordiates (x,y) were labeled as 'BAD'

GOOD Patches (clean): Square patches centered on true particle coordiates (x,y) were labeled as 'GOOD'

Code used : data_preprocessing.ipynb
	
**Output of Step 2:**

A dataset of labeled patches as 'BAD' or 'GOOD'
	
## Step 3: Adding FFT as a second view 

For each patch compute a FFT patch (the Fast Fourier Transform (FFT) of the patch converts the image from real space into frequency space to reveal repeating patterns and structural regularities)

Why this helps:

Crystalline ice produces characteristic ring patterns in frequency space.

Thick ice reduces high-frequency content, causing the signal to drop off.

Using both real-space and FFT information allows the model to mimic how cryo-EM experts visually assess ice quality.

Code used : computing_fft_patches.ipynb

**Output of Step 3:**

Each patch from step 2 becomes:
	
Input =[real_patch, fft_patch]
	
Label = GOOD or BAD
	
## Step 4: Training a CNN classifier 

Input: a patch and its FFT

Output: probability the patch is BAD (artifact-like)

**Output of Step 4:**

A trained CNN that can score any patch {Porb(BAD | patch)}

Code used : model_building.ipynb
	
## Step 5: Scan a whole micrograph to create an ice mask

Move a square window across the micrograph (sliding window)

For each location extract a patch (+ FFT)

The trained CNN outputs P(BAD)

Combine all probabilities into a 2D map over the micrograph

**Output of Step 5:**

An artifact probability map
	
A binary mask after thresholding:
	
1 = good region
	
0 = bad region (ice/artifacts)
	
