# Nii2NPZ

This repository provides a simple example script to preprocess medical imaging data from my previous paper — converting `.nii.gz` files into `.npz` format. It includes key preprocessing steps such as intensity normalization, cropping, and embedding metadata like age and disease condition.

## Features

- Converts NIfTI (`.nii.gz`) images to compressed `.npz` format  
- Applies intensity normalization to standardize image values  
- Crops the image and corresponding segmentations to a specified 3D region  
- Includes subject metadata: `age` and `disease_condition`  

## Output `.npz` Structure

Each saved `.npz` file contains the following keys:

- **`vol`**: Preprocessed image volume (3D NumPy array)
- **`seg`**: Anatomical segmentation (3D NumPy array)
- **`synth_seg`**: SynthSeg segmentation (3D NumPy array)
- **`age`**: Subject's age (float)
- **`disease_condition`**: Diagnosis label  
  - `0` = Healthy Control (HC)  
  - `1` = Alzheimer's Disease (AD)

