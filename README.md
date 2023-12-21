# Automating the Segmentation of X-ray Images using the U-Net Model

## Introduction
This repository contains a series of experiments aimed at automating the segmentation of X-ray images utilizing a U-Net deep learning model. The U-Net architecture is particularly effective for medical image segmentation tasks, and these experiments are designed to explore its capabilities and limitations.

## Experiment Notebooks
We have conducted several experiments with different configurations and parameters. The notebooks detailing these experiments are as follows:

#### 128x128 Resolution Experiments:
- `128x128_250Epoch_WithSubsets.ipynb`: Training on 128x128 images for 250 epochs, including data subsets.
- `128x128_EarlyStop_WithSubsets.ipynb`: Training on 128x128 images with early stopping, including data subsets.

#### 64x64 Resolution Experiments:
- `64x64_250Epoch_WithSubsets.ipynb`: Training on 64x64 images for 250 epochs, including data subsets.
- `64x64_EarlyStop_WithSubsets.ipynb`: Training on 64x64 images with early stopping, including data subsets.

## Advanced and Initial Models
In addition to the experiment notebooks, we have also included:

- **Advanced Model (`Advanced_model.ipynb`):**
  This notebook contains a more sophisticated version of our U-Net model. It includes enhanced features and optimizations beyond our initial experiments.

- **Initial Model (`initial_model.py`):**
  This Python script includes the baseline U-Net model that was used as a starting point.
