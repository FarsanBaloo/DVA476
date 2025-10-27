

# Comparing ANN and CNN on EuroSAT and COIL-100 Datasets

This project investigates and compares the performance of an **Artificial Neural Network (ANN)** and a **Convolutional Neural Network (CNN)** using two distinct image datasets — **EuroSAT** and **COIL-100** — implemented in **PyTorch**.
The work was carried out as part of the course *DVA476 – Deep Learning for Industrial Imaging* at Mälardalen University.


## Overview

The objective of this project was to:

* Implement and train an ANN using extracted image features.
* Implement and train a CNN using raw image data.
* Compare their accuracy, generalization, and computational complexity.


## Model Architectures

### Artificial Neural Network (ANN)

* Input: Extracted features (Edges using Sobel,color histograms using HSV)
* Two hidden layers with ReLU activations and Dropout regularization
* Output layer: one neuron per class
* Loss: CrossEntropyLoss
* Optimizer: Adam

**Parameter count:**
~20,000 when trained on extracted features
~1.6 million when trained on raw pixel data

### Convolutional Neural Network (CNN)

* Two convolutional blocks: Conv → BatchNorm → ReLU → MaxPool
* Dropout for regularization
* Flatten → Fully connected layer (128 neurons) → Output layer
* Loss: CrossEntropyLoss
* Optimizer: Adam

**Parameter count:** ~1 million


## Key Findings

* Feature extraction significantly reduced ANN complexity (from 1.6M → 20K parameters).
* CNNs generalized better on spatially complex datasets such as COIL-100.
* ANNs performed well on simpler patterns when properly regularized.

