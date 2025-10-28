
# Comparing ANN and CNN for Satellite and Object Image Classification

**EuroSAT and COIL-100 Analysis using PyTorch**

This project explores and compares the performance of Artificial Neural Networks (ANN) and Convolutional Neural Networks (CNN) for satellite image classification of diverse landscapes and object recognition tasks using the EuroSAT and COIL-100 datasets.
All models were implemented in PyTorch as part of the course DVA476 – Deep Learning for Industrial Imaging at Mälardalen University.

## Overview

The purpose of this project was to:

* Develop and train an ANN using handcrafted image features extracted from each image.
* Implement and train a CNN directly on raw image data.
* Compare the models in terms of accuracy, generalization, and computational complexity.

This study provides insight into how neural networks learn from both engineered features* and spatially learned representations, applied to real-world data such as satellite imagery and industrial object images.


## Feature Extraction (for ANN)

For the ANN, each image was converted to a compact numerical feature vector using the following extraction pipeline:

* **Color features** (HSV histograms)
* **Texture/contrast** (grayscale histogram)
* **Edge-based features** (Sobel magnitude and edge density)

These features significantly reduced input dimensionality while retaining key visual information relevant for landscape and object classification.

## Model Architectures

### Artificial Neural Network (ANN)

* **Input:** Extracted features from the above pipeline
* **Hidden Layers:** Two fully connected layers (128 → 64) with ReLU activations and Dropout
* **Output:** One neuron per class
* **Loss:** CrossEntropyLoss
* **Optimizer:** Adam

**Parameter count:**

* ~20,000 when trained on extracted features
* ~1.6 million when trained directly on flattened raw pixels

This demonstrates the efficiency and interpretability benefits of feature-engineered approaches.

### Convolutional Neural Network (CNN)

* Two convolutional blocks: **Conv → BatchNorm → ReLU → MaxPool**
* Dropout regularization between convolutional and dense layers
* Flattened feature maps → **Fully connected (128 neurons)** → Output
* **Loss:** CrossEntropyLoss
* **Optimizer:** Adam

**Parameter count:** ~2.7 million


## Key Findings

* The **feature extraction pipeline** combining *color histograms*, *edge detection*, and *texture features* drastically reduced the ANN’s parameter count (1.6M → 20K).
* The **CNN** achieved better performance on spatially complex datasets (COIL-100), thanks to its ability to learn translation-invariant features.
* The **ANN** performed competitively on simpler, lower-variance data when regularized, highlighting the potential of lightweight models for constrained environments.
* While CNNs demand more compute (~2.7M parameters), they eliminate the need for manual feature engineering and scale better to complex visual domains.

