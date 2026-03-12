# Diabetic Retinopathy Detection using ConvMixer

A deep learning-based computer vision system for automated detection of diabetic retinopathy severity from retinal fundus images using the ConvMixer architecture.

This project implements an end-to-end TensorFlow pipeline for dataset preprocessing, model training, evaluation, and performance analysis.

---

# Project Overview

Diabetic retinopathy (DR) is a serious diabetes complication that can lead to blindness if not detected early. Automated detection using computer vision can help assist medical professionals by enabling fast and scalable screening.

This project builds a deep learning model that classifies retinal fundus images into multiple DR severity stages.

The system uses the **ConvMixer architecture**, a convolutional neural network inspired by vision transformers, to learn spatial representations from retinal images.

---

# Key Features

- Computer vision pipeline for medical image classification
- ConvMixer deep learning architecture
- Multi-class diabetic retinopathy severity detection
- TensorFlow training pipeline
- Confusion matrix performance evaluation
- Scalable dataset loading using TensorFlow datasets

---

# Dataset

The model is trained on a retinal fundus image dataset containing multiple stages of diabetic retinopathy.

Classes include:

- No DR
- Mild
- Moderate
- Severe
- Proliferative DR

Images are resized to **224 × 224 resolution** and processed using TensorFlow's data pipeline.

---

# Model Architecture

This project implements the **ConvMixer architecture**, which combines convolution operations with patch-based processing.

Architecture components:

1. Input Image
2. Patch Embedding using Conv2D
3. ConvMixer Blocks
   - Depthwise Convolution
   - Residual Connections
   - Pointwise Convolution
4. Global Average Pooling
5. Dense Classification Layer

ConvMixer configuration used in this project:

- Filters: 256
- Depth: 8 layers
- Kernel Size: 5
- Patch Size: 2
- Image Size: 224 × 224

---

# Project Structure
