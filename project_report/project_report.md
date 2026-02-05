# Project Report: Transfer Learning for Image Classification

## Introduction
Early detection of plant diseases is crucial for healthy crops and better food production. This repository implements a Deep Learning solution to automatically classify 38 different types of plant leaves (including both healthy and diseased samples) using the PlantVillage dataset.

Goal is to compare the performance of two popular CNN models: DenseNet-121 and EfficientNet B0. We explore how well these models learn using two different strategies: training just the classification head versus fine-tuning the model's deeper layers.

## Problem Definition
The goal of this project is to build a Deep Learning model that can correctly identify 38 plant conditions from pictures of single leaves. These plant conditions include 14 types of crops and various diseases. The main problem is telling diseases apart that look very similar and have the texture. The Deep Learning model has to be good at distinguishing between these diseases, with similar visual symptoms and texture patterns of the plant conditions.

To address this, we aim to implement and optimize advanced CNN architectures—DenseNet-121 and EfficientNet B0—to automate the detection process. This solution is designed to make disease diagnosis faster and more accessible, enabling early intervention to significantly decrease crop losses.

## Dataset
For dataset we used PlantVillage from Kaggle.

![Training dataset class distribution](images/training_dataset.png)

We faced a problem of imbalanced classes: some classes (like specific Tomato diseases) have thousands of images, while others (like Potato Healthy) may have significantly fewer.

We address this problem during training step.

## Model Architecture
* DenseNet-121

![DenseNet121 Model Architecture](images/DenseNet121_architecture.png)


* EfficientNet B0

![EfficientNet B0 Model Architecture](images/EfficientNet_B0_architecture.webp)



## Training

We looked at Transfer Learning to make these strong pre-trained models work with the data we have. We tried two ways of training the models:

* **Head-Only Training:** Freezing the main model and training only the final classification layer.

* **Fine-Tuning:** Unfreezing specific layers to refine the model's performance on leaf textures.

## Evaluation


![](images/loss_comparison_of_training.png)
![](images/accuracy_comparison_of_training.png)
![](images/global_accuracy.png)
![Models Comparison](images/model_performance.png)

## Conclusions