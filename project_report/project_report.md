# Project Report: Transfer Learning for Image Classification

## Introduction and Problem Definition

Early detection of plant diseases is essential for maintaining crop health and improving food production. In this project, we implement a deep learning–based image classification system that identifies 38 plant leaf conditions, including healthy and diseased samples, using the PlantVillage dataset. The goal is to evaluate and compare two pretrained CNN architectures, DenseNet-121 and EfficientNet-B0, using transfer learning.

The task is a multi-class classification problem where many disease categories exhibit high visual similarity, especially within the same crop species. This requires the model to capture fine-grained texture and color patterns rather than relying on coarse visual features. To address this challenge, we compare head-only training with fine-tuning of deeper layers, assessing how model adaptation impacts classification performance and generalization.

---

## Dataset

The dataset used in this project is PlantVillage, obtained from Kaggle. It contains 54,305 labeled RGB images with a resolution of 256x256 pixels of single plant leaves captured under relatively controlled conditions. The images cover 14 species of crops (12 of which have healthy leaf images) and 17 basic diseases, 17 basic diseases, 4 bacterial diseases, 2 diseases caused by mold (oomycete), 2 viral diseases and 1 disease caused by a mite. In total the dataset has 38 classes. Below we take a look at a few images from our dataset with their labels:  
![](images/Input_images_AI_lab.png)

An important question is: "How is our data distributed?". It can be answered by observing the tables below:

![](images/Ai_lab_train_distrib.png)

Test loader:

![](images/Ai_lab_test_distrib.png)



As we can see, the dataset exhibits class imbalance, where some disease categories (such as several tomato leaf diseases) contain thousands of samples, while others (for example, healthy potato leaves) are represented by significantly fewer images. This imbalance can bias learning toward majority classes and poses an additional challenge for reliable classification, particularly for underrepresented categories.

---

## Model Architecture

### DenseNet-121

DenseNet-121 is a deep convolutional neural network that uses dense connectivity between layers, where each layer receives feature maps from all preceding layers within the same dense block. This design allows for feature reuse and improves gradient flow, which leads to more stable training in deep networks while keeping the parameter count relatively low. Due to its ability to preserve and reuse fine-grained features, DenseNet-121 is well suited for tasks that require detailed texture discrimination, in our case: identifying visually similar plant diseases.

![DenseNet121 Model Architecture](images/DenseNet121_architecture.png)

---

### EfficientNet-B0

EfficientNet-B0 is built around the principle of compound scaling, which balances network depth, width, and input resolution to achieve high accuracy with reduced computational cost. Its architecture is composed of repeated MBConv (Mobile Inverted Bottleneck) blocks, which use depthwise separable convolutions and squeeze-and-excitation mechanisms to efficiently capture both spatial and channel-wise information. By stacking these MBConv blocks with varying kernel sizes and expansion ratios across stages, EfficientNet-B0 achieves strong representational power while remaining highly parameter-efficient.

| EfficientNet-B0: High-level architecture      | EfficientNet-B0: MBConv block structure        |
| --------------------------------------------- | ---------------------------------------------- |
| ![](images/EfficientNet_B0_architecture.webp) | ![](images/efficientnet_b0_classification.png) |

---

## Training

We usedTransfer Learning to make these strong pre-trained models work with the data we have. We tried two ways of training the models:

* **Head-Only Training:** Freezing the main model and training only the final newly added classification layer. This is more computationally efficient and reduces the risk of overfitting, however it limits the model's ability to adapt to domain-specific leaf textures.

* **Fine-Tuning:** Unfreezing specific layers to refine the model's performance on leaf textures. Here the model is allowed to refine high-level feature representations specifically for plant disease patterns, at the cost of increased training time (and possibly overfitting).

---

## Evaluation

### Loss Curve Comparison

The graphs below compare training and validation loss for DenseNet-121 and EfficientNet-B0 under head-only training and fine-tuning.

For both networks, head-only training shows a rapid initial loss drop followed by early saturation. Validation loss remains consistently lower than training loss, indicating stable optimization but limited representational adaptation due to frozen feature extractors.

![](images/loss_comparison_of_training.png)

In contrast, fine-tuned models achieve significantly lower final loss values and smoother convergence. The reduced gap between training and validation loss suggests improved generalization and better alignment of learned features with the PlantVillage domain. Minor fluctuations in validation loss toward later epochs indicate the onset of diminishing returns rather than instability.

---

### Global Accuracy Comparison

The following results summarize the maximum achieved training and validation accuracy for each model and training strategy.

![](images/accuracy_comparison_of_training.png)

Both architectures benefit significantly from fine-tuning, with validation accuracy improving by approximately 3–4 percentage points compared to head-only training. EfficientNet-B0 (fine-tuned) achieves the highest overall validation accuracy, marginally outperforming DenseNet-121 while using a more parameter-efficient design.

Below we can see the small gap between training and validation accuracy across all configurations, which confirms strong generalization and absence of severe overfitting.

![](images/global_accuracy.png)

---

### Quantitative Performance

This table provides a comparison of both architectures across training strategies, including learning rate, number of epochs, and peak training and validation accuracy.

![Models Comparison](images/model_performance.png)

Both DenseNet-121 and EfficientNet-B0 show a clear performance jump when moving from head-only training to fine-tuning. Head-only training converges quickly within 5 epochs but is limited in maximum achievable accuracy, which shows that frozen backbone features are insufficient for capturing fine-grained plant disease patterns.

Fine-tuning, performed with a lower learning rate and extended training, enables both models to reach near-saturated accuracy above 99% on both training and validation sets. EfficientNet-B0 slightly outperforms DenseNet-121 in terms of validation accuracy while maintaining a more parameter-efficient design.

---

## Conclusions

The results show that while head-only training provides a fast and stable baseline, fine-tuning significantly improves classification accuracy by allowing the models to adapt to domain-specific leaf textures. Fine-tuned models achieved validation accuracies above 99% with minimal generalization gaps, indicating strong learning and limited overfitting.

Between the two architectures, EfficientNet-B0 achieved slightly higher validation accuracy while remaining more computationally efficient. Our results confirm that transfer learning combined with selective fine-tuning is an effective approach for accurate and scalable plant disease detection.

---