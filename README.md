# EuroSAT Land Use Classification using Deep CNN Architectures

## Project Description

This project presents a **comparative evaluation of multiple convolutional neural network (CNN) architectures** for **land use and land cover classification** using the **EuroSAT satellite image dataset**.

The main objective is to analyze the **trade-offs between classification performance, model complexity, and deployment feasibility** across different deep learning architectures under a unified experimental setup.

The following models are implemented and evaluated:

* VGG19 (custom implementation, trained from scratch)
* Inception V1 (GoogLeNet)
* ResNet50
* MobileNetV2

---

## Dataset

* **Name:** EuroSAT Land Use Dataset
* **Image Type:** RGB satellite images
* **Native Resolution:** 64 × 64 × 3
* **Number of Classes:** 10
* **Task:** Multi-class image classification

---

## Model Architectures

### VGG19 (Custom Implementation)

* Implemented from scratch following the original VGG design principles
* Composed of **five convolutional blocks** using uniform **3×3 convolutions**
* Max pooling applied after each block
* Feature depth increases progressively:
  **64 → 128 → 256 → 512 → 512**
* Classification head consists of:

  * Flatten
  * Dense(512)
  * Dropout
  * Softmax output layer
* Trained from scratch on a limited subset of the dataset without pre-training

**Observation:**
The model failed to converge to competitive performance due to the absence of pre-training and the relatively small dataset size, which is a known limitation for deep architectures trained from scratch.

---

### Inception V1 (GoogLeNet)

* Built around **Inception modules** that apply multiple convolutional filters in parallel:

  * 1×1, 3×3, and 5×5 convolutions
* **1×1 convolutions** used for dimensionality reduction
* Uses **Global Average Pooling** instead of large fully connected layers
* Originally proposed with auxiliary classifiers to improve gradient flow; the main architecture was used in this implementation
* Pre-trained on ImageNet and fine-tuned on EuroSAT

---

### ResNet50

* Utilizes **residual learning** to enable very deep networks

* Skip connections compute:

  ```
  F(x) + x
  ```

* Employs **bottleneck residual blocks**:

  * 1×1 convolution for dimensionality reduction
  * 3×3 convolution for feature extraction
  * 1×1 convolution for dimensionality expansion

* Ends with **Global Average Pooling** followed by a Softmax classifier

* Pre-trained on ImageNet and fine-tuned on EuroSAT

---

### MobileNetV2

* Designed for **efficient and lightweight deployment**
* Uses **depthwise separable convolutions**
* Implements **inverted residuals with linear bottlenecks**
* Significantly fewer parameters compared to ResNet and Inception models
* Pre-trained on ImageNet and fine-tuned on EuroSAT

---

## Training Strategy

* **Transfer learning** applied to:

  * ResNet50
  * Inception V1
  * MobileNetV2
* **VGG19 trained from scratch** for comparison purposes
* Input images were resized to **224 × 224** to meet the input requirements of ImageNet pre-trained architectures.
* Final classification layer adjusted to match the **10 EuroSAT classes**

---

## Experimental Results

### Performance on Test Set

| Model        | Accuracy | Precision | Recall | F1-score |
| ------------ | -------- | --------- | ------ | -------- |
| ResNet50     | 96.15%   | 96.11%    | 95.84% | 95.93%   |
| Inception V1 | 94.69%   | 94.61%    | 94.36% | 94.41%   |
| MobileNetV2  | 91.83%   | 91.61%    | 91.19% | 91.30%   |
| VGG19        | <30%     | Low       | Low    | Low      |

**Note:**
The poor performance of VGG19 is expected, as deep CNNs typically require large-scale datasets or pre-training to generalize effectively. The remaining models benefit significantly from ImageNet initialization.

---

## Comparative Analysis

### ResNet50

* Best overall performance
* Strong ability to capture subtle texture and spatial patterns
* Computationally expensive and memory intensive

### Inception V1

* Efficient multi-scale feature extraction
* Complex architecture and harder to modify or extend

### MobileNetV2

* Lightweight and fast
* Suitable for edge and mobile deployment
* Slight accuracy degradation compared to heavier architectures

### VGG19

* Conceptually simple and easy to understand
* Extremely parameter-heavy
* Not suitable for small datasets without transfer learning

---

## Conclusion

**ResNet50 achieved the highest overall performance (96.15% test accuracy)** on the EuroSAT dataset.

This result can be attributed to:

1. Effective transfer learning from ImageNet
2. Deep feature extraction enabled by residual connections
3. Stable optimization despite increased network depth

Overall, the experiment highlights the importance of transfer learning when working with limited data and demonstrates how architectural design choices impact both performance and deployment feasibility.


