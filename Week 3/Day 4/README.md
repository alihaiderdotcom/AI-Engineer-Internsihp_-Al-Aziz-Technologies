# Week 3 - Day 4: Computer Vision Fundamentals & Convolutional Neural Networks

## Overview
Day 4 introduces Computer Vision (CV), image representation in digital memory, foundational image preprocessing with OpenCV, and the theory and implementation of Convolutional Neural Networks (CNNs) in PyTorch.

---

## 1. Images as Data
In computing, a digital image is represented as a multidimensional array (tensor) of discrete pixel intensities:
* **Grayscale Images:** 2D tensor of shape $(H, W)$ with values typically in range $[0, 255]$ (8-bit unsigned integer `uint8`).
* **Color Images:** 3D tensor of shape $(H, W, C)$, where $C=3$ for RGB/BGR color channels.
* **PyTorch Tensor Format:** PyTorch vision conventions follow channel-first order:
  $$\mathbf{X} \in \mathbb{R}^{B \times C \times H \times W}$$
  where $B$ is batch size, $C$ is channels, $H$ is height, and $W$ is width.
* **OpenCV Color Convention:** OpenCV loads images in **BGR** (Blue, Green, Red) format by default, whereas Matplotlib and PyTorch expect **RGB**. Conversion is achieved via:
  ```python
  rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
  ```

---

## 2. Image Preprocessing with OpenCV
Essential preprocessing operations before model inference include:
1. **Color Space Conversion:** `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` simplifies images to single-channel intensity maps.
2. **Resizing:** Transforming images to uniform input resolution:
   * Downsampling: `cv2.INTER_AREA` (avoids moiré artifacts).
   * Upsampling: `cv2.INTER_CUBIC` or `cv2.INTER_LINEAR`.
3. **Filtering & Smoothing:** `cv2.GaussianBlur(img, (k, k), sigma)` suppresses high-frequency noise.
4. **Edge Detection:** `cv2.Canny(gray, threshold1, threshold2)` detects sharp spatial intensity gradients.
5. **Pixel Normalization:**
   * Min-Max scaling: $x_{norm} = \frac{x}{255.0} \in [0.0, 1.0]$
   * Standardization: $z = \frac{x - \mu}{\sigma}$

---

## 3. Convolutional Neural Networks (CNNs)
Traditional fully connected networks struggle with images due to excessive parameter counts and lack of spatial translation invariance. CNNs solve this using parameter sharing and local receptive fields.

```
Input Image [1, 32, 32]
       │
   [ Conv2D (3x3) + BatchNorm + ReLU ]
       │
   [ MaxPool2D (2x2) ] ──> Downsampled Feature Maps [16, 16, 16]
       │
   [ Conv2D (3x3) + BatchNorm + ReLU ]
       │
   [ MaxPool2D (2x2) ] ──> Higher-level Feature Maps [32, 8, 8]
       │
    Flatten ──> [ Linear (2048 -> 64) + ReLU + Dropout ] ──> [ Output Logits ]
```

### Core CNN Building Blocks:
* **Convolution Layer (`nn.Conv2d`):** Slides learnable 2D filters across the input, computing dot products to produce **feature maps**.
  * Output dimension formula:
    $$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
    where $W$ is input size, $K$ is kernel size, $P$ is padding, and $S$ is stride.
* **Stride:** The step size the kernel shifts at each step ($S=1$ preserves resolution; $S > 1$ downsamples).
* **Padding:** Adding border pixels (typically zeros) around the input to control output spatial dimensions and preserve edge features.
* **Pooling Layer (`nn.MaxPool2d`):** Downsamples feature maps by taking the maximum value within each pooling window, reducing computational complexity and providing spatial translation invariance.
* **Hierarchical Representation:**
  * Shallow Conv layers detect low-level primitives (edges, gradients, corners).
  * Intermediate layers detect textures, patterns, and contours.
  * Deep layers capture class-specific semantic parts (shapes, object components).

---

## 4. Object Detection Overview
While **Image Classification** assigns a single label to an entire image, **Object Detection** identifies multiple objects along with their spatial boundaries:
* **Bounding Boxes:** Represented as $(x_{min}, y_{min}, x_{max}, y_{max})$ or $(x_{center}, y_{center}, width, height)$.
* **Intersection over Union (IoU):** Metric measuring overlap between predicted box $B_p$ and ground truth box $B_{gt}$:
  $$\text{IoU} = \frac{\text{Area}(B_p \cap B_{gt})}{\text{Area}(B_p \cup B_{gt})}$$
* **Mean Average Precision (mAP):** Standard benchmark metric assessing both localization precision and classification recall across IoU thresholds.
* **Popular Architectures:** YOLO (You Only Look Once), Faster R-CNN, SSD (Single Shot MultiBox Detector).

---

## 5. Image Augmentation
Data augmentation artificially expands the diversity of training samples to prevent overfitting:
* **Horizontal / Vertical Flips:** `cv2.flip(img, 1)`
* **Affine Rotations:** `cv2.warpAffine()`
* **Brightness & Contrast Adjustments:** `cv2.convertScaleAbs()`
* **Torchvision Transforms:** `transforms.RandomCrop`, `transforms.ColorJitter`

---

## 6. Hands-on Code & Implementations

1. [`cv_basics.py`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%204/cv_basics.py): Demonstrates OpenCV image loading, color conversions, resizing, blurring, Canny edge detection, and augmentations.
   * Output figure: [`cv_transformations.png`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%204/cv_transformations.png)
2. [`cnn_classifier.py`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%204/cnn_classifier.py): Implements a complete PyTorch CNN image classifier trained on synthetic geometric patterns with noise.
   * Output model: [`cnn_model.pth`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%204/cnn_model.pth)
   * Visual training curves: [`cnn_training_curves.png`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%204/cnn_training_curves.png)
   * Prediction grid: [`sample_predictions.png`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%204/sample_predictions.png)
