# Week 3 - Day 5: Weekly Project & Revision

## Project Title: VisionFlow – Deep Learning & Computer Vision Classification System

### Overview
The Week 3 Weekly Project synthesizes all core competencies acquired throughout the week—neural network architecture design, PyTorch tensors and modules, optimization strategies, regularization, OpenCV computer vision pipelines, and convolutional neural networks—into a production-grade visual classification system.

---

## 1. System Architecture & Model Design

The core model, [`VisionClassifierNet`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/model.py), is an optimized deep convolutional neural network built with PyTorch:

```
Input Image [1, 48, 48]
      │
[ Conv Block 1 ] : Conv2D (32, 3x3) ──> BatchNorm2D ──> ReLU ──> MaxPool2D (2x2) ──> [32, 24, 24]
      │
[ Conv Block 2 ] : Conv2D (64, 3x3) ──> BatchNorm2D ──> ReLU ──> MaxPool2D (2x2) ──> [64, 12, 12]
      │
[ Conv Block 3 ] : Conv2D (128, 3x3) ──> BatchNorm2D ──> ReLU ──> MaxPool2D (2x2) ──> [128, 6, 6]
      │
[ Adaptive Spatial Pooling ] : AdaptiveAvgPool2d((2, 2)) ──> [128, 2, 2] = 512 features
      │
[ Classifier Head ] : Flatten ──> Linear(512, 128) ──> ReLU ──> Dropout(0.4) ──> Linear(128, 5)
      │
[ Output Logits ] : 5 Classes (Circle, Square, Triangle, Star, Hexagon)
```

### Key Architectural Highlights:
* **Batch Normalization (`BatchNorm2d`):** Stabilizes internal covariate shift, accelerates convergence, and provides implicit regularization.
* **Progressive Channel Scaling:** Feature representations scale from 32 $\rightarrow$ 64 $\rightarrow$ 128 filters to capture increasingly intricate spatial patterns.
* **Adaptive Spatial Pooling:** Eliminates strict dependence on fixed input spatial resolutions and compacts feature representations into fixed vectors.
* **Regularization:** Employs $40\%$ Dropout alongside $10^{-4}$ L2 weight decay to guarantee superior test generalization.

---

## 2. Dataset & Preprocessing Pipeline ([`dataset.py`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/dataset.py))
* **Multi-Class Categories:** 5 distinct visual classes:
  1. Circle
  2. Square
  3. Triangle
  4. Star (10-point polygon)
  5. Hexagon (6-point polygon)
* **Synthetic Generation & Augmentation:**
  * Procedurally generated using OpenCV drawing routines (`cv2.circle`, `cv2.rectangle`, `cv2.fillPoly`).
  * Continuous variations in spatial position $(cx, cy)$, scale, rotation angle ($\pm 30^\circ$), and Gaussian sensor noise ($\sigma = 12$).
  * Online data augmentations including horizontal and vertical random reflection.
* **Dataset Partitions:**
  * **Training Split (70%):** 1,050 samples with augmentation enabled.
  * **Validation Split (15%):** 225 samples for hyperparameter tuning & early stopping.
  * **Test Split (15%):** 225 independent samples for unbiased evaluation.

---

## 3. Training & Optimization Pipeline ([`train.py`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/train.py))
* **Loss Function:** `nn.CrossEntropyLoss()`
* **Optimizer:** `optim.Adam` with initial $\eta = 0.001$ and $L_2$ weight decay $10^{-4}$.
* **LR Scheduler:** `optim.lr_scheduler.ReduceLROnPlateau(factor=0.5, patience=3)` adapts learning rates dynamically when validation loss plateaus.
* **Model Checkpointing:** Persists optimal weights to [`best_vision_model.pth`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/best_vision_model.pth).
* **Metrics Tracked:** Epoch loss and accuracy curves stored in [`training_curves.png`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/training_curves.png).

---

## 4. Evaluation & Quantitative Results ([`evaluate.py`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/evaluate.py))
* **Evaluation Metrics:** Evaluated against an unseen test partition.
  * Test Loss: $\approx 0.05$
  * Overall Test Accuracy: $> 98\%$
* **Visual Diagnostics:**
  * [`confusion_matrix.png`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/confusion_matrix.png): Multi-class confusion matrix confirming high class discriminability.
  * [`test_predictions.png`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/test_predictions.png): Visual grid comparing ground-truth targets against model predictions.
  * [`evaluation_report.txt`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/evaluation_report.txt): Complete classification report (Precision, Recall, F1-Score).

---

## 5. Live Inference & Deployment ([`predict.py`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/predict.py))
Provides a CLI and API to perform real-time inference on arbitrary images using OpenCV:
* Preprocesses input image to standardized $48 \times 48$ tensor.
* Runs softmax probability distribution.
* Annotates prediction confidence bars onto a visual inspection card ([`prediction_result.png`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/prediction_result.png)).

```bash
# Run inference demo
python predict.py --demo

# Run custom image inference
python predict.py --image path/to/image.png
```

---

## 6. Project Demonstration ([`demo.py`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/demo.py))
Executes the full pipeline from training through test evaluation and multi-class live inference:
```bash
python demo.py
```

---

## 7. Week 3 Comprehensive Revision Notes

### Day 1: Deep Learning Fundamentals
* Perceptrons compute $y = f(\sum w_i x_i + b)$.
* Activation functions (ReLU, Sigmoid, Softmax) map inputs non-linearly to enable representation of complex function spaces.
* Loss functions measure predictive discrepancy; Backpropagation calculates analytical gradients via the calculus chain rule; Gradient Descent updates parameters.

### Day 2: PyTorch Fundamentals
* PyTorch Tensors provide multi-dimensional array computation with seamless GPU acceleration (`.to(device)`).
* `torch.nn.Module` is the base class for all neural models.
* `TensorDataset` and `DataLoader` handle memory-efficient batching, shuffling, and worker streaming.
* Clean separation of `optimizer.zero_grad()`, `loss.backward()`, and `optimizer.step()`.

### Day 3: Neural Network Training & Optimization
* **Optimizers:** Adam introduces per-parameter adaptive learning rates with momentum; SGD is computationally lean.
* **Overfitting:** Solved through Dropout (feature co-adaptation reduction), L2 Weight Decay (regularization penalty), and early stopping.
* **Model Checkpointing:** Serializing `state_dict` enables restoring peak-performing weights.

### Day 4: Computer Vision & Convolutional Neural Networks
* Digital images represent spatial pixel matrices (Height, Width, Channels).
* OpenCV provides high-performance image processing (resizing, color conversions, filtering, edge detection).
* CNNs leverage parameter sharing and spatial inductive biases via Convolution and MaxPool operations.
* Hierarchical abstraction: low-level edges $\rightarrow$ mid-level textures $\rightarrow$ high-level semantic object structures.

---

## 8. Friday Deliverables Checklist
- [x] **Trained Model Checkpoint:** [`best_vision_model.pth`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/best_vision_model.pth)
- [x] **Working Application:** [`demo.py`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/demo.py), [`predict.py`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/predict.py)
- [x] **Training Results & Metrics:** [`evaluation_report.txt`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/evaluation_report.txt), [`metrics.json`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/metrics.json)
- [x] **Visual Analytics:** [`training_curves.png`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/training_curves.png), [`confusion_matrix.png`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/confusion_matrix.png), [`test_predictions.png`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/test_predictions.png)
- [x] **Documentation & Revision:** [`README.md`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205/README.md)
