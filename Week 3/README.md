# Week 3 - Deep Learning, PyTorch & Computer Vision

**Objective:** Master neural network architectures, PyTorch tensor computation and autograd workflows, deep learning optimization and regularization techniques, OpenCV computer vision pipelines, and convolutional neural networks (CNNs).

## Daily Breakdown

### Day 1: Deep Learning Fundamentals
- What is Deep Learning, Artificial Neurons, Perceptrons
- Network layers (Input, Hidden, Output), Weights, Biases
- Activation Functions: ReLU, Sigmoid, Softmax
- Forward propagation, Loss functions (MSE, BCE, CrossEntropy)
- Backpropagation and Gradient Descent
- Epochs, batch size, and learning rate
- **Deliverable:** Functional artificial neuron and feedforward neural network in Python ([`Week 3/Day 1`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%201))

**Progress:** Day 1 complete. ✓

### Day 2: PyTorch Fundamentals
- PyTorch core architecture, Tensors, and Tensor mathematical operations
- Hardware acceleration: GPU vs CPU (`torch.device`)
- Data abstractions: `Dataset`, `TensorDataset`, and `DataLoader`
- Defining models with `nn.Module` and layers
- Loss functions and Optimizers (`optim.SGD`, `optim.Adam`)
- Training loop, autograd, and validation loop
- Model state dict saving and loading (`torch.save` / `torch.load`)
- **Deliverable:** Trained and evaluated PyTorch neural network with model persistence ([`Week 3/Day 2`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%202))

**Progress:** Day 2 complete. ✓

### Day 3: Neural Network Training & Optimization
- End-to-end training and validation workflow
- Loss calculation and convergence dynamics
- Optimizers in-depth: SGD with Momentum vs. Adam
- Learning rate schedules, epochs, and batch sizing
- Diagnosing overfitting: Training loss vs. Validation loss divergence
- Regularization methods: Dropout layers and L2 Weight Decay
- Model checkpointing based on minimal validation loss
- Training visualization: Plotting loss and accuracy curves
- **Deliverable:** Comparative training experiments with checkpointing and loss/accuracy curves ([`Week 3/Day 3`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%203))

**Progress:** Day 3 complete. ✓

### Day 4: Computer Vision Fundamentals & CNNs
- Introduction to Computer Vision, image tensor representation `[B, C, H, W]`
- Image preprocessing with OpenCV: BGR/RGB/Gray conversions, resizing, smoothing, normalization
- Spatial filtering and Canny edge detection
- Convolutional Neural Network (CNN) principles: Convolution kernels, stride, padding
- Downsampling and translation invariance with Max Pooling
- Hierarchical feature extraction: low-level edges to high-level semantic shapes
- Overview of Object Detection (bounding boxes, anchor boxes, IoU, mAP)
- Image augmentation: horizontal/vertical flips, rotations, brightness shifts
- **Deliverable:** OpenCV preprocessing suite and PyTorch CNN image classifier ([`Week 3/Day 4`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%204))

**Progress:** Day 4 complete. ✓

### Day 5: Weekly Project & Revision
- **Project:** VisionFlow – Deep Learning & Computer Vision Classification System
- Multi-class visual dataset generation with OpenCV noise and geometric variation
- Deep CNN model with Conv2D, BatchNorm2D, ReLU, MaxPool2D, Adaptive Pooling, and Dropout
- Training pipeline with dynamic LR scheduling and model checkpointing (`best_vision_model.pth`)
- Comprehensive test evaluation: Loss, Accuracy, Classification Report, Confusion Matrix
- Real-time inference engine (`predict.py`) with visual prediction cards
- Complete Week 3 conceptual revision notes
- **Deliverable:** End-to-end working vision classification application, trained weights, evaluation report, and visual diagnostics ([`Week 3/Day 5`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205))

**Progress:** Day 5 complete. ✓

---

## Week 3 Deliverables Summary
- [x] **Day 1:** Deep Learning Fundamentals ([`Week 3/Day 1`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%201))
- [x] **Day 2:** PyTorch Fundamentals ([`Week 3/Day 2`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%202))
- [x] **Day 3:** Neural Network Training & Optimization ([`Week 3/Day 3`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%203))
- [x] **Day 4:** Computer Vision Fundamentals & CNNs ([`Week 3/Day 4`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%204))
- [x] **Day 5:** Weekly Project & Revision ([`Week 3/Day 5`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%205))

---

**Status:** Week 3 Complete (Days 1–5 Completed). ✓
