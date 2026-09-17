# Week 3 - Day 3: Neural Network Training & Optimization

## Overview
Day 3 focuses on practical neural network training dynamics, diagnosing and mitigating overfitting, understanding gradient-based optimization algorithms (SGD vs. Adam), implementing regularization strategies (Dropout and L2 Weight Decay), saving best-performing model checkpoints, and visualizing training curves.

---

## 1. The Neural Network Training Workflow
The complete end-to-end training cycle follows these standardized stages:

```
[ Data Preparation ] ──> [ Forward Pass ] ──> [ Loss Calculation ] ──> [ Backward Pass (Autograd) ] ──> [ Optimizer Step ] ──> [ Validation & Checkpointing ]
```

1. **Forward Propagation:** The input batch $\mathbf{X}$ is passed through the network layers to compute predicted class logits $\hat{\mathbf{y}}$.
2. **Loss Calculation:** The difference between predicted probabilities and ground truth labels is evaluated using a loss function.
3. **Backpropagation:** Gradients of the loss with respect to all trainable parameters ($\nabla_\theta \mathcal{L}$) are computed using the chain rule:
   $$\frac{\partial \mathcal{L}}{\partial w_{ij}} = \frac{\partial \mathcal{L}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z_j} \cdot \frac{\partial z_j}{\partial w_{ij}}$$
4. **Weight Update:** The optimizer updates network parameters $\theta \leftarrow \theta - \eta \cdot \Delta \theta$.
5. **Validation:** Evaluating model generalization on unseen data with gradients disabled (`torch.no_grad()`).

---

## 2. Loss Functions
* **Cross-Entropy Loss (`nn.CrossEntropyLoss`):** Combines `LogSoftmax` and Negative Log-Likelihood (`NLLLoss`) into a single numerically stable function:
  $$\mathcal{L}_{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$
* **Binary Cross-Entropy (`nn.BCELoss` / `nn.BCEWithLogitsLoss`):** Used for single-label or multi-label binary tasks:
  $$\mathcal{L}_{BCE} = -[y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})]$$
* **Mean Squared Error (`nn.MSELoss`):** Standard for continuous regression outputs:
  $$\mathcal{L}_{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2$$

---

## 3. Optimizers: SGD vs. Adam

| Feature | Stochastic Gradient Descent (SGD) | Adam (Adaptive Moment Estimation) |
| :--- | :--- | :--- |
| **Update Rule** | $w \leftarrow w - \eta \nabla \mathcal{L}$ | Tracks first moment (mean) and second moment (uncentered variance) |
| **Momentum** | Adds velocity vector ($v \leftarrow \gamma v + \eta \nabla \mathcal{L}$) to overcome local minima | Inherent first-order momentum ($m_t$) and second-order RMSprop ($v_t$) |
| **Learning Rate** | Uniform across all dimensions | Per-parameter adaptive learning rate |
| **Convergence Speed** | Slower; sensitive to learning rate tuning | Rapid initial convergence; robust default (`lr=0.001` or `lr=0.005`) |
| **Use Case** | Often achieves slightly better generalization in vision with LR scheduling | Standard default choice for modern deep learning architectures |

```python
# SGD with Momentum
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Adam Optimizer
optimizer_adam = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

---

## 4. Hyperparameters: Learning Rate, Epochs, and Batches
* **Learning Rate ($\eta$):** The step size towards the loss minimum. Too large causes divergence or oscillation; too small leads to slow convergence or getting trapped in saddle points.
* **Epoch:** A complete iteration through the entire training dataset.
* **Batch Size:** The number of samples processed before updating model weights. Mini-batches (e.g., 32, 64) strike the ideal balance between memory efficiency and stochastic gradient regularizing effects.

---

## 5. Overfitting & Regularization Techniques

### What is Overfitting?
Overfitting occurs when a neural network memorizes training data noise and idiosyncrasies rather than learning generalizable underlying patterns.
* **Diagnosis:** Training loss steadily decreases while validation loss plateaus and begins increasing; large gap between training accuracy (e.g., 99%) and validation accuracy (e.g., 78%).

### Regularization Solutions
1. **Dropout (`nn.Dropout(p)`):**
   * During training, randomly zeros out activations with probability $p$.
   * Forces neurons to learn robust, co-independent features.
   * Automatically disabled during evaluation via `model.eval()`.
2. **L2 Regularization (Weight Decay):**
   * Adds a penalty proportional to the squared magnitude of weights:
     $$\mathcal{L}_{reg} = \mathcal{L}_{orig} + \frac{\lambda}{2}\sum w^2$$
   * Penalizes overly complex decision boundaries and large weights.
   * Directly specified via `weight_decay=1e-3` in PyTorch optimizers.
3. **Early Stopping & Checkpointing:**
   * Monitoring validation loss and saving only the model weights that achieve the minimum validation error.

---

## 6. Model Checkpointing
PyTorch allows persisting model checkpoints containing weights, optimizer state, and current training epoch:
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': best_val_loss,
}
torch.save(checkpoint, 'best_checkpoint.pth')
```

---

## 7. Hands-on Experiments & Results

The experiment script [`train_experiments.py`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%203/train_experiments.py) benchmarks three distinct model setups:
1. **Baseline Model (No Regularization):** Suffers from overfitting as training loss drops while validation loss diverges.
2. **Regularized Model (Dropout + Weight Decay):** Maintains low generalization gap and stable validation accuracy.
3. **SGD with Momentum:** Compares convergence trajectory and stability against Adam.

### Outputs Generated:
- [`training_curves.png`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%203/training_curves.png): 4-panel visual comparison of loss and accuracy trajectories.
- [`training_results.json`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%203/training_results.json): Quantitative summary metrics across all experiments.
- [`best_checkpoint.pth`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%203/best_checkpoint.pth): Serialized state dictionary of the top-performing model.

### How to Run:
```bash
python train_experiments.py
```
