# Week 3 - Day 2: PyTorch Fundamentals

## 1. What is PyTorch?
PyTorch is an open-source machine learning library developed by Facebook's AI Research lab (FAIR). It is widely used for applications such as computer vision and natural language processing. PyTorch provides two high-level features:
*   Tensor computation (like NumPy) with strong GPU acceleration.
*   Deep neural networks built on a tape-based autograd system.

---

## 2. Tensors & Operations
Tensors are the core data structure in PyTorch, similar to NumPy's `ndarray`, but they can run on GPUs to accelerate computing.

### Basic Operations
*   **Creation:** `torch.tensor()`, `torch.zeros()`, `torch.ones()`, `torch.rand()`
*   **Mathematical Operations:** Addition, subtraction, matrix multiplication (`torch.matmul` or `@`), and element-wise multiplication.
*   **Reshaping:** `tensor.view()`, `tensor.reshape()`, `tensor.squeeze()`, `tensor.unsqueeze()`

---

## 3. GPU vs CPU
PyTorch allows seamless switching between CPU and GPU hardware accelerators (CUDA for NVIDIA GPUs, MPS for Apple Silicon).
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tensor = torch.tensor([1.0, 2.0]).to(device)
```

---

## 4. Datasets and DataLoaders
PyTorch provides standard data primitives to decouple data loading from model training:
*   `Dataset` (`torch.utils.data.Dataset` / `TensorDataset`): Encapsulates samples and their corresponding labels.
*   `DataLoader` (`torch.utils.data.DataLoader`): Wraps an iterable around the `Dataset` to enable automatic batching, shuffling, and multi-threaded data loading.

```python
from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```

---

## 5. Building Models with `nn.Module`
All neural network architectures in PyTorch subclass `torch.nn.Module`:
1.  **`__init__()`**: Define network layers and operations (e.g., `nn.Linear`, `nn.ReLU`, `nn.Sigmoid`).
2.  **`forward(x)`**: Define the forward computation flow taking input tensor `x` and returning output logits or activations.

```python
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.hidden = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        self.output = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.output(self.relu(self.hidden(x))))
```

---

## 6. Loss Functions and Optimizers
*   **Loss Functions (`torch.nn`):** Quantify prediction error:
    *   `nn.BCELoss()`: Binary Cross Entropy for binary classification.
    *   `nn.CrossEntropyLoss()`: Softmax + Negative Log Likelihood for multi-class classification.
    *   `nn.MSELoss()`: Mean Squared Error for regression.
*   **Optimizers (`torch.optim`):** Update model weights via gradient descent:
    *   `optim.SGD(model.parameters(), lr=0.1)`
    *   `optim.Adam(model.parameters(), lr=0.001)`

---

## 7. The Training & Validation Loop
A complete PyTorch training iteration consists of:
1.  Set model to training mode: `model.train()`
2.  Clear previous gradients: `optimizer.zero_grad()`
3.  Forward pass: `predictions = model(batch_X)`
4.  Compute loss: `loss = criterion(predictions, batch_y)`
5.  Backward pass (Autograd): `loss.backward()`
6.  Update parameters: `optimizer.step()`
7.  Validation pass with disabled gradient tracking: `with torch.no_grad(): model.eval()`

---

## 8. Saving and Loading Models
Weights and parameters are saved as a dictionary (State Dict):
*   **Save:** `torch.save(model.state_dict(), "simple_model.pth")`
*   **Load:**
    ```python
    model = SimpleClassifier()
    model.load_state_dict(torch.load("simple_model.pth"))
    model.eval()
    ```

---

## 9. Hands-on Execution
The accompanying script [`pytorch_demo.py`](file:///home/ali-son-of-zulfiqar/Documents/AI%20Engineer%20Internsihp_%20Al%20Aziz%20Technologies/Week%203/Day%202/pytorch_demo.py) trains a synthetic binary classifier, validates it, saves the weights, loads them back, and tests single-sample inference.
